"""Make the models again from the models as they were before these steps, running only the steps whose code changed.

    python tools/build_models.py [--all] [--from STEP] [--until STEP] [--skip STEP,STEP] [--keep STEP,STEP] [--install] [--beamng] [--profile]

The track's models in track/models (and what AC's layouts take of them in track/ac: their model lists and AI
lines) are made by a chain of steps, each building on the one before, from the hand-made models in source/models
and AC's own files in source/track, copied there first (the steps are not made to run twice over their own work)
(STEPS:
terrain (the ground's heights, from tools/reference/terrain: the original heights, the filtered, the edits; first, so
the roads and everything after lie on it, and an edit to it runs them all), scan_materials, build_roads, restyle_paddock, drape_track, lay_pavers, merge_floor,
build_outside --paddock (the paddock's buildings), place_trees (the trees where track.json places them), place_objects (what the road editor places, from
tools/reference/objects.json), check_normals, check_holes; merge_floor makes the terrain (the roads, the rumbles, the ground) one
surface, proved as it is made, and stops the build at the first merge that does not come out whole; the last
checks the paint lies on the roads. What stands on the map (the pit boxes, the grid, the poles, the tyre walls,
the buildings) is placed by the road editor, top-race-editor, as data: not driven on, never checked for gaps). This starts from the hand-made models and runs them in order, but keeps, in
build/cache, the files each step changed: a step whose code (its script and the tools it imports,
and its data) and whose input (everything before it) are as they were last time is not run, its
files are put back as it left them. So a change to the buildings runs build_outside and the steps
after it; a change to the roads, restyle_paddock and on. Then the AC track is built (only the
models that changed, as build_tracks.py does), --install copies it into the game, --beamng packs
the BeamNG archive too.

--all runs every step; --from STEP runs that one and those after it, whatever the cache holds;
--until STEP stops after that one (the models as it leaves them, no track built: to work on the next);
--skip STEP,STEP leaves those out (what they make as the models had it before them; the steps after them
cached apart from a build with them).
--keep STEP,STEP puts back what those made last time without running them, whatever changed since (for a change
known not to touch what they make: a quick try, not a build to ship; the steps after them run as their keys say);
but only where what it changed is now as it was before it last ran (each file's content, kept with it): otherwise
it runs (put back, it would undo a change a step before it made since).
One build runs at a time: build/build.lock holds the running one's pid, start and arguments (JSON), and
another stops while that pid lives (a lock whose run is gone is taken over).
Each step's output goes to build/cache/<step>.log; with --profile each step run is timed by
Python's profiler too, into build/cache/<step>.prof (python -m pstats build/cache/<step>.prof).
"""
import ast, glob, hashlib, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
CACHE = os.path.join(ROOT, 'build', 'cache')
HAND = os.path.join(ROOT, 'source', 'models')        # (the hand-made models the steps start from)
MADE_DIR = os.path.join(ROOT, 'track')                # (what they make: track/models, and AC's own in track/ac)
AC_FILES = ('*/ai/*.ai', '*/data/*.ai')   # (AC's files of source/track the steps change, copied to track/ac: the AI lines)
# each step: its script, its arguments, and the data it reads beyond the models (globs from ROOT)
STEPS = [('terrain', ['build'], ['tools/reference/track.json', 'tools/reference/terrain/terrain.json', 'tools/reference/terrain/*.npz',   # (the ground, first: the rest lies on it)
                                 'tools/reference/elevation/google_*.json', 'tools/reference/surroundings/*.json',
                                 'tools/reference/surroundings/*.jpg', 'source/textures/GoogleMap.jpg']),
         ('scan_materials', [], ['tools/reference/materials/**/*']),
         ('build_roads', [], ['tools/reference/track.json', 'tools/reference/roads.json', 'tools/reference/retired.json']),
         ('restyle_paddock', [], []),
         ('drape_track', [], ['tools/reference/surroundings/*', 'tools/reference/elevation/*']),
         ('lay_pavers', [], []),
         ('merge_floor', [], ['tools/reference/track.json', 'tools/reference/roads.json']),
         ('build_outside', ['--paddock'], ['tools/reference/site/*', 'tools/reference/*.png']),   # (the paddock's buildings, on the ground as it lies now)
         ('place_trees', [], ['tools/reference/track.json', 'tools/reference/trees.json', 'tools/reference/templates/*']),   # (the trees where they are placed)
         ('place_objects', [], ['tools/reference/track.json', 'tools/reference/objects.json']),   # (what the road editor places: spots, poles, marks)
         ('check_normals', ['--fix'], []),
         ('check_holes', [], [])]          # (the last: no holes, or no track built from them)
# what the steps change, watched for changes (the rest of source/ and the reference they write)
STEP_DATA = {s: d for s, _, d in STEPS}
WATCH = ['source', 'track', os.path.join('tools', 'reference', 'elevation'), os.path.join('tools', 'reference', 'terrain')]
# made by the steps outside track/, out before the first
MADE = ['tools/reference/elevation/draped.json']
NOT_INPUT = ('tools/reference/elevation/draped.json',)


def git(*args):
    return subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def imports(name, seen):
    """The tools a tool imports, itself with them, its name to its path, all the way down."""
    path = os.path.join(TOOLS, name + '.py')
    if name in seen or not os.path.exists(path):
        return seen
    seen[name] = path
    tree = ast.parse(open(path, encoding='utf-8').read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports(a.name.split('.')[0], seen)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            imports(n.module.split('.')[0], seen)
    return seen


def data_print(f, rel):
    """A data file's part of a step's key: a description's (.json) by what it says (an edit that keeps its
    size is still an edit); anything else, big (pictures, scans), by its name and size (not its time: some
    steps write their data again)."""
    if f.endswith('.json'):
        return '%s %s' % (rel, hashlib.sha1(open(f, 'rb').read()).hexdigest())
    return '%s %d' % (rel, os.path.getsize(f))


def code_key(name, args, data):
    """A step's code and data as one hash: its scripts' contents, its data's names and sizes."""
    h = hashlib.sha1(json.dumps(args).encode())
    for _, path in sorted(imports(name, {}).items()):
        h.update(path.encode())
        h.update(open(path, 'rb').read())
    for pattern in data:
        for f in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
            rel = os.path.relpath(f, ROOT).replace(os.sep, '/')
            if os.path.isfile(f) and rel not in NOT_INPUT:
                h.update(data_print(f, rel).encode())
    return h.hexdigest()


def code_files(name, args, data):
    """What a step's key is made of, each part's own fingerprint: {what: sha1}."""
    out = {'args': hashlib.sha1(json.dumps(args).encode()).hexdigest()}
    for _, path in sorted(imports(name, {}).items()):
        out[os.path.relpath(path, ROOT).replace(os.sep, '/')] = hashlib.sha1(open(path, 'rb').read()).hexdigest()
    for pattern in data:
        for f in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
            rel = os.path.relpath(f, ROOT).replace(os.sep, '/')
            if os.path.isfile(f) and rel not in NOT_INPUT:
                out[rel] = data_print(f, rel)
    return out


_SUMS = {}


def digest(rel, stamp):
    """A watched file's content's sha1 (kept by its size and time: a file not touched is not read again)."""
    key = (rel, tuple(stamp))
    if key not in _SUMS:
        h = hashlib.sha1()
        with open(os.path.join(ROOT, rel), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                h.update(chunk)
        _SUMS[key] = h.hexdigest()
    return _SUMS[key]


def state():
    """Every watched file, its size and time."""
    out = {}
    for top in WATCH:
        for dirpath, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                p = os.path.join(dirpath, f)
                st = os.stat(p)
                out[os.path.relpath(p, ROOT).replace(os.sep, '/')] = (st.st_size, st.st_mtime_ns)
    return out


def reset():
    """track/ made again from the hand-made: the models of source/models in track/models, AC's files the steps change
    (AC_FILES of source/track) in track/ac; what the steps make out."""
    for sub in ('models', 'ac'):
        shutil.rmtree(os.path.join(MADE_DIR, sub), ignore_errors=True)
    shutil.copytree(HAND, os.path.join(MADE_DIR, 'models'))
    src = os.path.join(ROOT, 'source', 'track')
    for pattern in AC_FILES:
        for f in glob.glob(os.path.join(src, pattern)):
            dst = os.path.join(MADE_DIR, 'ac', os.path.relpath(f, src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(f, dst)
    for f in MADE:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            os.remove(p)


def base_key():
    """What the steps start from, as a key: each hand-made model's and AC file's name, size and time."""
    src = os.path.join(ROOT, 'source', 'track')
    files = sorted(glob.glob(os.path.join(HAND, '*'))) + sorted(f for p in AC_FILES for f in glob.glob(os.path.join(src, p)))
    return hashlib.sha1(json.dumps([(os.path.relpath(f, ROOT).replace(os.sep, '/'), os.path.getsize(f), os.stat(f).st_mtime_ns)
                                    for f in files]).encode()).hexdigest()


def restore(step):
    """A step's files put back as it left them."""
    d = os.path.join(CACHE, step)
    m = json.load(open(os.path.join(d, 'manifest.json')))
    for rel in m['changed']:
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(d, 'files', rel), dst)
    for rel in m['deleted']:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            os.remove(p)


def run(step, args, key, profile=False):
    """A step run, the files it changed kept for next time. Its time."""
    before = state()
    sums = {f: digest(f, v) for f, v in before.items()}
    t = time.time()
    log = os.path.join(CACHE, step + '.log')
    with open(log, 'w', encoding='utf-8') as out:
        prof = ['-m', 'cProfile', '-o', os.path.join(CACHE, step + '.prof')] if profile else []
        r = subprocess.run([sys.executable] + prof + [os.path.join(TOOLS, step + '.py')] + args, cwd=ROOT, stdout=out,
                           stderr=subprocess.STDOUT)
    if r.returncode != 0 and step != 'check_normals':        # (it exits 1 for the faces it leaves, as the karts')
        print(open(log, encoding='utf-8', errors='replace').read()[-3000:])
        sys.exit('%s failed (exit %d): %s' % (step, r.returncode, os.path.relpath(log, ROOT)))
    took = time.time() - t
    after = state()
    changed = sorted(f for f, v in after.items() if before.get(f) != v)
    deleted = sorted(f for f in before if f not in after)
    d = os.path.join(CACHE, step)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for rel in changed:
        dst = os.path.join(d, 'files', rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    json.dump(dict(key=key, changed=changed, deleted=deleted, seconds=round(took, 1), code=code_files(step, args, STEP_DATA[step]),
                   inputs={f: sums.get(f) for f in changed}),
              open(os.path.join(d, 'manifest.json'), 'w'), indent=1)
    return took


def main():
    args = sys.argv[1:]
    names = [s for s, _, _ in STEPS]
    start = len(STEPS)
    if '--all' in args:
        start = 0
    if '--from' in args:
        step = args[args.index('--from') + 1]
        if step not in names:
            sys.exit('no step %s: %s' % (step, ', '.join(names)))
        start = names.index(step)
    if '--install' in args and 'acs.exe' in subprocess.run(['tasklist'], capture_output=True, text=True).stdout.lower():
        sys.exit('Assetto Corsa is running: close it before --install.')
    skip = set(args[args.index('--skip') + 1].split(',')) if '--skip' in args else set()
    if skip - set(names):
        sys.exit('no step %s: %s' % (', '.join(sorted(skip - set(names))), ', '.join(names)))
    keep = set(args[args.index('--keep') + 1].split(',')) if '--keep' in args else set()
    if keep - set(names):
        sys.exit('no step %s: %s' % (', '.join(sorted(keep - set(names))), ', '.join(names)))
    until = args[args.index('--until') + 1] if '--until' in args else None
    if until and until not in names:
        sys.exit('no step %s: %s' % (until, ', '.join(names)))
    os.makedirs(CACHE, exist_ok=True)
    reset()
    key = base_key()
    t0 = time.time()
    before = []                                                # (each step done so far: when it was made, what it changed)
    for i, (step, sargs, data) in enumerate(STEPS):
        if step in skip:
            key = hashlib.sha1((key + 'skipped ' + step).encode()).hexdigest()
            print('%-16s skipped' % step)
            if step == until:
                break
            continue
        key = hashlib.sha1((key + code_key(step, sargs, data)).encode()).hexdigest()
        mf = os.path.join(CACHE, step, 'manifest.json')
        clash = []
        if step in keep and os.path.exists(mf):
            was = json.load(open(mf))
            if 'inputs' in was:
                now = state()
                clash = sorted(f for f in was['changed'] if (digest(f, now[f]) if f in now else None) != was['inputs'].get(f))
            else:                                               # (made before it kept them: by when the steps before it were made)
                made_at = os.path.getmtime(mf)
                clash = sorted({f for t, files in before for f in files if t > made_at} & set(was.get('changed', [])))
        if clash:
            print('%-16s runs (not kept): a step before it changed %s' % (step, ', '.join(os.path.basename(f) for f in clash[:4])))
        if step in keep and os.path.exists(mf) and not clash:
            restore(step)
            key = json.load(open(mf))['key']                    # (the steps after it as they were after it last time)
            print('%-16s kept as last made' % step)
        elif i < start and os.path.exists(mf) and json.load(open(mf))['key'] == key:
            restore(step)
            print('%-16s as before' % step)
        else:
            if i < start and os.path.exists(mf):
                # (why it runs: what of it is not as it was, or, all of it as it was, a step before it ran)
                was = json.load(open(mf)).get('code') or {}
                now = code_files(step, sargs, data)
                diff = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k)) if was else ['(no record of its parts)']
                print('%-16s runs: %s' % (step, ', '.join(diff[:8]) or 'a step before it ran'))
            start = min(start, i)                              # (and every step after it)
            print('%-16s %5.0f s' % (step, run(step, sargs, key, '--profile' in args)))

        if os.path.exists(mf):
            before.append((os.path.getmtime(mf), set(json.load(open(mf)).get('changed', []))))
        if step == until:
            break
    if until:
        print('stopped after %s in %.0f s' % (until, time.time() - t0))
        return
    build = [sys.executable, os.path.join(TOOLS, 'build_tracks.py'), 'ac'] + (['--install'] if '--install' in args else [])
    subprocess.run(build, cwd=ROOT, check=True)
    if '--beamng' in args:
        subprocess.run([sys.executable, os.path.join(TOOLS, 'build_tracks.py'), 'beamng', '--pack'], cwd=ROOT, check=True)
    print('models made in %.0f s' % (time.time() - t0))


LOCK = os.path.join(ROOT, 'build', 'build.lock')    # (one build at a time, whoever starts it)


def alive(pid):
    out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/NH'], capture_output=True, text=True).stdout
    return str(pid) in out.split()


def lock():
    """build/build.lock made, holding this run's pid, start and arguments; stops if another build holds it
    (a lock left by a run that is gone, as a crash leaves it, is taken over)."""
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.load(open(LOCK))
            except (OSError, ValueError):
                held = {}
            if isinstance(held.get('pid'), int) and alive(held['pid']):
                sys.exit('another build is running (pid %d, since %s: %s); wait for it, or remove %s if it is not'
                         % (held['pid'], held.get('started'), ' '.join(held.get('args', [])), LOCK))
            os.remove(LOCK)
            continue
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'args': sys.argv[1:]}, f)
        return
    sys.exit('could not take %s' % LOCK)


if __name__ == '__main__':
    lock()
    try:
        main()
    finally:
        os.remove(LOCK)
