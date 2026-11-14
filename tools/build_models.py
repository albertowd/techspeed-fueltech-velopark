"""Make the models again from the committed ones, running only the steps whose code changed.

    python tools/build_models.py [--all] [--from STEP] [--install] [--beamng] [--profile]

The models in source/models are made by a chain of steps, each building on the one before (STEPS:
organize_models, scan_materials, restyle_paddock, drape_track, build_outside, place_karts,
check_normals). This starts from the committed models and runs them in order, but keeps, in
build/cache, the files each step changed: a step whose code (its script and the tools it imports,
and its data) and whose input (everything before it) are as they were last time is not run, its
files are put back as it left them. So a change to the buildings runs build_outside and the steps
after it; a change to the roads, restyle_paddock and on. Then the AC track is built (only the
models that changed, as build_tracks.py does), --install copies it into the game, --beamng packs
the BeamNG archive too.

--all runs every step; --from STEP runs that one and those after it, whatever the cache holds.
Each step's output goes to build/cache/<step>.log; with --profile each step run is timed by
Python's profiler too, into build/cache/<step>.prof (python -m pstats build/cache/<step>.prof).
"""
import ast, glob, hashlib, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
CACHE = os.path.join(ROOT, 'build', 'cache')
# each step: its script, its arguments, and the data it reads beyond the models (globs from ROOT)
STEPS = [('organize_models', [], []),
         ('scan_materials', [], ['tools/reference/materials/**/*']),
         ('restyle_paddock', [], []),
         ('drape_track', [], ['tools/reference/surroundings/*', 'tools/reference/elevation/*']),
         ('build_outside', [], ['tools/reference/site/*', 'tools/reference/*.png']),
         ('place_karts', [], []),
         ('check_normals', ['--fix'], [])]
# what the steps change, watched for changes (the rest of source/ and the reference they write)
WATCH = ['source', os.path.join('tools', 'reference', 'elevation')]
# made by the steps, out before the first (as the committed models were before them)
MADE = ['source/models/velopark_lines.glb', 'source/models/velopark_runoffs.glb', 'source/models/velopark_trees.glb',
        'source/models/velopark_trees.json', 'tools/reference/elevation/draped.json']
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


def code_key(name, args, data):
    """A step's code and data as one hash: its scripts' contents, its data's sizes and times."""
    h = hashlib.sha1(json.dumps(args).encode())
    for _, path in sorted(imports(name, {}).items()):
        h.update(path.encode())
        h.update(open(path, 'rb').read())
    for pattern in data:
        for f in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
            rel = os.path.relpath(f, ROOT).replace(os.sep, '/')
            if os.path.isfile(f) and rel not in NOT_INPUT:
                st = os.stat(f)
                h.update(('%s %d %d' % (rel, st.st_size, st.st_mtime_ns)).encode())
    return h.hexdigest()


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
    """The models, the track's config and the layouts' AI lines as committed; what the steps make out."""
    ai = git('ls-files', 'source/track/*.ai').split()
    git('checkout', '--', 'source/models', 'source/track/extension/ext_config.ini', *ai)
    for f in MADE:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            os.remove(p)


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
    for rel in changed:
        dst = os.path.join(d, 'files', rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    json.dump(dict(key=key, changed=changed, deleted=deleted, seconds=round(took, 1)),
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
    os.makedirs(CACHE, exist_ok=True)
    reset()
    key = hashlib.sha1(git('rev-parse', 'HEAD:source/models').encode()).hexdigest()
    t0 = time.time()
    for i, (step, sargs, data) in enumerate(STEPS):
        key = hashlib.sha1((key + code_key(step, sargs, data)).encode()).hexdigest()
        mf = os.path.join(CACHE, step, 'manifest.json')
        if i < start and os.path.exists(mf) and json.load(open(mf))['key'] == key:
            restore(step)
            print('%-16s as before' % step)
            continue
        start = min(start, i)                                  # (and every step after it)
        print('%-16s %5.0f s' % (step, run(step, sargs, key, '--profile' in args)))
    build = [sys.executable, os.path.join(TOOLS, 'build_tracks.py'), 'ac'] + (['--install'] if '--install' in args else [])
    subprocess.run(build, cwd=ROOT, check=True)
    if '--beamng' in args:
        subprocess.run([sys.executable, os.path.join(TOOLS, 'build_tracks.py'), 'beamng', '--pack'], cwd=ROOT, check=True)
    print('models made in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
