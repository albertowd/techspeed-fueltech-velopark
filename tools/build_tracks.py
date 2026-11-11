"""Build the track for each game from its sources in source/.

    python tools/build_tracks.py [ac] [beamng] [--layout vp_1500] [--pack] [--install] [--all]

    ac        build/ac/content/tracks/techspeed_fueltech_velopark: each source/models/<name>.glb as
              <name>.kn5, with source/track copied beside them
    beamng    build/beamng/levels/techspeed_fueltech_velopark, from the AC build: one layout
              (--layout, vp_1500 by default), unpacked, to try from BeamNG's mods/unpacked;
              --pack also packs build/dist/techspeed-fueltech-velopark-<version>-beamng.zip
    neither   both

Only what changed since the last build is made again: a model whose .glb or textures are as
they were keeps its kn5, and BeamNG keeps the shapes of the kn5s that stayed the same. --all
makes everything again.

--install copies what changed of the AC build into the game (the folder in release.ps1, or
AC_TRACKS).
"""
import json, os, shutil, sys, time

import gltf_source
import kn5tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = 'techspeed_fueltech_velopark'
SOURCE = os.path.join(ROOT, 'source')
BUILD = os.path.join(ROOT, 'build')
AC_OUT = os.path.join(BUILD, 'ac', 'content', 'tracks', TRACK)
GAME_TRACKS = os.environ.get('AC_TRACKS', r'D:\SteamLibrary\steamapps\common\assettocorsa\content\tracks')


def stamp(path):
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def same_file(a, b):
    if not os.path.exists(b):
        return False
    sa, sb = os.stat(a), os.stat(b)
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def sync(src, dst, keep=lambda rel: False):
    """dst made like src: changed files copied, files src lacks removed, unless keep(rel)."""
    n = 0
    for base, dirs, files in os.walk(src):
        rel = os.path.relpath(base, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            a, b = os.path.join(base, f), os.path.join(dst, rel, f)
            if not same_file(a, b):
                shutil.copy2(a, b)
                n += 1
    for base, dirs, files in os.walk(dst, topdown=False):
        rel = os.path.relpath(base, dst)
        for f in files:
            r = os.path.normpath(os.path.join(rel, f))
            if not os.path.exists(os.path.join(src, r)) and not keep(r):
                os.remove(os.path.join(base, f))
                n += 1
        if rel != '.' and not os.listdir(base) and not os.path.exists(os.path.join(src, rel)):
            os.rmdir(base)
    return n


def build_ac(everything=False):
    t0 = time.time()
    models = os.path.join(SOURCE, 'models')
    os.makedirs(AC_OUT, exist_ok=True)
    state_path = os.path.join(BUILD, 'ac', 'models.json')
    state = {} if everything or not os.path.exists(state_path) else json.load(open(state_path))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(models) if f.endswith('.glb'))
    made = 0
    for stem in stems:
        glb = os.path.join(models, stem + '.glb')
        sig = [stamp(glb)] + [[os.path.relpath(t, SOURCE)] + stamp(t) for t in gltf_source.texture_files(glb)]
        out = os.path.join(AC_OUT, stem + '.kn5')
        if state.get(stem) == sig and os.path.exists(out):
            continue
        t = time.time()
        kn5tree.save(gltf_source.load_kn5(glb), out)
        state[stem] = sig
        made += 1
        print('  %-32s %.1f s' % (stem + '.kn5', time.time() - t))
    state = {k: v for k, v in state.items() if k in stems}
    # the kn5s are this build's own; the rest is source/track as it is
    copied = sync(os.path.join(SOURCE, 'track'), AC_OUT, keep=lambda rel: rel.endswith('.kn5') and os.path.splitext(rel)[0] in stems)
    json.dump(state, open(state_path, 'w'), indent=1)
    print('AC: %d of %d models made, %d track files updated, in %.1f s: %s' % (made, len(stems), copied, time.time() - t0, AC_OUT))


def main():
    args = sys.argv[1:]
    games = [a for a in args if a in ('ac', 'beamng')] or ['ac', 'beamng']
    layout = args[args.index('--layout') + 1] if '--layout' in args else 'vp_1500'
    everything = '--all' in args
    if 'ac' in games or not os.path.exists(AC_OUT):
        build_ac(everything)
    if '--install' in args:
        dest = os.path.join(GAME_TRACKS, TRACK)
        print('AC: %d files updated in %s' % (sync(AC_OUT, dest), dest))
    if 'beamng' in games:
        import export_beamng
        out = os.path.join(BUILD, 'beamng')
        if everything and os.path.exists(os.path.join(out, 'models.json')):
            os.remove(os.path.join(out, 'models.json'))
        t = time.time()
        export_beamng.export(layout, AC_OUT, out, pack='--pack' in args)
        print('BeamNG: %.1f s' % (time.time() - t))


if __name__ == '__main__':
    main()
