"""Build the track for each game from its sources in source/.

    python tools/build_tracks.py [ac] [beamng] [--layout vp_1500] [--pack] [--install] [--all]

    ac        build/ac/content/tracks/techspeed_fueltech_velopark: each track/models/<name>.glb (as
              tools/build_models.py makes them) as <name>.kn5, with source/track copied beside them (the
              layouts' AI lines as track/ac has them), each layout's model list (models_<layout>.ini) as
              track.json lists its models, and the Custom Shaders Patch config
              made from AC's own settings there and the track's lights and screens (tools/ac_lights.py)
    beamng    build/beamng/levels/techspeed_fueltech_velopark, from track/models and the source as
              tools/export_beamng.py makes it: one layout (--layout, vp_1500 by default), unpacked, to
              try from BeamNG's mods/unpacked; --pack also packs
              build/dist/techspeed_fueltech_velopark_beamng.zip
    neither   both

Only what changed since the last build is made again: a model whose .glb or textures are as
they were keeps its kn5, and BeamNG keeps the shapes of the models that stayed the same. --all
makes everything again.

--install copies what changed of the AC build into the game (the folder in release.ps1, or
AC_TRACKS).
"""
import json, os, shutil, sys, time

import build_paths  # noqa: F401  (the material library set for the models: tools/materials.py)
import ac_lights
import ac_places
import track_data
import gltf_source
import kn5tree
import materials

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = 'techspeed_fueltech_velopark'
SOURCE = os.path.join(ROOT, 'source')
MADE = os.path.join(ROOT, 'track')             # (what the model steps make: track/models; AC's files they change, track/ac)
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


def sync(src, dst, keep=lambda rel: False, own=()):
    """dst made like src: changed files copied, files src lacks removed, unless keep(rel); the files own (relative
    paths) neither copied nor removed, written by the build itself."""
    n = 0
    for base, dirs, files in os.walk(src):
        rel = os.path.relpath(base, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            a, b = os.path.join(base, f), os.path.join(dst, rel, f)
            if os.path.normpath(os.path.join(rel, f)) in own:
                continue
            if not same_file(a, b):
                shutil.copy2(a, b)
                n += 1
    for base, dirs, files in os.walk(dst, topdown=False):
        rel = os.path.relpath(base, dst)
        for f in files:
            r = os.path.normpath(os.path.join(rel, f))
            if not os.path.exists(os.path.join(src, r)) and not keep(r) and r not in own:
                os.remove(os.path.join(base, f))
                n += 1
        if rel != '.' and not os.listdir(base) and not os.path.exists(os.path.join(src, rel)):
            os.rmdir(base)
    return n


def build_ac(everything=False):
    t0 = time.time()
    models = os.path.join(MADE, 'models')
    ac_made = os.path.join(MADE, 'ac')
    over = sorted(os.path.normpath(os.path.relpath(os.path.join(b, f), ac_made)) for b, _, fs in os.walk(ac_made) for f in fs
                  if not f.startswith('models_'))
    lists = track_data.layouts()                                       # (each layout's models, track.json's)
    lists_rel = {'models_%s.ini' % lay for lay in lists}
    os.makedirs(AC_OUT, exist_ok=True)
    state_path = os.path.join(BUILD, 'ac', 'models.json')
    state = {} if everything or not os.path.exists(state_path) else json.load(open(state_path))
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(models) if f.endswith('.glb'))
    placed = ac_places.places()                # (the layouts' places: AC's own nodes made from them here)
    made = 0
    for stem in stems:
        glb = os.path.join(models, stem + '.glb')
        sig = [stamp(glb), stamp(materials.LIBRARY) if os.path.exists(materials.LIBRARY) else None,
               json.dumps(placed.get(stem), sort_keys=True) if stem in placed else None] +               [[os.path.relpath(t, SOURCE)] + stamp(t) for t in gltf_source.texture_files(glb)]
        out = os.path.join(AC_OUT, stem + '.kn5')
        if state.get(stem) == sig and os.path.exists(out):
            continue
        t = time.time()
        k = gltf_source.load_kn5(glb)
        if stem in placed:
            ac_places.add(k, placed[stem])
        for tex in k['textures']:
            tex['data'] = materials.ac_dds(tex['name'], tex['data'])      # (kept as PNG: AC's DDS made here)
        kn5tree.save(k, out)
        state[stem] = sig
        made += 1
        print('  %-32s %.1f s' % (stem + '.kn5', time.time() - t))
    state = {k: v for k, v in state.items() if k in stems}
    # the kn5s and the Custom Shaders Patch config are this build's own; the layouts' model lists and AI lines as the
    # steps made them (track/ac); the rest is source/track as it is
    copied = sync(os.path.join(SOURCE, 'track'), AC_OUT, keep=lambda rel: rel.endswith('.kn5') and os.path.splitext(rel)[0] in stems,
                  own=set(over) | lists_rel | {os.path.normpath(ac_lights.REL)})
    for rel in over:
        a, b = os.path.join(ac_made, rel), os.path.join(AC_OUT, rel)
        if not same_file(a, b):
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copy2(a, b)
            copied += 1
    for lay, names in lists.items():
        missing = [m for m in names if m not in stems]
        if missing:
            sys.exit('%s lists %s, which track/models does not have' % (lay, ', '.join(missing)))
        p = os.path.join(AC_OUT, 'models_%s.ini' % lay)
        text = track_data.ac_models(names)
        if not os.path.exists(p) or open(p, 'rb').read().decode('utf-8') != text:
            open(p, 'w', encoding='utf-8', newline='').write(text)
            copied += 1
    copied += ac_lights.write(AC_OUT, models)
    for layout, here in placed.items():
        ac_places.pitboxes(AC_OUT, layout, here.get('pits', 0))
    json.dump(state, open(state_path, 'w'), indent=1)
    print('AC: %d of %d models made, %d track files updated, in %.1f s: %s' % (made, len(stems), copied, time.time() - t0, AC_OUT))


def main():
    args = sys.argv[1:]
    games = [a for a in args if a in ('ac', 'beamng')] or ['ac', 'beamng']
    layout = args[args.index('--layout') + 1] if '--layout' in args else 'vp_1500'
    everything = '--all' in args
    if 'ac' in games:
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
        export_beamng.export(layout, out, pack='--pack' in args)
        print('BeamNG: %.1f s' % (time.time() - t))


if __name__ == '__main__':
    main()
