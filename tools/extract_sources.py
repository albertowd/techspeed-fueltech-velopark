"""Turn the built track in content/ into the track's sources, once:

    python tools/extract_sources.py [track folder] [--force]

    source/models/<name>.glb     each kn5, as a glTF binary (see gltf_source.py)
    source/textures/<file>       their textures, each once; one that two models hold
                                 different versions of goes in source/textures/<model>/
    source/track/                the rest of the track as it ships: layouts, AI lines, ui,
                                 the Custom Shaders Patch config, README and changelog

Every model is read back and checked to give the same kn5 bytes. It stops if source/ is there
already, as that is the track from then on; --force writes over it.
"""
import hashlib, os, shutil, sys, time

import gltf_source
import kn5tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'source')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    track = args[0] if args else os.path.join(ROOT, 'content', 'tracks', 'techspeed_velopark')
    if os.path.exists(SOURCE) and '--force' not in sys.argv:
        sys.exit('%s is there already: it is the track now. Pass --force to write over it.' % SOURCE)
    models, textures, rest = (os.path.join(SOURCE, d) for d in ('models', 'textures', 'track'))
    for d in (models, textures, rest):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    files = sorted(os.listdir(track))
    kn5s = [f for f in files if f.lower().endswith('.kn5')]
    # the main model first, so the texture it holds of a name is the one without a folder
    kn5s.sort(key=lambda f: (f != 'velopark.kn5', f))
    held = {}           # texture file name -> hash of what source/textures/<name> holds
    t0 = time.time()
    for fn in kn5s:
        stem = os.path.splitext(fn)[0]
        raw = open(os.path.join(track, fn), 'rb').read()
        k = kn5tree.parse(raw)

        def uri(t):
            h = hashlib.sha1(t['data']).hexdigest()
            if held.setdefault(t['name'], h) == h:
                path = os.path.join(textures, t['name'])
                rel = '../textures/' + t['name']
            else:
                os.makedirs(os.path.join(textures, stem), exist_ok=True)
                path = os.path.join(textures, stem, t['name'])
                rel = '../textures/%s/%s' % (stem, t['name'])
            if not os.path.exists(path):
                open(path, 'wb').write(t['data'])
            return rel
        doc, binary = gltf_source.to_gltf(k, uri)
        out = os.path.join(models, stem + '.glb')
        gltf_source.write_glb(out, doc, binary)
        back = kn5tree.dump(gltf_source.load_kn5(out))
        if back != raw:
            sys.exit('%s does not come back the same from %s' % (fn, out))
        print('%-32s %4d nodes, %2d textures, same bytes back' % (fn, sum(1 for _ in kn5tree.walk(k['root'])), len(k['textures'])))
    for f in files:
        if f in kn5s:
            continue
        src = os.path.join(track, f)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(rest, f))
    print('%d models, %d texture files and the track files in %s, in %.0f s' % (
        len(kn5s), sum(len(fs) for _, _, fs in os.walk(textures)), SOURCE, time.time() - t0))


if __name__ == '__main__':
    main()
