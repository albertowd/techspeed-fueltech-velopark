"""The paddock's asphalt the 0.5.1 track left round the vp_1000's T3 and T4 made ground, in source/models/velopark_kerbs.glb.

    python tools/clear_runoffs.py

Building 0.5.1 took the paddock's asphalt off the vp_1000's run-offs after K1_T3 and K1_T4 and
laid run-offs of asphalt of its own there (1ROAD_RUNOFF_*), which stay; but it left pieces of
the paddock's asphalt (1ROAD_PADDOCK_REST_*), some of them past the tyre walls, which on the real
track are ground. As the ground has holes where they lie, each becomes ground in their place:
the ground's material and photo, mapped as the ground's is (flat from above, as 1GRASS_0 is,
found from it), named 1GRASS_ so both games drive it as grass, and listed in the Custom Shaders
Patch config's GRASS_MESHES so grass grows on it. And what is left of the paddock's own asphalt
there (source/models/velopark_paddock.glb), under the ground and the run-offs, out of sight but
for its edges, goes: its faces within AREA with another surface over or on them. Run again, it
finds them cleared.
"""
import os, re

import numpy as np

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
KERBS = os.path.join(bp.MODELS, 'velopark_kerbs.glb')
CONFIG = os.path.join(ROOT, 'source', 'track', 'extension', 'ext_config.ini')
LEFT = ('1ROAD_PADDOCK_REST_K1_T3_EXIT', '1ROAD_PADDOCK_REST_K1_T4_EXIT', '1ROAD_PADDOCK_REST_K1_T3')
GROUND = '1GRASS_0'
AREA = (-85., 65., 225., 315.)      # x and z: round K1_T3 and K1_T4
COVER = .5               # metres over or under a face another surface lies, to cover it
PHOTO = 'Background'     # the ground's material


def photo_mapping(track):
    """The ground photo's uv as a function of x and z, fitted to the ground's points."""
    n = next(n for n, m in bp.meshes(track) if n['name'] == GROUND)
    V = n['verts'].astype(float)
    A = np.c_[V[:, 0], V[:, 2], np.ones(len(V))]
    coef = np.linalg.lstsq(A, V[:, 6:8], rcond=None)[0]
    worst = np.abs(A @ coef - V[:, 6:8]).max()
    if worst > 1e-4:
        raise SystemExit('the ground photo is not mapped flat from above (off by %g)' % worst)
    return lambda P: np.c_[P[:, 0], P[:, 2], np.ones(len(P))] @ coef


def under_cover(k, track):
    """The paddock's faces within AREA that another surface covers, taken out; how many."""
    pad = gltf_source.load_kn5(bp.PADDOCK)
    tris = []
    for model in (track, k):
        for n, m in bp.meshes(model):
            if n['name'][:1].isdigit() and 'WALL' not in n['name'] and len(n['faces']):
                t = bp.triangles(n, m)
                c = t[:, :, [0, 2]].mean(1) if len(t) else np.zeros((0, 2))
                t = t[(c[:, 0] > AREA[0] - 5) & (c[:, 0] < AREA[1] + 5) & (c[:, 1] > AREA[2] - 5) & (c[:, 1] < AREA[3] + 5)]
                if len(t):
                    tris.append(t)
    cover = bp.Surface(np.concatenate(tris))
    gone_n = 0
    for n, m in bp.meshes(pad):
        if not n['name'].startswith(bp.ASPHALT) or not len(n['faces']):
            continue
        V = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
        F = n['faces'].astype(int)
        c = V[F].mean(1)
        inside = (c[:, 0] > AREA[0]) & (c[:, 0] < AREA[1]) & (c[:, 2] > AREA[2]) & (c[:, 2] < AREA[3])
        gone = np.zeros(len(F), bool)
        for i in np.nonzero(inside)[0]:
            y = cover.y(c[i, 0], c[i, 2])
            gone[i] = not np.isnan(y) and abs(y - c[i, 1]) < COVER
        if gone.any():
            bp.split(n, gone)
            gone_n += gone.sum()
    if gone_n:
        bp.save(pad, bp.PADDOCK)
    return gone_n


def main():
    k = gltf_source.load_kn5(KERBS)
    track = gltf_source.load_kn5(bp.TRACK)
    nodes = [n for n, m in bp.meshes(k) if n['name'] in LEFT]
    hidden = under_cover(k, track)
    if hidden:
        print('  %d faces of the paddock\'s asphalt under the ground there out' % hidden)
    if not nodes:
        print('the paddock\'s leftovers are cleared already' if not hidden else 'the paddock\'s pieces were ground already')
        return
    uv_of = photo_mapping(track)
    mat = next(i for i, m in enumerate(k['materials']) if m['name'] == PHOTO)
    names = []
    for n in nodes:
        V = n['verts']
        U = uv_of(V[:, :3].astype(float))
        T = gltf_source.tangents(V[:, :3].astype(float), V[:, 3:6].astype(float), U, n['faces'].astype(int))
        n['verts'] = np.c_[V[:, :6], U, T].astype(V.dtype)
        n['material'] = mat
        n['name'] = '1GRASS_WAS_' + n['name'][len('1ROAD_'):]
        names.append(n['name'])
        print('  %s ground' % n['name'])
    bp.save(k, KERBS)
    # grass growing on them
    raw = open(CONFIG, 'rb').read().decode('utf-8')
    m = re.search(r'(?m)^GRASS_MESHES = (.*?)\s*$', raw)
    have = [x.strip() for x in m.group(1).split(',')]
    add = [x for x in names if x not in have]
    if add:
        raw = raw[:m.start(1)] + ', '.join(have + add) + raw[m.end(1):]
        open(CONFIG, 'wb').write(raw.encode('utf-8'))
    print('%d pieces of the paddock\'s asphalt made ground' % len(names))


if __name__ == '__main__':
    main()
