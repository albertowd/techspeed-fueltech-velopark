"""Render the 0.5.1 corners straight from the kn5 files build_051.py builds."""
import math, os, sys

import numpy as np
from PIL import Image

import kn5io
import build_preview as bp

TRACK = os.path.join(bp.ROOT, 'build', 'ac', 'velopark-0.5.1', 'content', 'tracks', 'techspeed_velopark')
OUT = os.path.join(bp.OUT, 'render', 'test051')
CACHE = os.path.join(OUT, 'tex')


def ac2b(p):
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)


def load(fn, centre, radius, skip=()):
    bpy = bp.bpy_mod()
    m = kn5io.Model(os.path.join(TRACK, fn))
    mats = {}
    tris = 0
    for mesh in m.meshes:
        if mesh['name'].startswith(skip):
            continue
        v, f = mesh['v'], mesh['f']
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if not len(f):
            continue
        c = v[f, :3].mean(1)
        near = np.linalg.norm(c[:, [0, 2]] - centre, axis=1) < radius
        if not near.any():
            continue
        f = f[near]
        mat = m.materials[mesh['mat']]
        if mat['name'] not in mats:
            tex = mat['tex'].get('txDiffuse')
            path = os.path.join(CACHE, '%s_%s.png' % (fn, mat['name'].replace(' ', '_')))
            if tex and not os.path.exists(path):
                raw = m.textures[tex]
                n = int.from_bytes(raw[4:8], 'little')
                payload = raw[8 + n + 4:]
                tmp = path + '.' + tex.split('.')[-1]
                open(tmp, 'wb').write(payload)
                Image.open(tmp).convert('RGB').save(path)
            mats[mat['name']] = bp.mat_img(mat['name'], path, rough=.8) if tex else bp.mat_flat(mat['name'], (.5, .5, .5))
        used = np.unique(f)
        remap = np.full(len(v), -1)
        remap[used] = np.arange(len(used))
        uv = np.stack([v[used, 6], 1 - v[used, 7]], 1)
        o = bp.add_mesh(mesh['name'], ac2b(v[used, :3]), remap[f], mats[mat['name']], uv=uv, smooth_angle=40)
        o.data.validate()
        tris += len(f)
    return tris


def shot(name, centre, cams, radius=45):
    bpy, sc = bp.reset((1600, 900), exposure=.3)
    bp.world(bp.lin((110, 150, 200)), bp.lin((185, 200, 215)), bp.lin((228, 226, 218)), .8)
    load('velopark.kn5', centre, radius, skip=('OBJ', 'AC_', '1WALL'))
    load('velopark_kerbs.kn5', centre, radius)
    load('velopark_poles.kn5', centre, radius + 30)
    load('velopark_tyres.kn5', centre, radius, skip=('1WALL',))
    bp.sun(200, 35, 3.0, color=(1, .92, .8))
    for k, (loc, look, lens) in enumerate(cams):
        bp.camera(ac2b(np.array([loc]))[0], ac2b(np.array([look]))[0], lens=lens)
        bp.render(sc, os.path.join('test051', '%s_%d' % (name, k)))


if __name__ == '__main__':
    os.makedirs(CACHE, exist_ok=True)
    # positions are in track coordinates: x, height, z
    jobs = dict(
        t3=((193, 37), [((180, 6, 22), (193, -.6, 37), 30),
                        ((205, 1.0, 22), (196, -.7, 38), 34),
                        ((193, 45, 37.1), (193, -.6, 37), 40)]),
        t2=((11, -2), [((-4, 1.2, 14), (9, .1, -3), 30), ((11, 40, -1.9), (11, 0, -2), 40)]),
        kink=((-2, 15), [((-10, 1.0, 40), (-2, 0, 15), 32), ((-2, 30, 15.1), (-2, 0, 15), 40)]),
        t3end=((193, 37), [((186.5, .9, 47.5), (191, -.7, 41.5), 30), ((200, .7, 33), (195.5, -.7, 36), 28)]),
        entry=((-7, 40), [((-14, 1.2, 60), (-7, .1, 40), 30), ((-7, 25, 40.1), (-7, 0, 40), 40)]),
        amateur=((32, 6), [((10, 1.2, 25), (35, .2, 5), 30), ((32, 45, 6.1), (32, 0, 6), 40)]),
    )
    want = sys.argv[1:] or list(jobs)
    for k in want:
        centre, cams = jobs[k]
        shot(k, np.array(centre, float), cams)
