"""The trees of the track's sources down on the ground, their trunks into it.

    python tools/plant_trees.py

A tree is a few crossed boards (ksTree), its texture's trunk running down the middle of them;
but the trunk ends a little over the bottom of its texture (the broad leaf trees' by 3.5% of
it, about 0.75 m on a tree 21 m high), so a tree whose boards stand on the ground, or just in
it, shows its trunk ending in the air. This lowers each tree, as a whole, until the lowest of its
trunk that shows is DEPTH under the ground round the trunk, in every model of source/models with
trees. Trees standing deep enough already are left as they are, and so are those the 0.5.0 track
put out of sight far under the ground. Run again, it finds the trees planted and changes nothing.
"""
import glob, io, os

import numpy as np
from PIL import Image

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
DEPTH = .1           # metres under the ground the lowest of a trunk that shows goes
TRUNK = .5           # metres round a tree's middle the ground is taken, as wide as its trunk
BURIED = 2.          # a tree whose boards stand further under the ground than this is out of sight
SMALL = .005         # metres; a tree that would go down less than this stays
ALPHA = .5           # how opaque a texture's pixel is to show


def bottom_gap(data):
    """How much of a tree texture's height is left under its trunk, as a fraction: the rows
    under the lowest that shows in its middle."""
    a = np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'))[..., 3] / 255
    h, w = a.shape
    rows = np.nonzero((a[:, w // 2 - w // 20:w // 2 + w // 20 + 1] > ALPHA).any(1))[0]
    return (h - 1 - rows.max()) / h if len(rows) else 0.


def pieces(F, n):
    label = np.arange(n)
    while True:
        low = label[F].min(1)
        before = label.copy()
        for k in range(3):
            np.minimum.at(label, F[:, k], low)
        label = label[label]
        if (label == before).all():
            return label


def main():
    files = sorted(glob.glob(os.path.join(bp.MODELS, '*.glb')))
    models = {f: gltf_source.load_kn5(f) for f in files}
    # the ground: every surface a car drives on, of every model
    ground = []
    for k in models.values():
        for n, m in bp.meshes(k):
            if n['name'][:1].isdigit() and 'WALL' not in n['name'] and len(n['faces']):
                ground.append(bp.triangles(n, m))
    ground = bp.Surface(np.concatenate(ground))
    ring = TRUNK * np.c_[np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))]
    ring = np.r_[[[0, 0]], ring]
    total = moved = 0
    for f, k in models.items():
        texd = {t['name']: t['data'] for t in k['textures']}
        gaps = {}
        changed = False
        for n, m in bp.meshes(k):
            mat = k['materials'][n['material']]
            if mat.get('shader') != 'ksTree' or not len(n['faces']):
                continue
            if not np.allclose(m[:3, :3], np.eye(3)):
                print('%s %s is turned or scaled: its trees left as they are' % (os.path.basename(f), n['name']))
                continue
            tex = next((t['name'] for t in mat['textures'] if t['sampler'] == 'txDiffuse'), None)
            if tex not in gaps:
                gaps[tex] = bottom_gap(texd[tex]) if tex in texd else 0.
            V = n['verts']
            P = V[:, :3].astype(float) + m[3, :3]
            label = pieces(n['faces'].astype(int), len(V))
            dy = np.zeros(len(V))
            for t in np.unique(label[n['faces'][:, 0].astype(int)]):
                sel = label == t
                p = P[sel]
                total += 1
                mid = (p[:, [0, 2]].min(0) + p[:, [0, 2]].max(0)) / 2
                g = ground.ys(mid + ring)
                if np.isnan(g).all():
                    continue
                g = np.nanmin(g)
                foot = p[:, 1].min()
                if foot < g - BURIED:
                    continue
                shows = foot + gaps[tex] * np.ptp(p[:, 1])
                down = shows - (g - DEPTH)
                if down >= SMALL:
                    dy[sel] = -down
                    moved += 1
            if (dy != 0).any():
                V[:, 1] += dy.astype(V.dtype)
                bp.sphere(n)
                changed = True
        if changed:
            bp.save(k, f)
            print('trees down in %s' % os.path.relpath(f, ROOT))
    if moved:
        print('%d of %d trees down, their trunks into the ground' % (moved, total))
    else:
        print('the trees are planted already: %d trees' % total)


if __name__ == '__main__':
    main()
