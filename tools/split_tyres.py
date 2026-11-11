"""Gaps in the tyre walls where something stands in them, in source/models/velopark_tyres.glb.

    python tools/split_tyres.py

The tyre wall by the vp_1000's start (TYRES_50) ran through its flag marshal's cones and the
front of his tent (tools/build_outside.py's marshal()). This takes out of the tyre walls each
tyre whose middle stands in one of GAPS (x0, x1, z0, z1), in the near meshes and the far, and
the faces of their walls to hit there, so the wall stops either side. Run again, it finds the
gaps open.
"""
import numpy as np

import build_paths as bp
import gltf_source
import kn5tree
from plant_trees import pieces

GAPS = {'the vp_1000 marshal': (-86.5, -82.5, 93.4, 98.4)}


def inside(c, box):
    x0, x1, z0, z1 = box
    return (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > z0) & (c[:, 1] < z1)


def main():
    k = gltf_source.load_kn5(bp.TYRES)
    taken, emptied = 0, []
    for n, m in bp.meshes(k):
        if 'TYRE' not in n['name'] or not len(n['faces']):
            continue
        V = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
        if not any(inside(V[:, [0, 2]], b).any() for b in GAPS.values()):
            continue
        F = n['faces'].astype(int)
        if n['name'].startswith('1WALL'):
            # a wall to hit: its faces whose middle is in a gap
            c = V[F].mean(1)[:, [0, 2]]
            out = np.zeros(len(F), bool)
            for b in GAPS.values():
                out |= inside(c, b)
        else:
            lab = pieces(F, len(V))
            flab = lab[F[:, 0]]
            ids = np.unique(flab)
            c = np.array([V[lab == t][:, [0, 2]].mean(0) for t in ids])
            gone = np.zeros(len(ids), bool)
            for b in GAPS.values():
                gone |= inside(c, b)
            out = np.isin(flab, ids[gone])
        if not out.any():
            continue
        taken += out.sum()
        if out.all():
            emptied.append(n)
        else:
            bp.split(n, out)
        print('  %-20s %5d faces out' % (n['name'], out.sum()))
    if not taken:
        print('the gaps in the tyre walls are open')
        return
    drop = {id(x) for x in emptied}
    for parent, _ in kn5tree.walk(k['root']):
        parent['children'] = [c for c in parent['children'] if id(c) not in drop]
    bp.save(k, bp.TYRES)


if __name__ == '__main__':
    main()
