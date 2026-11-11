"""R29's pavers laid flat, the grass round them brought up to them, in source/models/velopark.glb.

    python tools/level_pavers.py

The paved run-off by L1500_C (1KERB_29, R29 on the track map) rises with R4 beside it, but its
side away from the road bends down as much as 27 cm to the grass, which a kart drops over. This
lays each of PAVERS on the plane that fits it best (its points within ROAD_SEAM of the road under
them, the road's edge, kept as they are), and the near ground's points within EDGE metres of it
on that plane at its edge, eased back to the ground they were over BLEND metres more. Run again,
it finds them on their planes.
"""
import sys

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import distance, points

import build_paths as bp
import gltf_source

PAVERS = ('1KERB_29',)
ROAD_SEAM = .03          # metres: a paver point this near the road's height is on its edge
EDGE, BLEND = 1.5, 4.    # metres round the pavers the ground meets them, and eases back over
FLAT = .02               # metres off its plane under which a paving is flat already


def plane_of(P):
    A = np.c_[P[:, 0], P[:, 2], np.ones(len(P))]
    keep = np.ones(len(P), bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A[keep], P[keep, 1], rcond=None)
        r = P[:, 1] - A @ c
        keep = r > -.08          # leave out the edge bent down, which is what is to be undone
    return lambda x, z: c[0] * x + c[1] * z + c[2]


def main():
    k = gltf_source.load_kn5(bp.TRACK)
    nodes = {n['name']: (n, m) for n, m in bp.meshes(k)}
    road = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(k) if n['name'].startswith('1ROAD_') and len(n['faces'])]))
    gn, gm = nodes['1GRASS_0']
    GV = gn['verts']
    G = (np.c_[GV[:, :3].astype(float), np.ones(len(GV))] @ gm)[:, :3]
    changed = False
    for name in PAVERS:
        n, m = nodes[name]
        V = n['verts']
        P = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ m)[:, :3]
        plane = plane_of(P)
        off = P[:, 1] - plane(P[:, 0], P[:, 2])
        r = road.ys(P[:, [0, 2]])
        on_road = ~np.isnan(r) & (np.abs(P[:, 1] - r) < ROAD_SEAM)
        move = ~on_road & (np.abs(off) > FLAT)
        if not move.any():
            print('%s is flat already' % name)
            continue
        P[move, 1] = plane(P[move, 0], P[move, 2])
        V[:, :3] = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(m))[:, :3].astype(V.dtype)
        bp.sphere(n)
        # the ground round it up to its edge, eased back
        outline = unary_union([Polygon(t[:, [0, 2]]).buffer(.01) for t in bp.triangles(n, m) if Polygon(t[:, [0, 2]]).area > 1e-4])
        d = distance(outline, points(G[:, [0, 2]]))
        near = d < EDGE + BLEND
        t = np.clip((d[near] - EDGE) / BLEND, 0, 1)
        w = 1 - t * t * (3 - 2 * t)
        G[near, 1] += w * (plane(G[near, 0], G[near, 2]) - G[near, 1]) * (G[near, 1] < plane(G[near, 0], G[near, 2]) + .02)
        print('%s: %d points onto its plane (they were %.2f to %.2f m off it), %d of the ground round it' % (
            name, move.sum(), off[move].min(), off[move].max(), (w > 0).sum()))
        changed = True
    if changed:
        GV[:, :3] = (np.c_[G, np.ones(len(G))] @ np.linalg.inv(gm))[:, :3].astype(GV.dtype)
        bp.sphere(gn)
        bp.save(k, bp.TRACK)


if __name__ == '__main__':
    main()
