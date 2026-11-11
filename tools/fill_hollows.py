"""The near ground's points sunk under the ground round them lifted to it, in source/models/velopark.glb.

    python tools/fill_hollows.py

The near ground (1GRASS_0) has a trench in it inside the north straight, 2 m wide and up to
1.7 m deep, along x 18 from z 260 to 310, which from the track looks like a hole in the ground;
which is the one hollow it fills (AREAS). This lifts every point there more than SUNK under the
median of the ground's points within NEAR metres of it to that median, again until none is left;
but none within ROAD_EDGE of a road or kerb, where the ground tucks under
the road's edge on purpose (lifted, it stands over the road with a hole under it, as by the
vp_1000's start). The points the ground shares with its neighbours (at the same place) move
together. Run again, it finds none and changes nothing.
"""
import numpy as np
from scipy.spatial import cKDTree

import build_paths as bp
import gltf_source

GROUND = '1GRASS_0'
SUNK = .3                # metres under the ground round it
ROAD_EDGE = 1.5          # metres from a road or kerb: its skirt, tucked under the road's edge, not a hollow
NEAR = 4.                # metres round it
# where it looks: the trench inside the north straight (x0, x1, z0, z1). Elsewhere a point lower
# than the ground round it is as made: a skirt under a road, the ground round path A and D
AREAS = [(9., 27., 254., 314.)]


def main():
    k = gltf_source.load_kn5(bp.TRACK)
    n, m = next((n, m) for n, m in bp.meshes(k) if n['name'] == GROUND)
    V = n['verts']
    P = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ m)[:, :3]
    # the ground's places, each once, and the points at each
    U, inv = np.unique(np.round(P[:, [0, 2]], 3), axis=0, return_inverse=True)
    inv = inv.ravel()
    y = np.full(len(U), -1e9)
    np.maximum.at(y, inv, P[:, 1])
    tree = cKDTree(U)
    near = tree.query_ball_point(U, NEAR)
    roads = np.concatenate([(np.c_[r['verts'][:, :3].astype(float), np.ones(len(r['verts']))] @ rm)[:, [0, 2]]
                            for f in (bp.TRACK, bp.KERBS, bp.PADDOCK) for r, rm in bp.meshes(gltf_source.load_kn5(f))
                            if len(r['faces']) and r['name'].startswith(('1ROAD', '1KERB'))])
    edge = cKDTree(roads).query(U)[0] < ROAD_EDGE
    looked = np.zeros(len(U), bool)
    for x0, x1, z0, z1 in AREAS:
        looked |= (U[:, 0] > x0) & (U[:, 0] < x1) & (U[:, 1] > z0) & (U[:, 1] < z1)
    edge |= ~looked
    lifted = set()
    for _ in range(20):
        changed = False
        for i, nb in enumerate(near):
            if edge[i]:
                continue
            others = [j for j in nb if j != i]
            if len(others) < 5:
                continue
            med = float(np.median(y[others]))
            if y[i] < med - SUNK:
                y[i] = med
                lifted.add(i)
                changed = True
        if not changed:
            break
    if not lifted:
        print('no hollows in the near ground')
        return
    P[:, 1] = y[inv]
    V[:, :3] = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(m))[:, :3].astype(V.dtype)
    bp.sphere(n)
    bp.save(k, bp.TRACK)
    print('%d points of the near ground lifted out of hollows' % len(lifted))


if __name__ == '__main__':
    main()
