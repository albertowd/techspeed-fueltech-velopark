"""The kids track's roads and kerbs set back on the ground round them, after the levelling, in
source/models/velopark_kids.glb, and its AI lines with them.

    python tools/settle_kids.py

The ground under and round the kids track is a few large triangles of the near ground
(1GRASS_0 of velopark.glb), with no corner within 12 m of its roads. tools/level_ground.py and
tools/flatten_roads.py moved those corners and the kids roads each by the field where it stands,
so the triangles' flat faces came to lie up to 0.3 m over the roads (they lay 1 cm under them),
and the kids track sank out of sight under the grass. tools/settle_objects.py set what stands
there (its paint, kerb sides, tyre walls, cones and start) on that ground, but not the roads and
kerbs, as it leaves the surfaces driven on where they are. This lifts every point of the kids
roads and kerbs (1ROAD_KIDS_, 1ROAD_KERB_KIDS_) by one smooth correction: at each point of its
roads, what takes the road to 1 cm over the ground there, as it stood, spread over the kerbs from
the road points nearest them, so they keep their shape; and the kids layout's AI lines by the
same. Marked done in tools/reference/elevation/settled_kids.json.
"""
import datetime, glob, json, os

import numpy as np
from scipy.spatial import cKDTree

import build_paths as bp
import gltf_source
import level_ground as lg

DONE = os.path.join(lg.ev.OUT, 'settled_kids.json')
KIDS = os.path.join(bp.MODELS, 'velopark_kids.glb')
OVER = .01               # metres the roads lie over the ground, as they did
NEAR = 8                 # road points the correction at a point is the mean of


def world(n, m):
    V = n['verts'][:, :3].astype(float)
    return (np.c_[V, np.ones(len(V))] @ m)[:, :3]


def main():
    if os.path.exists(DONE):
        print('the kids track is settled already (%s)' % os.path.relpath(DONE, bp.ROOT))
        return
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(gltf_source.load_kn5(bp.TRACK))
                                        if 'GRASS' in n['name'][:8] and len(n['faces'])]))
    k = gltf_source.load_kn5(KIDS)
    roads = [(n, m) for n, m in bp.meshes(k) if n['name'].startswith('1ROAD_KIDS') and len(n['verts'])]
    R = np.concatenate([world(n, m) for n, m in roads])
    g = ground.ys(R[:, [0, 2]])
    ok = ~np.isnan(g)
    want = g[ok] + OVER - R[ok, 1]
    tree = cKDTree(R[ok][:, [0, 2]])

    def lift(x, z):
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        dist, i = tree.query(np.c_[x, z], k=NEAR)
        w = 1 / np.maximum(dist, .1)
        return (want[i] * w).sum(1) / w.sum(1)
    print('  the kids roads lay %+.2f to %+.2f m off the ground (median %+.2f)' % (-want.max(), -want.min(), -np.median(want)))
    moved = 0
    for n, m in bp.meshes(k):
        if not n['name'].startswith(('1ROAD_KIDS', '1ROAD_KERB_KIDS')) or not len(n['verts']):
            continue
        W = world(n, m)
        W[:, 1] += lift(W[:, 0], W[:, 2])
        n['verts'][:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(m))[:, :3].astype(n['verts'].dtype)
        bp.sphere(n)
        moved += len(W)
    bp.save(k, KIDS)
    lines = sorted(glob.glob(os.path.join(lg.TRACK, 'vp_kids', 'ai', '*.ai')) + glob.glob(os.path.join(lg.TRACK, 'vp_kids', 'data', '*.ai')))
    for f in lines:
        lg.move_ai(f, lift)
    print('  %d points of the kids roads and kerbs lifted, and %d AI lines' % (moved, len(lines)))
    json.dump(dict(settled=datetime.date.today().isoformat(), over_m=OVER), open(DONE, 'w'), indent=1)


if __name__ == '__main__':
    main()
