"""Everything standing on the ground set back on it, after the levelling, in every source.

    python tools/settle_objects.py

tools/level_ground.py and tools/flatten_roads.py moved every point by their fields where it
stands. The ground away from the roads is of large triangles, whose faces lie flat between their
corners: so between them it rose as the field does at its corners, not as it does under them,
and what stands there, moved as the field does there, stands up to 0.6 m off it, or in it (the
tyre walls along the west side and by the vp_1000's gantry most). This moves every point of
everything that is not the ground (nor a road, kerb or other surface driven on) by what the
ground under it moved less what the field moved it: back on the ground as it stood. The trees
are each stood on the ground where they stand (their meshes and BeamNG's forest's stands), the
Custom Shaders Patch's lights moved as their poles. tools/build_outside.py's model is built on
the ground as it is: run that again after. Marked done in tools/reference/elevation/settled.json.
"""
import datetime, glob, json, os, re

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

import build_paths as bp
import build_trees as bt
import flatten_roads as fr
import gltf_source
import kn5tree
import level_ground as lg

DONE = os.path.join(lg.ev.OUT, 'settled.json')
GRID = 1.                # metres, the grid the correction is worked out on
EXTENT = ((-300., 300.), (-200., 460.))
SKIP = ('velopark_outside.glb',)     # built on the ground as it is
GROUND_MODELS = ('velopark', 'velopark_kerbs', 'velopark_paddock', 'velopark_kids')
TREE_NEAR = 9.           # metres from a tree's stand its parts are its


def driven(name):
    """A surface: the ground, a road, a kerb (named for AC's physics, but its walls), or grass."""
    return 'GRASS' in name or (re.match(r'^\d+[A-Z]', name) is not None and not name[1:].startswith('WALL'))


def ground_tris():
    out = []
    for f in GROUND_MODELS:
        for n, m in bp.meshes(gltf_source.load_kn5(os.path.join(bp.MODELS, f + '.glb'))):
            if len(n['faces']) and n['visible'] and ('GRASS' in n['name'] or n['name'].startswith('1ROAD')):
                t = bp.triangles(n, m)
                if len(t):
                    out.append(t)
    return np.concatenate(out)


def main():
    if os.path.exists(DONE):
        print('everything is settled already (%s)' % os.path.relpath(DONE, bp.ROOT))
        return
    d = np.load(lg.FIELD)
    F = (d['x'], d['z'], d['dy'])
    flat = fr.change(F) if os.path.exists(fr.DONE) else (lambda x, z: np.zeros(np.size(x)))

    def T(x, z):
        return lg.at(F, x, z) + flat(x, z)
    t = ground_tris()
    P = t.reshape(-1, 3)
    tT = t.copy()
    tT[:, :, 1] = T(P[:, 0], P[:, 2]).reshape(-1, 3)
    ground, moved = bp.Surface(t), bp.Surface(tT)
    (xa, xb), (za, zb) = EXTENT
    X, Z = np.arange(xa, xb + .1, GRID), np.arange(za, zb + .1, GRID)
    XX, ZZ = np.meshgrid(X, Z)
    M = moved.ys(np.c_[XX.ravel(), ZZ.ravel()])
    D = np.where(np.isnan(M), 0., M - T(XX.ravel(), ZZ.ravel())).reshape(XX.shape)

    def delta(x, z):
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        return map_coordinates(D, [(z - Z[0]) / GRID, (x - X[0]) / GRID], order=1, cval=0.)
    print('  the ground moved %+.2f to %+.2f m from the field under what stands on it' % (D.min(), D.max()))

    # the trees, each on the ground where it stands
    forest = json.load(open(lg.FOREST))
    S = np.array([[s[1], s[3]] for s in forest['trees']])
    gy = ground.ys(S)
    shift = np.where(np.isnan(gy), 0., gy - np.array([s[2] for s in forest['trees']]))
    for s, v in zip(forest['trees'], shift):
        s[2] += float(v)
    json.dump(forest, open(lg.FOREST, 'w'), indent=0)
    stands = cKDTree(S)

    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        if os.path.basename(f) in SKIP:
            continue
        k = gltf_source.load_kn5(f)
        n_moved = [0]

        def walk(n, parent):
            if n['kind'] == 1:
                M4 = np.array(n['matrix'], float).reshape(4, 4)
                m = M4 @ parent
                if not np.allclose(M4, np.eye(4)):
                    if not driven(n['name']):
                        o = m[3, :3]
                        M4[3, :3] += np.array([0., float(delta([o[0]], [o[2]])[0]), 0.]) @ np.linalg.inv(parent[:3, :3])
                        n['matrix'] = list(M4.ravel())
                        n_moved[0] += 1
                    return
                for c in n['children']:
                    walk(c, m)
            elif n['kind'] == 2 and len(n['verts']) and not driven(n['name']):
                V = n['verts']
                W = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ parent)[:, :3]
                if n['name'].startswith((bt.PREFIX, '1WALL_' + bt.PREFIX)):
                    dist, i = stands.query(W[:, [0, 2]])
                    up = np.where(dist < TREE_NEAR, shift[i], delta(W[:, 0], W[:, 2]))
                else:
                    up = delta(W[:, 0], W[:, 2])
                if np.abs(up).max() < 1e-4:
                    return
                W[:, 1] += up
                V[:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(parent))[:, :3].astype(V.dtype)
                bp.sphere(n)
                n_moved[0] += len(V)
        walk(k['root'], np.eye(4))
        if n_moved[0]:
            if os.path.basename(f) == 'velopark_extra.glb':
                bt.save(k, f)
            else:
                bp.save(k, f)
            print('  %-30s %7d points' % (os.path.basename(f), n_moved[0]))
    print('  %d lights' % lg.move_csp(lg.CSP, delta))
    print('  %d trees stood on the ground, by %+.2f to %+.2f m' % (len(shift), shift.min(), shift.max()))
    json.dump(dict(settled=datetime.date.today().isoformat(), grid_m=GRID), open(DONE, 'w'), indent=1)


if __name__ == '__main__':
    main()
