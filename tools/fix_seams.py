"""Steps between two road surfaces taken out, a seam at a time, in the track's sources.

    python tools/fix_seams.py

Where two roads of the 0.5.0 track meet, they meet in places a few centimetres apart in height, a
lip a kart trips on or a drop it falls off; and the ground round the paddock's asphalt, which
tools/join_paddock.py eased into the roads, was left as it was, a gap between them. Each of SEAMS names a road eased into another where
they meet: its points on its edge where the other goes on from it go to the other's height there,
and its points within REACH of them go up or down as the nearest of those, less further off, as a
smooth curve. What shares a point with it there (the grass beside it, a kerb's bed) goes exactly as
far (but the road it is eased into), so no new step opens beside it; the paint and the kerbs lying on it, where it is the surface
on top, go with it. Ground eased keeps its edges where it meets anything else, and moves only other
ground with it. Run again, it finds the seams level.
"""
import glob, os

import numpy as np
from scipy.spatial import cKDTree

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
# (model, surface eased, model, the surfaces it is eased into)
SEAMS = [('velopark', '1ROAD_24', 'velopark', ('1ROAD_17',)),
         # the ground round the paddock's asphalt, which tools/join_paddock.py eased into the roads
         ('velopark', '1GRASS_0', 'velopark_paddock', ('1ROAD_PADDOCK', '1ROAD_PADDOCK_2'))]
ROUNDS = 4               # times the seams are eased over, as easing one can move another's edge
REACH = 2.               # metres from the seam over which a road is eased
LEVEL = .005             # metres apart at most that a seam counts as level
GROUND_LEVEL = .015      # and of ground, where a centimetre does not show
SHARE = .01              # metres apart in x and z two points count as one
RIDE = .06               # metres over a road the paint and kerbs on it lie
GROUND_PIECES = ('GRASS', 'SKIRT', 'BED', 'FOLD')     # what shares the ground's edges and goes with it
RIDERS = ('LinePaint', 'GridNumbers', 'KerbRamp', 'KerbConcrete', 'KerbFloor', 'KerbRound', 'Rumble1')


def seam_points(mover, keeper):
    """The mover's points on its edge where the keeper goes on from it, and how far each must go."""
    V = mover['verts'][:, :3].astype(float)
    F = mover['faces'].astype(int)
    key = np.unique(np.round(V, 3), axis=0, return_inverse=True)[1].ravel()
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    third = np.r_[F[:, 2], F[:, 0], F[:, 1]]
    s = np.sort(key[e], axis=1)
    _, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    edge = cnt[inv.ravel()] == 1
    # not where it stands on edge: points one above another
    xz = np.unique(np.round(V[:, [0, 2]], 3), axis=0, return_inverse=True)[1].ravel()
    lo, hi = np.full(xz.max() + 1, np.inf), np.full(xz.max() + 1, -np.inf)
    np.minimum.at(lo, xz, V[:, 1])
    np.maximum.at(hi, xz, V[:, 1])
    upright = (hi - lo)[xz] > .01
    out = {}
    for (a, b), c in zip(e[edge], third[edge]):
        d = (V[b] - V[a])[[0, 2]]
        n = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
        if n @ (V[c] - V[a])[[0, 2]] > 0:
            n = -n
        for v in (a, b):
            if upright[v]:
                continue
            y = keeper.y(*(V[v, [0, 2]] + n * .05))
            if not np.isnan(y):
                out[v] = y - V[v, 1]
    return out, np.unique(e[edge])


def main():
    files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(bp.MODELS, 'velopark*.glb'))}
    models = {f: gltf_source.load_kn5(p) for f, p in files.items()}
    changed = set()
    for round_ in range(ROUNDS):
        if not ease_all(models, changed, round_):
            break
    for f in sorted(changed):
        bp.save(models[f], files[f])


def ease_all(models, changed, round_):
    """Every seam eased once; whether any was not level."""
    any_ = False
    for mf, mname, kf, kname in SEAMS:
        mover = next(n for n, m in bp.meshes(models[mf]) if n['name'] == mname)
        keeper = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(models[kf]) if n['name'] in kname])
        # its faces lying flat, not the skirts down its sides
        up = np.cross(keeper[:, 1] - keeper[:, 0], keeper[:, 2] - keeper[:, 0])
        keeper = bp.Surface(keeper[np.abs(up[:, 1]) > .5 * np.linalg.norm(up, axis=1)])
        seam, rim = seam_points(mover, keeper)
        ground = 'GRASS' in mname
        if not seam or max(abs(v) for v in seam.values()) <= (GROUND_LEVEL if 'GRASS' in mname else LEVEL):
            if not round_:
                print('%s into %s: level' % (mname, ', '.join(kname)))
            continue
        any_ = True
        V = mover['verts']
        P = V[:, :3].astype(float)
        ids = np.array(sorted(seam))
        dys = np.array([seam[i] for i in ids])
        tree = cKDTree(P[ids][:, [0, 2]])

        def field(xz):
            d, j = tree.query(xz, k=min(3, len(ids)))
            d, j = np.atleast_2d(d), np.atleast_2d(j)
            w = 1 / np.maximum(d, 1e-3)
            near = (w * dys[j]).sum(1) / w.sum(1)
            f = np.clip(1 - d[:, 0] / REACH, 0, 1)
            return near * f * f * (3 - 2 * f)
        dy = field(P[:, [0, 2]])
        if ground:
            # the ground's edges where it meets anything else stay where they are
            dy[rim] = 0.
        dy[ids] = dys
        moved = np.abs(dy) > 1e-4
        old = P.copy()
        top = bp.Surface(bp.triangles(mover, np.eye(4)))
        # what shares its moved points goes as far; what lies on it, where it is on top, goes with it
        shared = cKDTree(old[moved][:, [0, 2]])
        others = [(f, n, m, k) for f, k in models.items() for n, m in bp.meshes(k) if n is not mover and len(n['verts'])
                  and not (f == kf and n['name'] in kname)]
        roads = bp.Surface(np.concatenate([bp.triangles(n, m) for f, n, m, k in others
                                           if n['name'][:1].isdigit() and 'WALL' not in n['name'] and len(n['faces'])
                                           and len(bp.triangles(n, m))]))
        for f, n, m, k in others:
            if not np.allclose(m, np.eye(4)):
                continue
            R = n['verts']
            rxz = R[:, [0, 2]].astype(float)
            d, j = shared.query(rxz)
            same = (d < SHARE) & (np.abs(R[:, 1] - old[moved][j, 1]) < .03)
            if ground and not any(w in n['name'] for w in GROUND_PIECES):
                # the ground moves only other ground with it, no road, kerb or tyre
                same[:] = False
            mat = k['materials'][n['material']]['name']
            ride = np.zeros(len(R), bool)
            if mat in RIDERS and not ground:
                near = np.nonzero(tree.query(rxz)[0] < REACH)[0]
                if len(near):
                    y = top.ys(rxz[near])
                    over = roads.ys(rxz[near])
                    on = ~np.isnan(y) & (R[near, 1] > y - .005) & (R[near, 1] < y + RIDE) & ~(over > y + .003)
                    ride[near[on]] = True
            go = same | ride & ~same
            if go.any():
                add = np.where(same, dy[moved][j], 0.)
                add[ride & ~same] = field(rxz[ride & ~same])
                R[go, 1] += add[go].astype(R.dtype)
                bp.sphere(n)
                changed.add(f)
        V[:, 1] += dy.astype(V.dtype)
        bp.sphere(mover)
        changed.add(mf)
        print('%s into %s: %d points eased, the step was up to %.1f cm' % (mname, ', '.join(kname), moved.sum(), np.abs(dys).max() * 100))
    return any_


if __name__ == '__main__':
    main()
