"""The levelling made the same across every road, so no road is twisted by it, in every source.

    python tools/flatten_roads.py [--dry-run]

tools/level_ground.py moved every point by a field from Google's heights, 9.5 m apart: smooth
along the roads, but tilting them across a little too, one way and then the other, up to 2%
along the north straight, which twists it. This moves every point near the roads again, by what
makes the field the same across them: on and beside a road, the field smoothed over LONG
metres, as it is at the racing line (the average of it at the layouts' AI lines' points round
it, within SPREAD metres, which is the same straight across the road and has no step where the
lines part or meet);
easing back to the field as it is from ON metres from the nearest line to OFF. The roads keep
the camber they had and the levelling's heights along them; the ground beside them moves with
them. Everything moves as tools/level_ground.py moves it: every model's points, the AC_ nodes,
the AI lines, the Custom Shaders Patch's lights and the forest. Marked done in
tools/reference/elevation/flatten_roads.json: run again, it changes nothing.
"""
import datetime, glob, json, os, struct, sys

import numpy as np
from scipy.spatial import cKDTree

import build_paths as bp
import gltf_source
import level_ground as lg

DONE = os.path.join(lg.ev.OUT, 'flatten_roads.json')
SPREAD = 4.              # metres, the racing line's points averaged round a point (sigma)
ON, OFF = 7., 20.        # metres from the nearest line: wholly the line's field, wholly the field's
GRID = 1.                # metres, the grid the change is worked out on
LONG = 40.               # metres the field is smoothed over on the roads


def lines():
    """Every point of every layout's AI lines, the pits' too."""
    out = []
    for f in sorted(glob.glob(os.path.join(lg.TRACK, '*', 'ai', '*.ai'))):
        b = open(f, 'rb').read()
        _, n, _, _ = struct.unpack_from('<4i', b, 0)
        out.append(np.frombuffer(b, '<f4', n * 5, 16).reshape(-1, 5)[:, [0, 2]].astype(float))
    P = np.concatenate(out)
    # thinned to a point every half metre or so
    return np.unique(np.round(P * 2) / 2, axis=0)


def change(F):
    """The function of x, z the points move by: the line's field less the field, eased out."""
    L = lines()
    tree = cKDTree(L)
    # the field smoothed over LONG along the lines: across a road it barely changes, so where
    # several layouts' lines run side by side the average of them is the same
    from scipy.ndimage import gaussian_filter
    X, Z, dy = F
    dyL = lg.at((X, Z, gaussian_filter(dy, LONG / lg.CELL)), L[:, 0], L[:, 1])

    def delta(x, z):
        x, z = np.atleast_1d(x).ravel(), np.atleast_1d(z).ravel()
        Q = np.c_[x, z]
        out = np.zeros(len(Q))
        d, _ = tree.query(Q)
        near = d < OFF
        if near.any():
            idx = tree.query_ball_point(Q[near], 3 * SPREAD)
            here = lg.at(F, x[near], z[near])
            line = np.empty(near.sum())
            for k, (q, ii) in enumerate(zip(Q[near], idx)):
                if not ii:
                    line[k] = here[k]
                    continue
                ii = np.array(ii)
                w = np.exp(-(np.sum((L[ii] - q) ** 2, 1)) / (2 * SPREAD ** 2))
                line[k] = (w @ dyL[ii]) / w.sum()
            t = np.clip((OFF - d[near]) / (OFF - ON), 0, 1)
            out[near] = t * t * (3 - 2 * t) * (line - here)
        return out
    # worked out once on a GRID metre grid round the lines, and taken from it between its points
    lo = L.min(0) - OFF - 2
    hi = L.max(0) + OFF + 2
    X, Z = np.arange(lo[0], hi[0], GRID), np.arange(lo[1], hi[1], GRID)
    XX, ZZ = np.meshgrid(X, Z)
    D = delta(XX.ravel(), ZZ.ravel()).reshape(XX.shape)

    def on_grid(x, z):
        from scipy.ndimage import map_coordinates
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        return map_coordinates(D, [(z - Z[0]) / GRID, (x - X[0]) / GRID], order=1, cval=0.)
    return on_grid


def main():
    if os.path.exists(DONE):
        print('the roads are flattened already (%s)' % os.path.relpath(DONE, bp.ROOT))
        return
    if not os.path.exists(lg.DONE):
        sys.exit('run tools/level_ground.py first')
    d = np.load(lg.FIELD)
    delta = change((d['x'], d['z'], d['dy']))
    # how much it takes out across the north straight, and the most it moves anything
    xs = np.arange(-150, 51, 10.)
    probe = np.r_[[delta(xs, np.full(len(xs), 288.))], [delta(xs, np.full(len(xs), 296.))]]
    print('  moves by %+.3f to %+.3f m across the north straight' % (probe.min(), probe.max()))
    if '--dry-run' in sys.argv:
        return
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        k = gltf_source.load_kn5(f)
        n = lg.move_model(k, delta)
        bp.save(k, f)
        print('  %-30s %7d points' % (os.path.basename(f), n))
    ai = sorted(glob.glob(os.path.join(lg.TRACK, '*', 'ai', '*.ai')) + glob.glob(os.path.join(lg.TRACK, '*', 'data', '*.ai')))
    for f in ai:
        lg.move_ai(f, delta)
    print('  %d AI lines, %d lights, %d trees' % (len(ai), lg.move_csp(lg.CSP, delta), lg.move_forest(lg.FOREST, delta)))
    json.dump(dict(flattened=datetime.date.today().isoformat(), spread_m=SPREAD, on_m=ON, off_m=OFF), open(DONE, 'w'), indent=1)


if __name__ == '__main__':
    main()
