"""The roads along the south of the track brought down onto Google's heights, smooth, in every
source: the vp_1500's south straight and the road beside it, by the pit lane's junction.

    python tools/smooth_south.py [--dry-run]

After tools/level_ground.py and tools/flatten_roads.py the south straight still rode over a crest
the old track had, at x = 122, where its grade turned from +0.6% to -1.3% in 5 m, and then fell
early, 10 to 15 cm under Google's ground, while Google's stays flat to x = 145 and then dips
smoothly by half a metre to x = 170; the road beside it, at z = 30, had one too, at x = 107, 29 cm
over Google's. The levelling smoothed its field over 15 m, so a crest that short stayed.

This moves the roads there onto Google's heights, smoothed over SMOOTH metres (so the 10 m grid's
corners don't show): at every layout's AI line points in the area (the pit lanes' too), what the
road needs to rise or fall, averaged over SPREAD metres round each point, the same straight
across the road, so it keeps its camber; easing back to nothing from ON to OFF metres from the
nearest line, and over EDGE metres inside the area's edges (AREA). Everything moves with it, as
tools/level_ground.py moves it: every model's points, the AC_ nodes, the AI lines, the Custom
Shaders Patch's lights and the forest. tools/build_outside.py again after it, to build what it
makes on the ground as it is now. Marked done in tools/reference/elevation/smooth_south.json:
run again, it changes nothing.
"""
import datetime, glob, json, os, struct, sys

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

import build_paths as bp
import gltf_source
import level_ground as lg

DONE = os.path.join(lg.ev.OUT, 'smooth_south.json')
AREA = (70., 190., -45., 50.)   # x0, x1, z0, z1: where the roads are brought onto Google's; it
                                # ends before T3's hairpin, where Google's has the plantation's tops
EDGE = 20.               # metres inside the area's edges the change grows from nothing
SMOOTH = 8.              # metres Google's heights are smoothed over (sigma)
SPREAD = 2.5             # metres the lines' points are averaged round a point (sigma)
ON, OFF = 7., 20.        # metres from the nearest line: wholly the line's change, none
GRID = 1.                # metres, the grid the change is worked out on


def google():
    """Google's heights as the track's y, smoothed over SMOOTH metres, as a function of x, z."""
    g = json.load(open(lg.latest('google_2*.json')))
    R = np.array(g['rows'])
    off = json.load(open(lg.DONE))['offset_m']
    f = LinearNDInterpolator(R[:, :2], R[:, 4] - off)
    x0, x1, z0, z1 = AREA
    X, Z = np.arange(x0 - 40, x1 + 40, GRID), np.arange(z0 - 40, z1 + 40, GRID)
    H = f(*np.meshgrid(X, Z))
    H = gaussian_filter(np.nan_to_num(H, nan=np.nanmean(H)), SMOOTH / GRID)
    return lambda x, z: map_coordinates(H, [(np.asarray(z) - Z[0]) / GRID, (np.asarray(x) - X[0]) / GRID], order=1, mode='nearest')


def lines():
    """Every layout's AI lines' points in the area, every half metre or so."""
    out = []
    for f in sorted(glob.glob(os.path.join(lg.TRACK, '*', 'ai', '*.ai'))):
        b = open(f, 'rb').read()
        _, n, _, _ = struct.unpack_from('<4i', b, 0)
        out.append(np.frombuffer(b, '<f4', n * 5, 16).reshape(-1, 5)[:, [0, 2]].astype(float))
    P = np.unique(np.round(np.concatenate(out) * 2) / 2, axis=0)
    x0, x1, z0, z1 = AREA
    return P[(P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > z0) & (P[:, 1] < z1)]


def change():
    """The function of x, z the points move by."""
    track = gltf_source.load_kn5(bp.TRACK)
    road = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                      if n['name'].startswith('1ROAD_') and len(n['faces'])]))
    L = lines()
    y = road.ys(L)
    L, y = L[~np.isnan(y)], y[~np.isnan(y)]
    need = google()(L[:, 0], L[:, 1]) - y
    tree = cKDTree(L)
    x0, x1, z0, z1 = AREA

    def edge(x, z):
        t = np.clip(np.minimum.reduce([x - x0, x1 - x, z - z0, z1 - z]) / EDGE, 0, 1)
        return t * t * (3 - 2 * t)

    def delta(x, z):
        Q = np.c_[x, z]
        out = np.zeros(len(Q))
        d, _ = tree.query(Q)
        near = d < OFF
        for k, (q, ii) in zip(np.nonzero(near)[0], zip(Q[near], tree.query_ball_point(Q[near], 3 * SPREAD))):
            if not ii:
                continue
            ii = np.array(ii)
            w = np.exp(-(np.sum((L[ii] - q) ** 2, 1)) / (2 * SPREAD ** 2))
            t = np.clip((OFF - d[k]) / (OFF - ON), 0, 1)
            out[k] = t * t * (3 - 2 * t) * (w @ need[ii]) / w.sum()
        return out * edge(x, z)
    X, Z = np.arange(x0, x1 + GRID, GRID), np.arange(z0, z1 + GRID, GRID)
    XX, ZZ = np.meshgrid(X, Z)
    D = delta(XX.ravel(), ZZ.ravel()).reshape(XX.shape)

    def on_grid(x, z):
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        return map_coordinates(D, [(z - Z[0]) / GRID, (x - X[0]) / GRID], order=1, cval=0.)
    return on_grid, D


def main():
    if os.path.exists(DONE):
        print('the south roads are smoothed already (%s)' % os.path.relpath(DONE, bp.ROOT))
        return
    delta, D = change()
    print('  moves by %+.3f to %+.3f m' % (D.min(), D.max()))
    if '--dry-run' in sys.argv:
        return
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        k = gltf_source.load_kn5(f)
        n = lg.move_model(k, delta)
        if n:
            bp.save(k, f)
            print('  %-30s %7d points' % (os.path.basename(f), n))
    ai = sorted(glob.glob(os.path.join(lg.TRACK, '*', 'ai', '*.ai')) + glob.glob(os.path.join(lg.TRACK, '*', 'data', '*.ai')))
    for f in ai:
        lg.move_ai(f, delta)
    print('  %d AI lines, %d lights, %d trees' % (len(ai), lg.move_csp(lg.CSP, delta), lg.move_forest(lg.FOREST, delta)))
    json.dump(dict(smoothed=datetime.date.today().isoformat(), area=AREA, smooth_m=SMOOTH, spread_m=SPREAD, on_m=ON, off_m=OFF),
              open(DONE, 'w'), indent=1)


if __name__ == '__main__':
    main()
