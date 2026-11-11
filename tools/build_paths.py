"""Paths A and D across the infield as asphalt between their tyre walls, in the track's sources.

    python tools/build_paths.py

Two paths cross the infield, lined with tyres: A from R20 down beside the start straight to
the vp_1500 pits at R25, and D from R19 across to R20. The paddock had them as thin strips of
its asphalt, laid where the ground took them: D partly under its tyre wall, A over a hump with
its sides twisted, and neither reaching the roads at their ends. This takes those strips out of
source/models/velopark_paddock.glb into the ground of source/models/velopark.glb, which has
holes where they lay, as 1GRASS_PATHS; and lays each path as a mesh of its own in the paddock,
1ROAD_PATH_<name> with the paddock's asphalt: along a centre line between the tyre walls, as wide as WIDTH, its
height eased along it and tilted across it no more than the ground needs, over the ground
everywhere, and widened at each end into the road it meets, under that road; but A's south end
runs on straight past its tyre walls into the pit exit, as a road merging with it, the pit exit
running the same way there just before the start straight. Its asphalt is mapped as most of the
paddock's is, flat from above. join_paddock.py
then eases their ends into the roads' surfaces, as it does the rest of the paddock's asphalt.
Where a path eases lower than the old strip lay, the strip goes down under it. Run again, it
finds the paths laid and changes nothing.
"""
import os, sys, urllib.parse

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import gltf_source
import kn5tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'source', 'models')
PADDOCK = os.path.join(MODELS, 'velopark_paddock.glb')
TRACK = os.path.join(MODELS, 'velopark.glb')
KERBS = os.path.join(MODELS, 'velopark_kerbs.glb')
TYRES = os.path.join(MODELS, 'velopark_tyres.glb')
ASPHALT = '1ROAD_PADDOCK'
GROUND, FILLER = '1GRASS_0', '1GRASS_PATHS'     # the track's ground, and the old strips in it
# the paths, in AC's (x, z) along their middle, each end a little into the road it meets, and
# their width: A between its two tyre walls, 4.4 m apart; D past the tyre ring by the tree,
# 2 m from the tyre wall on its other side
PATHS = {'A': (((3.5, 227.9), (-1., 226.1), (-4.8, 224.6), (-8.4, 222.9), (-11.8, 220.6), (-14.4, 217.8),
                (-16.5, 214.7), (-18.1, 211.6), (-19.5, 208.1), (-20.8, 204.5), (-21.9, 200.8), (-23., 197.1),
                (-23.9, 193.4), (-24.7, 189.5), (-25.4, 185.7), (-25.9, 181.9), (-26.3, 178.), (-26.5, 174.),
                (-26.6, 170.1), (-26.8, 166.1), (-26.6, 162.2), (-26.1, 158.5), (-24.4, 152.1), (-21.9, 147.1),
                (-19.8, 143.7), (-17.8, 140.3), (-16.5, 137.8), (-15.2, 135.2), (-13.9, 132.2), (-12.7, 129.8),
                (-11.72, 127.8)), 3.6),
         'D': (((50., 237.), (47., 236.6), (42.4, 235.9), (36.5, 235.), (32.5, 234.4), (29.6, 234.2), (26.5, 234.2),
                (22., 233.5), (17., 232.5), (13.4, 231.9), (10., 231.5)), 2.8)}
STEP = .5            # metres between the rows across a path
COLUMNS = 9          # points across it, edge to edge
FLARE, FLARE_LEN = 2., 6.     # how much wider each side a path ends in a road, and over how long
# a path's ends shaped otherwise, {(name, 0 first or 1 last): {side: (flare, narrow)}}: how much
# wider that side gets where the end leaves its road (FLARE where not given), and how far from
# the middle it narrows to by the end, inside the road (not at all where None). A runs on past
# its tyre walls into the pit exit as a lane merging with it, no wider, and narrows once in it,
# so its edge runs into the pit exit's before the pit exit meets the start straight
ENDS = {('A', 1): {1: (0., .5), -1: (0., .5)}}
LIFT = .02           # over the ground at the least
SINK = .01           # under the road where a path runs into it
MAX_TILT = .06       # the most a path leans across
TWIST = 5.           # metres over which its lean eases from one to another
BEND = 3.            # metres over which its height may bend as the ground does
SKIRT = .4           # how far out its sides go down into the ground
DROP = .15           # and how far under the ground beside them: steeply, so little of them shows over it
CLEAR = .3           # how far short of a tyre wall a path's widened end stops
UNDER = .03          # how far under a path the old strips go where it is lower than they were
STRIP = 1.6          # how far past a path's side the old strip's faces go


def meshes(k):
    return [(n, m) for n, m in kn5tree.walk(k['root']) if n['kind'] == 2]


def triangles(n, m):
    f = n['faces'].astype(int)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    p = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
    return p[f]


class Surface:
    """The top of some triangles over points, through a grid of 4 m cells; nan where none."""
    CELL = 4.

    def __init__(s, tris):
        s.t = tris
        lo = np.floor(tris[:, :, [0, 2]].min(1) / s.CELL).astype(int)
        hi = np.floor(tris[:, :, [0, 2]].max(1) / s.CELL).astype(int)
        s.grid = {}
        for k in range(len(tris)):
            for gx in range(lo[k, 0], hi[k, 0] + 1):
                for gz in range(lo[k, 1], hi[k, 1] + 1):
                    s.grid.setdefault((gx, gz), []).append(k)

    def y(s, x, z):
        best = np.nan
        for k in s.grid.get((int(np.floor(x / s.CELL)), int(np.floor(z / s.CELL))), []):
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = s.t[k]
            d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
            if abs(d) < 1e-12:
                continue
            l1 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d
            l2 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / d
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-6:
                h = l1 * ay + l2 * by + l3 * cy
                if not h <= best:
                    best = h
        return best

    def ys(s, P):
        return np.array([s.y(x, z) for x, z in P])


def flat_uv(node):
    """The uv most of a mesh's faces have, mapped flat from above, as a function of points: a
    paddock's asphalt is mapped so in patches, some of them turned, which no single plane
    through all its points fits."""
    v = node['verts'].astype(float)
    f = node['faces'].astype(int)
    P, U = v[f][:, :, [0, 2]], v[f][:, :, 6:8]
    dP = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], 1)
    dU = np.stack([U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]], 1)
    ok = np.abs(np.linalg.det(dP)) > .1
    J = np.linalg.solve(dP[ok], dU[ok])
    keys, inverse, counts = np.unique(np.round(J.reshape(-1, 4), 3), axis=0, return_inverse=True, return_counts=True)
    same = inverse.ravel() == np.argmax(counts)
    J = np.median(J[same], axis=0)
    # and where it starts, the texture repeating every whole uv
    off = (U[ok][same][:, 0] - P[ok][same][:, 0] @ J) % 1
    off = np.median(np.where(off > .5, off - 1, off), axis=0)
    return lambda Q: Q[:, [0, 2]] @ J + off


def centre_line(pts):
    """The line through the points as a Catmull-Rom curve, every STEP along it, with its
    directions."""
    pts = np.asarray(pts, float)
    ext = np.r_[[2 * pts[0] - pts[1]], pts, [2 * pts[-1] - pts[-2]]]
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        for t in np.linspace(0, 1, 40, endpoint=False):
            dense.append(.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                               + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    dense.append(pts[-1])
    dense = np.array(dense)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    at = np.linspace(0, s[-1], int(np.ceil(s[-1] / STEP)) + 1)
    c = np.c_[np.interp(at, s, dense[:, 0]), np.interp(at, s, dense[:, 1])]
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return at, c, d


def smooth_over(e, floor, fixed, length):
    """A smooth line near e, over floor and through fixed (each nan where it does not hold),
    as close to e as bending no more than over about length metres lets it: least squares of
    its distance from e and of its bending, kept over floor and on fixed."""
    n = len(e)
    free = np.isnan(e)
    w = np.where(free, 0., 1.)
    bend = np.zeros((n - 2, n))
    i = np.arange(n - 2)
    bend[i, i], bend[i, i + 1], bend[i, i + 2] = 1, -2, 1
    A = np.r_[np.diag(w), bend * (length / STEP) ** 2]
    b = np.r_[np.where(free, 0, e), np.zeros(n - 2)]
    lo = np.where(np.isnan(floor), -np.inf, floor)
    hi = np.full(n, np.inf)
    on = ~np.isnan(fixed)
    lo[on], hi[on] = fixed[on] - 1e-6, fixed[on] + 1e-6
    return lsq_linear(A, b, bounds=(lo, hi), method='bvls').x


def lay(name, pts, width, ground, road, walls):
    """The path's grid: its points (rows, columns, 3), a skirt each side."""
    s, c, d = centre_line(pts)
    n = np.c_[-d[:, 1], d[:, 0]]
    on = ~np.isnan(road.ys(c))
    if on[len(on) // 2] or not (on[0] and on[-1]):
        sys.exit('path %s must start and end in a road and not run in one between' % name)
    # where each end leaves its road, and the width flaring out to there
    first = np.argmin(on)
    last = len(on) - 1 - np.argmin(on[::-1])
    edge_s = (s[first], s[last])
    sides = []
    for sgn in (-1, 1):
        # how far this side reaches: widened round where each end leaves its road, and
        # narrowed from there to the end where the end says so
        half = np.full(len(s), width / 2)
        for end, e in enumerate(edge_s):
            flare = ENDS.get((name, end), {}).get(sgn, (FLARE, None))[0]
            u = np.clip(1 - np.abs(s - e) / FLARE_LEN, 0, 1)
            half += flare * u * u * (3 - 2 * u)
        half[:first] = half[first]
        half[last + 1:] = half[last]
        for end, k, out in ((0, first, s < s[first]), (1, last, s > s[last])):
            narrow = ENDS.get((name, end), {}).get(sgn, (FLARE, None))[1]
            if narrow is not None and out.any():
                span = s[-1] - s[k] if end else s[k] - s[0]
                u = np.clip(np.abs(s - s[k]) / span, 0, 1) * out
                half -= (half[k] - narrow) * u * u * (3 - 2 * u)
        # widened only as far as the tyre walls leave room, CLEAR short of them; the path's own
        # width goes under the tyres where they stand closer, as they stand on it there
        room = half.copy()
        for i in range(len(s)):
            ray = LineString([c[i], c[i] + sgn * n[i] * (half[i] + CLEAR)])
            hit = ray.intersection(walls)
            if not hit.is_empty:
                room[i] = Point(c[i]).distance(hit) - CLEAR
        least = np.minimum(width / 2, half)
        side = np.maximum(least, np.minimum(half, room))
        sides.append(np.minimum(gaussian_filter1d(side, 1. / STEP, mode='nearest'), np.maximum(least, room)))
    t = np.linspace(-1, 1, COLUMNS)
    span = np.where(t[None, :] < 0, sides[0][:, None], sides[1][:, None]) * t[None, :]
    P = c[:, None, :] + n[:, None, :] * span[:, :, None]
    flat = P.reshape(-1, 2)
    g = ground.ys(flat).reshape(P.shape[:2])
    r = road.ys(flat).reshape(P.shape[:2])
    off = np.isnan(r)
    # the tilt across: the ground's, eased along and kept small
    tilt = np.full(len(s), np.nan)
    for i in range(len(s)):
        k = off[i] & ~np.isnan(g[i])
        if k.sum() >= 3:
            x = span[i][k]
            tilt[i] = np.polyfit(x, g[i][k], 1)[0]
    ok = ~np.isnan(tilt)
    tilt = np.interp(s, s[ok], tilt[ok])
    tilt = np.clip(gaussian_filter1d(tilt, TWIST / STEP, mode='nearest'), -MAX_TILT, MAX_TILT)
    across = span * tilt[:, None]
    # the middle's height: as near the ground across as it eases, less the tilt, and where it
    # runs in a road, the road's; the ground goes down under it where it is higher (sink)
    v = np.where(off & ~np.isnan(g), g + LIFT - across, np.nan)
    target = np.full(len(s), np.nan)
    k = ~np.isnan(v).all(1)
    target[k] = np.nanmean(v[k], axis=1)
    fixed = np.full(len(s), np.nan)
    fixed[on] = r[on, COLUMNS // 2] - SINK
    target[on] = np.nan
    mid = smooth_over(target, np.full(len(s), np.nan), fixed, BEND)
    y = mid[:, None] + across
    y = np.where(off, y, r - SINK)
    # the sides go down into the ground where the path stands over it
    out = c[:, None, :] + n[:, None, :] * np.c_[-(sides[0] + SKIRT), sides[1] + SKIRT][:, :, None]
    go = ground.ys(out.reshape(-1, 2)).reshape(out.shape[:2])
    ro = road.ys(out.reshape(-1, 2)).reshape(out.shape[:2])
    ys = np.where(np.isnan(go), y[:, [0, -1]], np.minimum(go - DROP, y[:, [0, -1]]))
    ys = np.where(np.isnan(ro), ys, np.minimum(ys, ro - SINK))
    grid = np.concatenate([out[:, :1], P, out[:, 1:]], axis=1)
    height = np.concatenate([ys[:, :1], y, ys[:, 1:]], axis=1)
    pos = np.dstack([grid[:, :, 0], height, grid[:, :, 1]])
    lift = np.where(off & ~np.isnan(g), y - g, np.nan)
    grade = np.abs(np.diff(mid)) / STEP
    print('path %s: %.0f m, %d rows; over the ground by %.2f..%.2f m, grade up to %.1f%%, tilt up to %.1f%%'
          % (name, s[-1], len(s), np.nanmin(lift), np.nanmax(lift), 100 * grade[first:last].max(),
             100 * np.abs(tilt[first:last + 1]).max()))
    return pos


def top_of(path):
    """A path's faces, its skirts left out."""
    V, F = path['verts'][:, :3].astype(float), path['faces'].astype(int)
    inner = ~np.isin(np.arange(len(V)) % (COLUMNS + 2), (0, COLUMNS + 1))
    return V[F[inner[F].all(1)]]


def sink(grounds, paths):
    """The ground down UNDER below the paths wherever it would show through them: first its
    points under them, then the points of any of its faces still over them, at the paths'
    points and the middles of their faces."""
    # the paths' tops, their skirts left out, as those go into the ground
    tops, probe = [], []
    for p in paths:
        t = top_of(p)
        tops.append(t)
        probe += [t.reshape(-1, 3), t.mean(1)]
    top = Surface(np.concatenate(tops))
    probe = np.concatenate(probe)
    lo, hi = top.t[:, :, [0, 2]].min((0, 1)) - 1, top.t[:, :, [0, 2]].max((0, 1)) + 1
    moved = outside = 0
    for g in grounds:
        V = g['verts']
        P = V[:, [0, 2]].astype(float)
        near = (P >= lo).all(1) & (P <= hi).all(1)
        idx = np.nonzero(near)[0]
        over = top.ys(P[idx])
        k = ~np.isnan(over) & (V[idx, 1] > over - UNDER)
        V[idx[k], 1] = (over[k] - UNDER).astype(V.dtype)
        moved += k.sum()
    for _ in range(10):
        worst = 0
        for g in grounds:
            V, F = g['verts'], g['faces'].astype(int)
            t = V[F][:, :, :3].astype(float)
            c = t[:, :, [0, 2]]
            sel = np.nonzero((c.max(1) >= lo).all(1) & (c.min(1) <= hi).all(1))[0]
            surf = Surface(t[sel])
            for x, y, z in probe:
                for j in surf.grid.get((int(np.floor(x / surf.CELL)), int(np.floor(z / surf.CELL))), []):
                    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = surf.t[j]
                    d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
                    if abs(d) < 1e-12:
                        continue
                    l1 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d
                    l2 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / d
                    l3 = 1 - l1 - l2
                    if min(l1, l2, l3) < -1e-6 or l1 * ay + l2 * by + l3 * cy <= y - UNDER / 2:
                        continue
                    worst = max(worst, l1 * ay + l2 * by + l3 * cy - y)
                    for q in F[sel[j]]:
                        if V[q, 1] > y - UNDER:
                            V[q, 1] = y - UNDER
                            outside += 1
        if worst == 0:
            break
    else:
        sys.exit('the ground still shows through the paths, by up to %.3f m' % worst)
    for g in grounds:
        sphere(g)
    print('the ground lowered under the paths at %d points, and %d more round them' % (moved, outside))


def mesh(template, name, pos, uv_of):
    rows, cols = pos.shape[:2]
    P = pos.reshape(-1, 3)
    idx = np.arange(len(P)).reshape(rows, cols)
    F = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            F += [(a, b, d), (a, d, c)]
    F = np.array(F, int)
    # facing up
    t = P[F]
    down = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 1] < 0
    F[down] = F[down][:, ::-1]
    N = np.zeros_like(P)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    U = uv_of(P)
    T = gltf_source.tangents(P, N, U, F)
    node = dict(template)
    node.update(name=name, children=[], verts=np.c_[P, N, U, T].astype('<f4'), faces=F.astype('<u2'))
    sphere(node)
    return node


def sphere(node):
    P = node['verts'][:, :3].astype(float)
    mid = (P.min(0) + P.max(0)) / 2
    node['sphere'] = [float(x) for x in mid] + [float(np.linalg.norm(P - mid, axis=1).max())]


def split(node, out):
    """The faces marked out taken from the mesh: their vertices and faces, numbered afresh."""
    v, f = node['verts'], node['faces'].astype(int)

    def part(sel):
        used = np.unique(f[sel])
        remap = np.full(len(v), -1)
        remap[used] = np.arange(len(used))
        return v[used], remap[f[sel]]
    taken = part(out)
    node['verts'], kept = part(~out)
    node['faces'] = kept.astype('<u2')
    sphere(node)
    return taken


def save(k, path):
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris[t['name']])
    gltf_source.write_glb(path, doc, binary)


def main():
    pad = gltf_source.load_kn5(PADDOCK)
    track = gltf_source.load_kn5(TRACK)
    kerbs = gltf_source.load_kn5(KERBS)
    asphalt, am = next((n, m) for n, m in meshes(pad) if n['name'] == ASPHALT)
    if not np.allclose(am, np.eye(4)):
        sys.exit('%s is not at the model\'s origin' % ASPHALT)
    # the paddock's asphalt is mapped flat from above, the most of it the same way
    uv_of = flat_uv(asphalt)
    v = asphalt['verts'].astype(float)

    # the old strips out of the paddock's asphalt, and into the track's ground: the ground has
    # holes where they lay, which they fill as grass now, its photo on them
    zones = [LineString(pts).buffer(w / 2 + FLARE + STRIP) for pts, w in PATHS.values()]
    f = asphalt['faces'].astype(int)
    cen = v[f][:, :, [0, 2]].mean(1)
    gone = np.zeros(len(f), bool)
    for z in zones:
        gone |= contains_xy(z, cen[:, 0], cen[:, 1])
    if not gone.any():
        if any(n['name'].startswith('1ROAD_PATH_') for n, _ in meshes(pad)):
            print('the paths are laid in %s already' % os.path.relpath(PADDOCK, ROOT))
            return
        sys.exit('no strips of %s where the paths go' % ASPHALT)
    if any(c['name'] == FILLER for c in track['root']['children']):
        sys.exit('%s has %s already, from paths laid before' % (os.path.relpath(TRACK, ROOT), FILLER))
    grass, gm = next((n, m) for n, m in meshes(track) if n['name'] == GROUND)
    if not np.allclose(gm, np.eye(4)):
        sys.exit('%s is not at the model\'s origin' % GROUND)
    g = grass['verts'].astype(float)
    gc, *_ = np.linalg.lstsq(np.c_[g[:, 0], g[:, 2], np.ones(len(g))], g[:, 6:8], rcond=None)
    moved, faces = split(asphalt, gone)
    P = moved[:, :3].astype(float)
    U = np.c_[P[:, 0], P[:, 2], np.ones(len(P))] @ gc
    filler = dict(grass)
    filler.update(name=FILLER, children=[], faces=faces.astype('<u2'),
                  verts=np.c_[P, moved[:, 3:6], U, gltf_source.tangents(P, moved[:, 3:6].astype(float), U, faces)].astype('<f4'))
    track['root']['children'].append(filler)
    print('%d faces of the old strips out of %s, into %s as %s'
          % (gone.sum(), ASPHALT, os.path.relpath(TRACK, ROOT), FILLER))

    # the ground a path stands on, the old strips in it or not, and the roads it runs into
    box = np.array([min(np.min(np.array(p)[:, i]) for p, _ in PATHS.values()) for i in (0, 1)]) - 15, \
        np.array([max(np.max(np.array(p)[:, i]) for p, _ in PATHS.values()) for i in (0, 1)]) + 15

    def near(t):
        p = t[:, :, [0, 2]]
        return t[(p.max(1) >= box[0]).all(1) & (p.min(1) <= box[1]).all(1)]
    ground, road = [], []
    for k in (track, kerbs, pad):
        for n, m in meshes(k):
            nm = n['name']
            if not nm[:1].isdigit() or 'WALL' in nm or nm.startswith('1ROAD_PATH_'):
                continue
            t = near(triangles(n, m))
            if not len(t):
                continue
            (road if k is not pad and nm[1:].startswith('ROAD') else ground).append(t)
    ground, road = Surface(np.concatenate(ground)), Surface(np.concatenate(road))
    # the tyre walls along the paths, from above
    tyres = gltf_source.load_kn5(TYRES)
    walls = unary_union([Polygon(a).buffer(.01) for n, m in meshes(tyres)
                         if n['name'].startswith('1WALL') for a in near(triangles(n, m))[:, :, [0, 2]]])

    root = pad['root']
    laid = []
    for name, (pts, width) in PATHS.items():
        node = mesh(asphalt, '1ROAD_PATH_' + name, lay(name, pts, width, ground, road, walls), uv_of)
        root['children'].append(node)
        laid.append(node)
    grounds = [n for n, _ in meshes(track) if n['name'] in (GROUND, FILLER)]
    sink(grounds, laid)
    # the tyres standing where the ground went down, down with it, the walls in them too
    after = Surface(np.concatenate([near(triangles(n, np.eye(4))) for n in grounds] + [top_of(p) for p in laid]
                                   + [near(triangles(n, m)) for n, m in meshes(pad)
                                      if n['name'][:1].isdigit() and 'WALL' not in n['name'] and not n['name'].startswith('1ROAD_PATH_')]))
    settled = 0
    for n, m in meshes(tyres):
        if not np.allclose(m, np.eye(4)):
            continue
        V = n['verts']
        P = V[:, [0, 2]].astype(float)
        idx = np.nonzero((P >= box[0]).all(1) & (P <= box[1]).all(1))[0]
        if not len(idx):
            continue
        drop = after.ys(P[idx]) - ground.ys(P[idx])
        k = drop < -.005
        if k.any():
            V[idx[k], 1] += drop[k].astype(V.dtype)
            sphere(n)
            settled += k.sum()
    print('%d points of the tyres down with the ground' % settled)
    if settled:
        save(tyres, TYRES)
    save(track, TRACK)
    save(pad, PADDOCK)
    print('paths %s in %s' % (', '.join(PATHS), os.path.relpath(PADDOCK, ROOT)))


if __name__ == '__main__':
    main()
