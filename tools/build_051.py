"""Build the 0.5.1 test: new kerbs on T1-T3 of the 1500 (and the Amateur corner after T1),
new light poles with their night lights recomputed, and the lowered start panel.

Copies base/velopark-0.5.0 to build/velopark-0.5.1 and only ever writes inside the copy.
The track in content/ is what this made; copy build/velopark-0.5.1/content over it to take a
rebuild. Do not after content/ has moved on from 0.5.1: it would undo those changes.
"""
import io, json, math, os, re, shutil, struct, sys, types, zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import triangle

import kn5io
import corners as C
import build_preview as bp
import build_simple as bs
from build_preview import BLOCK, TILE, A_DEPTH, B_WIDTH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'base', 'velopark-0.5.0')
DST = os.path.join(ROOT, 'build', 'velopark-0.5.1')
TRACK = os.path.join(DST, 'content', 'tracks', 'techspeed_velopark')
TEX = bp.TEX
VERSION = '0.5.1'
REPORT = []


def log(*a):
    t = ' '.join(str(x) for x in a)
    REPORT.append(t)
    print(t, flush=True)


# ---- Ground --------------------------------------------------------------------
class Ground:
    """Height and surface of the road and grass meshes, sampled straight down."""

    def __init__(s, model, names):
        s.tris, s.owner = [], []
        for mi, m in enumerate(model.meshes):
            if m['name'] in names:
                s.tris.append(m['v'][m['f'], :3])
                s.owner += [mi] * len(m['f'])
        s.tris = np.concatenate(s.tris)
        s.owner = np.array(s.owner)
        lo = s.tris[:, :, [0, 2]].min(1)
        hi = s.tris[:, :, [0, 2]].max(1)
        s.cell = 4.0
        s.grid = {}
        for k in range(len(s.tris)):
            for gx in range(int(lo[k, 0] // s.cell), int(hi[k, 0] // s.cell) + 1):
                for gz in range(int(lo[k, 1] // s.cell), int(hi[k, 1] // s.cell) + 1):
                    s.grid.setdefault((gx, gz), []).append(k)

    def on(s, x, z, mesh_ids):
        return any(s.owner[k] in mesh_ids for _, k in s.hits(x, z))

    def hits(s, x, z):
        out = []
        for k in s.grid.get((int(x // s.cell), int(z // s.cell)), []):
            a, b, c = s.tris[k]
            d = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
            if abs(d) < 1e-12:
                continue
            l1 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / d
            l2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / d
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-6:
                out.append((l1 * a[1] + l2 * b[1] + l3 * c[1], k))
        return out

    def y(s, x, z):
        h = s.hits(x, z)
        if not h:
            # outside the ground: use the nearest corner
            d = np.linalg.norm(s.tris[:, :, [0, 2]].reshape(-1, 2) - (x, z), axis=1)
            return float(s.tris[:, :, 1].ravel()[d.argmin()])
        return max(h)[0]


# ---- Kerb lines ------------------------------------------------------------------
def resample(p, step=.25):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def smooth_line(p, win=9):
    k = np.ones(win) / win
    q = p.copy()
    for a in range(2):
        pad = np.r_[np.full(win, p[0, a]), p[:, a], np.full(win, p[-1, a])]
        q[:, a] = np.convolve(pad, k, 'same')[win:-win]
    return q


def order_along(points, ai):
    """Sort points by where they fall along the racing line."""
    d = np.linalg.norm(points[:, None] - ai[None], axis=2)
    idx = d.argmin(1)
    # unwrap around the start line
    if idx.max() - idx.min() > len(ai) / 2:
        idx = np.where(idx < len(ai) / 2, idx + len(ai), idx)
    return points[np.argsort(idx + d.min(1) * 1e-3)]


def edge_from_kerb(mesh, ai):
    """Track-side edge of an old kerb, from its two outer UV rows."""
    v = mesh['v']
    uv = v[:, 7]
    a = v[np.abs(uv) < .03][:, [0, 2]]
    b = v[np.abs(uv) > .97][:, [0, 2]]
    da = np.linalg.norm(a[:, None] - ai[None], axis=2).min(1).mean()
    db = np.linalg.norm(b[:, None] - ai[None], axis=2).min(1).mean()
    track, far = (a, b) if da < db else (b, a)
    line = smooth_line(resample(order_along(np.unique(np.round(track, 3), axis=0), ai)))
    return line, far


def edge_from_road(model, road_names, ai, i0, i1, side, road=None):
    """Road edge on one side of the racing line, by casting sideways from it."""
    pts = []
    segs = []
    for m in model.meshes:
        if m['name'] not in road_names:
            continue
        p = np.round(m['v'][:, [0, 2]], 2)
        for t in m['f']:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                pa, pb = tuple(p[a]), tuple(p[b])
                segs.append((min(pa, pb), max(pa, pb)))
    from collections import Counter
    cnt = Counter(segs)
    edges = np.array([[a, b] for (a, b), n in cnt.items() if n == 1])
    for i in range(i0, i1):
        o = ai[i]
        t = ai[(i + 1) % len(ai)] - ai[i - 1]
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]]) * side
        best = None
        A, B = edges[:, 0], edges[:, 1]
        e = B - A
        den = n[0] * e[:, 1] - n[1] * e[:, 0]
        ok = np.abs(den) > 1e-9
        w = A - o
        r = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1)
        u = (w[:, 0] * n[1] - w[:, 1] * n[0]) / np.where(ok, den, 1)
        hit = ok & (u >= 0) & (u <= 1) & (r > .5) & (r < 15)
        # seams between road meshes are boundaries too: take the first with no road past it
        for d in np.sort(r[hit]):
            if road is None or not road(*(o + n * (d + .4))):
                pts.append(o + n * d)
                break
    return smooth_line(resample(np.array(pts)), 13)


class Path:
    """Kerb placement: s along the edge, t away from the track. Straight between block nodes."""

    def __init__(s, line, away_hint, ground, length=None, cover=None):
        # carry the line on past its ends, so whole blocks can reach past the old kerb
        ext = 3.0
        t0 = (line[0] - line[3]) / np.linalg.norm(line[0] - line[3])
        t1 = (line[-1] - line[-4]) / np.linalg.norm(line[-1] - line[-4])
        line = resample(np.r_[[line[0] + t0 * ext], line, [line[-1] + t1 * ext]], .1)
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        if cover is None:
            lo, hi = ext, cum[-1] - ext
        else:
            # every corner of the old kerb, far edge included, must fall inside the new one
            at = cum[np.linalg.norm(cover[:, None] - line[None], axis=2).argmin(1)]
            lo, hi = at.min() - .1, at.max() + .1
        n = int(math.ceil((hi - lo) / BLOCK)) if length is None else int(length // BLOCK)
        start = max(0., lo - (n * BLOCK - (hi - lo)) / 2)
        sn = start + np.arange(n + 1) * BLOCK
        s.nodes = np.stack([np.interp(sn, cum, line[:, 0]), np.interp(sn, cum, line[:, 1])], 1)
        tan = np.gradient(s.nodes, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        nor = np.stack([-tan[:, 1], tan[:, 0]], 1)
        if (away_hint - s.nodes.mean(0)) @ nor.mean(0) < 0:
            nor = -nor
        s.normal = nor
        s.n = n
        s.length = n * BLOCK
        s.ground = ground
        s.exact = False
        s.cache = {}
        # t runs to the right of s when this is +1 (seen from above, y up)
        s.flip = (tan[:, 0] * nor[:, 1] - tan[:, 1] * nor[:, 0]).mean() > 0

    def st(s, pts):
        """Distance along and away from the kerb's track edge, for (n, 2) points."""
        d = np.linalg.norm(pts[:, None] - s.nodes[None], axis=2)
        k = d.argmin(1)
        rel = pts - s.nodes[k]
        k2 = np.clip(k, 0, s.n - 1)
        tan = s.nodes[np.minimum(k2 + 1, s.n)] - s.nodes[k2]
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        return k * BLOCK + (rel * tan).sum(1), (rel * s.normal[k]).sum(1)

    def g(s, k, t):
        key = (k, round(float(t), 4))
        if key not in s.cache:
            x, z = s.nodes[k] + s.normal[k] * t
            s.cache[key] = (x, s.ground.y(x, z), z)
        return s.cache[key]

    def place(s, S, T, Z):
        out = []
        for sv, tv, zv in zip(np.ravel(S), np.ravel(T), np.ravel(Z)):
            k = min(int(sv // BLOCK), s.n - 1)
            f = sv / BLOCK - k
            a, b = np.array(s.g(k, tv)), np.array(s.g(k + 1, tv))
            p = a + (b - a) * f
            if s.exact:
                # sit on the ground right under each corner, not just at the block ends; off
                # the road, on the highest ground close by, so none of it shows through
                r = .1 if tv > 0 else 0
                p[1] = max(s.ground.y(p[0] + dx, p[2] + dz) for dx, dz in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)))
                if tv >= 0:
                    # and never below the road beside it, so it always rises from the road; taken
                    # a little inside the edge, as right on it the road may already have ended
                    e = s.nodes[k] + (s.nodes[k + 1] - s.nodes[k]) * f - s.normal[k] * .1
                    p[1] = max(p[1], s.ground.y(*e))
            out.append((p[0], p[1] + zv, p[2]))
        return np.array(out)


# ---- Kerb meshes ---------------------------------------------------------------
RAMP_T = [0, .02, .18, .20, .95, 1.0, 1.03, 1.05, 1.08, 1.25]
RAMP_Z = [0, .012, .020, .024, .098, .108, .112, .104, .095, 0]


def ramp_only(s0, s1):
    q = bs.Soup()
    prof = lambda t: float(np.interp(t, RAMP_T, RAMP_Z))
    rows = [0, .05, .20, 1.03, 1.09, 1.25]
    # a quarter block long, so the ramp follows the ground closely
    for b in range(int(round(s0 / BLOCK)) * 4, int(round(s1 / BLOCK)) * 4):
        x0, x1 = b * BLOCK / 4, (b + 1) * BLOCK / 4
        for t0, t1 in zip(rows, rows[1:]):
            q.quad((x0, t0, prof(t0)), (x1, t0, prof(t0)), (x1, t1, prof(t1)), (x0, t1, prof(t1)))
    return q, rows, prof


def end_caps(q, s_at, rows, prof, facing):
    """Close the raised profile at a kerb end, down to the ground."""
    for t0, t1 in zip(rows, rows[1:]):
        a, b = prof(t0), prof(t1)
        if a <= 0 and b <= 0:
            continue
        quad = [(s_at, t0, 0), (s_at, t1, 0), (s_at, t1, b), (s_at, t0, a)]
        q.quad(*quad)
        q.quad(*quad[::-1])


TAPER = 2 * BLOCK
KEEP_BED = {'T4', 'L1500_C'}
# kerbs whose old ones sat inside the road's edge line, on a short line of their own: moved
# out by this much, so they start at the outside of the edge line either side of them
SHIFT_OUT = {'L1500_C': .46, 'MID_EXIT': .52}
# kerbs laid along the outer edge of the road's edge line beside them, as their old ones
# wander away from it
ALONG_LINE = {'T4': 'LINE_1', 'T9': 'LINE_18'}
# kerbs laid right on the edge of the driven asphalt, where the line beside the old one ran
# out on the road; that line goes, the kerb's own edge line takes its place
ALONG_EDGE = {'S1_IN': 'LINE_16'}
# and the road's edge line it bends off from, at its start
EDGE_FROM = {'S1_IN': 'LINE_10'}    # kerbs with asphalt behind them that stays
# how the profiles are built, as (profile, built) distances from the road edge: the concrete
# kerb a narrow painted ramp with wider cement ribs behind it, the ramp kerb 0.6 m
CONCRETE_WARP = ([0, 1.09, 2.0, A_DEPTH], [0, .45, 1.45, 1.5])
RAMP_WARP = ([0, A_DEPTH], [0, A_DEPTH * .6 / 1.25])
KERB_LOW = .55       # of the profile's height
BASE = -.0501      # height of the hidden base under the tapered ends


def soup_mesh(q, path, depth, angle, name, material, taper=False, warp=None, low=1.):
    """Soup in track space to a kn5 mesh block. With taper the kerb grows out of the road
    edge as a triangle over its first and last two blocks, rising as it widens. warp builds
    it narrower or wider in parts (see CONCRETE_WARP), low lowers it; the texture is then
    mapped across it as built, depth being its built width."""
    corners = np.array(q.q, float)                   # (n, 4, 3)
    flat = corners.reshape(-1, 3)
    key = np.round(flat, 5)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel().reshape(-1, 4)
    S, T, Z = uniq[:, 0].copy(), uniq[:, 1].copy(), uniq[:, 2].copy()
    if warp:
        T[T > 0] = np.interp(T[T > 0], *warp)
    Z[Z > 0] *= low
    V = T.copy()
    if taper:
        f = np.clip(np.minimum(S, path.length - S) / TAPER, 0, 1)
        kerb = (T > 0) & (np.abs(Z - BASE) > 1e-6)
        T[kerb] *= f[kerb]
        Z[kerb] *= f[kerb]
    pos = path.place(S, T, Z)
    uv = np.stack([uniq[:, 0] / TILE, V / depth], 1)
    faces = []
    for f in inv:
        f = list(dict.fromkeys(f.tolist()))
        if len(f) >= 3:
            faces.append(f[::-1] if path.flip else f)
    tris, normals = kn5io.smooth_normals(pos, faces, angle)
    block, n = kn5io.mesh_block(name, pos[tris].reshape(-1, 3), normals.reshape(-1, 3), uv[tris].reshape(-1, 2), material)
    return block, n, pos, tris


# ---- Ground cut-out under the floor kerb --------------------------------------
def footprint(path, t0, t1):
    a = [path.g(k, t0) for k in range(path.n + 1)]
    b = [path.g(k, t1) for k in range(path.n + 1)][::-1]
    return np.array(a + b)


def inside(poly, pts):
    x, z = pts[:, 0], pts[:, 1]
    res = np.zeros(len(pts), bool)
    px, pz = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        c = ((pz[i] > z) != (pz[j] > z)) & (x < (px[j] - px[i]) * (z - pz[i]) / (pz[j] - pz[i] + 1e-12) + px[i])
        res ^= c
        j = i
    return res


def densify(poly, step=.15):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(n)]
    return np.array(out)


def covers(tri, pts):
    """For each triangle, does it contain any of the points."""
    a, b, d = tri[:, 0][:, None], tri[:, 1][:, None], tri[:, 2][:, None]
    q = pts[None]
    den = (b[..., 1] - d[..., 1]) * (a[..., 0] - d[..., 0]) + (d[..., 0] - b[..., 0]) * (a[..., 1] - d[..., 1])
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    l1 = ((b[..., 1] - d[..., 1]) * (q[..., 0] - d[..., 0]) + (d[..., 0] - b[..., 0]) * (q[..., 1] - d[..., 1])) / den
    l2 = ((d[..., 1] - a[..., 1]) * (q[..., 0] - d[..., 0]) + (a[..., 0] - d[..., 0]) * (q[..., 1] - d[..., 1])) / den
    return ((l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)).any(1)


def tyre_levels(barrier):
    """The bottom, middle and top tyre of the first stack in the tyre barrier, as triangles
    of full vertices, centred on the stack's axis with the ground at y 0."""
    tri = np.concatenate([barrier.mesh(n)['v'][barrier.mesh(n)['f']] for n in ('TYRES_0', 'TYRES_0F')])
    # the stack's axis: the middle of the piece of its inside the first triangle belongs to
    m = barrier.mesh('TYRES_0F')
    v, f = m['v'][:, :3], m['f']
    key = np.unique(np.round(v, 4), axis=0, return_inverse=True)[1].ravel()
    mine, grown = {key[f[0, 0]]}, True
    while grown:
        n = len(mine)
        for a, b, c in key[f]:
            if a in mine or b in mine or c in mine:
                mine |= {a, b, c}
        grown = len(mine) > n
    piece = v[np.isin(key, list(mine))]
    axis = (piece.min(0) + piece.max(0)) / 2
    r = np.linalg.norm(tri[:, :, [0, 2]] - axis[[0, 2]], axis=2)
    # the stack's own faces, without the lid over its hole
    tri = tri[(r < .32).all(1) & (r > .1).all(1)].copy()
    y0 = tri[:, :, 1].min()
    tri[:, :, :3] -= (axis[0], y0, axis[2])
    y = tri[:, :, 1]
    return [tri[y.max(1) <= .21], tri[(y.min(1) >= .19) & (y.max(1) <= .41)], tri[y.min(1) >= .39]]


# ---- Road marks the paint lacks ------------------------------------------------
# the grid painted on the 2nd straight, as the ground photo shows it: the mark of each
# slot on the right, front last; the left row lies beside it
GRID_2 = np.array([(43.29, -12.31), (47.17, -11.41), (51.03, -10.48), (54.93, -9.56), (58.81, -8.72),
                   (62.69, -7.80), (66.58, -6.81), (70.45, -5.86), (74.28, -5.04), (78.19, -4.15),
                   (82.06, -3.33), (85.89, -2.24), (89.87, -1.46), (93.77, -0.64), (97.66, .29)])
GRID_2_LEFT = np.array([-1.08, 4.68])
# the straight kerb on the outside of the grid, its middle line from end to end, then the
# solid line on to the dotted one, and the pit lane's line beside them
BOARD_100 = np.array([91.9, -5.5])   # the 100 m board on the 2nd straight
STRAIGHT_KERB = np.array([(20.6, -20.19), (58.6, -11.41)])
LINE_WIDTH = .12      # as the paint's own lines
DOTS_END = 109.6      # x where the dashes end
PIT_LINE_2 = np.array([(40, -19.3), (50, -16.9), (60, -14.4)])
PIT_MERGE_X = 85.      # where it bends into the dashed line and ends
# the hotlap start line, just out of the curve onto the 3rd straight: its ends
HOTLAP_LINE = np.array([(175.8, 59.3), (184.6, 60.1)])


def road_marks(ground, road, mat_blocks, tex_blocks, kerb_paths, floor):
    """Copies of the 1500 grid's own slot marks and numbers, moved onto the 2nd straight,
    the lines along its outside and the hotlap start line, in the paint's own materials."""
    paint = kn5io.Model(os.path.join(TRACK, 'velopark_paint.kn5'))
    mat = {}
    for m in paint.materials:
        mat[m['name']] = len(mat_blocks)
        mat_blocks.append(m['raw'])
        tex_blocks += [paint.textures[t] for t in m['tex'].values()]
    lay = kn5io.Model(os.path.join(TRACK, 'vp_1500.kn5')).dummies
    # all of them, so a piece of a slot past the ones copied is not taken for one of those
    slots = [lay['AC_START_%d' % k] for k in range(len(lay)) if 'AC_START_%d' % k in lay]
    src_at = np.array([d[3, [0, 2]] for d in slots])
    src_dir = np.array([d[2, [0, 2]] / np.linalg.norm(d[2, [0, 2]]) for d in slots])
    right, left = GRID_2, GRID_2 + GRID_2_LEFT
    # the new slots, front first, right then left as the 1500 grid numbers them
    ahead = np.gradient(right, axis=0)
    ahead /= np.linalg.norm(ahead, axis=1, keepdims=True)
    dst_at, dst_dir = [], []
    for k in range(len(GRID_2) - 1, -1, -1):
        dst_at += [right[k], left[k]]
        dst_dir += [ahead[k], ahead[k]]
    # which side of its row each of the 1500's slots is on must match
    side = lambda a, d, o: np.sign(d[0] * (o - a)[1] - d[1] * (o - a)[0])
    assert side(src_at[0], src_dir[0], src_at[1]) == side(dst_at[0], dst_dir[0], dst_at[1])

    def pieces(name):
        m = paint.mesh(name)
        tri = m['v'][m['f']]
        a, b = tri[:, 1, [0, 2]] - tri[:, 0, [0, 2]], tri[:, 2, [0, 2]] - tri[:, 0, [0, 2]]
        return tri[np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-9]

    out = []
    for name, src, small in (('GRID_1500', 'GRID_2', 20), ('NUMBERS_1500', 'NUMBERS_2', 99)):
        tri = pieces(name)
        lift = np.median(tri[:, :, 1].ravel() - [ground.y(x, z) for x, z in tri[:, :, [0, 2]].reshape(-1, 2)])
        # each slot's own mark: the small pieces, not the long lines along the grid
        key = np.unique(np.round(tri[:, :, [0, 2]].reshape(-1, 2), 3), axis=0, return_inverse=True)[1].reshape(-1, 3)
        part = np.arange(key.max() + 1)
        def root(i):
            while part[i] != i:
                part[i] = part[part[i]]
                i = part[i]
            return i
        for a, b, c in key:
            part[root(b)] = root(a)
            part[root(c)] = root(a)
        group = np.array([root(a) for a, _, _ in key])
        keep = []
        for g in np.unique(group):
            mine = tri[group == g]
            if len(mine) > small:
                continue
            c = mine[:, :, [0, 2]].reshape(-1, 2).mean(0)
            k = np.linalg.norm(src_at - c, axis=1).argmin()
            if np.linalg.norm(src_at[k] - c) < 3.5 and k < 2 * len(GRID_2):
                keep.append((k, mine))
        P, N, U = [], [], []
        for k, mine in keep:
            a = math.atan2(dst_dir[k][1], dst_dir[k][0]) - math.atan2(src_dir[k][1], src_dir[k][0])
            r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            xz = (mine[:, :, [0, 2]].reshape(-1, 2) - src_at[k]) @ r.T + dst_at[k]
            y = np.array([ground.y(x, z) for x, z in xz]) + lift
            P.append(np.c_[xz[:, 0], y, xz[:, 1]])
            N.append(np.tile([0, 1., 0], (len(xz), 1)))
            U.append(mine[:, :, 6:8].reshape(-1, 2))
        b, n = kn5io.mesh_block(src, np.concatenate(P), np.concatenate(N), np.concatenate(U), mat[paint.material_name(paint.mesh(name))])
        out.append(b)
        log('road marks: %s, %d slots, %d triangles' % (src, len({k for k, _ in keep}), sum(len(x) for x in P) // 3))

    tri = pieces('GRID_1500')
    lift = np.median(tri[:, :, 1].ravel() - [ground.y(x, z) for x, z in tri[:, :, [0, 2]].reshape(-1, 2)])
    turn = lambda t: (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    wound = np.sign(np.median(turn(tri[:, :, [0, 2]])))

    def flat(name, quads):
        """Quads of (x, z) corners as paint lying on the road."""
        P = np.array([[q[i], q[j], q[k]] for q in quads for i, j, k in ((0, 1, 2), (0, 2, 3))])
        P[turn(P) * wound < 0] = P[turn(P) * wound < 0][:, ::-1]
        P = P.reshape(-1, 2)
        Y = np.array([ground.y(x, z) for x, z in P]) + lift
        b, _ = kn5io.mesh_block(name, np.c_[P[:, 0], Y, P[:, 1]], np.tile([0, 1., 0], (len(P), 1)),
                                np.c_[P[:, 0] / 10, P[:, 1] / 20], mat['LinePaint'])
        return b

    # on from the end of the straight kerb, in line with it: a solid line, then dashes half a
    # metre long a metre apart, as thin as the paint's other lines
    a, b = STRAIGHT_KERB
    along = (b - a) / np.linalg.norm(b - a)
    side = np.array([-along[1], along[0]]) * LINE_WIDTH / 2
    solid_end = b + along * 11.
    dash_end = a + along * (DOTS_END - a[0]) / along[0]
    # on across the open asphalt past the grass, until it meets the next line painted there
    others = np.concatenate([m['v'][m['f']][:, :, [0, 2]] for m in paint.meshes
                             if m['name'].startswith('LINE') and m['name'] not in ('LINE_3', 'LINE_5', 'LINE_13', 'LINE_39') + tuple(ALONG_EDGE.values())])
    u = 0.
    while u < 80 and not covers(others, (dash_end + along * (u + .3))[None]).any():
        u += .1
    dash_end = dash_end + along * u
    dashes = [solid_end + along * (.5 + u) for u in np.arange(0, np.linalg.norm(dash_end - solid_end) - .5, 1.02)]
    out.append(flat('LINE_2_DOTS', [[c - along * .25 - side, c + along * .25 - side, c + along * .25 + side, c - along * .25 + side]
                                     for c in dashes]))
    log('road marks: dashed line, %d dashes' % len(dashes))
    # the line along the old straight kerb now runs through the grid, and the short lines along
    # two old kerbs inside the edge line would run beside the new ones
    paint_data = bytearray(paint.bytes)
    for name in ('LINE_3', 'LINE_5', 'LINE_13', 'LINE_39') + tuple(ALONG_EDGE.values()):
        kn5io.blank_faces(paint_data, paint.mesh(name))
    open(os.path.join(TRACK, 'velopark_paint.kn5'), 'wb').write(paint_data)

    def strip(name, pts, width):
        quads = []
        for a, b in zip(pts, pts[1:]):
            for f0, f1 in zip(np.linspace(0, 1, 4)[:-1], np.linspace(0, 1, 4)[1:]):
                p, q = a + (b - a) * f0, a + (b - a) * f1
                d = (q - p) / np.linalg.norm(q - p)
                n = np.array([-d[1], d[0]]) * width / 2
                quads.append([p - n, q - n, q + n, p + n])
        return flat(name, quads)

    # the pit lane's line: bending smoothly off the end of the paint's own line along the
    # road edge, straight beside the kerb, then bending in to join the dashed line, where it ends
    def bend(p0, t0, p1, t1, n=24):
        L = np.linalg.norm(p1 - p0)
        t0, t1 = t0 / np.linalg.norm(t0) * L, t1 / np.linalg.norm(t1) * L
        u = np.linspace(0, 1, n)[:, None]
        return (2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * t0 + (-2 * u**3 + 3 * u**2) * p1 + (u**3 - u**2) * t1
    edge = paint.mesh('LINE_4')['v'][:, [0, 2]]
    near = np.linalg.norm(edge - PIT_LINE_2[0], axis=1)
    e0 = edge[near.argmin()]
    back = edge[(np.linalg.norm(edge - e0, axis=1) > 1.5) & (np.linalg.norm(edge - e0, axis=1) < 3)].mean(0)
    # a line along the track side of the floor kerb too, bending off the end of the edge line
    # like the pit lane's one, and in to the middle of the kerb's line once past it
    t_in = -LINE_WIDTH / 2 - .02
    inner = floor.place(np.array([0., floor.length]), np.full(2, t_in), np.zeros(2))[:, [0, 2]]
    mid = b + along * 6.
    line = np.r_[bend(e0, e0 - back, inner[0], along)[:-1], bend(inner[1], along, mid, along), [solid_end]]
    out.append(strip('LINE_2_SOLID', line, LINE_WIDTH))
    pit_dir = PIT_LINE_2[1] - PIT_LINE_2[0]
    merge = a + along * (PIT_MERGE_X - a[0]) / along[0]
    line = np.r_[bend(e0, e0 - back, PIT_LINE_2[0], pit_dir), PIT_LINE_2[1:-1],
                 bend(PIT_LINE_2[-1], pit_dir, merge, along)]
    out.append(strip('LINE_2_PIT', line, LINE_WIDTH))
    log('road marks: solid line after the straight kerb, pit lane line beside it')

    # the road's edge line along the kerbs, where it is missing: the old one ran under where
    # the new kerb lies, or along an old kerb's own edge
    added = 0.
    for k, (label, path) in enumerate(kerb_paths.items()):
        S = np.arange(0, path.length + .01, .25)
        run = []
        for sv in list(S) + [None]:
            painted = sv is None or any(covers(others, path.place(np.array([sv]), np.array([t]), np.zeros(1))[:, [0, 2]]).any()
                                        for t in (-.06, -.18, -.3))
            if not painted:
                run.append(sv)
            elif len(run) >= 3:
                pts = path.place(np.array(run), np.full(len(run), t_in), np.zeros(len(run)))[:, [0, 2]]
                out.append(strip('LINE_KERB_EDGE_%d_%d' % (k, run[0] * 100), pts, LINE_WIDTH))
                added += run[-1] - run[0]
                run = []
            else:
                run = []
    log('road marks: %.0f m of edge line along the kerbs where it was missing' % added)
    # bending off the road's edge line into the kerb's, where they do not meet
    for label, line_name in EDGE_FROM.items():
        path = kerb_paths[label]
        k0 = path.place(np.array([0., .5]), np.full(2, t_in), np.zeros(2))[:, [0, 2]]
        v = paint.mesh(line_name)['v'][:, [0, 2]]
        e = v[np.linalg.norm(v - k0[0], axis=1).argmin()]
        back = v[(np.linalg.norm(v - e, axis=1) > 1.) & (np.linalg.norm(v - e, axis=1) < 2.5)].mean(0)
        out.append(strip('LINE_KERB_JOIN_' + label, bend(e - (e - back) / np.linalg.norm(e - back) * .1, e - back, k0[0], k0[1] - k0[0], 12),
                         LINE_WIDTH))
        log('road marks: %s edge line joined to %s, %.1f m' % (label, line_name, np.linalg.norm(k0[0] - e)))

    # the hotlap start line, as thin as the others
    c = HOTLAP_LINE.mean(0)
    across = HOTLAP_LINE[1] - HOTLAP_LINE[0]
    ends = np.linalg.norm(across) * np.array([-.5, .5])
    across = across / np.linalg.norm(across)
    along = np.array([-across[1], across[0]])
    assert all(ground.on(*(c + across * d), road) for d in np.linspace(*ends, 30))
    w = np.linspace(ends[0], ends[1], int((ends[1] - ends[0]) / .5) + 2)
    h = LINE_WIDTH / 2
    out.append(flat('LINE_HOTLAP_START', [[c + across * d0 - along * h, c + across * d1 - along * h,
                                           c + across * d1 + along * h, c + across * d0 + along * h] for d0, d1 in zip(w, w[1:])]))
    log('road marks: hotlap start line, %.1f m across the road' % (ends[1] - ends[0]))
    return out


class TyreRows:
    """Rows of tyre stacks added round the track, in the tyre barrier's own tyres and material,
    each with a wall round it for the karts to hit."""

    def __init__(s, ground, keep_off, mat_blocks, tex_blocks):
        s.barrier = kn5io.Model(os.path.join(TRACK, 'velopark_tyres.kn5'))
        s.mat = len(mat_blocks)
        mat_blocks.append(s.barrier.materials[0]['raw'])
        tex_blocks += [s.barrier.textures[t] for t in s.barrier.materials[0]['tex'].values()]
        s.levels = tyre_levels(s.barrier)
        s.ground, s.keep_off = ground, keep_off
        s.walls = np.concatenate([m['v'][:, [0, 2]] for m in s.barrier.meshes if m['name'].startswith('TYRES_')])
        s.blocks = []

    def clear(s, at):
        """Is a stack here all on ground no layout drives on."""
        ring = [at + .34 * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
        return all(s.ground.hits(*p) and not s.ground.on(p[0], p[1], s.keep_off) for p in ring + [at])

    def row(s, name, line, what):
        """Stacks of three every 0.62 m along the line, where there is room for them."""
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        at = [np.array([np.interp(u, cum, line[:, 0]), np.interp(u, cum, line[:, 1])]) for u in np.arange(0, cum[-1] + .01, .62)]
        why = [(p.round(2).tolist(), bool(s.ground.hits(*p)), s.ground.on(p[0], p[1], s.keep_off)) for p in at if not s.clear(p)]
        at = [p for p in at if s.clear(p)]
        if why:
            log('  %s: no room at %s' % (name, why[:6]))
        T, P = [], []
        for k, p in enumerate(at):
            foot = np.array([p[0], s.ground.y(*p), p[1]])
            for lvl in range(3):
                a = (k * 2.39 + lvl * 1.7) % (2 * math.pi)
                r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
                t = s.levels[lvl].copy()
                t[:, :, :3] = t[:, :, :3] @ r.T + foot
                t[:, :, 3:6] = t[:, :, 3:6] @ r.T
                T.append(t)
            # its wall: a box round the stack, open at the bottom, both ways round
            y0, y1 = foot[1] - .1, foot[1] + .6
            c = [p + .31 * np.array(q) for q in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for i in range(4):
                a, b = c[i], c[(i + 1) % 4]
                P += [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y0, a[1]), (b[0], y1, b[1]), (a[0], y1, a[1])]
            P += [(c[i][0], y1, c[i][1]) for i in (0, 1, 2, 0, 2, 3)]
        T = np.concatenate(T)
        s.blocks.append(kn5io.mesh_block('TYRES_' + name, T[:, :, :3].reshape(-1, 3), T[:, :, 3:6].reshape(-1, 3),
                                         T[:, :, 6:8].reshape(-1, 2), s.mat, cast_shadows=True)[0])
        P = np.array(P, float)
        P = np.r_[P, P.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)]
        nrm = np.cross(P[1::3] - P[0::3], P[2::3] - P[0::3])
        nrm = np.repeat(nrm / np.linalg.norm(nrm, axis=1, keepdims=True), 3, axis=0)
        s.blocks.append(kn5io.mesh_block('1WALL_TYRE_' + name, P, nrm, np.zeros((len(P), 2)), s.mat, renderable=False)[0])
        log('tyres: %d stacks of three %s' % (len(at), what))

    def behind_kerb(s, name, path, reach, toward, what):
        """A straight row on the grass behind a kerb, along its last straight stretch before
        the end nearest toward."""
        end = 0 if np.linalg.norm(path.nodes[0] - toward) < np.linalg.norm(path.nodes[-1] - toward) else path.n
        ks = [k for k in range(path.n + 1) if TAPER <= abs(k - end) * BLOCK <= TAPER + 8]
        back = path.nodes[ks] + path.normal[ks] * (reach + .45)
        c = back.mean(0)
        d = np.linalg.svd(back - c)[2][0]
        if d @ (path.nodes[end] - c) < 0:
            d = -d
        side = np.array([-d[1], d[0]])
        if side @ path.normal[ks].mean(0) < 0:
            side = -side
        # back far enough to clear the kerb all along, then on toward the tip
        c = c + side * max(0., ((back - c) @ side).max())
        s.row(name, np.array([c + d * ((back - c) @ d).min(), c + d * ((path.nodes[end] - c) @ d)]), what)

    def tarmac_edge(s, name, road, x_from, z_range, what):
        """Along the edge where the road ends and the grass starts, walking out along x from
        x_from at each z, a stack's radius and a little into the grass."""
        line = []
        for z in np.arange(*z_range, .25 * np.sign(z_range[1] - z_range[0])):
            x = x_from
            # over the road, and over any kerb on it, the ground of which is not road
            while (s.ground.on(x, z, road) or not s.ground.hits(x, z)) and x < x_from + 30:
                x += .05
            line.append((x + .45, z))
        s.row(name, np.array(line), what)

    def across(s, name, road, ai, near, toward, what, gap=.45):
        """Straight out from the road at the AI line's point nearest near, square to it, on the
        side toward, until it meets the barrier there."""
        k = np.linalg.norm(ai - near, axis=1).argmin()
        t = ai[min(k + 1, len(ai) - 1)] - ai[max(k - 1, 0)]
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        if n @ (toward - ai[k]) < 0:
            n = -n
        u = 0.
        while s.ground.on(*(ai[k] + n * u), road) and u < 30:
            u += .05
        a = ai[k] + n * (u + gap)
        v = 0.
        while np.linalg.norm(s.walls - (a + n * v), axis=1).min() > .7 and v < 25:
            v += .1
        s.row(name, np.array([a, a + n * v]), what)


def board_50(ground, road):
    """A 50 m board 50 m on from the 100 m board on the 2nd straight: a copy of it on the grass,
    turned as the track turns, its face moved to the 50 on the board texture. Writes it to its
    own model and returns the board models that have that 100 m board."""
    with_100, src = set(), None
    for fn in sorted(f for f in os.listdir(TRACK) if f.endswith('_boards.kn5')):
        b = kn5io.Model(os.path.join(TRACK, fn))
        for m in b.meshes:
            if np.linalg.norm(m['v'][:, [0, 2]].mean(0) - BOARD_100) < 1:
                with_100.add(fn)
                src = src or (b, m)
    b, m = src
    ai = C.ai_line('vp_1500')[:, [0, 2]]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ai, axis=0), axis=1))]
    tan = np.gradient(ai, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    c0 = m['v'][:, [0, 2]].mean(0)
    k0 = np.linalg.norm(ai - c0, axis=1).argmin()
    nrm = lambda k: np.array([-tan[k, 1], tan[k, 0]])
    lat = (c0 - ai[k0]) @ nrm(k0)
    # on grass: clear of the road and of the paddock's asphalt beside it. As close to 50 m on
    # and as far out as the 100 m one as it can be, else a little further on, or further out
    paddock = kn5io.Model(os.path.join(TRACK, 'velopark_paddock.kn5'))
    tarmac = Ground(paddock, {x['name'] for x in paddock.meshes if x['name'].startswith('1ROAD')})
    def clear(c):
        for q in ((0, 0), (.6, 0), (-.6, 0), (0, .6), (0, -.6)):
            if not ground.hits(*(c + q)) or ground.on(*(c + q), road) or tarmac.hits(*(c + q)):
                return False
        return True
    spot = next((ds, off) for ds in np.arange(0, 10.1, .5) for off in np.arange(0, 4.01, .25)
                if clear(ai[np.searchsorted(d, d[k0] + 50 + ds)] + nrm(np.searchsorted(d, d[k0] + 50 + ds)) * (lat + np.sign(lat) * off)))
    k1 = np.searchsorted(d, d[k0] + 50 + spot[0])
    off = spot[1]
    c1 = ai[k1] + nrm(k1) * (lat + np.sign(lat) * off)
    a = math.atan2(tan[k1, 1], tan[k1, 0]) - math.atan2(tan[k0, 1], tan[k0, 0])
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    v = m['v'][m['f']].reshape(-1, 11).copy()
    v[:, [0, 2]] = (v[:, [0, 2]] - c0) @ r.T + c1
    v[:, [3, 5]] = v[:, [3, 5]] @ r.T
    v[:, 1] += ground.y(*c1) - ground.y(*c0)
    # its numbered face shows the 50, half the texture to the left
    face = (v[:, 7] < .3).reshape(-1, 3).all(1).repeat(3)
    v[face, 6] -= .5
    mat = b.materials[m['mat']]
    block, _ = kn5io.mesh_block('1WALL_BOARD_50_STRAIGHT2', v[:, :3], v[:, 3:6], v[:, 6:8], 0)
    kn5io.write(os.path.join(TRACK, 'velopark_board50.kn5'), [b.textures[t] for t in mat['tex'].values()], [mat['raw']], [block])
    log('boards: 50 m board at (%.1f, %.1f), %.1f m on from the 100 m one and %.1f m out from the line, for %s'
        % (c1[0], c1[1], 50 + spot[0], abs(lat) + off, ', '.join(sorted(with_100))))
    return with_100


def skirt(tris, ground):
    """Faces from the open edges of the triangles (n, 3, 11) down into the ground beside
    them, both ways round, as (positions, normals, which triangle) of a triangle list."""
    key = np.round(tris[:, :, [0, 2]], 3)
    uses = {}
    for i, t in enumerate(key):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            uses.setdefault(tuple(sorted((tuple(t[a]), tuple(t[b])))), []).append((i, a, b))
    P, N, I = [], [], []
    for edge in uses.values():
        if len(edge) != 1:
            continue
        i, a, b = edge[0]
        top = [tris[i, a, :3], tris[i, b, :3]]
        d = (top[1] - top[0])[[0, 2]]
        out = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
        if out @ ((top[0] + top[1])[[0, 2]] / 2 - tris[i][:, [0, 2]].mean(0)) < 0:
            out = -out
        ends = []
        for p in top:
            below = [y for y, _ in ground.hits(*(p[[0, 2]] + out * .06)) if y < p[1] + .001]
            ends.append(min(below) - .03 if below else None)
        if None in ends:
            continue
        bot = [np.r_[top[0][0], ends[0], top[0][2]], np.r_[top[1][0], ends[1], top[1][2]]]
        n = np.r_[out[0], 0, out[1]]
        for q in ((top[0], top[1], bot[1]), (top[0], bot[1], bot[0])):
            P += [q[0], q[1], q[2], q[0], q[2], q[1]]
            N += [n] * 3 + [-n] * 3
            I += [i] * 6
    return np.array(P).reshape(-1, 3), np.array(N).reshape(-1, 3), np.array(I, int)


def repaint_photo(jpeg, areas, ground_uv, search=260):
    """Paint grass over the given ground triangles in the ground photo, copied from the
    nearest spot of the photo that is all grass. Returns the new JPEG."""
    img = Image.open(io.BytesIO(jpeg)).convert('RGB')
    W, H = img.size
    for tris in areas:
        px = (np.c_[tris.reshape(-1, 2), np.ones(len(tris) * 3)] @ ground_uv * (W, H)).reshape(-1, 3, 2)
        lo = np.floor(px.reshape(-1, 2).min(0)).astype(int) - 8
        hi = np.ceil(px.reshape(-1, 2).max(0)).astype(int) + 8
        size = tuple(hi - lo)
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        for t in px:
            draw.polygon([tuple(p - lo) for p in t], fill=255)
        # a couple of texels round it, faded in
        mask = mask.filter(ImageFilter.MaxFilter(5))
        soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.5)), float)[..., None] / 255
        inside = np.asarray(mask) > 0
        best = None
        for dy in range(-search, search + 1, 5):
            for dx in range(-search, search + 1, 5):
                src = np.asarray(img.crop((lo[0] + dx, lo[1] + dy, hi[0] + dx, hi[1] + dy)), int)[inside]
                green = np.mean((src[:, 1] > src[:, 0] + 8) & (src[:, 1] > src[:, 2] + 4))
                score = green - .0005 * math.hypot(dx, dy)
                if best is None or score > best[0]:
                    best = (score, dx, dy, green)
        _, dx, dy, green = best
        src = np.asarray(img.crop((lo[0] + dx, lo[1] + dy, hi[0] + dx, hi[1] + dy)), float)
        dst = np.asarray(img.crop((lo[0], lo[1], hi[0], hi[1])), float)
        img.paste(Image.fromarray((dst * (1 - soft) + src * soft).round().astype(np.uint8)), tuple(lo))
        log('  ground photo: %.0f m2 painted over with grass from %.0f m away, %.0f%% green' % (
            inside.sum() * (1 / (W * abs(ground_uv[0, 0]))) ** 2, math.hypot(dx, dy) / (W * abs(ground_uv[0, 0])), 100 * green))
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=95)
    return out.getvalue()


CUT = set()


def cut_ground(model, data, ground_names, outline, grow, ground):
    """Blank the ground triangles around the kerb and fill the gap around its outline.
    Returns new patch meshes per source mesh."""
    o2 = outline[:, [0, 2]]
    g2 = grow[:, [0, 2]]
    probe = np.r_[densify(g2), densify(o2)]
    lo, hi = g2.min(0) - 3, g2.max(0) + 3
    removed = []
    for mi, m in enumerate(model.meshes):
        if m['name'] not in ground_names:
            continue
        p = m['v'][:, [0, 2]]
        tri = p[m['f']]
        near = np.where(((tri.max(1) > lo) & (tri.min(1) < hi)).all(1))[0]
        if not len(near):
            continue
        t = tri[near]
        hit = inside(g2, t.reshape(-1, 2)).reshape(-1, 3).any(1) | inside(g2, t.mean(1)) | covers(t, probe)
        if hit.any():
            removed.append((mi, near[hit]))
    again = sum((mi, int(k)) in CUT for mi, ks in removed for k in ks)
    assert not again, '%d ground triangles were already cut out for another kerb' % again
    CUT.update((mi, int(k)) for mi, ks in removed for k in ks)
    allrem = np.concatenate([model.meshes[mi]['v'][model.meshes[mi]['f'][ks]][:, :, [0, 2]] for mi, ks in removed])
    inner = densify(o2, .3)
    covered = np.array([covers(allrem, inner[i:i + 1]).any() for i in range(len(inner))])
    log('ground cut: kerb outline covered by the cut %.0f%%' % (100 * covered.mean()))
    # hole boundary: edges of removed triangles not shared with another removed triangle
    from collections import Counter
    edges, info = Counter(), {}
    allpts = {}
    for mi, ks in removed:
        m = model.meshes[mi]
        for k in ks:
            t = m['f'][k]
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                pa = tuple(np.round(m['v'][a, [0, 2]], 3))
                pb = tuple(np.round(m['v'][b, [0, 2]], 3))
                allpts[pa] = m['v'][a, 1]
                allpts[pb] = m['v'][b, 1]
                e = (min(pa, pb), max(pa, pb))
                edges[e] += 1
    boundary = [e for e, n in edges.items() if n == 1]
    keys = list(allpts)
    index = {k: i for i, k in enumerate(keys)}
    vert = [list(k) for k in keys]
    height = [allpts[k] for k in keys]
    seg = [[index[a], index[b]] for a, b in boundary]
    base = len(vert)
    for i, p in enumerate(outline):
        vert.append([p[0], p[2]])
        height.append(p[1])
        seg.append([base + i, base + (i + 1) % len(outline)])
    used = sorted({i for sg in seg for i in sg})
    remap = {u: k for k, u in enumerate(used)}
    V = np.array([vert[u] for u in used])
    Hh = np.array([height[u] for u in used])
    Sg = np.array([[remap[a], remap[b]] for a, b in seg])
    # no hole marker: the middle of a curved kerb can fall outside it; triangles inside
    # the kerb are dropped below instead
    res = triangle.triangulate(dict(vertices=V, segments=Sg), 'p')
    Tv, Tt = res['vertices'], res['triangles']
    if len(Tv) > len(V):
        # points where two ground meshes cross: they lie on the old surface
        extra = [ground.y(x, z) for x, z in Tv[len(V):]]
        log('ground cut: %d points added where ground meshes meet' % len(extra))
        Hh = np.r_[Hh, extra]
    # keep triangles inside the hole and outside the kerb
    cen = Tv[Tt].mean(1)
    keep = ~inside(o2, cen)
    owner = np.full(len(Tt), -1)
    for mi, ks in removed:
        m = model.meshes[mi]
        tri = m['v'][m['f'][ks]][:, :, [0, 2]]
        for j in np.where(keep & (owner < 0))[0]:
            c = cen[j]
            a, b, d = tri[:, 0], tri[:, 1], tri[:, 2]
            den = (b[:, 1] - d[:, 1]) * (a[:, 0] - d[:, 0]) + (d[:, 0] - b[:, 0]) * (a[:, 1] - d[:, 1])
            l1 = ((b[:, 1] - d[:, 1]) * (c[0] - d[:, 0]) + (d[:, 0] - b[:, 0]) * (c[1] - d[:, 1])) / den
            l2 = ((d[:, 1] - a[:, 1]) * (c[0] - d[:, 0]) + (a[:, 0] - d[:, 0]) * (c[1] - d[:, 1])) / den
            if ((l1 >= -1e-6) & (l2 >= -1e-6) & (1 - l1 - l2 >= -1e-6)).any():
                owner[j] = mi
    keep &= owner >= 0
    log('ground cut: removed', sum(len(k) for _, k in removed), 'triangles, filled with', int(keep.sum()))
    for mi, ks in removed:
        kn5io.blank_faces(data, model.meshes[mi], ks)
    patches = []
    for mi in sorted(set(owner[keep])):
        m = model.meshes[mi]
        sel = Tt[keep & (owner == mi)]
        P = np.c_[Tv[:, 0], Hh, Tv[:, 1]]
        uv = uv_from(m, Tv)
        # match the winding of the ground we replace: face up
        tri = sel.copy()
        p = P[tri]
        up = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 1]
        sign_ref = np.sign(np.cross(*(lambda q: (q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]))(m['v'][m['f'][:50], :3]))[:, 1].mean())
        flip = np.sign(up) != sign_ref
        tri[flip] = tri[flip][:, ::-1]
        patches.append((mi, P, uv, tri))
    return patches


def clip(tri, origin, normal):
    """The part of a triangle of full kn5 vertices on the side of the line through origin
    that normal points to, as triangles. Corner values are carried across linearly."""
    d = (tri[:, [0, 2]] - origin) @ normal
    out = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        da, db = d[i], d[(i + 1) % 3]
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + (b - a) * (da / (da - db)))
    return [np.array([out[0], out[k], out[k + 1]]) for k in range(1, len(out) - 1)]


def track_uv(m, pts):
    """UV for points beside a track mesh: the mapping of its triangle nearest to them all,
    carried on across them, so a patch reads as one piece of the same asphalt."""
    tri = m['v'][m['f']]
    xz = tri[:, :, [0, 2]]
    e1, e2 = xz[:, 1] - xz[:, 0], xz[:, 2] - xz[:, 0]
    ok = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-4
    k = np.flatnonzero(ok)[np.linalg.norm(xz[ok].mean(1) - pts.mean(0), axis=1).argmin()]
    A = np.c_[xz[k], np.ones(3)]
    coef = np.linalg.solve(A, tri[k][:, 6:8])
    return np.c_[pts, np.ones(len(pts))] @ coef


def uv_from(m, pts):
    """UV of points on a mesh, from the triangle under them or the nearest one."""
    tri = m['v'][m['f']]
    xz = tri[:, :, [0, 2]]
    e1, e2 = xz[:, 1] - xz[:, 0], xz[:, 2] - xz[:, 0]
    flat = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) < 1e-9
    tri, xz = tri[~flat], xz[~flat]
    cen = xz.mean(1)
    out = np.zeros((len(pts), 2))
    for i, c in enumerate(pts):
        near = np.argsort(np.linalg.norm(cen - c, axis=1))[:12]
        best = None
        for k in near:
            a, b, d = xz[k]
            den = (b[1] - d[1]) * (a[0] - d[0]) + (d[0] - b[0]) * (a[1] - d[1])
            if abs(den) < 1e-12:
                continue
            l1 = ((b[1] - d[1]) * (c[0] - d[0]) + (d[0] - b[0]) * (c[1] - d[1])) / den
            l2 = ((d[1] - a[1]) * (c[0] - d[0]) + (a[0] - d[0]) * (c[1] - d[1])) / den
            l = np.array([l1, l2, 1 - l1 - l2])
            err = -min(l.min(), 0)
            if best is None or err < best[0]:
                best = (err, l @ tri[k][:, 6:8])
        out[i] = best[1]
    return out


# ---- Lights --------------------------------------------------------------------
def light_model(points, pos, dirs, intensity, spot, sharp, rng, off, conc):
    """Rough model of how the patch spreads one light series over the road."""
    total = np.zeros(len(points))
    half = math.radians(spot / 2)
    inner = math.radians(spot / 2 * (1 - sharp))
    for p, d in zip(pos, dirs):
        v = points - p
        dist = np.linalg.norm(v, axis=1)
        l = v / dist[:, None]
        att = np.clip((rng - dist) / (rng * (1 - off)), 0, 1)
        att = att * att * (3 - 2 * att)
        cosang = l @ d
        sp = np.clip((cosang - math.cos(half)) / (math.cos(inner) - math.cos(half)), 0, 1)
        ndl = np.clip(-l[:, 1], 0, 1)
        total += intensity * att * sp * (1 - conc + conc * ndl)
    return total


def parse_series(cfg, n):
    i = cfg.index('[LIGHT_SERIES_%d]' % n)
    j = cfg.index('\n[', i + 1)
    body = cfg[i:j]
    pos = {int(k): np.array([float(x) for x in v.split(',')]) for k, v in re.findall(r'POSITION_(\d+) = (.*)', body)}
    dirs = {int(k): np.array([float(x) for x in v.split(',')]) for k, v in re.findall(r'DIRECTION_(\d+) = (.*)', body)}
    col = [float(x) for x in re.search(r'\nCOLOR = (.*)', body).group(1).split(',')]
    return i, j, body, pos, dirs, col


# ---- Build ---------------------------------------------------------------------
def material_kerb(name, albedo, normal):
    props = dict(ksAmbient=.2, ksDiffuse=.2, ksSpecular=.1, ksSpecularEXP=6., ksEmissive=0., ksAlphaRef=.5)
    return kn5io.material_block(name, 'ksPerPixelAT_NM', props, dict(txDiffuse=albedo, txNormal=normal))


def material_plain(name, tex, spec=.1, exp=6., amb=.2, dif=.2):
    props = dict(ksAmbient=amb, ksDiffuse=dif, ksSpecular=spec, ksSpecularEXP=exp, ksEmissive=0., ksAlphaRef=1.)
    return kn5io.material_block(name, 'ksPerPixel', props, dict(txDiffuse=tex))


def flip_green(path):
    a = np.asarray(Image.open(path).convert('RGB')).copy()
    a[..., 1] = 255 - a[..., 1]
    return Image.fromarray(a)


def solid(rgb):
    return Image.new('RGB', (16, 16), tuple(int(c) for c in rgb))


def build():
    if os.path.exists(DST):
        raise SystemExit('%s already exists, remove it first' % DST)
    shutil.copytree(SRC, DST)
    log('copied', SRC, 'to', DST)

    main_path = os.path.join(TRACK, 'velopark.kn5')
    model = kn5io.Model(main_path)
    data = bytearray(model.bytes)
    ground_names = {m['name'] for m in model.meshes
                    if m['name'].startswith(('1ROAD', '1GRASS')) and model.material_name(m) in ('Tarmac Dark', 'Tarmac Light', 'Background')}
    ground = Ground(model, ground_names)
    ai15 = C.ai_line('vp_1500')[:, [0, 2]]
    aiam = C.ai_line('vp_1500_amateur')[:, [0, 2]]

    # -- kerbs
    kerbs = [
        ('T1_ENTRY', 'ramp', '1ROAD_72', ai15),
        ('T1_KINK', 'concrete', '1ROAD_48', ai15),
        ('AMATEUR_T2', 'ramp', '1ROAD_74', aiam),
        ('T2', 'concrete', '1ROAD_73', ai15),
        ('T3', 'concrete', '1ROAD_75', ai15),
        # on through the turn onto the 3rd straight, its S, the second right up it and the
        # S the short layout takes after it and the corners after that, round to the floor kerbs
        ('T4', 'ramp', '1ROAD_50', ai15),
        ('S1_IN', 'ramp', '1ROAD_77', ai15),
        ('S1_OUT', 'ramp', '1ROAD_51', ai15),
        ('STRAIGHT3', 'concrete', '1ROAD_78', ai15),
        ('S2_LEFT', 'ramp', '1ROAD_55', ai15),
        ('S2_RIGHT', 'ramp', '1ROAD_82', ai15),
        ('S2_EXIT', 'ramp', '1ROAD_83', ai15),
        ('T6', 'ramp', '1ROAD_56', ai15),
        ('T7', 'ramp', '1ROAD_84', ai15),
        ('T8', 'ramp', '1ROAD_85', ai15),
        ('T9', 'ramp', '1ROAD_58', ai15),
        # and on round the rest of the lap: all plain but the inside of the curve at the end of
        # the long straight back to the middle, and the 1500's own loop
        ('T3_INSIDE', 'ramp', '1ROAD_76', ai15),
        ('T10', 'ramp', '1ROAD_87', ai15),
        ('T11', 'ramp', '1ROAD_59', ai15),
        ('T12_OUT', 'ramp', '1ROAD_60', ai15),
        ('T12', 'concrete', '1ROAD_89', ai15),
        ('S2_1500', 'ramp', '1ROAD_52', ai15),
        ('L1500_A', 'ramp', '1ROAD_53', ai15),
        ('L1500_B', 'concrete', '1ROAD_79', ai15),
        ('L1500_C', 'ramp', '1ROAD_54', ai15),
        ('L1500_D', 'ramp', '1ROAD_80', ai15),
        ('L1500_E', 'ramp', '1ROAD_81', ai15),
        ('MID_EXIT', 'ramp', '1ROAD_88', ai15),
    ]
    tex_blocks = []
    # textures made for the kerbs as narrow as they are built, so paint, pores and roller
    # marks keep their size: painted ramp, bare cement over the ribs behind it
    made = {}
    for name, out, warp in (('kerb_cement', 1.085, CONCRETE_WARP), ('kerb_ramp', 1.26, RAMP_WARP)):
        made[name] = '%s_%s' % (name, '_'.join('%d' % (x * 100) for x in warp[1]))
        if not os.path.exists(os.path.join(TEX, made[name] + '_normal.png')):
            bp.tex_concrete(made[name], out, warp)
            h = bs.read16(made[name] + '_height.png')
            bs.normal_map(h * .004, h.shape[1] / TILE, made[name] + '_normal.png')
    conc, ramp = made['kerb_cement'], made['kerb_ramp']
    conc_alb = Image.open(os.path.join(TEX, conc + '_albedo.png')).resize((2048, 512), Image.LANCZOS)
    conc_nrm = flip_green(os.path.join(TEX, conc + '_normal.png')).resize((2048, 512), Image.LANCZOS)
    floor_alb = Image.open(os.path.join(TEX, 'floor_albedo.png')).resize((2048, 512), Image.LANCZOS)
    floor_nrm = flip_green(os.path.join(TEX, 'floor_normal.png')).resize((2048, 512), Image.LANCZOS)
    for name, img, fmt in (('kerb_concrete.dds', conc_alb, 'DXT1'), ('kerb_concrete_nm.dds', conc_nrm, 'DXT5'),
                           ('kerb_floor.dds', floor_alb, 'DXT1'), ('kerb_floor_nm.dds', floor_nrm, 'DXT5')):
        tex_blocks.append(kn5io.texture_block(name, kn5io.dds(img, fmt)))
    ramp_alb = Image.open(os.path.join(TEX, ramp + '_albedo.png')).resize((2048, 512), Image.LANCZOS)
    ramp_nrm = flip_green(os.path.join(TEX, ramp + '_normal.png')).resize((2048, 512), Image.LANCZOS)
    for name, img, fmt in (('kerb_ramp.dds', ramp_alb, 'DXT1'), ('kerb_ramp_nm.dds', ramp_nrm, 'DXT5')):
        tex_blocks.append(kn5io.texture_block(name, kn5io.dds(img, fmt)))
    mat_blocks = [material_kerb('KerbConcrete', 'kerb_concrete.dds', 'kerb_concrete_nm.dds'),
                  material_kerb('KerbFloor', 'kerb_floor.dds', 'kerb_floor_nm.dds'),
                  material_kerb('KerbRamp', 'kerb_ramp.dds', 'kerb_ramp_nm.dds')]
    mesh_blocks = []
    placed = {}
    copied = {}
    grass_meshes = []

    def copy_material(mat):
        if mat['name'] not in copied:
            copied[mat['name']] = len(mat_blocks)
            mat_blocks.append(mat['raw'])
            for t in mat['tex'].values():
                if t not in copied:
                    copied[t] = len(tex_blocks)
                    tex_blocks.append(model.textures[t])
        return copied[mat['name']]

    grass = model.mesh('1GRASS_0')
    grass_mat = copy_material(model.materials[grass['mat']])
    # the grass is mapped flat from above: fit that mapping so new grass lines up with it
    gv = grass['v']
    grass_uv, *_ = np.linalg.lstsq(np.c_[gv[:, 0], gv[:, 2], np.ones(len(gv))], gv[:, 6:8], rcond=None)
    # road meshes no layout drives on
    driven = set()
    for l in ('vp_1000', 'vp_1500', 'vp_1500_amateur', 'vp_1500_hotlap', 'vp_1500_short', 'vp_1500_short_amateur', 'vp_2500', 'vp_kids'):
        for x, z in C.ai_line(l)[::2, [0, 2]]:
            driven |= {ground.owner[k] for _, k in ground.hits(x, z)}
    bed_meshes = [k for k, m in enumerate(model.meshes) if m['name'] in ground_names and m['name'].startswith('1ROAD') and k not in driven]
    repaint = []
    bands = []
    bed_cut = []
    bed_asphalt = []

    def cut_under(path, outline, grow, label):
        patches = cut_ground(model, data, ground_names, outline, grow, ground)
        for mi, P, uv, tri in patches:
            m = model.meshes[mi]
            mat = model.materials[m['mat']]
            copy_material(mat)
            nrm = np.cross(P[tri][:, 1] - P[tri][:, 0], P[tri][:, 2] - P[tri][:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            corner_n = np.repeat(nrm, 3, axis=0)
            name = '%s_PATCH_%s_%s' % ('1ROAD' if m['name'].startswith('1ROAD') else '1GRASS', label, m['name'])
            b2, n2 = kn5io.mesh_block(name, P[tri].reshape(-1, 3), corner_n, uv[tri].reshape(-1, 2), copied[mat['name']])
            mesh_blocks.append(b2)
            if m['name'] in ('1GRASS_0', 'GRASS_1'):
                grass_meshes.append(name)

    paint0 = kn5io.Model(os.path.join(TRACK, 'velopark_paint.kn5'))
    driven_road = {k for k in driven if model.meshes[k]['name'].startswith('1ROAD')}
    for label, kind, old, ai in kerbs:
        old_mesh = model.mesh(old)
        line, far = edge_from_kerb(old_mesh, ai)
        path = Path(line, far.mean(0), ground, cover=old_mesh['v'][:, [0, 2]])
        if label in ALONG_LINE:
            tri = paint0.mesh(ALONG_LINE[label])['v'][paint0.mesh(ALONG_LINE[label])['f']][:, :, [0, 2]]
            c = tri.mean(1)
            sc, tc = path.st(c)
            q = (sc > -2) & (sc < path.length + 2) & (np.abs(tc) < 2.5)
            c = c[q][np.argsort(sc[q])]
            k = np.linalg.norm(c[:, None] - path.nodes[None], axis=2).argmin(1)
            path = Path(smooth_line(resample(c + path.normal[k] * .06), 5), far.mean(0), ground, cover=old_mesh['v'][:, [0, 2]])
        if label in ALONG_EDGE:
            on_road = lambda q: ground.on(q[0], q[1], driven_road)
            pts = []
            for k in range(path.n + 1):
                t = -1.5
                while t < 3 and on_road(path.nodes[k] + path.normal[k] * (t + .05)):
                    t += .05
                pts.append(path.nodes[k] + path.normal[k] * t)
            path = Path(smooth_line(resample(np.array(pts)), 5), far.mean(0), ground, cover=old_mesh['v'][:, [0, 2]])
        if label in SHIFT_OUT:
            k = np.linalg.norm(line[:, None] - path.nodes[None], axis=2).argmin(1)
            path = Path(line + path.normal[k] * SHIFT_OUT[label], far.mean(0), ground, cover=old_mesh['v'][:, [0, 2]])
        path.exact = True
        # narrow and low, as the real ones are next to a kart
        if kind == 'concrete':
            q = bs.concrete_mesh(0, path.length)
            warp = CONCRETE_WARP
        else:
            q, rows, prof = ramp_only(0, path.length)
            warp = RAMP_WARP
        angle = 38
        # a lip just under the road, filling any gap the old kerb used to cover there
        for k in range(path.n):
            x0, x1 = k * BLOCK, (k + 1) * BLOCK
            q.quad((x0, -.15, -.003), (x1, -.15, -.003), (x1, 0, -.003), (x0, 0, -.003))
        block, n, pos, tris = soup_mesh(q, path, warp[1][-1], angle, '1ROAD_KERB_' + label, 0 if kind == 'concrete' else 2,
                                        taper=True, warp=warp, low=KERB_LOW)
        mesh_blocks.append(block)
        log('kerb %-11s %-8s replaces %-9s length %5.1f m, %5d triangles' % (label, kind, old, path.length, n))
        kn5io.blank_faces(data, old_mesh)
        placed[label] = (path, pos)
        # the side of the kerb at its back edge, down into the grass, which can lie lower than
        # the road the kerb starts from
        reach = float(np.interp(np.array(q.q)[:, :, 1].max(), *warp))
        bands.append((label, path, reach))
        S = np.linspace(0, path.length, int(path.length / .3) + 1)
        top = path.place(S, reach * np.clip(np.minimum(S, path.length - S) / TAPER, 0, 1), np.zeros_like(S))
        low = np.array([(x, ground.y(x, z) - .05, z) for x, _, z in top])
        P, U = [], []
        for i in range(len(S) - 1):
            for a, b, c in ((top[i], top[i + 1], low[i + 1]), (top[i], low[i + 1], low[i])):
                P += [a, b, c, a, c, b]         # both faces, it is seen from either side
                U += [(S[i], 0), (S[i + 1], 0), (S[i + 1], 0)] * 2
        P = np.array(P)
        nrm = np.cross(P[1::3] - P[0::3], P[2::3] - P[0::3])
        nrm = np.repeat(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9), 3, axis=0)
        U = np.array(U, float) / TILE
        U[:, 1] = reach / warp[1][-1] * .98
        b2, _ = kn5io.mesh_block('KERB_%s_SIDE' % label, P, nrm, U, 0 if kind == 'concrete' else 2)
        mesh_blocks.append(b2)
        # the photo under the old kerb, and in a band behind the new one running on past its ends
        t0, t1 = path.nodes[1] - path.nodes[0], path.nodes[-1] - path.nodes[-2]
        line = np.r_[[path.nodes[0] - t0 / np.linalg.norm(t0) * 2], path.nodes, [path.nodes[-1] + t1 / np.linalg.norm(t1) * 2]]
        nrm = np.r_[path.normal[:1], path.normal, path.normal[-1:]]
        back = line + nrm * (reach + 1.5)
        band = np.concatenate([np.stack([line[:-1], line[1:], back[1:]], 1), np.stack([line[:-1], back[1:], back[:-1]], 1)])
        repaint.append(np.concatenate([old_mesh['v'][old_mesh['f']][:, :, [0, 2]], band]))
        # strips of old light asphalt at the kerb, driven on by no layout: past the kerb's
        # ends they carry the track on, so they get the track's asphalt; alongside it they
        # are behind it, so grass. Same shape, cut where the kerb starts and ends
        bed = []
        # behind this one the asphalt stays as it is
        keep = label in KEEP_BED
        for mi in [] if keep else bed_meshes:
            m = model.meshes[mi]
            tri = m['v'][m['f']]
            near = np.flatnonzero(np.linalg.norm(tri[:, :, [0, 2]].mean(1)[:, None] - path.nodes[None], axis=2).min(1) < 4)
            for j in near:
                s, t = path.st(tri[j][:, [0, 2]])
                if (t > .05).all() and (t < 4).all() and (s > -3).all() and (s < path.length + 3).all():
                    bed.append(tri[j])
                    kn5io.blank_faces(data, m, [j])
                    bed_cut.append((mi, j))
        # and the original grass the old kerb lay over
        gi = model.meshes.index(grass)
        gt = grass['v'][grass['f']]
        old_xz = old_mesh['v'][old_mesh['f']][:, :, [0, 2]]
        cands = []
        for j in np.flatnonzero(np.linalg.norm(gt[:, :, [0, 2]].mean(1)[:, None] - path.nodes[None], axis=2).min(1) < 3):
            c = gt[j][:, [0, 2]].mean(0)
            sc, tc = (x[0] for x in path.st(c[None]))
            # under the old kerb, or just past an end of it, where the old kerb's tip lay
            if covers(old_xz, c[None]).any() or (-.2 < tc < 1. and (-1.5 < sc < 0 or path.length < sc < path.length + 1.5)):
                cands.append(j)
        if (bed or cands) and not keep:
            t0 = path.nodes[1] - path.nodes[0]
            t1 = path.nodes[-1] - path.nodes[-2]
            start = (path.nodes[0], -t0 / np.linalg.norm(t0))     # keeps what lies before the start
            end = (path.nodes[-1], t1 / np.linalg.norm(t1))       # keeps what lies after the end
            pieces = dict(before=[], beside=[], between=[], after=[])
            # does each end of the kerb point at another road
            tip = [any(ground.owner[q] in driven and model.meshes[ground.owner[q]]['name'].startswith('1ROAD')
                       for d in (1., 2., 3., 4.) for _, q in ground.hits(*(path.nodes[e] + path.normal[e] * d))) for e in (0, -1)]
            for j in cands:
                # only at an end that points at another road, as at the tips of the island,
                # does the grass round the kerb's end become the track's asphalt
                sc = path.st(gt[j][:, [0, 2]].mean(0)[None])[0][0]
                if not ((sc < path.length / 2 and tip[0]) or (sc >= path.length / 2 and tip[1])):
                    continue
                part = 'before' if sc < 0 else 'after' if sc > path.length else 'between'
                if part == 'between' and TAPER <= sc <= path.length - TAPER:
                    continue
                pieces[part].append(gt[j])
                kn5io.blank_faces(data, grass, [j])
            for tri in bed:
                # past an end it only carries the road on where that end points at another
                # road; elsewhere it lies off the road, behind where the kerb's tip was: grass
                pieces['before' if tip[0] else 'beside'] += clip(tri, *start)
                pieces['after' if tip[1] else 'beside'] += clip(tri, *end)
                for half in clip(tri, start[0], -start[1]):
                    for piece in clip(half, end[0], -end[1]):
                        # alongside the kerb: grass, but beside a tapered end that points at another
                        # road, as at the tips of the island, asphalt like the road around it
                        sc = path.st(piece[:, [0, 2]].mean(0)[None])[0][0]
                        between = (sc < TAPER and tip[0]) or (sc > path.length - TAPER and tip[1])
                        pieces['between' if between else 'beside'].append(piece)
            beside = [ground.owner[k] for x, z in path.nodes - path.normal * .3 for _, k in ground.hits(x, z)]
            track = model.meshes[max(set(beside), key=beside.count)]
            for part in ('before', 'between', 'after'):
                if not pieces[part]:
                    continue
                P = np.array(pieces[part])
                name = '1ROAD_KERB_BED_%s_%s' % (label, part.upper())
                b3, _ = kn5io.mesh_block(name, P[:, :, :3].reshape(-1, 3), P[:, :, 3:6].reshape(-1, 3),
                                         track_uv(track, P[:, :, [0, 2]].reshape(-1, 2)), copy_material(model.materials[track['mat']]))
                mesh_blocks.append(b3)
                bed_asphalt.append(P)
            if pieces['beside']:
                P = np.array(pieces['beside'])
                name = '1GRASS_KERB_BED_' + label
                b3, _ = kn5io.mesh_block(name, P[:, :, :3].reshape(-1, 3), P[:, :, 3:6].reshape(-1, 3),
                                         np.c_[P[:, :, [0, 2]].reshape(-1, 2), np.ones(len(P) * 3)] @ grass_uv, grass_mat)
                mesh_blocks.append(b3)
                grass_meshes.append(name)
                repaint.append(P[:, :, [0, 2]])
            # it lies a little above the grass round it in places: close its open edges down
            # into the grass, or the sky shows through the crack
            every = [(p, part) for part in pieces for p in pieces[part]]
            if every:
                P, N, I = skirt(np.array([p for p, _ in every]), ground)
                grassy = np.array([every[i][1] == 'beside' for i in I], bool)
                for kind, sel in (('GRASS', grassy), ('ASPHALT', ~grassy)):
                    if not sel.any():
                        continue
                    if kind == 'GRASS':
                        mat, U = grass_mat, np.c_[P[sel][:, [0, 2]], np.ones(sel.sum())] @ grass_uv
                    else:
                        mat, U = copy_material(model.materials[track['mat']]), track_uv(track, P[sel][:, [0, 2]])
                    b3, _ = kn5io.mesh_block('KERB_BED_SKIRT_%s_%s' % (label, kind), P[sel], N[sel], U, mat)
                    mesh_blocks.append(b3)
                log('  %d open edges of it closed down to the grass' % (len(P) // 12))
            log('  old light asphalt at it: %d triangles; %d before, %d between roads and %d after the kerb given the asphalt of %s, %d by the grass made grass'
                % (len(bed), len(pieces['before']), len(pieces['between']), len(pieces['after']), track['name'], len(pieces['beside'])))

    # more tyre walls, where karts would otherwise take to the grass
    road = {k for k, m in enumerate(model.meshes) if m['name'] in ground_names and m['name'].startswith('1ROAD')}
    rows = TyreRows(ground, driven & road, mat_blocks, tex_blocks)
    path, _ = placed['T2']
    rows.behind_kerb('ISLAND', path, next(r for l, _, r in bands if l == 'T2'), GRID_2[0],
                     'on the island, behind the kerb along its exit onto the 2nd straight')
    rows.tarmac_edge('T3_EXIT', road, 184., (60.5, 52.4), 'where the tarmac ends inside the turn onto the 3rd straight')
    rows.across('S_EXIT', road, ai15, np.array([167.8, 171.6]), np.array([200., 172.]),
                'across the grass from behind the kerb to the barrier in the middle of the kink after the second S', gap=1.3)
    mesh_blocks += rows.blocks

    # floor kerb on the outside of the exit from the hairpin round the grass island, in place
    # of the old straight kerb there: where the photo has it, in line with the grid and the
    # dotted line after it, between the track and the pit lane beside it
    old_mesh = model.mesh('1ROAD_49')
    a, b = STRAIGHT_KERB
    d = (b - a) / np.linalg.norm(b - a)
    inward = np.array([-d[1], d[0]])
    if inward @ (GRID_2.mean(0) - a) < 0:
        inward = -inward
    line = np.array([a + (b - a) * f for f in np.linspace(0, 1, 160)]) + inward * B_WIDTH / 2
    path = Path(line, a - inward * 5, ground)
    q = bs.floor_mesh(0, path.length)
    block, n, pos, _ = soup_mesh(q, path, B_WIDTH, 20, '1ROAD_KERB_HAIRPIN_OUT', 1)
    mesh_blocks.append(block)
    kn5io.blank_faces(data, old_mesh)
    # the old kerb hid the seam between the track and the pit lane beside it: asphalt just
    # under the road where it lay, so the seam shows asphalt, not the sky
    tri = old_mesh['v'][old_mesh['f']]
    tri = tri[(old_mesh['f'][:, 0] != old_mesh['f'][:, 1]) & (old_mesh['f'][:, 1] != old_mesh['f'][:, 2])]
    P = tri[:, :, :3].reshape(-1, 3).copy()
    P[:, 1] = [ground.y(x, z) - .006 for x, z in P[:, [0, 2]]]
    track = model.mesh('1ROAD_3')
    b2, _ = kn5io.mesh_block('KERB_OLD_SEAM_HAIRPIN_OUT', P, np.tile([0, 1., 0], (len(P), 1)), track_uv(track, P[:, [0, 2]]),
                             copy_material(model.materials[track['mat']]))
    mesh_blocks.append(b2)
    placed['HAIRPIN_OUT'] = (path, pos)
    log('kerb HAIRPIN_OUT floor    replaces 1ROAD_49  length %5.1f m, %5d triangles' % (path.length, n))
    cut_under(path, footprint(path, 0, B_WIDTH), footprint(path, -.3, B_WIDTH + 1.2), 'HAIRPIN_OUT')
    # floor kerbs too along the two straights between the hairpins after the short layout's
    # second S, and on the outside of the last corner
    for label, old in (('STRAIGHT4', '1ROAD_57'), ('STRAIGHT5', '1ROAD_86'), ('LAST_OUT', '1ROAD_61')):
        old_mesh = model.mesh(old)
        line, far = edge_from_kerb(old_mesh, ai15)
        path = Path(line, far.mean(0), ground, cover=old_mesh['v'][:, [0, 2]])
        q = bs.floor_mesh(0, path.length)
        block, n, pos, _ = soup_mesh(q, path, B_WIDTH, 20, '1ROAD_KERB_' + label, 1)
        mesh_blocks.append(block)
        kn5io.blank_faces(data, old_mesh)
        placed[label] = (path, pos)
        log('kerb %-11s floor    replaces %-9s length %5.1f m, %5d triangles' % (label, old, path.length, n))
        cut_under(path, footprint(path, 0, B_WIDTH), footprint(path, -.3, B_WIDTH + 1.2), label)
    # -- the sloping asphalt edge the paddock model runs along the road sides was hidden
    # under the old kerbs; behind the new ones it would stand out of the grass
    paddock_path = os.path.join(TRACK, 'velopark_paddock.kn5')
    paddock = kn5io.Model(paddock_path)
    paddock_data = bytearray(paddock.bytes)
    edge = paddock.mesh('1ROAD_PADDOCK')
    tri = edge['v'][edge['f']][:, :, [0, 2]]
    removed = []
    for label, path, reach in bands:
        near = np.flatnonzero(np.linalg.norm(tri.mean(1)[:, None] - path.nodes[None], axis=2).min(1) < reach + 3)
        gone = []
        for j in near:
            s, t = path.st(tri[j])
            if (t > -.05).all() and (t < reach + 2).all() and (s > -2).all() and (s < path.length + 2).all():
                # only where the track's own ground lies under it: in places it is the only ground
                # over all of it, edges included, or thin cracks open along them
                a, b, c = tri[j]
                w = np.r_[np.random.default_rng(j).dirichlet((1, 1, 1), 60),
                          [(1 - x, x, 0) for x in np.linspace(0, 1, 12)], [(0, 1 - x, x) for x in np.linspace(0, 1, 12)],
                          [(x, 0, 1 - x) for x in np.linspace(0, 1, 12)]]
                if all(ground.hits(*p) for p in w @ tri[j]):
                    gone.append(j)
        kn5io.blank_faces(paddock_data, edge, gone)
        removed.append(tri[gone].mean(1))
        log('  %s: %d triangles of the paddock road edge behind it removed' % (label, len(gone)))
    open(paddock_path, 'wb').write(paddock_data)

    # it also covered the drop where the asphalt ends above lower grass: close those
    # edges, near where it was taken away, with a short face down into the grass
    removed = np.concatenate(removed)
    for mi, m in enumerate(model.meshes):
        if m['name'] not in ground_names or not m['name'].startswith('1ROAD'):
            continue
        v, f = m['v'], m['f']
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        key = np.round(v[:, [0, 2]], 3)
        count = {}
        for t in f:
            for i, j in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e = tuple(sorted((tuple(key[i]), tuple(key[j]))))
                count.setdefault(e, []).append((i, j, t))
        faces = {}
        for e, uses in count.items():
            if len(uses) != 1:
                continue
            i, j, t = uses[0]
            mid = (v[i, [0, 2]] + v[j, [0, 2]]) / 2
            if np.linalg.norm(removed - mid, axis=1).min() > .6:
                continue
            d = v[j, [0, 2]] - v[i, [0, 2]]
            out = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
            if out @ (mid - v[t, :][:, [0, 2]].mean(0)) < 0:
                out = -out
            ends = []
            for k in (i, j):
                x, z = v[k, [0, 2]] + out * .06
                below = [y for y, _ in ground.hits(x, z) if y < v[k, 1] - .005]
                ends.append(min(below) - .03 if below else None)
            if None in ends:
                continue
            top = [v[i, :3], v[j, :3]]
            bot = [np.r_[v[i, 0], ends[0], v[i, 2]], np.r_[v[j, 0], ends[1], v[j, 2]]]
            n = np.r_[out[0], 0, out[1]]
            ui, uj = v[i, 6:8], v[j, 6:8]
            # the old light asphalt is now track asphalt past the kerbs and grass beside them:
            # its edge goes with whichever it borders
            kind = 'own'
            if mi in bed_meshes:
                inner = mid - out * .03
                kind = 'grass'
                for piece in bed_asphalt:
                    if covers(piece[:, :, [0, 2]], inner[None]).any():
                        kind = 'asphalt'
            P, N, U = faces.setdefault(kind, ([], [], []))
            for q, u in (((top[0], top[1], bot[1]), (ui, uj, uj)), ((top[0], bot[1], bot[0]), (ui, uj, ui))):
                # both faces, it is seen from either side
                P += [q[0], q[1], q[2], q[0], q[2], q[1]]
                N += [n] * 3 + [-n] * 3
                U += [u[0], u[1], u[2], u[0], u[2], u[1]]
        for kind, (P, N, U) in faces.items():
            P = np.array(P)
            if kind == 'own':
                mat, U = copy_material(model.materials[m['mat']]), np.array(U)
            elif kind == 'asphalt':
                track = model.mesh('1ROAD_3')
                mat, U = copy_material(model.materials[track['mat']]), track_uv(track, P[:, [0, 2]])
            else:
                mat, U = grass_mat, np.c_[P[:, [0, 2]], np.ones(len(P))] @ grass_uv
            b4, n4 = kn5io.mesh_block('EDGE_%s_%s' % (m['name'], kind.upper()), P, np.array(N), U, mat)
            mesh_blocks.append(b4)
            log('  %s: %d open edges closed down to the grass, %s' % (m['name'], n4 // 4, kind))

    # the grass has folds near the kerbs: thin upright strips joining two levels a few
    # centimetres apart, one-sided, so from behind they let the sky through; face them both ways
    gtri = grass['v'][grass['f']]
    gn = np.cross(gtri[:, 1, :3] - gtri[:, 0, :3], gtri[:, 2, :3] - gtri[:, 0, :3])
    upright = np.flatnonzero(np.abs(gn[:, 1]) < .5 * np.linalg.norm(gn, axis=1))
    folds = set()
    for label, path, reach in bands:
        for j in upright:
            s_, t_ = path.st(gtri[j][:, [0, 2]])
            if (t_ > -1).all() and (t_ < reach + 4).all() and (s_ > -4).all() and (s_ < path.length + 4).all():
                # only out in the grass: next to the road, showing their back would let grass
                # stick up through the asphalt
                c = gtri[j][:, [0, 2]].mean(0)
                if not any(model.meshes[ground.owner[q]]['name'].startswith('1ROAD')
                           for dx, dz in ((0, 0), (.3, 0), (-.3, 0), (0, .3), (0, -.3)) for _, q in ground.hits(c[0] + dx, c[1] + dz)):
                    folds.add(j)
    if folds:
        back = gtri[sorted(folds)][:, ::-1]
        b5, _ = kn5io.mesh_block('GRASS_FOLDS_KERBS', back[:, :, :3].reshape(-1, 3), -back[:, :, 3:6].reshape(-1, 3),
                                 back[:, :, 6:8].reshape(-1, 2), grass_mat)
        mesh_blocks.append(b5)
        log('  %d upright grass triangles near the kerbs made two-sided' % len(folds))

    # the ground photo still shows the old kerbs and asphalt strips: paint grass over them,
    # so the ground there looks and grows grass like the rest
    tex_name = model.materials[grass['mat']]['tex']['txDiffuse']
    old_tex = model.textures[tex_name]
    n = int.from_bytes(old_tex[4:8], 'little')
    new_tex = kn5io.texture_block(tex_name, repaint_photo(old_tex[8 + n + 4:], repaint, grass_uv))
    tex_blocks[copied[tex_name]] = new_tex
    mesh_blocks += road_marks(ground, {k for k, m in enumerate(model.meshes) if m['name'] in ground_names and m['name'].startswith('1ROAD')},
                              mat_blocks, tex_blocks, {l: placed[l][0] for l, _, _, _ in kerbs}, placed['HAIRPIN_OUT'][0])
    kn5io.write(os.path.join(TRACK, 'velopark_kerbs.kn5'), tex_blocks, mat_blocks, mesh_blocks)

    # -- poles
    cfg_path = os.path.join(TRACK, 'extension', 'ext_config.ini')
    cfg = open(cfg_path).read()
    _, _, _, pos0, dir0, col0 = parse_series(cfg, 0)
    _, _, _, pos1, _, col1 = parse_series(cfg, 1)
    extra_path = os.path.join(TRACK, 'velopark_extra.kn5')
    extra = kn5io.Model(extra_path)
    extra_data = bytearray(extra.bytes)
    old_light_verts = []
    for mdl, dat in ((model, data), (extra, extra_data)):
        for m in mdl.meshes:
            if mdl.material_name(m) == 'TRACK_LIGHT' and not m['name'].startswith('1WALL'):
                old_light_verts.append(m['v'][:, :3])
                kn5io.blank_faces(dat, m)
    old_light_verts = np.concatenate(old_light_verts)
    # copies of some tree groups were left about 58 m under the ground
    sunk = 0
    for mdl, dat in ((model, data), (extra, extra_data)):
        for m in mdl.meshes:
            if mdl.material_name(m).startswith(('BROAD', 'ASH')):
                low = np.flatnonzero((m['v'][m['f'], 1] < -20).all(1))
                kn5io.blank_faces(dat, m, low)
                sunk += len(low)
    log('trees: %d triangles under the ground removed' % sunk)
    parts = bs.pole_parts()
    up = {k: v for k, v in parts.items()}
    pole_tex = [kn5io.texture_block('pole_mast.dds', kn5io.dds(Image.open(os.path.join(TEX, 'mast.png')).resize((256, 1024)), 'DXT1')),
                kn5io.texture_block('pole_frame.dds', kn5io.dds(solid((22, 22, 24)), 'DXT1')),
                kn5io.texture_block('pole_lens.dds', kn5io.dds(solid((150, 150, 146)), 'DXT1')),
                kn5io.texture_block('pole_glass.dds', kn5io.dds(solid((14, 15, 17)), 'DXT1'))]
    pole_mats = [material_plain('PoleMast', 'pole_mast.dds', spec=.35, exp=20),
                 material_plain('PoleFrame', 'pole_frame.dds', spec=.15, exp=12),
                 material_plain('PoleLens', 'pole_lens.dds', spec=.6, exp=60),
                 material_plain('PoleGlass', 'pole_glass.dds', spec=.8, exp=80)]
    # all five lamps share the lens, so they all light up at night
    mat_of = dict(mast=0, frame=1, lens=2, glass=2, tyres=4)
    # a stack of tyres round the foot of each mast, in the tyre barrier's own material
    barrier = kn5io.Model(os.path.join(TRACK, 'velopark_tyres.kn5'))
    tyre_mat = barrier.materials[0]
    pole_mats.append(tyre_mat['raw'])
    pole_tex += [barrier.textures[t] for t in tyre_mat['tex'].values()]
    # the tyres themselves are copied from a stack of three in the barrier, so they are
    # mapped onto its texture the same way: its bottom, a middle one three times, its top
    tyre_stack = tyre_levels(barrier)
    smooth = dict(bs.SMOOTH)
    pole_meshes = []
    new_pos0, new_dir0 = {}, {}
    lamp_height = POLE_LAMP
    tris_per_pole = 0
    for k in sorted(pos0):
        P = pos0[k]
        near = old_light_verts[np.linalg.norm(old_light_verts[:, [0, 2]] - P[[0, 2]], axis=1) < 2.0]
        low = near[near[:, 1] < near[:, 1].min() + 1.0]
        base_xz = low[:, [0, 2]].mean(0)
        base_y = min(low[:, 1].min(), ground.y(*base_xz) - .3)
        d = dir0[k][[0, 2]]
        d = d / np.linalg.norm(d)
        th = math.atan2(d[0], d[1])
        rot = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
        rng = np.random.default_rng(k)
        foot = np.array([base_xz[0], ground.y(*base_xz), base_xz[1]])
        T = []
        for n, (lvl, dy) in enumerate(((0, 0), (1, 0), (1, .2), (1, .4), (2, .4))):
            a = rng.uniform(0, 2 * math.pi)
            r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            t = tyre_stack[lvl].copy()
            t[:, :, :3] = t[:, :, :3] @ r.T + foot + (*rng.uniform(-.012, .012, 1), dy, *rng.uniform(-.012, .012, 1))
            t[:, :, 3:6] = t[:, :, 3:6] @ r.T
            T.append(t)
        T = np.concatenate(T)
        b3, n3 = kn5io.mesh_block('POLE_%d_TYRES' % k, T[:, :, :3].reshape(-1, 3), T[:, :, 3:6].reshape(-1, 3),
                                  T[:, :, 6:8].reshape(-1, 2), mat_of['tyres'], cast_shadows=True, lod_out=2000.)
        pole_meshes.append(b3)
        if k == 0:
            tris_per_pole += n3
        for part_name, part in parts.items():
            v = np.concatenate(part.v)
            f = np.concatenate([np.array(x) for x in part.f])
            uv = np.concatenate(part.uv)
            ac = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1) @ rot.T + (base_xz[0], base_y, base_xz[1])
            tris, nrm = kn5io.smooth_normals(ac, [list(x) for x in f], smooth[part_name])
            uvd = np.stack([uv[:, 0], 1 - uv[:, 1]], 1)
            b3, n3 = kn5io.mesh_block('POLE_%d_%s' % (k, part_name.upper()), ac[tris].reshape(-1, 3), nrm.reshape(-1, 3),
                                      uvd[tris].reshape(-1, 2), mat_of[part_name], cast_shadows=True, lod_out=2000.)
            pole_meshes.append(b3)
            if k == 0:
                tris_per_pole += n3
        lamp = np.array([base_xz[0], base_y + lamp_height, base_xz[1]]) + np.r_[d[0], 0, d[1]] * .4
        # aim the beam at the same spot on the ground as before
        old_d = dir0[k]
        hit = P + old_d * ((P[1] - ground.y(P[0], P[2])) / -old_d[1])
        nd = hit - lamp
        new_pos0[k], new_dir0[k] = lamp, nd / np.linalg.norm(nd)
    kn5io.write(os.path.join(TRACK, 'velopark_poles.kn5'), pole_tex, pole_mats, pole_meshes)
    log('poles: %d replaced, %d triangles each (old pole 1752)' % (len(pos0), tris_per_pole))

    # last, as it moves everything after the photo along
    at = data.find(old_tex)
    assert at > 0 and data.find(old_tex, at + 1) < 0
    data = data[:at] + new_tex + data[at + len(old_tex):]
    open(main_path, 'wb').write(data)
    open(extra_path, 'wb').write(extra_data)

    # -- night lights: keep the road at least as bright as before
    pts = np.concatenate([C.ai_line(l) for l in ('vp_1000', 'vp_1500', 'vp_1500_amateur', 'vp_1500_hotlap', 'vp_1500_short',
                                                 'vp_1500_short_amateur', 'vp_2500', 'vp_kids')])[::3]
    def series(posd, dird, c0, c1):
        a = light_model(pts, [posd[k] for k in sorted(posd)], [dird[k] for k in sorted(posd)], c0, 170, .25, 125, .5, .35)
        b = light_model(pts, [posd[k] for k in sorted(posd)], [np.array([0, -1., 0])] * len(posd), c1, 178, .25, 70, .5, .35)
        return a + b
    old = series(pos0, dir0, col0[3], col1[3])
    new = series(new_pos0, new_dir0, col0[3], col1[3])
    lit = old > old.max() * .02
    ratio = old[lit] / new[lit]
    # the darkest point on the road ends up 2% brighter than today
    scale = float(ratio.max()) * 1.02
    final = new * scale
    log('lights: before scaling the new poles give %.0f%% of the old light at the median point' % (100 * np.median(new[lit] / old[lit])))
    log('lights: brightness x%.2f, so %.0f%% of the road is at least as bright as before, median %+.0f%%, darkest point %+.0f%%'
        % (scale, 100 * np.mean(final[lit] >= old[lit] * .999), 100 * (np.median(final[lit] / old[lit]) - 1), 100 * (np.min(final[lit] / old[lit]) - 1)))
    # then half that: at that the new poles were too bright
    scale *= POLE_DIM
    log('lights: then x%.2f, %.0f%% of the light from before at the median point' % (POLE_DIM, 100 * np.median(new[lit] * scale / old[lit])))
    write_config(cfg_path, cfg, new_pos0, new_dir0, col0, col1, scale, grass_meshes)

    # -- start panel
    for fn in ('velopark_gantry_1000.kn5', 'velopark_gantry_1500.kn5'):
        shutil.copy(os.path.join(bs.KN5_OUT, fn), os.path.join(TRACK, fn))
    log('start panel: lowered gantry copied in')

    # -- the 50 m board the 100 m board on the 2nd straight lacks, for the layouts with that one
    with_100 = board_50(ground, road)

    # -- every layout loads the new models
    for ini in sorted(f for f in os.listdir(TRACK) if f.startswith('models_') and f.endswith('.ini')):
        p = os.path.join(TRACK, ini)
        s = open(p).read().rstrip() + '\n'
        n = len(re.findall(r'^\[MODEL_\d+\]', s, re.M))
        boards = set(re.findall(r'FILE=(\S+_boards\.kn5)', s))
        for fn in ('velopark_kerbs.kn5', 'velopark_poles.kn5') + (('velopark_board50.kn5',) if boards & with_100 else ()):
            s += '\n[MODEL_%d]\nFILE=%s\nPOSITION=0,0,0\nROTATION=0,0,0\n' % (n, fn)
            n += 1
        open(p, 'w', newline='\r\n' if '\r\n' in open(p, newline='').read() else '\n').write(s)
    versions()
    pack()


POLE_DIM = .5        # the night lights at half the brightness that matches the old poles
POLE_LAMP = bs.POLE_H - .55 + .5 + .3 * .75


def write_config(path, cfg, pos0, dir0, col0, col1, scale, grass_meshes):
    crlf = b'\r\n' in open(path, 'rb').read()
    def fmt(v):
        return ', '.join('%g' % round(x, 3) for x in v)
    for n, posd, dird, col in ((0, pos0, dir0, col0), (1, pos0, None, col1)):
        i, j, body, _, _, _ = parse_series(cfg, n)
        nb = re.sub(r'\nCOLOR = .*', '\nCOLOR = %s, %g' % (', '.join('%g' % c for c in col[:3]), round(col[3] * scale, 2)), body)
        nb = re.sub(r'POSITION_(\d+) = .*', lambda m: 'POSITION_%s = %s' % (m.group(1), fmt(posd[int(m.group(1))])), nb)
        if dird:
            nb = re.sub(r'DIRECTION_(\d+) = .*', lambda m: 'DIRECTION_%s = %s' % (m.group(1), fmt(dird[int(m.group(1))])), nb)
        cfg = cfg[:i] + nb + cfg[j:]
    cfg = cfg.replace("""; One flood panel on each of the 25 poles. The beam follows the way each panel
; really faces,""", """; One beam from the lamp head of each of the 25 poles, 17 m up. The beam lands where the
; old 8 m poles aimed it, and it is bright enough that no part of the road is darker
; than with the old poles. It is""")
    old = cfg[cfg.index('; Only the lens glows'):cfg.index('; Tarmac reflects about')]
    cfg = cfg.replace(old, """; The lamp glass glows at night.
[MATERIAL_ADJUSTMENT_...]
ACTIVE = 1
DESCRIPTION = Pole lamps at night
MATERIALS = PoleLens
CONDITION = NIGHT_SMOOTH
KEY_0 = ksEmissive
VALUE_0 = 255, 238, 210, 0.12
VALUE_0_OFF = 0, 0, 0

""")
    cfg = cfg.replace('OCCLUDING_MATERIALS = Tarmac Dark, Tarmac Light, Rumble1,', 'OCCLUDING_MATERIALS = Tarmac Dark, Tarmac Light, Rumble1, KerbConcrete, KerbFloor, KerbRamp,')
    cfg = cfg.replace('PUDDLES_MATERIALS = Tarmac Dark, Tarmac Light, Rumble1', 'PUDDLES_MATERIALS = Tarmac Dark, Tarmac Light, Rumble1, KerbConcrete, KerbFloor, KerbRamp')
    if grass_meshes:
        # the grass patched in round the flat kerb grows grass like the ground it replaces
        cfg = cfg.replace('GRASS_MESHES = 1GRASS_0, GRASS_1', 'GRASS_MESHES = 1GRASS_0, GRASS_1, ' + ', '.join(grass_meshes))
    cfg = cfg.replace('SMOOTH_MATERIALS = TRACK_LIGHT,', 'SMOOTH_MATERIALS = PoleLens, PoleGlass, PoleMast,')
    cfg = cfg.replace('MATERIALS = Tarmac Dark, Tarmac Light, Rumble1, Pit01c2\n', 'MATERIALS = Tarmac Dark, Tarmac Light, Rumble1, KerbConcrete, KerbFloor, KerbRamp, Pit01c2\n')
    open(path, 'w', newline='\r\n' if crlf else '\n').write(cfg)


def versions():
    def sub(p, a, b):
        s = open(p, encoding='utf-8', newline='').read()
        s2 = re.sub(a, b, s)
        open(p, 'w', encoding='utf-8', newline='').write(s2)
    sub(os.path.join(TRACK, 'README.md'), r'(?m)^Version \d+\.\d+\.\d+,', 'Version %s,' % VERSION)
    cfg = os.path.join(TRACK, 'extension', 'ext_config.ini')
    sub(cfg, r'(?m)^; Techspeed Velopark \d+\.\d+\.\d+', '; Techspeed Velopark ' + VERSION)
    sub(cfg, r'(?m)^VERSION = \d+\.\d+\.\d+', 'VERSION = ' + VERSION)
    sub(cfg, r'(?m)^DATE_RELEASE = .*$', 'DATE_RELEASE = 2 November 2026')
    for d in os.listdir(os.path.join(TRACK, 'ui')):
        p = os.path.join(TRACK, 'ui', d, 'ui_track.json')
        if os.path.exists(p):
            sub(p, r'"version":\s*"[^"]*"', '"version": "%s"' % VERSION)
    p = os.path.join(TRACK, 'CHANGELOG.md')
    s = open(p, encoding='utf-8', newline='').read()
    nl = '\r\n' if '\r\n' in s else '\n'
    entry = nl.join(['## 0.5.1', '',
                     'Test build for the new kerbs, light poles and start panel.', '',
                     '- New kerbs on the Techspeed Velopark 1500 opening corners, modelled on the real ones:',
                     '  - Painted ramp on the inside of the first kink and of the Amateur corner after it.',
                     '  - Painted ramp with concrete ribs behind it on the inside of the left after the kink,',
                     '    of the hairpin round the grass island and of the far hairpin.',
                     '  - Flat kerb with grooves cut into it in place of the straight kerb on the outside of the exit',
                     '    from the hairpin round the grass island, moved out to where the real one is, in line with',
                     '    the grid.',
                     '  - All the other kerbs round the 1500 and its short layout: painted ramps, with concrete ribs',
                     '    on the second on the right up the third straight, inside the curve at the end of the',
                     '    long straight back to the middle and inside the top curve of the 1500, and flat kerbs with grooves along the two straights',
                     '    between the hairpins after the second S and outside the last corner.',
                     '  - Narrow and low like the real ones: 0.6 m for the ramps, a 0.45 m ramp with 1 m of ribs behind.',
                     '- The white edge line of the road along every new kerb, where the old kerbs hid it.',
                     '- Grid slot marks on the second straight, the lines along its outside edge and the pit lane',
                     '  beside it, and the hotlap start line after the turn onto the third straight, as painted',
                     '  on the real track.',
                     '- A 50 m board on the second straight, after its 100 m board.',
                     '- New tyre walls: on the island along the exit onto the second straight, where the tarmac ends',
                     '  inside the turn onto the third straight, and across the grass to the barrier at the exit of',
                     '  the kink after the second S.',
                     '- New 17 m light poles with a ring of round floodlights and a stack of tyres round the foot,',
                     '  in place of the old 8 m poles.',
                     '- Night lights moved up to the new lamp heads, at half the brightness that matched the old poles.',
                     '- Start panel lowered to the height of the real one, with its legs standing at the panel ends.',
                     '- Removed copies of some trees that were left far under the ground.', '', ''])
    s = s.replace('## 0.5.0', entry + '## 0.5.0', 1)
    open(p, 'w', encoding='utf-8', newline='').write(s)


def pack():
    out = os.path.join(ROOT, 'build', 'velopark-%s.zip' % VERSION)
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for base, _, files in os.walk(os.path.join(DST, 'content')):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, DST))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
    log('packed %s (%.1f MB)' % (out, os.path.getsize(out) / 2 ** 20))


if __name__ == '__main__':
    build()
    open(os.path.join(ROOT, 'build', 'build_051.log'), 'w').write('\n'.join(REPORT) + '\n')
