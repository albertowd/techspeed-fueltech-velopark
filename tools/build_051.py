"""Build the 0.5.1 test: new kerbs on T1-T3 of the 1500 (and the Amateur corner after T1),
new light poles with their night lights recomputed, and the lowered start panel.

Copies base/velopark-0.5.0 to build/velopark-0.5.1 and only ever writes inside the copy.
The track in content/ is what this made; copy build/velopark-0.5.1/content over it to take a
rebuild. Do not after content/ has moved on from 0.5.1: it would undo those changes.
"""
import glob, io, json, math, os, re, shutil, struct, sys, types

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
        if not h and getattr(s, 'below', None) is not None and s.below.hits(x, z):
            # in a hole in it, where the ground below (the paddock's asphalt) is what shows
            return s.below.y(x, z)
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

    def __init__(s, line, away_hint, ground, length=None, cover=None, anchor=None):
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
        # one end held where the line ends, as where two pieces of one old kerb meet
        if anchor == 'start':
            lo = ext
        elif anchor == 'end':
            hi = cum[-1] - ext
        n = int(math.ceil((hi - lo) / BLOCK)) if length is None else int(length // BLOCK)
        start = max(0., lo - (n * BLOCK - (hi - lo)) / 2)
        if anchor == 'start':
            start = lo
        elif anchor == 'end':
            start = max(0., hi - n * BLOCK)
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
KEEP_BED = {'T4', 'L1500_C', 'K1_T1_ENTRY', 'K1_T2_OUT', 'K1_T3_EXIT', 'K1_T4_ENTRY', 'K1_T4_EXIT', 'K1_T5_EXIT',
            'K1_T6_EXIT', 'K1_T7_EXIT', 'K1_T8_OUT'}
# old kerbs that run on round two corners, built as two: the old kerb, where on it the two
# meet, and which piece each label is, the one before that point or the one after it
SPLIT_AT = {'1ROAD_63': (-52.6, 233.9)}
PIECE = {'K1_T2_OUT': 0, 'K1_T3': 1}
# kerbs whose old ones sat inside the road's edge line, on a short line of their own: moved
# out by this much, so they start at the outside of the edge line either side of them
SHIFT_OUT = {'L1500_C': .46, 'MID_EXIT': .52}
# kerbs laid along the outer edge of the road's edge line beside them, as their old ones
# wander away from it
ALONG_LINE = {'T4': 'LINE_1', 'T9': 'LINE_18', 'T7': 'LINE_15'}
# kerbs laid right on the outer edge of the asphalt their old ones lay on, as those started in
# on the road; the line beside the old one goes, the kerb's own edge line takes its place
ALONG_EDGE = {'S1_IN': 'LINE_16'}
# and the road's edge line it bends off from, at its start
EDGE_FROM = {'S1_IN': 'LINE_10'}    # kerbs with asphalt behind them that stays
# floor kerbs laid straight, as their old ones wander, in line with the grid's lines beside
# them, and the line along their track side with them: the line, and how much shorter the
# kerb is at its back, the end by the grid's end, where the asphalt behind it narrows
STRAIGHTEN = {'LAST_OUT': ('LINE_44', 6., 'LINE_2')}
# (and the road's edge line it carries on in line with, from its front end)
# how the profiles are built, as (profile, built) distances from the road edge: the concrete
# kerb a narrow painted ramp with wider cement ribs behind it, the ramp kerb 0.6 m
CONCRETE_WARP = ([0, 1.09, 2.0, A_DEPTH], [0, .45, 1.45, 1.5])
RAMP_WARP = ([0, A_DEPTH], [0, A_DEPTH * .6 / 1.25])
KERB_LOW = .55       # of the profile's height
BASE = -.0501      # height of the hidden base under the tapered ends


def grow(S, length, ends):
    """How far the kerb has grown out of the road at S, 0 to 1, over TAPER from each end
    it tapers at: ends says whether it does at its start and at its end."""
    return np.clip(np.minimum(S if ends[0] else np.inf, length - S if ends[1] else np.inf) / TAPER, 0, 1)


RAMP_SCALE = RAMP_WARP[1][1] / RAMP_WARP[0][1]     # the ramp kerb as built, a metre of profile


def from_ramp(T, Z, prof_t, f, low):
    """A concrete kerb's profile grown out of the ramp kerb's: at f 0 its painted part is the
    ramp's painted part and its concrete lies down the ramp's back slope, at f 1 it is itself.
    T and Z as built, prof_t where each point lies in the profile."""
    paint = prof_t <= 1.09 + 1e-6
    tt = np.where(paint, prof_t * RAMP_SCALE, RAMP_T[-1] * RAMP_SCALE)
    zt = np.where(paint, np.interp(prof_t, RAMP_T, RAMP_Z) * low, 0.)
    return tt + (T - tt) * f, zt + (Z - zt) * f


def soup_mesh(q, path, depth, angle, name, material, taper=False, warp=None, low=1., morph=None):
    """Soup in track space to a kn5 mesh block. With taper the kerb grows out of the road
    edge as a triangle over its first and last two blocks, rising as it widens, or only at
    the ends a (start, end) pair says; at the end morph says it grows out of a ramp kerb
    instead (see from_ramp). warp builds
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
    if morph is not None:
        # at that end (0 its start, 1 its end) it grows out of the ramp kerb it carries on from
        kerb = (uniq[:, 1] > 0) & (np.abs(Z - BASE) > 1e-6)
        f = grow(S, path.length, (morph == 0, morph == 1))
        T[kerb], Z[kerb] = from_ramp(T[kerb], Z[kerb], uniq[kerb, 1], f[kerb], low)
    if taper:
        f = grow(S, path.length, (True, True) if taper is True else taper)
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


def stack_centres(m):
    """Where the stacks of a tyre barrier mesh stand, in x, z, in the order they are built."""
    f = m['f']
    key = np.unique(np.round(m['v'][:, :3], 3), axis=0, return_inverse=True)[1].ravel()
    parent = np.arange(key.max() + 1)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    # each tyre is a piece of its own: the pieces, then the tyres standing on one spot
    for a, b, c in key[f]:
        ra = root(a)
        parent[root(b)] = ra
        parent[root(c)] = ra
    piece = np.array([root(k) for k in key])
    v = m['v'][:, [0, 2]]
    stacks = []
    for r in dict.fromkeys(piece[f[:, 0]]):
        c = v[piece == r].mean(0)
        for st in stacks:
            if np.linalg.norm(st[0] / st[1] - c) < .3:
                st[0] += c
                st[1] += 1
                break
        else:
            stacks.append([c.copy(), 1])
    return np.array([c / n for c, n in stacks])


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
PIT_LINE_2 = np.array([(40, -19.3), (50, -16.9), (60, -14.4)])
PIT_MERGE_X = 85.      # where it bends into the line after the kerb and ends
# the line is dashed up to where this layout's cones along it start, and runs on into the
# edge line of the turn at the end of the straight
CONES_LAYOUT = 'vp_1500'
STRAIGHT_END = 'LINE_17'
# the hotlap start line, just out of the curve onto the 3rd straight: its ends
HOTLAP_LINE = np.array([(175.8, 59.3), (184.6, 60.1)])
# the 1500 grid's lines bend in towards the inside of the last turn where they end: they, the
# slots on them and the start places there are laid at the distance from the inside edge
# line they keep along the straight. Near where they end, and that edge line
GRID_END = np.array([8., 208.])
GRID_EDGE = 'LINE_37'
JOIN_BACK = 2.5      # how much of a road's edge line bends into a kerb's, where they meet
# the junction roads between the layouts, given edge lines along the grass on both sides
JUNCTIONS = ('1ROAD_12', '1ROAD_9', '1ROAD_10')
# the grass islands the kerbs at T2 and T3 curve round: their flat side runs along the road
# the track goes on by, from the end of the line at one tip of the kerb to the line at the
# other, and is given an edge line between them. The kerb, the road, and those two ends
ISLANDS = (('T2', '1ROAD_6', (9.2, 7.2), (24.2, -7.4)), ('T3', '1ROAD_3', (185.6, 44.4), (183.0, 29.6)))
# kerbs whose ends lie in the grass between two roads, as at the tip where the road from the
# last turn's outside meets the other: the old asphalt round their ends is grass, and the two
# roads' edge lines, the one along the kerb and the other's, meet in a corner there
GRASS_TIPS = {'T12_OUT'}
TIP_CORNERS = (('LINE_47', 'LINE_45', (46.7, 204.)),)
# the gaps in the tyre barrier round the inside of the vp_1500 pit entry: the barrier's stacks
# on either side of each, the one before, the last before the gap, the first after it and the
# one after that; and the short piece standing out of line in the second
PIT_ENTRY_GAPS = (((-29.9, 66.1), (-29.3, 65.8), (-28.2, 65.3), (-27.6, 65.1)),
                  ((-28.2, 65.3), (-27.6, 65.1), (-21.9, 66.1), (-21.5, 66.7)))
PIT_ENTRY_PIECE = ('TYRES_108', 'TYRES_108F', '1WALL_TYRE_108')
# the gap in the barrier round the inside of the vp_1000's T2, between the end of its long run
# and the short piece on round the kerb's end, as those above
K1_T2_GAP = ((-38.22, 215.27), (-38.19, 215.9), (-39.72, 219.53), (-40.21, 219.93))
# run-offs behind kerbs where the track's ground has a hole and the paddock's asphalt below
# shows in a shape of its own: laid as a strip of the track's asphalt along the kerb, growing
# out from its start and bending back in to it at its end, with grass round it. The kerb, the
# box the paddock's asphalt there lies in, how far out behind the kerb the strip reaches, and
# the barrier along it, laid again as far out behind the kerb all along as its last stack
# stands, the strip then reaching just past it
RUNOFFS = (('K1_T3_EXIT', (-80., 252., -36., 274.), None, 'TYRES_40'),
           ('K1_T4_EXIT', (-10., 262., 66., 308.), 3.45, None))
RUNOFF_IN, RUNOFF_OUT = 6., 8.      # how long the strip grows out and bends back in over
# and behind these kerbs grass, all the way
PADDOCK_GRASS = (('K1_T3', (-70., 233., -43., 249.)),)
# the grass showing through between the paddock's asphalt and the road out of the pit entry
# loop, in this box, and that road
PIT_ENTRY_GRASS = (-36., 56., -31., 65.)
PIT_ENTRY_ROAD = '1ROAD_26'


def road_marks(ground, road, names, mat_blocks, tex_blocks, kerb_paths, floor, straight, driven, all_kerbs, old_kerbs):
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
    # the paint copied onto the 2nd straight, as triangles of (x, z) corners
    copies = []
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
        copies.append(np.concatenate(P)[:, [0, 2]].reshape(-1, 3, 2))
        log('road marks: %s, %d slots, %d triangles' % (src, len({k for k, _ in keep}), sum(len(x) for x in P) // 3))

    tri = pieces('GRID_1500')
    lift = np.median(tri[:, :, 1].ravel() - [ground.y(x, z) for x, z in tri[:, :, [0, 2]].reshape(-1, 2)])
    turn = lambda t: (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    wound = np.sign(np.median(turn(tri[:, :, [0, 2]])))

    # the paint is see-through, so where two lines overlap the paint shows whiter: the lines
    # drawn here are gathered, as quads of (x, z) corners, and laid as one at the end
    drawn = []
    tarmac = Ground(kn5io.Model(os.path.join(TRACK, 'velopark_paddock.kn5')), {'1ROAD_PADDOCK'})

    def flat(name, quads):
        """Quads of (x, z) corners as paint lying on the road."""
        drawn.append((name, [np.asarray(q, float) for q in quads]))

    def lay(name, P):
        """Triangles of (x, z) corners as paint lying on the road."""
        P = np.array(P)
        P[turn(P) * wound < 0] = P[turn(P) * wound < 0][:, ::-1]
        P = P.reshape(-1, 2)
        Y = np.array([ground.y(x, z) for x, z in P])
        # on the paddock's asphalt where it lies just over the road, as past the grass beside
        # the 2nd straight's grid, or it hides the paint
        top = np.array([max([h for h, _ in tarmac.hits(x, z)], default=-1e9) for x, z in P])
        # a little over it, as it is sampled only at the corners
        Y = np.where((top > Y) & (top < Y + .05), top + .003, Y) + lift
        b, _ = kn5io.mesh_block(name, np.c_[P[:, 0], Y, P[:, 1]], np.tile([0, 1., 0], (len(P), 1)),
                                np.c_[P[:, 0] / 10, P[:, 1] / 20], mat['LinePaint'])
        return b

    # on from the end of the straight kerb, in line with it, to the end of the straight: solid,
    # dashed from just past where the pit lane's line joins it to where the cones beside it
    # start, then solid again on into the edge line of the turn at the end of the straight;
    # dashes half a metre long a metre apart, as thin as the paint's other lines
    a, b = STRAIGHT_KERB
    along = (b - a) / np.linalg.norm(b - a)
    side = np.array([-along[1], along[0]]) * LINE_WIDTH / 2
    across = side / np.linalg.norm(side)
    # the line along the old straight kerb now runs through the grid, and the short lines along
    # two old kerbs inside the edge line would run beside the new ones
    paint_data = bytearray(paint.bytes)
    for name in ('LINE_3', 'LINE_5', 'LINE_13', 'LINE_39') + tuple(ALONG_EDGE.values()):
        kn5io.blank_faces(paint_data, paint.mesh(name))
    # the turn's edge line starts off to the side and runs in onto the line: that start goes,
    # and the line runs on to where the edge line lies on it, and bends onto its middle there
    m = paint.mesh(STRAIGHT_END)
    t = m['v'][m['f']][:, :, [0, 2]].mean(1)
    tu, tw = (t - b) @ along, (t - b) @ across
    on = np.abs(tw) < .2
    meet = tu[on].min()
    lead = np.flatnonzero((tu < meet) & ~on)
    kn5io.blank_faces(paint_data, m, lead)
    v = m['v'][:, [0, 2]]
    vu = (v - b) @ along
    line_end = v[(vu > meet - .3) & (vu < meet + .7) & (np.abs((v - b) @ across) < .3)].mean(0)
    # the paint's lines, less those that go
    others = np.concatenate([m['v'][m['f'] if m['name'] != STRAIGHT_END else np.delete(m['f'], lead, 0)][:, :, [0, 2]]
                             for m in paint.meshes if m['name'].startswith('LINE')
                             and m['name'] not in ('LINE_3', 'LINE_5', 'LINE_13', 'LINE_39') + tuple(ALONG_EDGE.values())])
    # the cones along the line, on the side the pit lane joins from
    lay_cones = kn5io.Model(os.path.join(TRACK, CONES_LAYOUT + '.kn5'))
    cones = np.array([c['v'][:, [0, 2]].mean(0) for c in lay_cones.meshes
                      if lay_cones.materials[c['mat']]['name'] == 'ConeOrange' and not c['name'].startswith('AC_')])
    cu, cw = (cones - b) @ along, (cones - b) @ across
    near = (cu > 0) & (cu < meet) & (np.abs(cw) < 3)
    assert near.any(), 'no cones along the line after the straight kerb in ' + CONES_LAYOUT
    merge = a + along * (PIT_MERGE_X - a[0]) / along[0]
    dash_from, dash_to = (merge - b) @ along + 1., cu[near].min()
    # whole dashes, the gaps at both ends as long as those between them
    n = max(1, int(round((dash_to - dash_from - .52) / 1.02)))
    gap = (dash_to - dash_from - n * .5) / (n + 1)
    dashes = [b + along * (dash_from + gap + .25 + k * (.5 + gap)) for k in range(n)]
    solid_end, end_from = b + along * dash_from, b + along * dash_to

    def strip(name, pts, width):
        quads = []
        for a, b in zip(pts, pts[1:]):
            for f0, f1 in zip(np.linspace(0, 1, 4)[:-1], np.linspace(0, 1, 4)[1:]):
                p, q = a + (b - a) * f0, a + (b - a) * f1
                d = (q - p) / np.linalg.norm(q - p)
                n = np.array([-d[1], d[0]]) * width / 2
                quads.append([p - n, q - n, q + n, p + n])
        flat(name, quads)

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
    strip('LINE_2_SOLID', line, LINE_WIDTH)
    pit_dir = PIT_LINE_2[1] - PIT_LINE_2[0]
    merge = a + along * (PIT_MERGE_X - a[0]) / along[0]
    line = np.r_[bend(e0, e0 - back, PIT_LINE_2[0], pit_dir), PIT_LINE_2[1:-1],
                 bend(PIT_LINE_2[-1], pit_dir, merge, along)]
    strip('LINE_2_PIT', line, LINE_WIDTH)
    flat('LINE_2_DOTS', [[c - along * .25 - side, c + along * .25 - side, c + along * .25 + side, c - along * .25 + side]
                         for c in dashes])
    # a line width into the edge line: where they overlap goes when the lines are laid
    to = (line_end - end_from) / np.linalg.norm(line_end - end_from)
    strip('LINE_2_END', np.array([end_from, line_end + to * LINE_WIDTH]), LINE_WIDTH)
    log('road marks: solid line after the straight kerb, pit lane line joining it; dashed for %.1f m, %d dashes, '
        'from past where they join to the cones, then solid for %.1f m on into %s, its first %d triangles gone'
        % (dash_to - dash_from, n, np.linalg.norm(line_end - end_from), STRAIGHT_END, len(lead)))

    # the road's edge line along the kerbs, where it is missing: the old one ran under where
    # the new kerb lies, or along an old kerb's own edge
    added = 0.
    # the lines drawn here, as their middles: those not fixed are joined up with the lines
    # round them at their ends before they are drawn
    mine = []
    for k, (label, path) in enumerate(kerb_paths.items()):
        S = np.arange(0, path.length + .01, .25)
        run = []
        for sv in list(S) + [None]:
            painted = sv is None or any(covers(others, path.place(np.array([sv]), np.array([t]), np.zeros(1))[:, [0, 2]]).any()
                                        for t in (-.3, -.18, -.06, .06, .18))
            if not painted:
                run.append(sv)
            elif len(run) >= 5:
                pts = path.place(np.array(run), np.full(len(run), t_in), np.zeros(len(run)))[:, [0, 2]]
                mine.append(dict(name='LINE_KERB_EDGE_%d_%d' % (k, run[0] * 100), pts=pts, fixed=False))
                added += run[-1] - run[0]
                run = []
            else:
                run = []
    log('road marks: %.0f m of edge line along the kerbs where it was missing' % added)
    # bending off the road's edge line into the kerb's, where they do not meet
    for label, line_name in EDGE_FROM.items():
        # the last stretch of the road's edge line goes, and a bend from where it now ends
        # carries it on into the kerb's edge line, so the two run on as one
        path = kerb_paths[label]
        k0 = path.place(np.array([0., .5]), np.full(2, t_in), np.zeros(2))[:, [0, 2]]
        m = paint.mesh(line_name)
        v = m['v'][:, [0, 2]]
        e = v[np.linalg.norm(v - k0[0], axis=1).argmin()]
        tri = v[m['f']]
        gone = np.flatnonzero(np.linalg.norm(tri.mean(1) - e, axis=1) < JOIN_BACK)
        kn5io.blank_faces(paint_data, m, gone)
        left = np.unique(m['f'][np.setdiff1d(np.arange(len(m['f'])), gone)])
        d = np.linalg.norm(v[left] - e, axis=1)
        c = v[left][np.argsort(d)[:2]].mean(0)
        back = v[left][(d > d.min() + 1.) & (d < d.min() + 2.5)].mean(0)
        mine.append(dict(name='LINE_KERB_JOIN_' + label, pts=bend(c, c - back, k0[0], k0[1] - k0[0], 16), fixed=True))
        log('road marks: %s edge line joined to %s, its last %.1f m bent in, %.2f m across' % (label, line_name, np.linalg.norm(c - e), np.linalg.norm(k0[0] - e)))

    # the line along the straightened floor kerbs, straight with them
    for label, (line_name, _, front_name) in STRAIGHTEN.items():
        path = straight[label]
        # the old line along it, and the hook at the end of the road's edge line before it
        gone = 0
        for name in (line_name, front_name):
            m = paint.mesh(name)
            sc, tc = path.st(m['v'][m['f']][:, :, [0, 2]].mean(1))
            sel = np.flatnonzero((sc > -.3) & (sc < path.length + .3) & (tc > -1.5) & (tc < 1.))
            kn5io.blank_faces(paint_data, m, sel)
            gone += len(sel)
        S = np.linspace(0, path.length, int(path.length / 2) + 2)
        mine.append(dict(name='LINE_STRAIGHT_' + label, pts=path.place(S, np.full(len(S), t_in), np.zeros(len(S)))[:, [0, 2]], fixed=False))
        log('road marks: line along %s laid straight with it, %d triangles of %s and %s gone' % (label, gone, line_name, front_name))
    grid_end(paint, paint_data, ground)

    # edge lines on both sides of the junction roads between the layouts, where their edge
    # runs along the grass: not where they open onto another road or a kerb lies along them
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    for name in JUNCTIONS:
        J = names.index(name)
        tri = ground.tris[ground.owner == J][:, :, [0, 2]]
        a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        tri = tri[np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-9]
        shape = unary_union([Polygon(t) for t in tri]).buffer(.01).buffer(-.01)
        shape = max(getattr(shape, 'geoms', [shape]), key=lambda g: g.area)
        edge = shape.exterior
        # the line's middle, just inside the outline, followed all the way round
        ring = shape.buffer(-(LINE_WIDTH / 2 + .02)).exterior
        pts = np.array([ring.interpolate(u).coords[0] for u in np.arange(0, ring.length, .2)])
        free = []
        for p in pts:
            q = np.array(edge.interpolate(edge.project(Point(p))).coords[0])
            n = (q - p) / max(np.linalg.norm(q - p), 1e-9)
            # another road past it: a seam, or the junction's mouth; or a kerb along it
            seam = any(ground.owner[k] in road and ground.owner[k] != J for f in (.15, .4) for _, k in ground.hits(*(q + n * f)))
            kerb = any((lambda st: -.5 < st[0][0] < kp.length + .5 and -.6 < st[1][0] < 1.5)(kp.st(q[None])) for kp in kerb_paths.values())
            # or a line already painted along it
            painted = covers(others, np.array([p, p + n * .15, p - n * .15])).any()
            free.append(not seam and not kerb and not painted)
        # runs of free points round the ring, starting at one that is not, so none is split
        free = np.array(free)
        if free.all() or not free.any():
            log('road marks: %s, no edge along the grass found' % name)
            continue
        at = int(np.flatnonzero(~free)[0])
        pts, free = np.roll(pts, -at, axis=0), np.roll(free, -at)
        made, run, k = 0., [], 0
        for p, f in list(zip(pts, free)) + [(None, False)]:
            if f:
                run.append(p)
                continue
            if len(run) * .2 >= 2.:
                line = smooth_line(np.array(run), 5)
                mine.append(dict(name='LINE_JUNCTION_%s_%d' % (name, k), pts=line, fixed=False))
                made += np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
                k += 1
            run = []
        log('road marks: %s, %d edge lines along the junction road, %.0f m' % (name, k, made))
    # the islands' flat sides, along the road's outline between the two ends
    for _, name, a, b in ISLANDS:
        a, b = np.array(a), np.array(b)
        J = names.index(name)
        tri = ground.tris[ground.owner == J][:, :, [0, 2]]
        c = tri.mean(1)
        lo, hi = np.minimum(a, b) - 8, np.maximum(a, b) + 8
        tri = tri[(c > lo).all(1) & (c < hi).all(1)]
        e0, e1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        tri = tri[np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) > 1e-9]
        shape = unary_union([Polygon(t) for t in tri]).buffer(.01).buffer(-.01)
        shape = max(getattr(shape, 'geoms', [shape]), key=lambda g: g.area)
        ring = shape.buffer(-(LINE_WIDTH / 2 + .02)).exterior
        u0, u1 = sorted((ring.project(Point(a)), ring.project(Point(b))))
        # the shorter way round, between them
        us = np.arange(u0, u1, .2) if u1 - u0 < ring.length / 2 else np.r_[np.arange(u1, ring.length, .2), np.arange(0, u0, .2)]
        line = smooth_line(np.array([ring.interpolate(u).coords[0] for u in us]), 5)
        mine.append(dict(name='LINE_ISLAND_%s' % name, pts=line, fixed=False, corner=True))
        log('road marks: island along %s, edge line %.1f m' % (name, np.linalg.norm(np.diff(line, axis=0), axis=1).sum()))
    # the two roads' edge lines meeting at a tip of grass between them: on to a corner
    for name_a, name_b, near in TIP_CORNERS:
        (A, da), (B, db) = (paint_end(paint.mesh(n)['v'][:, [0, 2]], np.array(near)) for n in (name_a, name_b))
        M = np.array([da, -db]).T
        t = np.linalg.solve(M, B - A) if abs(np.linalg.det(M)) > .2 else None
        if t is not None and (t > -.2).all() and (t < 3).all():
            X = A + da * t[0]
            mine.append(dict(name='LINE_TIP_%s_A' % name_a, pts=np.array([A, X + da * LINE_WIDTH / 2]), fixed=True))
            mine.append(dict(name='LINE_TIP_%s_B' % name_a, pts=np.array([B, X + db * LINE_WIDTH / 2]), fixed=True))
        else:
            mine.append(dict(name='LINE_TIP_%s' % name_a, pts=np.array([A, B]), fixed=True))
        log('road marks: %s and %s meet in a corner at the tip between them' % (name_a, name_b))

    from shapely.geometry import LineString
    def polys(tri):
        e0, e1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        return [Polygon(t) for t in tri[np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) > 1e-7]]
    kerb_area = unary_union([Polygon(footprint(kp, 0, 1.2)[:, [0, 2]]).buffer(0) for kp in all_kerbs]
                            + [p for t in old_kerbs for p in polys(t)]).buffer(.35)

    # two of the paint's lines ending close by each other without meeting, with no kerb between:
    # on to where they cross, a hard corner, or straight across from one end to the other
    ends = []
    for m in paint.meshes:
        if not m['name'].startswith('LINE') or m['name'].startswith('LINE_KIDS') or m['name'] in sum(((a, b) for a, b, _ in TIP_CORNERS), ()):
            continue
        f = np.array(struct.unpack_from('<%dH' % m['f'].size, paint_data, m['ip'])).reshape(-1, 3)
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if not len(f):
            continue
        v = m['v'][:, [0, 2]]
        key = np.unique(np.round(v, 3), axis=0, return_inverse=True)[1].ravel()
        part = list(range(key.max() + 1))

        def root(i):
            while part[i] != i:
                part[i] = part[part[i]]
                i = part[i]
            return i
        for a, b, c in key[f]:
            part[root(b)] = root(a)
            part[root(c)] = root(a)
        grp = np.array([root(key[a]) for a, _, _ in f])
        for r in np.unique(grp):
            vv = v[np.unique(f[grp == r])]
            if np.ptp(vv, axis=0).max() < 1.5:
                continue
            far = vv[np.argmax(np.linalg.norm(vv - vv[0], axis=1))]
            for e in (far, vv[np.argmax(np.linalg.norm(vv - far, axis=1))]):
                pe = paint_end(vv, e)
                if pe is not None and np.linalg.norm(pe[0] - e) < 1:
                    ends.append((m['name'], r, pe[0], pe[1], vv))
    closed = 0
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            (na, ra, A, da, va), (nb, rb, B, db, vb) = ends[i], ends[j]
            if (na, ra) == (nb, rb) or np.linalg.norm(A - B) > 1.5:
                continue
            if min(np.linalg.norm(vb - A, axis=1).min(), np.linalg.norm(va - B, axis=1).min()) < .15:
                continue
            if LineString([A, B]).intersects(kerb_area.buffer(-.3)):
                continue
            M = np.array([da, -db]).T
            t = np.linalg.solve(M, B - A) if abs(np.linalg.det(M)) > .2 else None
            if t is not None and (t > -.2).all() and (t < 2).all():
                X = A + da * t[0]
                mine.append(dict(name='LINE_CLOSE_%d_A' % closed, pts=np.array([A, X + da * LINE_WIDTH / 2]), fixed=True))
                mine.append(dict(name='LINE_CLOSE_%d_B' % closed, pts=np.array([B, X + db * LINE_WIDTH / 2]), fixed=True))
            else:
                mine.append(dict(name='LINE_CLOSE_%d' % closed, pts=np.array([A, B]), fixed=True))
            closed += 1
    log('road marks: %d gaps between the ends of two of the paint lines closed' % closed)

    # the road's edge along the grass where no line runs and no kerb lies: given an edge line;
    # not behind a kerb, nor where a line runs along just inside it
    surface = unary_union(polys(ground.tris[np.isin(ground.owner, list(driven))][:, :, [0, 2]])).buffer(.02).buffer(-.02)
    # any asphalt at all, the strips beside the road no layout drives on too
    asphalt = unary_union(polys(ground.tris[np.isin(ground.owner, list(road))][:, :, [0, 2]])).buffer(.02)
    grassy = np.array([names[o].startswith('1GRASS') for o in ground.owner])
    grass = unary_union(polys(ground.tris[grassy][:, :, [0, 2]])).buffer(.01)
    grids = np.concatenate([m['v'][m['f']][:, :, [0, 2]] for m in paint.meshes if m['name'].startswith('GRID')])
    painted = unary_union(polys(others) + polys(grids) + [LineString(m['pts']).buffer(LINE_WIDTH / 2) for m in mine]
                          + [Polygon(q).buffer(0) for _, qs in drawn for q in qs] + [Point(c).buffer(.3) for c in dashes])
    near_line = painted.buffer(.3)
    from shapely.ops import nearest_points
    gaps, made = 0, 0.
    for g in getattr(surface, 'geoms', [surface]):
        if g.area < 20:
            continue
        for ring in [g.exterior] + list(g.interiors):
            pts = np.array([ring.interpolate(u).coords[0] for u in np.arange(0, ring.length, .25)])
            d = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
            n = np.c_[d[:, 1], -d[:, 0]] / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            # out of the road
            n *= np.where([g.contains(Point(p + q * .3)) for p, q in zip(pts, n)], -1, 1)[:, None]
            open_ = np.array([grass.contains(Point(p + q * .4)) and not asphalt.contains(Point(p + q * .4))
                              and not near_line.contains(Point(p)) and not kerb_area.contains(Point(p)) for p, q in zip(pts, n)])
            if open_.all() or not open_.any():
                continue
            at = int(np.flatnonzero(~open_)[0])
            pts, n, open_ = np.roll(pts, -at, 0), np.roll(n, -at, 0), np.roll(open_, -at)
            run = []
            for k in list(range(len(pts))) + [None]:
                if k is not None and open_[k]:
                    run.append(k)
                    continue
                if len(run) * .25 >= 1.:
                    r, q = pts[run], n[run]
                    behind = np.mean([any(kerb_area.contains(Point(p - v * t)) for t in (.5, 1., 1.5, 2.)) for p, v in zip(r, q)])
                    # a line just off to its side, not one that ends near it: the road runs on past
                    # its edge line there
                    mid = range(len(r) // 3, max(len(r) // 3 + 1, 2 * len(r) // 3))
                    side = []
                    for i in mid:
                        w = np.array(nearest_points(painted, Point(r[i]))[0].coords[0]) - r[i]
                        side.append(np.linalg.norm(w) < 1.5 and abs(w @ q[i]) > .7 * np.linalg.norm(w))
                    if behind <= .5 and np.mean(side) <= .5:
                        line = r - q * (LINE_WIDTH / 2 + .02)
                        line = smooth_line(line, 5) if len(line) >= 8 else line
                        mine.append(dict(name='LINE_EDGE_GAP_%d' % gaps, pts=line, fixed=False))
                        gaps += 1
                        made += len(r) * .25
                run = []
    log('road marks: %d gaps in the road\'s edge lines along the grass given a line, %.0f m' % (gaps, made))
    # the paint's own lines and marks where the road or the paddock's asphalt lies over them,
    # as the start of one past the finish line does: onto the top of it
    raised = 0
    for m in paint.meshes:
        if not m['name'].startswith(('LINE', 'GRID', 'NUMBERS')):
            continue
        v = np.frombuffer(bytes(paint_data[m['vp']:m['vp'] + len(m['v']) * 44]), '<f4').reshape(-1, 11)
        for i, (x, y, z) in enumerate(v[:, :3]):
            top = max([h for h, _ in ground.hits(x, z) + tarmac.hits(x, z)], default=-1e9)
            if y - .002 < top < y + .06:
                struct.pack_into('<f', paint_data, m['vp'] + i * 44 + 4, top + .004)
                raised += 1
    log('road marks: %d corners of the paint lying under the road raised onto it' % raised)
    open(os.path.join(TRACK, 'velopark_paint.kn5'), 'wb').write(paint_data)
    join_lines(mine, kn5io.Model(os.path.join(TRACK, 'velopark_paint.kn5')), bend)
    for line in mine:
        strip(line['name'], line['pts'], LINE_WIDTH)

    # the hotlap start line, as thin as the others
    c = HOTLAP_LINE.mean(0)
    across = HOTLAP_LINE[1] - HOTLAP_LINE[0]
    ends = np.linalg.norm(across) * np.array([-.5, .5])
    across = across / np.linalg.norm(across)
    along = np.array([-across[1], across[0]])
    assert all(ground.on(*(c + across * d), road) for d in np.linspace(*ends, 30))
    w = np.linspace(ends[0], ends[1], int((ends[1] - ends[0]) / .5) + 2)
    h = LINE_WIDTH / 2
    flat('LINE_HOTLAP_START', [[c + across * d0 - along * h, c + across * d1 - along * h,
                                c + across * d1 + along * h, c + across * d0 + along * h] for d0, d1 in zip(w, w[1:])])
    log('road marks: hotlap start line, %.1f m across the road' % (ends[1] - ends[0]))

    # the lines drawn here laid as one shape, less the paint already there: where two of them
    # overlap, or one runs into a line or mark of the paint's own, it is painted once, so their
    # joints show no whiter than the lines
    import shapely
    from shapely.strtree import STRtree
    under = []
    for m in paint.meshes:
        if not m['name'].startswith(('LINE', 'GRID', 'NUMBERS')):
            continue
        f = np.array(struct.unpack_from('<%dH' % m['f'].size, paint_data, m['ip'])).reshape(-1, 3)
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        under += polys(m['v'][f][:, :, [0, 2]])
    under = unary_union(under + [p for t in copies for p in polys(t)])
    shapes = [(name, Polygon(q).buffer(0)) for name, qs in drawn for q in qs]
    shapes = [(name, g) for name, g in shapes if g.area > 1e-6]
    tree = STRtree([g for _, g in shapes])
    whole = unary_union([g for _, g in shapes]).buffer(0)
    whole = whole.difference(under)
    by_name = {}
    for g in getattr(whole, 'geoms', [whole]):
        if g.area < 1e-4:
            continue
        tri = [np.array(t.exterior.coords[:3]) for t in shapely.constrained_delaunay_triangles(g).geoms]
        # split until short enough to follow the ground
        while True:
            big = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) > 1.]
            if not big:
                break
            tri = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) <= 1.]
            for p, q, r in big:
                pq, qr, rp = (p + q) / 2, (q + r) / 2, (r + p) / 2
                tri += [np.array(x) for x in ((p, pq, rp), (pq, q, qr), (rp, qr, r), (pq, qr, rp))]
        # named for the line it is most of
        name = shapes[tree.nearest(g.representative_point())][0]
        by_name.setdefault(name, []).extend(tri)
    count = 0
    for name, tri in by_name.items():
        # within the 16 bit indices of a mesh
        for k in range(0, len(tri), 20000):
            out.append(lay(name if k == 0 else '%s_%d' % (name, k // 20000), tri[k:k + 20000]))
        count += len(tri)
    log('road marks: %d lines drawn laid as one, %.2f m2 of them overlapping each other or the paint left out, '
        '%d triangles' % (len(drawn), sum(g.area for _, g in shapes) - whole.area, count))
    return out


def paint_end(v, E):
    """An end of a paint line near E: its middle, and the way the line runs out to it."""
    best = None
    for x in v[np.linalg.norm(v - E, axis=1) < 3.]:
        nb = v[np.linalg.norm(v - x, axis=1) < 1.2]
        side = np.linalg.norm(nb.mean(0) - x)
        if len(nb) >= 3 and side > .3 and (best is None or side > best[0]):
            best = (side, x)
    if best is None:
        return None
    c = v[np.linalg.norm(v - best[1], axis=1) < .16].mean(0)
    r = np.linalg.norm(v - c, axis=1)
    back = v[(r > .5) & (r < 1.5)]
    if not len(back):
        return None
    d = c - back.mean(0)
    return c, d / max(np.linalg.norm(d), 1e-9)


def trim(pts, length, end):
    """The polyline without its first (end 0) or last (end -1) stretch of that length."""
    if end == -1:
        return trim(pts[::-1], length, 0)[::-1]
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    a = np.array([np.interp(length, cum, pts[:, 0]), np.interp(length, cum, pts[:, 1])])
    return np.r_[[a], pts[cum > length + .01]]


def join_lines(mine, paint, bend):
    """Join the ends of the new lines to the lines round them. Ending near another line's end,
    it is cut back and a bend carries the other on into it; ending beside one, as at a
    junction's mouth, it bends in to run on along it. The bends are added to mine."""
    # the paint's lines still drawn, each a group of joined corners
    groups = []
    for m in paint.meshes:
        if not m['name'].startswith('LINE'):
            continue
        f, v = m['f'], m['v'][:, [0, 2]]
        t = v[f]
        a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        f = f[np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-9]
        if not len(f):
            continue
        key = np.unique(np.round(v, 3), axis=0, return_inverse=True)[1].ravel()
        part = list(range(key.max() + 1))

        def root(i):
            while part[i] != i:
                part[i] = part[part[i]]
                i = part[i]
            return i
        for x, y, z in key[f]:
            part[root(y)] = root(x)
            part[root(z)] = root(x)
        g = np.array([root(key[x]) for x, _, _ in f])
        for c in np.unique(g):
            groups.append(v[np.unique(f[g == c])])

    def outward(pts, end):
        q = pts if end == 0 else pts[::-1]
        run = np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))
        d = q[0] - q[min(np.searchsorted(run, .8) + 1, len(q) - 1)]
        return q[0], d / max(np.linalg.norm(d), 1e-9)


    def along(v, Q):
        """Which way a line runs at Q."""
        r = np.linalg.norm(v - Q, axis=1)
        # at least a few corners along it, where the paint has them far apart
        nb = v[r < max(1., np.sort(r)[min(5, len(r) - 1)] + .01)]
        return np.linalg.eigh(np.cov((nb - nb.mean(0)).T))[1][:, -1]

    done = set()
    ends, sides, corners = 0, 0, 0
    for i, line in enumerate(list(mine)):
        if line['fixed']:
            continue
        for end in (0, -1):
            if (i, end) in done:
                continue
            done.add((i, end))
            E, d = outward(line['pts'], end)
            if line.get('corner'):
                # straight on into the line it meets, as at the islands' tips: a sharp corner,
                # both run on to where they cross, cut back if past it
                best = None
                for v in groups + [m['pts'] for j, m in enumerate(mine) if j != i and not m.get('corner')]:
                    r = np.linalg.norm(v - E, axis=1)
                    k = r.argmin()
                    if r[k] < 2. and (best is None or r[k] < best[0]):
                        best = (r[k], v[k], v)
                if best is None:
                    continue
                r, Q, v = best
                u = along(v, Q)
                den = d[0] * u[1] - d[1] * u[0]
                t = ((Q - E)[0] * u[1] - (Q - E)[1] * u[0]) / den if abs(den) > .2 else None
                if t is None or not -1.5 < t < 2.5:
                    mine.append(dict(name=line['name'] + '_CORNER_%d' % (end + 1), pts=np.array([E, Q]), fixed=True))
                else:
                    X = E + d * t
                    if t < 0:
                        line['pts'] = trim(line['pts'], -t, end)
                        E = X
                    w = LINE_WIDTH / 2
                    if t > 0:
                        mine.append(dict(name=line['name'] + '_CORNER_%d' % (end + 1), pts=np.array([E, X + d * w]), fixed=True))
                    # the other on to the corner, if it ends short of it
                    c = v[np.linalg.norm(v - X, axis=1).argmin()]
                    c = v[np.linalg.norm(v - c, axis=1) < .16].mean(0)
                    if np.linalg.norm(c - X) > LINE_WIDTH:
                        back = (X - c) / np.linalg.norm(X - c)
                        mine.append(dict(name=line['name'] + '_CORNER_%d_ON' % (end + 1), pts=np.array([c - back * w, X + back * w]), fixed=True))
                corners += 1
                continue
            # another line's end facing this one, the paint's or another new one
            best = None
            for v in groups:
                if np.linalg.norm(v - E, axis=1).min() < 3.:
                    pe = paint_end(v, E)
                    if pe is not None and (best is None or np.linalg.norm(pe[0] - E) < best[0]):
                        best = (np.linalg.norm(pe[0] - E), pe[0], pe[1], None)
            for j, other in enumerate(mine):
                if j != i:
                    for e2 in (0, -1):
                        C, dc = outward(other['pts'], e2)
                        if best is None or np.linalg.norm(C - E) < best[0]:
                            best = (np.linalg.norm(C - E), C, dc, (j, e2))
            facing = best
            if best is not None and best[0] < 3.:
                gap, C, dc, who = best
                if dc @ d < -.3 and d @ (C - E) > -.5:
                    if gap > .05:
                        length = np.linalg.norm(np.diff(line['pts'], axis=0), axis=1).sum()
                        line['pts'] = trim(line['pts'], max(0., min(2. - gap, .4 * length)), end)
                        E2, d2 = outward(line['pts'], end)
                        # on from the other's end the way it ran, into this one's end the way it runs
                        mine.append(dict(name=line['name'] + '_JOIN_%d' % (end + 1), pts=bend(C, dc, E2, -d2, 16), fixed=True))
                        ends += 1
                    if who is not None:
                        done.add(who)
                    continue
            # beside a line: running much its way, bend in to run on along it; across it, as
            # at the tip of the grass between a junction and the road, straight on to meet it
            best = None
            for v in groups + [m['pts'] for j, m in enumerate(mine) if j != i]:
                r = np.linalg.norm(v - E, axis=1)
                k = r.argmin()
                if .15 < r[k] < 2. and d @ (v[k] - E) > 0 and (best is None or r[k] < best[0]):
                    best = (r[k], v[k], v)
            if best is not None:
                r, Q, v = best
                u = along(v, Q)
                u = u if u @ d > 0 else -u
                ahead = Q + u * max(1., r)
                runs_on = np.linalg.norm(v - ahead, axis=1).min() < .3
                pts = bend(E, d, ahead, u, 16) if u @ d > .8 and runs_on else np.array([E, Q])
                mine.append(dict(name=line['name'] + '_MEET_%d' % (end + 1), pts=pts, fixed=True))
                sides += 1
                continue
            # ending beside another's end at an angle, as at an island's tip: a corner
            if facing is not None and .03 < facing[0] < .6:
                mine.append(dict(name=line['name'] + '_CORNER_%d' % (end + 1), pts=np.array([E, facing[1]]), fixed=True))
                corners += 1
    log('road marks: new lines joined to the lines round them, %d end to end, %d into the side of one, %d at a corner' % (ends, sides, corners))


# the mouths of the roads a layout does not take, where its edge line is carried across: near
# where the line runs off along the road on each side, and whether it is a way in or out of the
# pits, where the line across is dashed
MOUTHS = {'vp_1000': (((-73.4, 97.5), (-70.7, 103.2), True),        # pit exit
                      ((-51.7, 193.45), (-58., 191.65), True),      # pit entry
                      ((4.5, 255.15), (-23., 255.45), False),       # into the 1500 before T4
                      ((56.6, 262.5), (45., 256.65), False),        # the hairpin at the end of the oval
                      ((-75.3, 284.6), (-90.3, 278.9), False),      # either side of the island at the top
                      ((-106., 279.15), (-121.6, 280.4), False))}


def hermite(p0, t0, p1, t1, n):
    """From p0 running along t0 to p1 running along t1, the tangents as long as the way between."""
    L = np.linalg.norm(p1 - p0)
    t0, t1 = t0 / np.linalg.norm(t0) * L, t1 / np.linalg.norm(t1) * L
    u = np.linspace(0, 1, n)[:, None]
    return (2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * t0 + (-2 * u**3 + 3 * u**2) * p1 + (u**3 - u**2) * t1


def layout_edges(layout, step=.25, reach=15.):
    """The edge lines of a layout, followed along its AI line on both sides. The line along the
    track's edge (the road's own, or a kerb's front) is looked for across each point of it; where
    it is missing, the line is carried on from where it last ran along the edge to where it runs
    along it again, and where it runs off along a road the layout does not take, across that
    road's mouth (MOUTHS). Returns the lines, as (x, z) points, each with whether it is for the
    layout alone (across a mouth) and whether it is dashed (into or out of the pits), and the
    faces of the paint's lines running off the edge where a line is carried on, by file and mesh."""
    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from track_map import triangles, KERBS
    ai = C.ai_line(layout)[:, [0, 2]]
    lo, hi = ai.min(0) - reach - 5, ai.max(0) + reach + 5
    near = lambda t: t[((t > lo).all(2) & (t < hi).all(2)).any(1)]
    models = {f: kn5io.Model(os.path.join(TRACK, f)) for f in ('velopark.kn5', 'velopark_kerbs.kn5', 'velopark_paint.kn5')}
    road, kerb, lines, owner = [], [], [], []
    for f, model in models.items():
        for m in model.meshes:
            n, mat = m['name'], model.material_name(m)
            if mat in KERBS:
                kerb.append(near(triangles(m)))
            elif n.startswith('1ROAD') and not any(w in n for w in ('PADDOCK', 'RUNOFF')):
                road.append(near(triangles(m)))
            elif n.startswith('LINE') and not n.startswith('LINE_KIDS'):
                lines.append(near(triangles(m)))
                owner.append((f, n))
    union = lambda ts: unary_union([Polygon(t) for t in np.concatenate(ts)]).buffer(.005)
    R, K, L = union(road), union(kerb), union(lines)
    for g in (R, K, L):
        shapely.prepare(g)
    corners = np.concatenate(lines).reshape(-1, 2)
    out, spurs = [], {}

    # across the mouths: from the line on one side, the way it runs there, into the other
    def on_line(p, toward):
        # the paint round it, in points a few centimetres apart
        g = np.mgrid[-1.5:1.5:.025, -1.5:1.5:.025].reshape(2, -1).T + p
        g = g[shapely.contains_xy(L, g[:, 0], g[:, 1])]
        assert len(g), 'no line near %s' % (p,)
        q = g[np.linalg.norm(g - p, axis=1).argmin()]
        nb = g[np.linalg.norm(g - q, axis=1) < 1.]
        u = np.linalg.eigh(np.cov((nb - nb.mean(0)).T))[1][:, -1]
        c = nb[np.abs((nb - q) @ u) < .03].mean(0)
        return c, u if u @ (toward - c) > 0 else -u
    bridges = []
    for k, (a, b, pits) in enumerate(MOUTHS.get(layout, ())):
        a, b = np.array(a), np.array(b)
        (pa, ua), (pb, ub) = on_line(a, b), on_line(b, a)
        pts = hermite(pa, ua, pb, -ub, max(8, int(np.linalg.norm(pb - pa) / step)))
        bridges.append(pts)
        out.append(dict(name='LINE_%s_MOUTH_%d' % (layout.upper(), k), pts=pts, own=True, dashed=pits))

    # the AI line every step, the way it runs and across it
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[ai, ai[:1]], axis=0), axis=1))]
    S = np.arange(0, cum[-1] - step / 2, step)
    A = np.c_[np.interp(S, cum, np.r_[ai[:, 0], ai[0, 0]]), np.interp(S, cum, np.r_[ai[:, 1], ai[0, 1]])]
    D = np.roll(A, -4, 0) - np.roll(A, 4, 0)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    N = np.c_[-D[:, 1], D[:, 0]]
    r = np.arange(0, reach, .02)
    for side in (1, -1):
        n = len(A)
        edge, line, at_kerb = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
        for i in range(n):
            P = A[i] + side * N[i] * r[:, None]
            track = shapely.contains_xy(R, P[:, 0], P[:, 1]) & ~shapely.contains_xy(K, P[:, 0], P[:, 1])
            if not track[0]:
                continue
            k = np.argmin(track)
            e = r[k] if not track.all() else reach
            at_kerb[i] = not track.all() and shapely.contains_xy(K, *P[k])
            on = shapely.contains_xy(L, P[:, 0], P[:, 1]) & (r < e + .3)
            edge[i] = e
            k = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            if len(k):
                mids = (r[k[0::2]] + r[k[1::2] - 1]) / 2
                line[i] = mids[np.abs(mids - e).argmin()]
        # the paint's own lines lie on the road's edge, a little either side of it
        good = (line > edge - .35) & (line < edge + .3)
        if good.all() or not good.any():
            continue
        # from a good point, so no stretch is cut in two at the start of the lap
        idx = np.roll(np.arange(n), -int(np.flatnonzero(good)[0]))
        g = good[idx]
        # stretches without it, those less than a metre apart as one
        runs = []
        for b in np.flatnonzero(~g):
            if runs and b - runs[-1][1] <= 1. / step + 1:
                runs[-1][1] = b
            else:
                runs.append([b, b])
        for i0, i1 in runs:
            ia, ib = i0 - 1, (i1 + 1) % n
            # from further back where it steps across, so it bends over gently: on from the
            # road's edge line along the grass into a kerb's set in from it, say
            far = max(1., 2.5 * abs(line[idx[ia]] - line[idx[ib]])) / step
            for _ in range(int(far)):
                if g[ia - 1] and ia - 1 > 0:
                    ia -= 1
                if g[(ib + 1) % n]:
                    ib = (ib + 1) % n
            ja, jb = idx[ia], idx[ib]
            j = idx[np.arange(ia, ia + (ib - ia) % n + 1) % n]
            e = np.where(np.isnan(edge[j]), reach, edge[j])
            # a mouth: carried across above
            Q = A[j] + side * N[j] * np.where(np.isnan(line[j]), e, line[j])[:, None]
            if any(np.linalg.norm(Q[:, None] - br[None], axis=2).min() < 3. for br in bridges):
                continue
            # on from the line either side, the way it runs there; under the point of a kerb
            # standing out past it, where the kerb hides it
            on = lambda i: A[idx[i % n]] + side * N[idx[i % n]] * line[idx[i % n]]
            pa, pb = on(ia), on(ib)
            ua = pa - on(ia - 2) if g[ia - 2] else D[ja] * side
            ub = on(ib + 2) - pb if g[(ib + 2) % n] else D[jb] * side
            pts = hermite(pa, ua, pb, ub, max(4, len(j)))
            out.append(dict(name='LINE_%s_GAP_%d' % (layout.upper(), len(out)), pts=pts, own=False, dashed=False))
            # the paint running off past it, off the edge
            for (f, name), tri in zip(owner, lines):
                if not len(tri):
                    continue
                c = tri.mean(1)
                box = ((c > pts.min(0) - 3) & (c < pts.max(0) + 3)).all(1)
                if not box.any():
                    continue
                c = c[box]
                k = np.linalg.norm(c[:, None] - pts[None], axis=2).argmin(1)
                along = np.gradient(pts, axis=0)[k]
                along /= np.linalg.norm(along, axis=1, keepdims=True)
                out_of = side * N[j[np.minimum(k * len(j) // len(pts), len(j) - 1)]]
                w = ((c - pts[k]) * out_of).sum(1)
                a = np.abs(((c - pts[k]) * along).sum(1))
                sel = (w > .2) & (w < 3.) & (a < .3) & (k > 0) & (k < len(pts) - 1)
                if sel.any():
                    m = models[f].mesh(name)
                    cen = m['v'][m['f']][:, :, [0, 2]].mean(1)
                    spurs.setdefault((f, name), set()).update(int(np.linalg.norm(cen - x, axis=1).argmin()) for x in c[sel])
    return out, spurs


def edge_lines(layout, tex_blocks, mat_blocks, mesh_blocks):
    """Lay the layout's edge lines where they are missing (layout_edges): those for every layout
    with the kerbs, written here to velopark_kerbs.kn5, those across the mouths of the roads it
    does not take in a model of its own, <layout>_lines.kn5; and the paint running off the edge
    where a line is carried on taken out. Returns the name of the layout's own model."""
    import shapely
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    from track_map import triangles, KERBS
    lines, spurs = layout_edges(layout)
    paint = kn5io.Model(os.path.join(TRACK, 'velopark_paint.kn5'))
    kerbs = kn5io.Model(os.path.join(TRACK, 'velopark_kerbs.kn5'))
    # what the paint lies on: the top of the road, the grass or the paddock there, not a kerb
    grounds = [Ground(m, {x['name'] for x in m.meshes if x['name'].startswith(('1ROAD', '1GRASS')) and m.material_name(x) not in KERBS})
               for m in (kn5io.Model(os.path.join(TRACK, f)) for f in ('velopark.kn5', 'velopark_kerbs.kn5', 'velopark_paddock.kn5'))]
    top = lambda x, z: max([h for g in grounds for h, _ in g.hits(x, z)], default=np.nan)
    # the paint's lines and how far over the ground they lie, the way they are wound
    every = np.concatenate([m['v'][m['f']] for mdl in (paint, kerbs) for m in mdl.meshes if m['name'].startswith('LINE')])
    pick = every[::max(1, len(every) // 2000), 0]
    lift = np.nanmedian([y - top(x, z) for x, y, z in pick[:, :3]])
    turn = lambda t: (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    wound = np.sign(np.median(turn(every[:, :, [0, 2]])))
    lo = np.min([l['pts'].min(0) for l in lines], 0) - 2
    hi = np.max([l['pts'].max(0) for l in lines], 0) + 2
    # the paint there, less what is taken out
    under = []
    for f, mdl in (('velopark_paint.kn5', paint), ('velopark_kerbs.kn5', kerbs)):
        for m in mdl.meshes:
            if m['name'].startswith('LINE'):
                t = m['v'][np.delete(m['f'], sorted(spurs.get((f, m['name']), ())), 0)][:, :, [0, 2]]
                a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
                t = t[(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-8) & (t.max(1) > lo).all(1) & (t.min(1) < hi).all(1)]
                under += [Polygon(x) for x in t]
    under = unary_union(under)

    def shape(line):
        pts = line['pts']
        if not line['dashed']:
            return LineString(pts).buffer(LINE_WIDTH / 2, cap_style='flat')
        # half a metre long a metre apart, as the dashes where the pit lane joins the 1500
        ls = LineString(pts)
        return unary_union([shapely.ops.substring(ls, u, min(u + .5, ls.length)).buffer(LINE_WIDTH / 2, cap_style='flat')
                            for u in np.arange(.25, ls.length - .2, 1.)])

    def mesh(name, geoms):
        # painted once where it runs into a line already there
        g = unary_union(geoms).difference(under)
        tri = []
        for p in getattr(g, 'geoms', [g]):
            if p.area < 1e-4:
                continue
            tri += [np.array(t.exterior.coords[:3]) for t in shapely.constrained_delaunay_triangles(p).geoms]
        # split until short enough to follow the ground
        while True:
            big = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) > 1.]
            if not big:
                break
            tri = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) <= 1.]
            for p, q, r in big:
                pq, qr, rp = (p + q) / 2, (q + r) / 2, (r + p) / 2
                tri += [np.array(x) for x in ((p, pq, rp), (pq, q, qr), (rp, qr, r), (pq, qr, rp))]
        P = np.array(tri)
        P[turn(P) * wound < 0] = P[turn(P) * wound < 0][:, ::-1]
        P = P.reshape(-1, 2)
        Y = np.array([top(x, z) for x, z in P])
        # off the ground's triangles, as on a seam: as high as the nearest corner on them
        on = ~np.isnan(Y)
        for i in np.flatnonzero(~on):
            Y[i] = Y[on][np.linalg.norm(P[on] - P[i], axis=1).argmin()]
        Y = Y + lift
        return P, Y, name

    mat = next(m for m in paint.materials if m['name'] == 'LinePaint')
    shared = [l for l in lines if not l['own']]
    own = [l for l in lines if l['own']]
    if shared:
        P, Y, name = mesh('LINE_%s_EDGE' % layout.upper(), [shape(l) for l in shared])
        b, _ = kn5io.mesh_block(name, np.c_[P[:, 0], Y, P[:, 1]], np.tile([0, 1., 0], (len(P), 1)),
                                np.c_[P[:, 0] / 10, P[:, 1] / 20], mat_blocks.index(mat['raw']))
        mesh_blocks.append(b)
    kn5io.write(os.path.join(TRACK, 'velopark_kerbs.kn5'), tex_blocks, mat_blocks, mesh_blocks)
    fn = None
    if own:
        P, Y, name = mesh('LINE_%s_MOUTHS' % layout.upper(), [shape(l) for l in own])
        b, _ = kn5io.mesh_block(name, np.c_[P[:, 0], Y, P[:, 1]], np.tile([0, 1., 0], (len(P), 1)),
                                np.c_[P[:, 0] / 10, P[:, 1] / 20], 0)
        fn = '%s_lines.kn5' % layout
        kn5io.write(os.path.join(TRACK, fn), [paint.textures[t] for t in mat['tex'].values()], [mat['raw']], [b])
    # the paint running off the edge where a line is carried on
    gone = 0
    for f in ('velopark_paint.kn5', 'velopark_kerbs.kn5'):
        path = os.path.join(TRACK, f)
        model = kn5io.Model(path)
        data = bytearray(model.bytes)
        for (g, name), faces in spurs.items():
            if g == f:
                kn5io.blank_faces(data, model.mesh(name), sorted(faces))
                gone += len(faces)
        open(path, 'wb').write(data)
    length = lambda ls: sum(np.linalg.norm(np.diff(l['pts'], axis=0), axis=1).sum() for l in ls)
    log('road marks: %s edge lines followed round it, %d gaps at the ends of kerbs closed, %.0f m; %d mouths of the roads '
        'it does not take lined across, %.0f m, %d of them dashed, into %s; %d triangles of paint running off the edge gone'
        % (layout, len(shared), length(shared), len(own), length(own), sum(l['dashed'] for l in own), fn, gone))
    return fn


def grid_end(paint, data, ground):
    """Lay the 1500 grid's lines, slots and numbers where they bend in at the last turn, and
    the start places on them, at the distance from the inside edge they keep along the
    straight: in the paint, in place, and in each layout."""
    from shapely.geometry import LineString, Point
    live = lambda m: m['f'][(m['f'][:, 0] != m['f'][:, 1]) & (m['f'][:, 1] != m['f'][:, 2]) & (m['f'][:, 0] != m['f'][:, 2])]
    # the inside edge line, as a chain of its corners from the far end of the turn round
    m = paint.mesh(GRID_EDGE)
    v = m['v'][np.unique(live(m))][:, [0, 2]]
    v = v[np.linalg.norm(v - GRID_END, axis=1) < 60]
    chain, left = [int(v[:, 0].argmax())], set(range(len(v)))
    left.discard(chain[0])
    while left:
        L = np.array(sorted(left))
        d = np.linalg.norm(v[L] - v[chain[-1]], axis=1)
        if d.min() > 3:
            break
        chain.append(int(L[d.argmin()]))
        left.discard(chain[-1])
    edge = LineString(smooth_line(v[chain], 9))
    def frame(s_):
        """The edge line's point at s, and the way out from it on the grid's side."""
        a, b = (np.array(edge.interpolate(x).coords[0]) for x in (s_ - .3, s_ + .3))
        t = (b - a) / np.linalg.norm(b - a)
        return np.array(edge.interpolate(s_).coords[0]), np.array([-t[1], t[0]]) * side

    def sd(q):
        """Along the edge line, and out from it, across it below 0."""
        S = np.array([edge.project(Point(x)) for x in q])
        D = np.array([(x - Q) @ n for x, (Q, n) in zip(q, map(frame, S))])
        return S, D
    side = 1.
    near = paint.mesh('GRID_1500')['v'][:, [0, 2]]
    near = near[np.linalg.norm(near - GRID_END, axis=1) < 15]
    side = np.sign(np.median(sd(near)[1]))
    # the grid's long lines
    g = paint.mesh('GRID_1500')
    f = live(g)
    key = np.unique(np.round(g['v'][:, [0, 2]], 3), axis=0, return_inverse=True)[1].ravel()
    part = list(range(key.max() + 1))

    def root(i):
        while part[i] != i:
            part[i] = part[part[i]]
            i = part[i]
        return i
    for a, b, c in key[f]:
        part[root(b)] = root(a)
        part[root(c)] = root(a)
    grp = np.array([root(key[a]) for a, _, _ in f])
    lines = []
    for r in np.unique(grp):
        idx = np.unique(f[grp == r])
        xz = g['v'][idx][:, [0, 2]]
        if (grp == r).sum() < 20 or np.linalg.norm(xz - GRID_END, axis=1).min() > 15:
            continue
        S, D = sd(xz)
        near = S < S.min() + 30
        S, D = S[near], D[near]
        keep = np.median(D[S > S.min() + 20])
        # its middle along it, from both its edges
        bins = np.arange(S.min(), S.max() + .5, .5)
        mid = []
        for b in bins:
            q = (S >= b - .4) & (S < b + .4)
            mid.append((D[q].min() + D[q].max()) / 2 if q.sum() >= 2 else np.nan)
        mid = np.array(mid)
        ok = ~np.isnan(mid)
        delta = keep - np.interp(bins, bins[ok], mid[ok])
        # only from where it starts to bend in, fading in over a metre
        off = np.abs(delta) > .08
        if not off.any():
            continue
        last = bins[np.flatnonzero(off)[-1]] + 1.
        delta = np.where(bins <= last, delta, 0) * np.clip((last + 1 - bins), 0, 1)
        lines.append((keep, bins, delta))
    if not lines:
        log('road marks: grid end, nothing to move')
        return
    lines.sort(key=lambda x: x[0])

    def move(xz):
        S, D = sd(xz)
        out = xz.copy()
        for k, (s_, d_, q) in enumerate(zip(S, D, xz)):
            dl = np.array([np.interp(s_, b, dd) for _, b, dd in lines])
            was = np.array([kp for kp, _, _ in lines]) - dl
            shift = np.interp(d_, was, dl)
            if abs(shift) < 1e-4:
                continue
            out[k] = q + frame(s_)[1] * shift
        return out
    def in_column(p1, refs):
        """Moved across to lie as far out as the ones in its column that stay where they are."""
        S, D = sd(p1[None])
        near = refs[np.abs(refs - D[0]) < .8]
        if not len(near):
            return p1
        Q, n = frame(S[0])
        return Q + n * np.median(near)

    # the start places that stay, as far out as each lies
    stay = kn5io.Model(os.path.join(TRACK, 'vp_1500.kn5')).dummies
    stay = np.array([w[3, [0, 2]] for n, w in stay.items() if n.startswith('AC_START')])
    stay = stay[np.linalg.norm(stay - GRID_END, axis=1) < 40]
    stay = sd(stay[np.linalg.norm(move(stay) - stay, axis=1) < 1e-3])[1]

    def turned(p0, fwd):
        """Where a start place moves to, and how far round it turns, to run along the lines."""
        p1 = move(p0[None])[0]
        if np.linalg.norm(p1 - p0) < 1e-3:
            return p0, 0.
        p1 = in_column(p1, stay)
        s_ = edge.project(Point(p1))
        tan = np.array(edge.interpolate(s_ + .5).coords[0]) - np.array(edge.interpolate(s_ - .5).coords[0])
        tan = tan if tan @ fwd > 0 else -tan
        return p1, math.atan2(tan[1], tan[0]) - math.atan2(fwd[1], fwd[0])
    def pieces(m):
        """The small pieces of a mesh, as their corners: a slot's mark, or a digit."""
        f = live(m)
        key = np.unique(np.round(m['v'][:, [0, 2]], 3), axis=0, return_inverse=True)[1].ravel()
        part[:] = list(range(key.max() + 1))
        for a, b, c in key[f]:
            part[root(b)] = root(a)
            part[root(c)] = root(a)
        grp = np.array([root(key[a]) for a, _, _ in f])
        return [np.unique(f[grp == r]) for r in np.unique(grp) if (grp == r).sum() <= 20]
    # each slot's mark moved whole to where its middle goes, and turned square to the lines
    # there; its number with it
    marks = []
    gm = paint.mesh('GRID_1500')
    # the marks that stay, as far out as each lies
    mids = [gm['v'][vs][:, [0, 2]].mean(0) for vs in pieces(gm)]
    mids = np.array([c for c in mids if np.linalg.norm(c - GRID_END) < 40])
    still = sd(mids[np.linalg.norm(move(mids) - mids, axis=1) < 1e-3])[1]
    for vs in pieces(gm):
        xz = gm['v'][vs][:, [0, 2]]
        c = xz.mean(0)
        if np.linalg.norm(c - GRID_END) > 25:
            continue
        c1 = move(c[None])[0]
        if np.linalg.norm(c1 - c) < 1e-3:
            continue
        c1 = in_column(c1, still)
        across = np.linalg.eigh(np.cov((xz - c).T))[1][:, -1]
        n1 = frame(edge.project(Point(c1)))[1]
        n1 = n1 if n1 @ across > 0 else -n1
        a = math.atan2(n1[1], n1[0]) - math.atan2(across[1], across[0])
        marks.append((vs, c, c1, np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])))
    moved = 0
    for name in ('GRID_1500', 'NUMBERS_1500'):
        m = paint.mesh(name)
        idx = np.flatnonzero(np.linalg.norm(m['v'][:, [0, 2]] - GRID_END, axis=1) < 25)
        old = m['v'][idx]
        new = move(old[:, [0, 2]])
        at = {i: k for k, i in enumerate(idx)}
        for vs in pieces(m):
            vs = [i for i in vs if i in at]
            if not vs or not marks:
                continue
            c = m['v'][vs][:, [0, 2]].mean(0)
            k = int(np.argmin([np.linalg.norm(mc - c) for _, mc, _, _ in marks]))
            _, mc, mc1, R = marks[k]
            if np.linalg.norm(mc - c) > 3.:
                continue
            for i in vs:
                new[at[i]] = mc1 + R @ (m['v'][i, [0, 2]] - mc)
        for i, a, b in zip(idx, old, new):
            if np.linalg.norm(a[[0, 2]] - b) > 1e-4:
                y = a[1] - ground.y(a[0], a[2]) + ground.y(*b)
                struct.pack_into('<3f', data, m['vp'] + i * 44, b[0], y, b[1])
                moved += 1
    log('road marks: grid end, %d lines laid on along the turn, %d corners of the grid and numbers moved, up to %.2f m'
        % (len(lines), moved, max(np.abs(d).max() for _, _, d in lines)))
    # the start places on them, turned to run along the lines there
    for path in sorted(glob.glob(os.path.join(TRACK, 'vp_*.kn5'))):
        lay = kn5io.Model(path)
        out = bytearray(lay.bytes)
        n = 0
        for name, (at, parent) in lay.dummy_at.items():
            if not name.startswith('AC_START'):
                continue
            w = lay.dummies[name].copy()
            p0 = w[3, [0, 2]]
            p1, a = turned(p0, w[2, [0, 2]])
            if np.linalg.norm(p1 - p0) < 1e-3:
                continue
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            for row in range(3):
                w[row, [0, 2]] = R @ w[row, [0, 2]]
            w[3, [0, 2]] = p1
            w[3, 1] += ground.y(*p1) - ground.y(*p0)
            struct.pack_into('<16f', out, at, *(w @ np.linalg.inv(parent)).ravel())
            n += 1
        if n:
            open(path, 'wb').write(out)
            log('road marks: grid end, %d start places moved in %s' % (n, os.path.basename(path)))


def island_holes(model, data, kerbs, area, step=1.):
    """Gaps in the ground of the main model, as built, and the new kerbs and grass, inside
    the area: triangles of (x, y, z) to fill them, y from the ground round them."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import shapely
    import struct
    lo, hi = np.array(area[:2]), np.array(area[2:])
    flats, solid = [], []
    for mdl, live in ((model, data), (kerbs, None)):
        for m in mdl.meshes:
            if not m['name'].startswith(('1ROAD', '1GRASS', 'KERB')):
                continue
            f = m['f'] if live is None else np.array(struct.unpack_from('<%dH' % m['f'].size, live, m['ip'])).reshape(-1, 3)
            t = m['v'][f][:, :, :3]
            c = t[:, :, [0, 2]].mean(1)
            t = t[(c > lo).all(1) & (c < hi).all(1)]
            e0, e1 = t[:, 1, [0, 2]] - t[:, 0, [0, 2]], t[:, 2, [0, 2]] - t[:, 0, [0, 2]]
            t = t[np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) > 1e-9]
            solid += [Polygon(x[:, [0, 2]]) for x in t]
            # the heights from the ground itself, not the kerbs standing on it
            if not m['name'].startswith(('1ROAD_KERB_', 'KERB')) or m['name'].startswith('1ROAD_KERB_BED'):
                flats.append(t.reshape(-1, 3))
    flats = np.concatenate(flats)
    cover = unary_union(solid).buffer(.005)
    out = []
    for g in getattr(cover, 'geoms', [cover]):
        for ring in g.interiors:
            hole = Polygon(ring)
            if hole.area < .05:
                continue
            # a little over the ground round it, so no crack opens along its edge
            tri = [np.array(t.exterior.coords[:3]) for t in shapely.constrained_delaunay_triangles(hole.buffer(.03)).geoms]
            # split until small enough to follow the ground
            while True:
                big = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) > step]
                if not big:
                    break
                tri = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) <= step]
                for a, b, c in big:
                    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
                    tri += [np.array(x) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
            P = np.array(tri).reshape(-1, 2)
            d = np.linalg.norm(flats[None, :, [0, 2]] - P[:, None], axis=2)
            w = 1 / np.maximum(d, .05) ** 2 * (d < 2.5)
            y = (w * flats[None, :, 1]).sum(1) / np.maximum(w.sum(1), 1e-9) - .005
            out.append((hole, np.c_[P[:, 0], y, P[:, 1]].reshape(-1, 3, 3)))
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

    def row(s, name, line, what, at=None, height=None):
        """Stacks of three every 0.62 m along the line, or at the points given, where there is
        room for them; or, given the height of the ground under them, at every one."""
        if at is None:
            seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
            cum = np.r_[0, np.cumsum(seg)]
            at = [np.array([np.interp(u, cum, line[:, 0]), np.interp(u, cum, line[:, 1])]) for u in np.arange(0, cum[-1] + .01, .62)]
        if height is None:
            why = [(p.round(2).tolist(), bool(s.ground.hits(*p)), s.ground.on(p[0], p[1], s.keep_off)) for p in at if not s.clear(p)]
            at = [p for p in at if s.clear(p)]
            if why:
                log('  %s: no room at %s' % (name, why[:6]))
            height = lambda p: s.ground.y(*p)
        T, P = [], []
        for k, p in enumerate(at):
            foot = np.array([p[0], height(p), p[1]])
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

    def bridge(s, name, before, a, b, after, what, height=None):
        """A gap in a barrier closed: on from its stack a to its stack b, along a curve carrying
        on the way it runs on either side, from the stacks before and after them, with the
        stacks as evenly spaced as the barrier's."""
        before, a, b, after = (np.array(p, float) for p in (before, a, b, after))
        L = np.linalg.norm(b - a)
        t0, t1 = (a - before) / np.linalg.norm(a - before) * L, (after - b) / np.linalg.norm(after - b) * L
        u = np.linspace(0, 1, 400)[:, None]
        curve = (2 * u**3 - 3 * u**2 + 1) * a + (u**3 - 2 * u**2 + u) * t0 + (-2 * u**3 + 3 * u**2) * b + (u**3 - u**2) * t1
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        n = max(2, int(round(cum[-1] / .62)))
        at = [np.array([np.interp(v, cum, curve[:, 0]), np.interp(v, cum, curve[:, 1])]) for v in cum[-1] * np.arange(1, n) / n]
        s.row(name, None, what, at, height)

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
    # the photo held once, padded so the search never reads past its sides
    pad = search + 16
    arr = np.pad(np.asarray(img), ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    del img
    # each area in tiles of the photo, so long kerbs bending round a corner are not one huge box
    tiles = []
    for tris in areas:
        px = (np.c_[tris.reshape(-1, 2), np.ones(len(tris) * 3)] @ ground_uv * (W, H)).reshape(-1, 3, 2)
        key = np.floor(px.mean(1) / 256).astype(int)
        for k in np.unique(key, axis=0):
            tiles.append(px[(key == k).all(1)])
    for px in tiles:
        lo = np.floor(px.reshape(-1, 2).min(0)).astype(int) - 8
        hi = np.ceil(px.reshape(-1, 2).max(0)).astype(int) + 8
        size = tuple(hi - lo)
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        for t in px:
            draw.polygon([tuple(p - lo) for p in t], fill=255)
        # a couple of texels round it, faded in
        mask = mask.filter(ImageFilter.MaxFilter(5))
        soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.5)), np.float32)[..., None] / 255
        iy, ix = np.nonzero(np.asarray(mask) > 0)
        if not len(iy):
            continue
        # judged on a sample of the texels, enough to tell grass from asphalt
        pick = np.random.default_rng(len(iy)).choice(len(iy), min(len(iy), 3000), replace=False)
        sy, sx = iy[pick] + lo[1] + pad, ix[pick] + lo[0] + pad
        best = None
        for dy in range(-search, search + 1, 5):
            for dx in range(-search, search + 1, 5):
                src = arr[sy + dy, sx + dx].astype(np.int16)
                green = np.mean((src[:, 1] > src[:, 0] + 8) & (src[:, 1] > src[:, 2] + 4))
                score = green - .0005 * math.hypot(dx, dy)
                if best is None or score > best[0]:
                    best = (score, dx, dy, green)
        _, dx, dy, green = best
        y0, x0 = lo[1] + pad, lo[0] + pad
        src = arr[y0 + dy:y0 + dy + size[1], x0 + dx:x0 + dx + size[0]].astype(np.float32)
        dst = arr[y0:y0 + size[1], x0:x0 + size[0]]
        dst[:] = (dst * (1 - soft) + src * soft).round().astype(np.uint8)
        log('  ground photo: %.0f m2 painted over with grass from %.0f m away, %.0f%% green' % (
            len(iy) * (1 / (W * abs(ground_uv[0, 0]))) ** 2, math.hypot(dx, dy) / (W * abs(ground_uv[0, 0])), 100 * green))
    out = io.BytesIO()
    Image.fromarray(arr[pad:pad + H, pad:pad + W]).save(out, format='JPEG', quality=95)
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
    # where the track's ground has holes, the paddock's asphalt under it is the ground
    ground.below = Ground(kn5io.Model(os.path.join(TRACK, 'velopark_paddock.kn5')), {'1ROAD_PADDOCK'})
    ai15 =C.ai_line('vp_1500')[:, [0, 2]]
    aiam = C.ai_line('vp_1500_amateur')[:, [0, 2]]
    ai10 = C.ai_line('vp_1000')[:, [0, 2]]

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
        # and round the 1000: concrete inside its corners, plain ramps elsewhere
        ('K1_T1_ENTRY', 'ramp', '1ROAD_90', ai10),
        ('K1_T1', 'concrete', '1ROAD_62', ai10),
        ('K1_T2', 'concrete', '1ROAD_91', ai10),
        ('K1_T2_OUT', 'ramp', '1ROAD_63', ai10),
        ('K1_T3', 'concrete', '1ROAD_63', ai10),
        ('K1_T3_EXIT', 'ramp', '1ROAD_92', ai10),
        ('K1_T4_JOIN', 'ramp', '1ROAD_64', ai10),
        ('K1_T4_ENTRY', 'ramp', '1ROAD_65', ai10),
        ('K1_T4', 'concrete', '1ROAD_93', ai10),
        ('K1_T4_EXIT', 'ramp', '1ROAD_66', ai10),
        ('K1_S_RIGHT', 'ramp', '1ROAD_94', ai10),
        ('K1_S_LEFT', 'concrete', '1ROAD_67', ai10),
        ('K1_T5_ENTRY', 'ramp', '1ROAD_99', ai10),
        ('K1_T5', 'concrete', '1ROAD_95', ai10),
        ('K1_T5_EXIT', 'ramp', '1ROAD_68', ai10),
        ('K1_T6', 'concrete', '1ROAD_69', ai10),
        ('K1_T6_EXIT', 'ramp', '1ROAD_96', ai10),
        ('K1_T7', 'concrete', '1ROAD_97', ai10),
        ('K1_T7_EXIT', 'ramp', '1ROAD_70', ai10),
        ('K1_T8', 'ramp', '1ROAD_98', ai10),
        ('K1_T8_OUT', 'ramp', '1ROAD_71', ai10),
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
    kerb_back = {}
    driven_road ={k for k in driven if model.meshes[k]['name'].startswith('1ROAD')}
    for label, kind, old, ai in kerbs:
        old_mesh = model.mesh(old)
        line, far = edge_from_kerb(old_mesh, ai)
        cover = old_mesh['v'][:, [0, 2]]
        old_tri = old_mesh['v'][old_mesh['f']]
        anchor = None
        if label in PIECE:
            # the line up to or on from where the two pieces meet, and the old kerb along it
            cut = int(np.linalg.norm(line - SPLIT_AT[old], axis=1).argmin())
            mine = lambda xz: (np.linalg.norm(xz[:, None] - line[None], axis=2).argmin(1) < cut) == (PIECE[label] == 0)
            cover, old_tri = cover[mine(cover)], old_tri[mine(old_tri[:, :, [0, 2]].mean(1))]
            far = far[mine(far)]
            line, anchor = (line[:cut + 1], 'end') if PIECE[label] == 0 else (line[cut:], 'start')
        path = Path(line, far.mean(0), ground, cover=cover, anchor=anchor)
        if label in ALONG_LINE:
            tri = paint0.mesh(ALONG_LINE[label])['v'][paint0.mesh(ALONG_LINE[label])['f']][:, :, [0, 2]]
            c = tri.mean(1)
            sc, tc = path.st(c)
            q = (sc > -2) & (sc < path.length + 2) & (np.abs(tc) < 2.5)
            c = c[q][np.argsort(sc[q])]
            k = np.linalg.norm(c[:, None] - path.nodes[None], axis=2).argmin(1)
            path = Path(smooth_line(resample(c + path.normal[k] * .06), 5), far.mean(0), ground, cover=cover)
        if label in ALONG_EDGE:
            # out from the old kerb's front to where the asphalt under it ends, the junction's
            # included, before the grass drops away
            asphalt = (driven_road | set(bed_meshes)) - {model.meshes.index(old_mesh)}
            pts = []
            for k in range(path.n + 1):
                t = -.5
                while t < 2 and ground.on(*(path.nodes[k] + path.normal[k] * (t + .05)), asphalt):
                    t += .05
                pts.append(path.nodes[k] + path.normal[k] * t)
            out = [float(x) for x in path.st(np.array(pts))[1]]
            log('kerb %s moved out %.2f..%.2f m, to the edge of the asphalt' % (label, min(out), max(out)))
            path = Path(smooth_line(resample(np.array(pts)), 5), far.mean(0), ground, cover=cover)
        if label in SHIFT_OUT:
            k = np.linalg.norm(line[:, None] - path.nodes[None], axis=2).argmin(1)
            path = Path(line + path.normal[k] * SHIFT_OUT[label], far.mean(0), ground, cover=cover)
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
        # where two pieces of one old kerb meet, the ramp runs on at full height to the concrete
        # kerb, closed down to the ground there, and the concrete kerb starts as the ramp does,
        # its painted part carrying the ramp on and its concrete growing out behind it as a
        # triangle, so karts riding the ramp meet one rumble and no square edge of concrete
        joint = 1 - PIECE[label] if label in PIECE else None
        morph = joint if kind == 'concrete' else None
        ends = (joint != 0, joint != 1)
        full = joint if kind != 'concrete' else None
        if full is not None:
            s_at = 0. if full == 0 else path.length
            prof = sorted({(round(c[1], 5), round(c[2], 5)) for quad in q.q for c in quad
                           if abs(c[0] - s_at) < 1e-6 and c[1] >= 0 and c[2] >= 0})
            for (t0, z0), (t1, z1) in zip(prof, prof[1:]):
                if z0 > 0 or z1 > 0:
                    cap = [(s_at, t0, 0), (s_at, t1, 0), (s_at, t1, z1), (s_at, t0, z0)]
                    q.quad(*cap)
                    q.quad(*cap[::-1])
        block, n, pos, tris = soup_mesh(q, path, warp[1][-1], angle, '1ROAD_KERB_' + label, 0 if kind == 'concrete' else 2,
                                        taper=ends, warp=warp, low=KERB_LOW, morph=morph)
        mesh_blocks.append(block)
        log('kerb %-11s %-8s replaces %-9s length %5.1f m, %5d triangles' % (label, kind, old, path.length, n))
        kn5io.blank_faces(data, old_mesh)
        placed[label] = (path, pos)
        # the side of the kerb at its back edge, down into the grass, which can lie lower than
        # the road the kerb starts from
        reach = float(np.interp(np.array(q.q)[:, :, 1].max(), *warp))
        bands.append((label, path, reach))

        def back_edge(S, path=path, reach=reach, ends=ends, morph=morph):
            """How far out the kerb's back edge lies at S along it."""
            t = reach * grow(S, path.length, ends)
            if morph is not None:
                ramp = RAMP_T[-1] * RAMP_SCALE
                t = ramp + (t - ramp) * grow(S, path.length, (morph == 0, morph == 1))
            return t
        kerb_back[label] = back_edge
        S = np.linspace(0, path.length, int(path.length / .3) + 1)
        top = path.place(S, back_edge(S), np.zeros_like(S))
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
        if label in ALONG_EDGE:
            # on the asphalt's edge the grass behind drops away into a ditch: a slope of grass
            # from the kerb's back edge down to it, so the ramp ends on grass like the others
            T = [0, .3, .6, 1.]
            rows = [top] + [path.place(S, reach * np.clip(np.minimum(S, path.length - S) / TAPER, 0, 1) + d, np.zeros_like(S))
                            for d in T[1:]]
            for r, d in zip(rows[1:], T[1:]):
                for p, t in zip(r, top):
                    g = ground.y(p[0], p[2])
                    p[1] = g if d == 1 else max(g + .01, t[1] + (g - t[1]) * d)
            P = []
            for r0, r1 in zip(rows, rows[1:]):
                for i in range(len(S) - 1):
                    P += [r0[i], r1[i + 1], r0[i + 1], r0[i], r1[i], r1[i + 1]]
            P = np.array(P)
            nrm = np.cross(P[1::3] - P[0::3], P[2::3] - P[0::3])
            if nrm[:, 1].mean() < 0:
                P = P.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
                nrm = -nrm
            nrm = np.repeat(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9), 3, axis=0)
            name = '1GRASS_KERB_FILL_' + label
            b2, _ = kn5io.mesh_block(name, P, nrm, np.c_[P[:, [0, 2]], np.ones(len(P))] @ grass_uv, grass_mat)
            mesh_blocks.append(b2)
            grass_meshes.append(name)
            log('  grass sloped from behind it down into the ditch, %.2f m at most' % max(t[1] - g[1] for t, g in zip(top, rows[-1])))
        # the photo under the old kerb, and in a band behind the new one running on past its ends
        t0, t1 = path.nodes[1] - path.nodes[0], path.nodes[-1] - path.nodes[-2]
        line = np.r_[[path.nodes[0] - t0 / np.linalg.norm(t0) * 2], path.nodes, [path.nodes[-1] + t1 / np.linalg.norm(t1) * 2]]
        nrm = np.r_[path.normal[:1], path.normal, path.normal[-1:]]
        back = line + nrm * (reach + 1.5)
        band = np.concatenate([np.stack([line[:-1], line[1:], back[1:]], 1), np.stack([line[:-1], back[1:], back[:-1]], 1)])
        repaint.append(np.concatenate([old_tri[:, :, [0, 2]], band]))
        # strips of old light asphalt at the kerb, driven on by no layout: past the kerb's
        # ends they carry the track on, so they get the track's asphalt; alongside it they
        # are behind it, so grass. Same shape, cut where the kerb starts and ends
        bed, bed_src = [], []
        # behind this one the asphalt stays as it is
        keep = label in KEEP_BED
        # the end where a piece of a split old kerb meets the other: what lies past it is the
        # other piece's, and stays as it was (0 its start, 1 its end)
        joint = 1 - PIECE[label] if label in PIECE else None
        for mi in [] if keep else bed_meshes:
            m = model.meshes[mi]
            tri = m['v'][m['f']]
            near = np.flatnonzero(np.linalg.norm(tri[:, :, [0, 2]].mean(1)[:, None] - path.nodes[None], axis=2).min(1) < 4)
            for j in near:
                s, t = path.st(tri[j][:, [0, 2]])
                if (t > .05).all() and (t < 4).all() and (s > -3).all() and (s < path.length + 3).all():
                    bed.append(tri[j])
                    bed_src.append(mi)
                    kn5io.blank_faces(data, m, [j])
                    bed_cut.append((mi, j))
        # and the original grass the old kerb lay over
        gi = model.meshes.index(grass)
        gt = grass['v'][grass['f']]
        old_xz = old_tri[:, :, [0, 2]]
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
            if label in GRASS_TIPS:
                tip = [False, False]
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
            own = {}
            for tri, mi in zip(bed, bed_src):
                # past an end it only carries the road on where that end points at another
                # road; elsewhere it lies off the road, behind where the kerb's tip was: grass
                for e, (plane, part) in enumerate(((start, 'before'), (end, 'after'))):
                    if e == joint:
                        own.setdefault(mi, []).extend(clip(tri, *plane))
                    else:
                        pieces[part if tip[e] else 'beside'] += clip(tri, *plane)
                for half in clip(tri, start[0], -start[1]):
                    for piece in clip(half, end[0], -end[1]):
                        # alongside the kerb: grass, but beside a tapered end that points at another
                        # road, as at the tips of the island, asphalt like the road around it; not
                        # inside the islands that are grass all the way to the road
                        sc = path.st(piece[:, [0, 2]].mean(0)[None])[0][0]
                        between = ((sc < TAPER and tip[0]) or (sc > path.length - TAPER and tip[1])) and label not in [l for l, *_ in ISLANDS]
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
            for mi, tris in own.items():
                src = model.meshes[mi]
                P = np.array(tris)
                b3, _ = kn5io.mesh_block('1ROAD_KERB_BED_%s_JOINT_%s' % (label, src['name']), P[:, :, :3].reshape(-1, 3),
                                         P[:, :, 3:6].reshape(-1, 3), P[:, :, 6:8].reshape(-1, 2),
                                         copy_material(model.materials[src['mat']]))
                mesh_blocks.append(b3)
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
    # the barrier round the inside of the vp_1500 pit entry broke off in two places, with a
    # short piece of its own standing out of line in the second gap: that goes, and the
    # barrier runs on through both
    tyres_path = os.path.join(TRACK, 'velopark_tyres.kn5')
    tyres = kn5io.Model(tyres_path)
    tyres_data = bytearray(tyres.bytes)
    # and the barriers laid again along run-offs go
    old_walls = [n for _, _, _, w in RUNOFFS if w for n in (w, w + 'F', '1WALL_TYRE_' + w[len('TYRES_'):])]
    for name in PIT_ENTRY_PIECE + tuple(old_walls):
        kn5io.blank_faces(tyres_data, tyres.mesh(name))
    open(tyres_path, 'wb').write(tyres_data)
    for k, stacks in enumerate(PIT_ENTRY_GAPS):
        rows.bridge('PIT_ENTRY_%d' % k, *stacks, 'closing gap %d in the barrier inside the vp_1500 pit entry' % k)
    # partly over the paddock's asphalt, where the track's ground has a hole
    rows.bridge('K1_T2', *K1_T2_GAP, 'closing the gap in the barrier inside the vp_1000 T2', height=lambda p: ground.y(*p))
    mesh_blocks += rows.blocks

    # the grass showing through between the paddock's asphalt and the road out of the vp_1500
    # pit entry loop, asphalt all round it: asphalt too, of that road, as high as the asphalt
    # round it, and the grass under it goes so none grows through
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import shapely
    lo, hi = np.array(PIT_ENTRY_GRASS[:2]), np.array(PIT_ENTRY_GRASS[2:])
    def flat_polys(tri):
        c = tri[:, :, [0, 2]].mean(1)
        tri = tri[(c > lo - 3).all(1) & (c < hi + 3).all(1)]
        e0, e1 = tri[:, 1, [0, 2]] - tri[:, 0, [0, 2]], tri[:, 2, [0, 2]] - tri[:, 0, [0, 2]]
        tri = tri[np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) > 1e-7]
        return tri, [Polygon(t[:, [0, 2]]) for t in tri]
    pad = kn5io.Model(os.path.join(TRACK, 'velopark_paddock.kn5')).mesh('1ROAD_PADDOCK')
    asphalt_tri = [flat_polys(m['v'][m['f']][:, :, :3]) for m in model.meshes if m['name'].startswith('1ROAD')]
    asphalt_tri.append(flat_polys(pad['v'][pad['f']][:, :, :3]))
    asphalt = unary_union([q for _, ps in asphalt_tri for q in ps])
    near = np.concatenate([t.reshape(-1, 3) for t, _ in asphalt_tri if len(t)])
    grass = model.mesh('1GRASS_0')
    gtri, gpolys = flat_polys(grass['v'][grass['f']][:, :, :3])
    showing = unary_union(gpolys).difference(asphalt)
    box = Polygon([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])
    # and the slivers along its edges
    pieces = [g for g in getattr(showing, 'geoms', [showing]) if g.area > .01 and box.contains(g.centroid)]
    assert pieces, 'no grass showing at the vp_1500 pit entry'
    hole = unary_union([g.buffer(.05) for g in pieces]).buffer(-.05).intersection(showing.buffer(.01))
    tri = [np.array(t.exterior.coords[:3]) for t in shapely.constrained_delaunay_triangles(hole.buffer(.03)).geoms]
    while True:
        big = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) > 1.]
        if not big:
            break
        tri = [t for t in tri if max(np.linalg.norm(t - np.roll(t, 1, 0), axis=1)) <= 1.]
        for a, b, c in big:
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            tri += [np.array(x) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
    P = np.array(tri).reshape(-1, 2)
    d = np.linalg.norm(near[None, :, [0, 2]] - P[:, None], axis=2)
    w = 1 / np.maximum(d, .05) ** 2 * (d < 2.5)
    y = (w * near[None, :, 1]).sum(1) / np.maximum(w.sum(1), 1e-9) - .003
    P3 = np.c_[P[:, 0], y, P[:, 1]].reshape(-1, 3, 3)
    # wound as the road is
    track = model.mesh(PIT_ENTRY_ROAD)
    rt = track['v'][track['f']][:, :, :3]
    facing = np.sign(np.median(np.cross(rt[:, 1] - rt[:, 0], rt[:, 2] - rt[:, 0])[:, 1]))
    flip = np.sign(np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0])[:, 1]) != facing
    P3[flip] = P3[flip][:, ::-1]
    P3 = P3.reshape(-1, 3)
    b5, _ = kn5io.mesh_block('1ROAD_PATCH_PIT_ENTRY', P3, np.tile([0, 1., 0], (len(P3), 1)),
                             track_uv(track, P3[:, [0, 2]]), copy_material(model.materials[track['mat']]))
    mesh_blocks.append(b5)
    gall = grass['v'][grass['f']][:, :, [0, 2]]
    under = [i for i, t in enumerate(gall) if Polygon(t).buffer(0).intersects(hole) and Polygon(t).buffer(0).area > 1e-7]
    # and the grass round it lying on the asphalt, as high as it or over it, hiding it
    pad_ground = Ground(kn5io.Model(os.path.join(TRACK, 'velopark_paddock.kn5')), {'1ROAD_PADDOCK'})
    road_ids = {k for k, x in enumerate(model.meshes) if x['name'].startswith('1ROAD')}
    for i, t in enumerate(grass['v'][grass['f']][:, :, :3]):
        c = t.mean(0)
        if i in under or not ((c[[0, 2]] > lo - 1).all() and (c[[0, 2]] < hi + 1).all()):
            continue
        hs = [h for h, _ in pad_ground.hits(c[0], c[2])] + [h for h, k in ground.hits(c[0], c[2]) if ground.owner[k] in road_ids]
        if hs and c[1] > max(hs) - .02:
            under.append(i)
    kn5io.blank_faces(data, grass, under)
    log('pit entry: %.1f m2 of grass between the asphalt made asphalt of %s, %d grass triangles under it gone'
        % (hole.area, PIT_ENTRY_ROAD, len(under)))

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
    straight_lines = {}
    for label, old in (('STRAIGHT4', '1ROAD_57'), ('STRAIGHT5', '1ROAD_86'), ('LAST_OUT', '1ROAD_61')):
        old_mesh = model.mesh(old)
        line, far = edge_from_kerb(old_mesh, ai15)
        cover = old_mesh['v'][:, [0, 2]]
        if label in STRAIGHTEN:
            # the way the grid's lines run along the straight beside it, away from where they
            # bend into the last turn
            c = line.mean(0)
            u = np.linalg.eigh(np.cov((line - c).T))[1][:, -1]
            g = paint0.mesh('GRID_1500')['v'][:, [0, 2]]
            g = g[(np.abs((g - c) @ np.array([-u[1], u[0]])) < 6) & (np.linalg.norm(g - GRID_END, axis=1) > 20)
                  & (np.abs((g - c) @ u) < np.abs((line - c) @ u).max())]
            # the grid's line nearest the kerb, fitted again without the slot marks beside it
            n = np.array([-u[1], u[0]])
            g = g[np.abs((g - c) @ n - np.median((g - c) @ n)) < 2.5]
            near = g[np.abs((g - c) @ n) <= np.sort(np.abs((g - c) @ n))[len(g) // 5] + .3]
            for _ in range(3):
                gc = near.mean(0)
                u2 = np.linalg.eigh(np.cov((near - gc).T))[1][:, -1]
                r = (near - gc) @ np.array([-u2[1], u2[0]])
                near = near[np.abs(r) < .25]
            u2 = u2 if u2 @ u > 0 else -u2
            n = np.array([-u2[1], u2[0]])
            along_, across = (line - c) @ u2, (line - c) @ n
            log('kerb %s laid straight, in line with the grid, %.1f degrees round from the old one, which wandered %.2f m either side'
                % (label, math.degrees(math.acos(min(1., abs(u @ u2)))), np.ptp(across - np.polyval(np.polyfit(along_, across, 1), along_)) / 2))
            # its edge line in line with the road's edge line before its front: that line's
            # straight run, as its last stretch hooks out to where the old kerb's back was
            v = paint0.mesh(STRAIGHTEN[label][2])['v'][:, [0, 2]]
            ahead = 1 if np.linalg.norm(line[0] - GRID_END) < np.linalg.norm(line[-1] - GRID_END) else -1
            a_front = along_.min() if ahead == 1 else along_.max()
            beyond = ((v - c) @ u2 - a_front) * -ahead
            run = v[(beyond > JOIN_BACK) & (beyond < 8) & (np.abs((v - c) @ n - np.median(across)) < 1.5)]
            fit = np.polyfit((run - c) @ u2, (run - c) @ n, 1)
            away = np.sign((far.mean(0) - c) @ n)
            offset = np.polyval(fit, a_front) + away * (LINE_WIDTH / 2 + .02)
            log('  moved %.2f m, so its edge line runs on from %s' % (offset - np.median(across), STRAIGHTEN[label][2]))
            # the line along it runs on past its front over the hook, to meet that line's run
            a0, a1 = along_.min(), along_.max()
            a0, a1 = (a0 - JOIN_BACK, a1) if ahead == 1 else (a0, a1 + JOIN_BACK)
            full = c + np.outer(np.linspace(a0, a1, len(line)), u2) + n * offset
            straight_lines[label] = Path(full, far.mean(0), ground)
            # shorter at its back
            back = 0 if np.linalg.norm(full[0] - GRID_END) < np.linalg.norm(full[-1] - GRID_END) else -1
            a0, a1 = along_.min(), along_.max()
            if back == 0:
                a0 += STRAIGHTEN[label][1]
            else:
                a1 -= STRAIGHTEN[label][1]
            line = c + np.outer(np.linspace(a0, a1, len(line)), u2) + n * offset
            cover = None
        path = Path(line, far.mean(0), ground, cover=cover)
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

    # -- run-offs: behind some kerbs the track's ground has a hole and the paddock's asphalt
    # below shows in a shape of its own; there it is a strip of road along the kerb, as the
    # run-offs elsewhere, with grass round it, or grass all the way
    import shapely
    from shapely.geometry import Polygon, box as shape_box
    from shapely.ops import unary_union

    def now(buf, m):
        """A mesh's triangles of full vertices as they are now, blanked ones left out, with
        their indices."""
        f = np.array(struct.unpack_from('<%dH' % m['f'].size, buf, m['ip'])).reshape(-1, 3)
        ok = np.flatnonzero((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]))
        return ok, m['v'][f[ok]]

    def flat(t):
        return Polygon(t[:, [0, 2]]).buffer(0)

    def parts(tri, shape):
        """The parts of a triangle of full vertices inside a shape, as triangles wound as it
        is, every value carried across it from its corners."""
        xz = tri[:, [0, 2]]
        A = np.c_[xz, np.ones(3)]
        if abs(np.linalg.det(A)) < 1e-9:
            return []
        coef = np.linalg.solve(A, tri)
        up = np.cross(tri[1, :3] - tri[0, :3], tri[2, :3] - tri[0, :3])[1] > 0
        out = []
        g = Polygon(xz).intersection(shape)
        for p in getattr(g, 'geoms', [g]):
            if p.geom_type != 'Polygon' or p.area < 1e-5:
                continue
            for t in shapely.constrained_delaunay_triangles(p).geoms:
                q = np.c_[np.array(t.exterior.coords[:3]), np.ones(3)] @ coef
                if (np.cross(q[1, :3] - q[0, :3], q[2, :3] - q[0, :3])[1] > 0) != up:
                    q = q[::-1]
                out.append(q)
        return out

    def emit(name, tris, mat, uv=None):
        P = np.array(tris)
        nrm = np.cross(P[:, 1, :3] - P[:, 0, :3], P[:, 2, :3] - P[:, 0, :3])
        nrm *= np.sign(nrm[:, 1:2]) + (nrm[:, 1:2] == 0)
        nrm = np.repeat(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9), 3, axis=0)
        U = P[:, :, 6:8].reshape(-1, 2) if uv is None else uv(P[:, :, [0, 2]].reshape(-1, 2))
        b, _ = kn5io.mesh_block(name, P[:, :, :3].reshape(-1, 3), nrm, U, mat)
        mesh_blocks.append(b)

    grass_at = lambda xz: np.c_[xz, np.ones(len(xz))] @ grass_uv
    reach_of = {l: r for l, _, r in bands}

    def kerb_shape(label):
        path, reach = placed[label][0], reach_of[label]
        S = np.linspace(0, path.length, int(path.length / .25) + 1)
        front = path.place(S, np.zeros_like(S), np.zeros_like(S))[:, [0, 2]]
        back = path.place(S, kerb_back[label](S), np.zeros_like(S))[:, [0, 2]]
        return Polygon(np.r_[front, back[::-1]]).buffer(0)

    def paddock_to_grass(label, area, strip):
        """In the box area, where the paddock's asphalt is the only ground, grass; where the
        strip is, nothing, the strip lies there. The rest of it as it was."""
        x0, z0, x1, z1 = area
        box_ = shape_box(x0, z0, x1, z1)
        # the track's own ground, as it is now, and the kerbs on it
        cover = []
        for m in model.meshes:
            if m['name'] in ground_names:
                _, t = now(data, m)
                c = t[:, :, [0, 2]].mean(1)
                t = t[(c[:, 0] > x0 - 5) & (c[:, 0] < x1 + 5) & (c[:, 1] > z0 - 5) & (c[:, 1] < z1 + 5)]
                cover += [flat(q) for q in t]
        near = [l for l in reach_of
                if (np.abs(placed[l][0].nodes - [(x0 + x1) / 2, (z0 + z1) / 2]) < [(x1 - x0) / 2 + 5, (z1 - z0) / 2 + 5]).all(1).any()]
        cover += [kerb_shape(l) for l in near]
        hole = box_.difference(unary_union(cover).buffer(.002))
        work = hole.union(strip) if strip is not None else hole
        ids, tri = now(paddock_data, pad_mesh)
        hit = [i for i, t in enumerate(tri) if flat(t).intersects(work)]
        grass_tris, rest = [], []
        for i in hit:
            t = tri[i]
            if strip is not None:
                left = Polygon(t[:, [0, 2]]).buffer(0).difference(strip)
            else:
                left = Polygon(t[:, [0, 2]]).buffer(0)
            grass_tris += parts(t, left.intersection(hole))
            rest += parts(t, left.difference(hole))
        kn5io.blank_faces(paddock_data, pad_mesh, ids[hit])
        # never higher than a kerb's back edge right behind it: eased down to it over 0.6 m
        back = []
        for l in near:
            path, reach = placed[l][0], reach_of[l]
            S = np.linspace(0, path.length, int(path.length / .1) + 1)
            back.append(path.place(S, kerb_back[l](S), np.zeros_like(S)))
        if back and grass_tris:
            back = np.concatenate(back)
            for t in grass_tris:
                d = np.linalg.norm(t[:, None, [0, 2]] - back[None, :, [0, 2]], axis=2)
                k = d.argmin(1)
                for c in range(3):
                    if d[c, k[c]] < .6 and t[c, 1] > back[k[c], 1]:
                        t[c, 1] = back[k[c], 1] + (t[c, 1] - back[k[c], 1]) * d[c, k[c]] / .6
        if grass_tris:
            name = '1GRASS_RUNOFF_' + label
            emit(name, grass_tris, grass_mat, grass_at)
            grass_meshes.append(name)
            repaint.append(np.array(grass_tris)[:, :, [0, 2]])
        if rest:
            emit('1ROAD_PADDOCK_REST_' + label, rest, copy_material(next(m for m in model.materials if m['name'] == 'Tarmac Dark')))
        log('  %.0f m2 of the paddock\'s asphalt showing made grass' % sum(Polygon(t[:, [0, 2]]).area for t in grass_tris))

    pad_mesh = paddock.mesh('1ROAD_PADDOCK')
    tyres0 = rows.barrier
    for label, area, width, wall in RUNOFFS:
        path, reach = placed[label][0], reach_of[label]
        S = np.linspace(0, path.length, int(path.length / .25) + 1)
        t_in = kerb_back[label](S)
        end = path.length
        if wall:
            # the barrier as far out behind the kerb all along as its last stack stands
            sa, ta = path.st(stack_centres(tyres0.mesh(wall)))
            t_wall = float(ta[sa.argmax()])
            width = t_wall + .5 - reach
            end = min(float(sa.max()) + RUNOFF_OUT, path.length)
        ease = lambda u: np.clip(u, 0, 1) ** 2 * (3 - 2 * np.clip(u, 0, 1))
        t_out = t_in + width * ease(S / RUNOFF_IN) * ease((end - S) / RUNOFF_OUT)
        sel = t_out - t_in > .01
        inner = path.place(S[sel], t_in[sel], np.zeros(sel.sum()))
        outer = path.place(S[sel], t_out[sel], np.zeros(sel.sum()))
        # off the road, and off the kerb
        beside = [ground.owner[k] for x, z in path.nodes - path.normal * .3 for _, k in ground.hits(x, z)]
        track = model.meshes[max(set(beside), key=beside.count)]
        road_now = []
        for m in model.meshes:
            if m['name'] in ground_names and m['name'].startswith('1ROAD'):
                _, t = now(data, m)
                road_now += [flat(q) for q in t if np.linalg.norm(q[:, [0, 2]].mean(0) - path.nodes, axis=1).min() < 8]
        strip = Polygon(np.r_[inner[:, [0, 2]], outer[::-1][:, [0, 2]]]).buffer(0)
        strip = strip.difference(unary_union(road_now + [kerb_shape(label)]))
        strip = shapely.segmentize(strip, .5)
        # its heights: the kerb's back edge along its inside, so it is never higher than the
        # kerb, the ground everywhere else
        known = {tuple(np.round(p[[0, 2]], 3)): p[1] for p in inner}
        tris = []
        for p in getattr(strip, 'geoms', [strip]):
            if p.geom_type != 'Polygon' or p.area < .01:
                continue
            for t in shapely.constrained_delaunay_triangles(p).geoms:
                q = np.array(t.exterior.coords[:3])
                y = [known.get(tuple(np.round(c, 3)), None) for c in q]
                y = [ground.y(*c) if h is None else h for c, h in zip(q, y)]
                v = np.zeros((3, 8))
                v[:, [0, 2]], v[:, 1] = q, y
                if np.cross(v[1, :3] - v[0, :3], v[2, :3] - v[0, :3])[1] < 0:
                    v = v[::-1]
                tris.append(v)
        emit('1ROAD_RUNOFF_' + label, tris, copy_material(model.materials[track['mat']]), lambda xz: track_uv(track, xz))
        log('run-off %s: a strip of %.0f m2 of road along it, %.1f m out from its back edge, back in to it %.1f m along it'
            % (label, strip.area, width, end))
        # what it lies over goes: the grass of the track's ground under it, cut round it
        for m in model.meshes:
            if m['name'] not in ground_names or m['name'].startswith('1ROAD'):
                continue
            ids, tri = now(data, m)
            hit = [i for i, t in enumerate(tri) if flat(t).intersects(strip)]
            if not hit:
                continue
            left = [q for i in hit for q in parts(tri[i], Polygon(tri[i][:, [0, 2]]).buffer(0).difference(strip))]
            kn5io.blank_faces(data, m, ids[hit])
            if left:
                name = '%s_RUNOFF_%s' % (m['name'], label)
                emit(name, left, copy_material(model.materials[m['mat']]))
                grass_meshes.append(name)
        # and the paddock's asphalt showing round it
        paddock_to_grass(label, area, strip)
        if wall:
            n0 = len(rows.blocks)
            Sw = np.arange(max(float(sa.min()), RUNOFF_IN), float(sa.max()) + .01, .62)
            at = path.place(Sw, np.full(len(Sw), t_wall), np.zeros(len(Sw)))[:, [0, 2]]
            rows.row('RUNOFF_' + label, None, 'along the run-off behind %s, %.1f m out from the road all along' % (label, t_wall),
                     at=list(at), height=lambda p: ground.y(*p))
            mesh_blocks += rows.blocks[n0:]
    for label, area in PADDOCK_GRASS:
        log('behind %s:' % label)
        paddock_to_grass(label, area, None)
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
    # the islands, grass all the way to the road now, still show the old asphalt in the photo
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union
    import shapely
    for label, name, a, b in ISLANDS:
        path = placed[label][0]
        hull = MultiPoint(np.r_[path.nodes, [a, b]]).convex_hull
        lo, hi = np.array(hull.bounds[:2]) - 2, np.array(hull.bounds[2:]) + 2
        tri = ground.tris[np.isin(ground.owner, list(driven_road))][:, :, [0, 2]]
        c = tri.mean(1)
        tri = tri[(c > lo).all(1) & (c < hi).all(1)]
        e0, e1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        tri = tri[np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) > 1e-9]
        shape = hull.difference(unary_union([Polygon(t) for t in tri]).buffer(.05))
        pieces = [np.array(t.exterior.coords[:3]) for g in getattr(shape, 'geoms', [shape]) if g.area > .05
                  for t in shapely.constrained_delaunay_triangles(g).geoms]
        if pieces:
            repaint.append(np.array(pieces))
            log('island at %s: %.0f m2 of the photo painted grass' % (label, shape.area))
    tex_name = model.materials[grass['mat']]['tex']['txDiffuse']
    old_tex = model.textures[tex_name]
    n = int.from_bytes(old_tex[4:8], 'little')
    new_tex = kn5io.texture_block(tex_name, repaint_photo(old_tex[8 + n + 4:], repaint, grass_uv))
    tex_blocks[copied[tex_name]] = new_tex
    mesh_blocks += road_marks(ground, {k for k, m in enumerate(model.meshes) if m['name'] in ground_names and m['name'].startswith('1ROAD')},
                              [m['name'] for m in model.meshes],
                              mat_blocks, tex_blocks, {l: placed[l][0] for l, _, _, _ in kerbs}, placed['HAIRPIN_OUT'][0],
                              straight_lines, driven_road, [p for p, _ in placed.values()],
                              # the old kerbs still there, as they are left
                              [m['v'][np.array(struct.unpack_from('<%dH' % m['f'].size, data, m['ip'])).reshape(-1, 3)][:, :, [0, 2]]
                               for m in model.meshes if model.material_name(m).startswith('Rumble')])
    kn5io.write(os.path.join(TRACK, 'velopark_kerbs.kn5'), tex_blocks, mat_blocks, mesh_blocks)
    # the islands round the kerbs at T2 and T3 left gaps in the grass between the kerb and the
    # road past them, where the paddock's asphalt below shows: grass over them, so each
    # island is grass all the way to the road
    for k, (_, name, a, b) in enumerate(ISLANDS):
        a, b = np.array(a), np.array(b)
        holes = island_holes(model, data, kn5io.Model(os.path.join(TRACK, 'velopark_kerbs.kn5')),
                             np.r_[np.minimum(a, b) - 6, np.maximum(a, b) + 6])
        for j, (hole, P) in enumerate(holes):
            area = hole.area
            # facing up
            n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            P[n[:, 1] < 0] = P[n[:, 1] < 0][:, ::-1]
            P = P.reshape(-1, 3)
            name = '1GRASS_ISLAND_%d_%d' % (k, j)
            b2, _ = kn5io.mesh_block(name, P, np.tile([0, 1., 0], (len(P), 1)), np.c_[P[:, [0, 2]], np.ones(len(P))] @ grass_uv, grass_mat)
            mesh_blocks.append(b2)
            grass_meshes.append(name)
            log('island %d: gap of %.1f m2 in the grass filled' % (k, area))
            # the paddock's asphalt under the gap lies as high as the grass in places and
            # would show through it
            from shapely.geometry import Polygon
            pad = paddock.mesh('1ROAD_PADDOCK')
            tri = pad['v'][pad['f']][:, :, [0, 2]]
            lo, hi = np.array(hole.bounds[:2]) - 1, np.array(hole.bounds[2:]) + 1
            near = np.flatnonzero(((tri.max(1) > lo) & (tri.min(1) < hi)).all(1))
            gone = [i for i in near if Polygon(tri[i]).area > 1e-9 and Polygon(tri[i]).intersects(hole.buffer(.1))]
            kn5io.blank_faces(paddock_data, pad, gone)
            log('  %d triangles of the paddock under it removed' % len(gone))
    open(paddock_path, 'wb').write(paddock_data)
    kn5io.write(os.path.join(TRACK, 'velopark_kerbs.kn5'), tex_blocks, mat_blocks, mesh_blocks)
    # the vp_1000's edge lines, followed all round it: across the mouths of the roads it does not
    # take, in a model of its own
    own_lines = {'vp_1000': edge_lines('vp_1000', tex_blocks, mat_blocks, mesh_blocks)}

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
    # then a quarter of that: at that the new poles were too bright
    scale *= POLE_DIM
    log('lights: then x%.2f, %.0f%% of the light from before at the median point' % (POLE_DIM, 100 * np.median(new[lit] * scale / old[lit])))
    write_config(cfg_path, cfg, new_pos0, new_dir0, col0, col1, scale, grass_meshes)

    # -- start panel
    for fn in ('velopark_gantry_1000.kn5', 'velopark_gantry_1500.kn5'):
        shutil.copy(os.path.join(bs.KN5_OUT, fn), os.path.join(TRACK, fn))
    log('start panel: lowered gantry copied in')
    # each where the ground photo has it, off its layout's start line, and a still copy of it
    # for the layouts that do not use it
    for fn, feet in GANTRY_FEET.items():
        place_gantry(os.path.join(TRACK, fn), np.array(feet, float), ground)
        still_gantry(os.path.join(TRACK, fn), os.path.join(TRACK, fn.replace('.kn5', '_still.kn5')))

    # -- the 50 m board the 100 m board on the 2nd straight lacks, for the layouts with that one
    with_100 = board_50(ground, road)

    # -- every layout loads the new models
    for ini in sorted(f for f in os.listdir(TRACK) if f.startswith('models_') and f.endswith('.ini')):
        p = os.path.join(TRACK, ini)
        s = open(p).read().rstrip() + '\n'
        n = len(re.findall(r'^\[MODEL_\d+\]', s, re.M))
        boards = set(re.findall(r'FILE=(\S+_boards\.kn5)', s))
        # the start panels it does not use, standing still
        still = tuple(fn.replace('.kn5', '_still.kn5') for fn in GANTRY_FEET if 'FILE=' + fn not in s)
        own = own_lines.get(ini[len('models_'):-len('.ini')])
        for fn in ('velopark_kerbs.kn5', 'velopark_poles.kn5') + (('velopark_board50.kn5',) if boards & with_100 else ()) + still + ((own,) if own else ()):
            s += '\n[MODEL_%d]\nFILE=%s\nPOSITION=0,0,0\nROTATION=0,0,0\n' % (n, fn)
            n += 1
        open(p, 'w', newline='\r\n' if '\r\n' in open(p, newline='').read() else '\n').write(s)
    versions()


# where the ground photo has the feet of each start panel: those at the end of the board on the
# left of the timing line, then those on the right. Both stand off their layout's start line,
# the 1000's well past it, over the middle of the pit exit. The 1500's stands on its shadow in
# the photo, 5 m before the line, and set to the right so its left feet clear the tyres there
GANTRY_FEET = {'velopark_gantry_1500.kn5': ((-16.9, 105.0), (2.15, 103.1)),
               'velopark_gantry_1000.kn5': ((-87.4, 105.1), (-67.4, 104.3))}


def gantry_feet(m):
    """The middles of the feet at each end of a start panel, from its leg walls, in the
    order of GANTRY_FEET."""
    walls = np.array([x['v'][:, [0, 2]].mean(0) for x in m.meshes if x['name'].startswith('1WALL_GANTRY')])
    board = m.mesh('GANTRY_BOARD')['v'][:, [0, 2]]
    axis = np.linalg.svd(board - board.mean(0), full_matrices=False)[2][0]
    u = (walls - board.mean(0)) @ axis
    return walls[u < 0].mean(0), walls[u > 0].mean(0)


def place_gantry(path, feet, ground):
    """Turn and move a start panel, all of it, so its feet stand where given, on the ground
    there as it stood on the ground before."""
    m = kn5io.Model(path)
    data = bytearray(m.bytes)
    a, b = gantry_feet(m)
    if np.linalg.norm(a - feet[0]) + np.linalg.norm(b - feet[1]) > np.linalg.norm(a - feet[1]) + np.linalg.norm(b - feet[0]):
        a, b = b, a
    th = math.atan2(*(feet[1] - feet[0])[::-1]) - math.atan2(*(b - a)[::-1])
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    c0, c1 = (a + b) / 2, feet.mean(0)
    board = m.mesh('GANTRY_BOARD')['v'][:, [0, 2]].mean(0)
    dy = ground.y(*((board - c0) @ r.T + c1)) - ground.y(*board)
    R = np.array([[r[0, 0], 0, r[0, 1]], [0, 1, 0], [r[1, 0], 0, r[1, 1]]])
    for x in m.meshes:
        d = np.frombuffer(m.bytes, '<f4', count=len(x['v']) * 11, offset=x['vp']).reshape(-1, 11).copy()
        xz = (d[:, [0, 2]] - c0) @ r.T + c1
        d[:, 0], d[:, 2], d[:, 1] = xz[:, 0], xz[:, 1], d[:, 1] + dy
        d[:, 3:6] = d[:, 3:6] @ R.T
        d[:, 8:11] = d[:, 8:11] @ R.T
        data[x['vp']:x['vp'] + d.nbytes] = d.astype('<f4').tobytes()
        # its bounding sphere's middle, after the material, the layer and the lod distances
        at = x['ip'] + x['ic'] * 2 + 16
        sx, sy, sz = struct.unpack_from('<3f', data, at)
        q = (np.array([sx, sz]) - c0) @ r.T + c1
        struct.pack_into('<3f', data, at, q[0], sy + dy, q[1])
    open(path, 'wb').write(data)
    log('start panel: %s moved %.1f m and turned %.1f degrees to where the photo has it, %.2f m up'
        % (os.path.basename(path), np.linalg.norm(c1 - c0), math.degrees(th), dy))


def still_gantry(src, dst):
    """A copy of a start panel that stays dark: its lights, flag lamps and board renamed so
    neither the game's start lights nor the board script or its glow find them."""
    b = open(src, 'rb').read()
    for old, new in ((b'KS_START_LIGHT', b'NO_START_LIGHT'), (b'GANTRY_FLAG', b'STATIC_FLAG'), (b'GantryBoard', b'GantryStill')):
        assert len(old) == len(new)
        b = b.replace(old, new)
    open(dst, 'wb').write(b)
    log('start panel: still copy %s' % os.path.basename(dst))


POLE_DIM = .25       # the night lights at a quarter of the brightness that matches the old poles
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
    cfg = night_buildings(cfg)
    open(path, 'w', newline='\r\n' if crlf else '\n').write(cfg)


def night_buildings(cfg):
    """The paddock buildings' night response, after the road's."""
    if 'DESCRIPTION = Buildings at night' in cfg:
        return cfg
    at = cfg.index('; ---- Start gantry board')
    return cfg[:at] + """; The paddock buildings reflect three times what the road does by day (ksDiffuse 0.6, a
; light wall texture) and stand right under the paddock wash lights, so at night they
; glowed brighter than anything else. They have no lights of their own. At night they
; take about the road's response, less for the light walls. Daytime values are untouched.
[MATERIAL_ADJUSTMENT_...]
ACTIVE = 1
DESCRIPTION = Buildings at night
MATERIALS = BuildingWall, BuildingRoof, BuildingSolar, BuildingDoor, BuildingDark
CONDITION = NIGHT_SMOOTH
KEY_0 = ksDiffuse
VALUE_0 = 0.25
VALUE_0_OFF = 0.6
KEY_1 = ksAmbient
VALUE_1 = 0.15
VALUE_1_OFF = 0.3

""" + cfg[at:]


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


if __name__ == '__main__':
    build()
    open(os.path.join(ROOT, 'build', 'build_051.log'), 'w').write('\n'.join(REPORT) + '\n')
