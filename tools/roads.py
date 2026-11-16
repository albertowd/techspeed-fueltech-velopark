"""Roads built from their descriptions (see ROADS.md): a road's middle and its width each side of it,
its mesh made from them as the track's own roads are made.

A road (a dict, as in tools/reference/roads.json):

    {"name": "1ROAD_26", "material": "Tarmac Dark", "map": "along" | "flat",
     "points": [[x, z, left, right], ...]}

its middle a smooth curve through its points (x, z), its edges left and right of it (metres, seen
going along it from its first point), the widths changing smoothly between its points. Or, in place
of its points, its middle as a curve of cubic Bezier pieces (as the road editor (top-race-editor) edits them):

    "curve": [{"p": [x, z], "in": [dx, dz], "out": [dx, dz], "left": w, "right": w}, ...]

each node a point the middle goes through, its handles (in, out: from the point, metres) the way it
comes in and goes on, its widths there, eased from one node's to the next's.

Its mesh: lines along it every COL metres out from its middle each side (its edges the last; a wider
road has more of them, never wider columns), each keeping a point every ROW metres of its own length
(fewer inside a curve, more outside: no cell folds, none loses the curve), all at its fine rows,
square to its middle; the strips between the lines zipped through those rows, every point shared. Its top ROAD over the ground (the drape lays it so); the ramp from its edges down to the ground
is the drape's (its 1GRASS_EDGE_RAMP), as for every road.
"""
import json, os

import numpy as np

import build_paths as bp

COL = .5              # metres: a column's width
ROW = .1              # metres between rows along a road (as its centre line's points: smooth)
TURN = 6.             # degrees: a line turning more than this keeps a point more
FINE = .1             # metres between a road's fine rows, square to its middle, which its lines keep points at
ROAD = .02            # metres a road's top is over the ground
TEX = 5.              # metres: the asphalt's texture, each way, mapped along a road
ROADS = os.path.join(bp.ROOT, 'tools', 'reference', 'roads.json')
HSTEP = 1.            # metres along a road between the rows of its heights
END_BLEND = 4.        # metres from a road's end over which its rows turn into its end row
RUNOFF_SMOOTH = 3.    # metres a run-off's outer edge's heights are smoothed over along it
SMOOTH_DRAWN = 2.5    # metres a drawn centre line is smoothed over
BLEND = 8.            # metres a branch's centre line eases into its road's at each end over
POINT_STEP = .1       # metres between a branch's points along its centre line (a smooth curve)
LINE_PAD, LINE_W, LINE_LIFT = .2, .12, .006
GAP_FILL, MIN_RUN, OVERLAP = 1., 1.5, .2   # metres: a line's gaps filled, its pieces kept, on past a fork     # metres: an edge line's middle in from the edge, its width, over the road


def spline(P, n=8):
    """A smooth curve through points P (k, d): Catmull-Rom (centripetal), n points a span. (m, d)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        t = np.linspace(0., 1., n * (len(P) - 1) + 1)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    Q = np.r_[[2 * P[0] - P[1]], P, [2 * P[-1] - P[-2]]]
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        d = lambda a, b: max(np.linalg.norm((b - a)[:2]) ** .5, 1e-6)
        t0, t1 = 0., d(p0, p1)
        t2, t3 = t1 + d(p1, p2), t1 + d(p1, p2) + d(p2, p3)
        for t in np.linspace(t1, t2, n, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-1])
    return np.array(out)


_CENTRES, _FINE = {}, {}


def bezier(curve, step=.02):
    """A curve of Bezier pieces (see above) as points along it (x, z, left, right), about step metres
    apart; and where along them each node is (indices)."""
    out, at, n = [], [0], 0
    for a, b in zip(curve[:-1], curve[1:]):
        P0, P3 = np.asarray(a['p'], float), np.asarray(b['p'], float)
        P1, P2 = P0 + np.asarray(a.get('out', (0., 0.)), float), P3 + np.asarray(b.get('in', (0., 0.)), float)
        L = np.linalg.norm(P1 - P0) + np.linalg.norm(P2 - P1) + np.linalg.norm(P3 - P2)
        t = np.linspace(0., 1., max(int(L / step), 2) + 1)[:, None]
        Q = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
        e = t[:, 0] * t[:, 0] * (3 - 2 * t[:, 0])               # (the widths eased from node to node)
        W = np.c_[a['left'] + (b['left'] - a['left']) * e, a['right'] + (b['right'] - a['right']) * e]
        seg = np.c_[Q, W]
        if out:
            seg = seg[1:]
        out.append(seg)
        n += len(seg)
        at.append(n - 1)
    C = np.concatenate(out)
    return C, np.array(at)


def centre(road):
    """A road's middle as points close along it (x, z, left, right), its length to each (s), and how far
    along each of its points (or its curve's nodes) is: from its points (spline) or its curve (bezier)."""
    key = json.dumps(road.get('curve') or [q[:4] for q in road['points']])
    if key not in _CENTRES:
        if road.get('curve'):
            C, at = bezier(road['curve'])
        else:
            C = spline(np.asarray([q[:4] for q in road['points']], float), 16)
            at = np.arange(len(road['points'])) * 16
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1))]
        if len(_CENTRES) > 200:
            _CENTRES.clear()
        _CENTRES[key] = (C, s, s[np.clip(at, 0, len(s) - 1)])
    return _CENTRES[key]


def fine(road, raw=False):
    """A road's fine rows: square to its middle every FINE along it: their middles (x, z), the way to
    their left (unit, x, z), their left and right widths (inside a curve tighter than the road is wide,
    no farther out than the curve's middle: inside_widths; raw: as described, which its heights are
    kept across), how far along."""
    key = json.dumps([road.get('curve') or [q[:4] for q in road['points']], road.get('start_left'), road.get('end_left')])
    if key in _FINE:
        return _FINE[key][1] if raw else _FINE[key][0]
    C, s, _ = centre(road)
    keep = np.r_[True, np.diff(s) > 1e-9]
    C, s = C[keep], s[keep]
    xz = C[:, :2]
    t = np.gradient(xz, s, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    at = np.r_[np.arange(0., s[-1], FINE), s[-1]]
    mid = np.c_[np.interp(at, s, xz[:, 0]), np.interp(at, s, xz[:, 1])]
    tt = np.c_[np.interp(at, s, t[:, 0]), np.interp(at, s, t[:, 1])]
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    left = ends_turned(road, np.c_[-tt[:, 1], tt[:, 0]], at)
    wl0, wr0 = np.interp(at, s, C[:, 2]), np.interp(at, s, C[:, 3])
    wl, wr = inside_widths(tt, at, wl0, wr0)
    if len(_FINE) > 200:
        _FINE.clear()
    _FINE[key] = ((mid, left, wl, wr, at), (mid, left, wl0, wr0, at))
    return _FINE[key][1] if raw else _FINE[key][0]


INSIDE = .97          # of a curve's radius: as far out as a road's side inside it goes (no fold)
BASE = .2             # metres each way a road's turning is measured over (a kink at a node no tight curve)


def inside_widths(tt, at, wl, wr):
    """A road's widths (wl, wr at its fine rows, its way along them tt) where it turns tighter than it is
    wide: the side inside the curve no farther out than INSIDE of the curve's radius (its edge closing
    to a fan round the curve's middle, never folding back over itself)."""
    from scipy.ndimage import maximum_filter1d
    if len(at) < 3:
        return wl, wr
    h = np.unwrap(np.arctan2(tt[:, 1], tt[:, 0]))
    # (turning, + to its left: measured over BASE metres each way at least, so a small kink at a node is
    # no tight curve; as the road editor's page measures it)
    a = np.clip(np.searchsorted(at, at - BASE, side='right') - 1, 0, len(at) - 1)
    b = np.clip(np.searchsorted(at, at + BASE, side='left'), 0, len(at) - 1)
    k = (h[b] - h[a]) / np.maximum(at[b] - at[a], 1e-9)
    k = np.where(np.abs(k) > 1e3, 0., k)
    kl = maximum_filter1d(np.maximum(k, 0.), 5)
    kr = maximum_filter1d(np.maximum(-k, 0.), 5)
    wl = np.where(kl > 1e-6, np.minimum(wl, INSIDE / np.maximum(kl, 1e-6)), wl)
    wr = np.where(kr > 1e-6, np.minimum(wr, INSIDE / np.maximum(kr, 1e-6)), wr)
    return wl, wr


def ends_turned(road, left, at):
    """Its rows' way across (left, unit, at how far along each) turned, within END_BLEND of each end, into
    its end rows' (road['start_left'], road['end_left']: the row a road ends on is the row the next
    begins on, square to neither maybe): the last row the same as the next road's first."""
    out = left.copy()
    for key, d in (('start_left', at - at[0]), ('end_left', at[-1] - at)):
        if key not in road:
            continue
        w = np.clip(1. - d / END_BLEND, 0., 1.)
        w = w * w * (3 - 2 * w)
        e = np.asarray(road[key], float)
        out = out * (1 - w)[:, None] + e[None, :] * w[:, None]
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


def lines(wl, wr):
    """A road's lines along it, left edge to right edge, as offsets at each fine row (+ to the left):
    its left edge, its columns every COL out from the middle (each on the edge where the road is not
    so wide there), its middle, the same to its right."""
    k_l = int(np.ceil(wl.max() / COL - 1e-6))
    k_r = int(np.ceil(wr.max() / COL - 1e-6))
    out = [wl]
    out += [np.minimum(k * COL, wl) for k in range(k_l - 1, 0, -1)]
    out += [np.zeros_like(wl)]
    out += [-np.minimum(k * COL, wr) for k in range(1, k_r)]
    out += [-wr]
    return out


def pick(Q, ang):
    """Of a line's points at the fine rows (Q, (n, 2)), those it keeps: one every ROW along it (its own
    length: fewer inside a curve, more outside), closer where it turns more than TURN, its ends."""
    keep = [0]
    run, a0 = 0., ang[0]
    for r in range(1, len(Q)):
        run += np.linalg.norm(Q[r] - Q[r - 1])
        if run >= ROW - 1e-6 or abs(ang[r] - a0) > TURN or r == len(Q) - 1:
            keep.append(r)
            run, a0 = 0., ang[r]
    if len(keep) > 2 and np.linalg.norm(Q[keep[-1]] - Q[keep[-2]]) < ROW * .3:
        keep.pop(-2)                                            # (no sliver at its end)
    return keep


def mesh(road, height=None):
    """A road's mesh: points (x, y, z), their mapping (u, v), faces (facing up). Its top as high as its
    heights say (road['heights']), or height(x, z) + ROAD when given. Its lines along it each keep their own
    points (see pick), the strips between them zipped through the fine rows: no point but is shared."""
    mid, left, wl, wr, at = fine(road)
    tt = np.c_[left[:, 1], -left[:, 0]]
    ang = np.degrees(np.unwrap(np.arctan2(tt[:, 1], tt[:, 0])))
    P, U, kept, SO = [], [], [], []
    for off in lines(wl, wr):
        Q = mid + left * off[:, None]
        keep = pick(Q, ang)
        idx = []
        for r in keep:
            idx.append(len(P))
            P.append((Q[r, 0], 0., Q[r, 1]))
            SO.append((at[r], off[r]))
            if road.get('map', 'along') == 'along':
                U.append(((off[r] - wl[r]) * road.get('u_per_m', 1. / TEX), at[r] * road.get('v_per_m', 1. / TEX)))
            else:
                U.append((Q[r, 0] / TEX, Q[r, 1] / TEX))
        kept.append((np.array(keep), idx))
    F = []
    for (ra, ia), (rb, ib) in zip(kept[:-1], kept[1:]):
        # (two lines zipped along: on along the one whose next point is at the nearer fine row, a
        # quad split the same way where both have one at the same)
        i = j = 0
        while i < len(ia) - 1 or j < len(ib) - 1:
            na = ra[i + 1] if i < len(ia) - 1 else np.inf
            nb = rb[j + 1] if j < len(ib) - 1 else np.inf
            if na < nb:
                F.append((ia[i], ib[j], ia[i + 1]))
                i += 1
            elif nb < na:
                F.append((ia[i], ib[j], ib[j + 1]))
                j += 1
            else:
                F += [(ia[i], ib[j], ib[j + 1]), (ia[i], ib[j + 1], ia[i + 1])]
                i += 1
                j += 1
    P, U, F = np.array(P), np.array(U), np.array(F, int)
    # (the strips where a column runs on the edge have no width: out, and the points left over with them)
    a = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 1]
    F = F[np.abs(a) > 1e-6]
    used = np.unique(F)
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    SO = np.array(SO)[used]
    P, U, F = P[used], U[used], remap[F]
    if road.get('heights') and height is None:
        P[:, 1] = height_at(road, SO[:, 0], SO[:, 1])
    elif height is not None:
        P[:, 1] = height(P[:, 0], P[:, 2]) + ROAD
    up = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)
    return P, U, F


def control_s(road):
    """How far along a road its points (or its curve's nodes) are (its middle's length to each)."""
    return centre(road)[2]


def across(wl, wr):
    """Where a road's heights are kept across it at a point, left edge to right (+ to the left): every
    COL in from its left edge, its right edge the last."""
    return np.r_[wl - np.arange(0., wl + wr - 1e-6, COL), -wr]


def height_at(road, at, off):
    """A road's top's height at points how far along it (at) and across it (off, + to the left), from
    its heights (road['heights']: every 'step' metres along it, its heights across it there, see across)."""
    h = road['heights']
    step, rows = h['step'], h['rows']
    mid, left, wl, wr, s = fine(road, raw=True)
    at, off = np.atleast_1d(np.asarray(at, float)), np.atleast_1d(np.asarray(off, float))
    st = np.r_[np.arange(len(rows) - 1) * step, s[-1]]          # (every step, its last at its very end)
    i = np.clip(np.searchsorted(st, at, side='right') - 1, 0, len(rows) - 2)
    t = np.clip((at - st[i]) / np.maximum(st[i + 1] - st[i], 1e-9), 0., 1.)
    W = [(np.interp(a, s, wl), np.interp(a, s, wr)) for a in st]

    # (a row at a time: every point between rows k and k + 1 read off both at once)
    out = np.zeros(len(at))
    for k in np.unique(i):
        sel = i == k
        o = off[sel]
        ys = []
        for kk in (k, k + 1):
            a, row = across(*W[kk]), np.asarray(rows[kk], float)
            if len(row) != len(a):
                a = np.linspace(a[0], a[-1], len(row))         # (heights kept for another width: spread across it)
            ys.append(np.interp(o, a[::-1], row[::-1]))
        out[sel] = ys[0] * (1 - t[sel]) + ys[1] * t[sel]
    return out


def record_heights(road, height):
    """A road's heights from height(x, z) (its top's: the ground's as the drape lays roads, plus
    ROAD): every HSTEP metres along it, across it (see across), to a millimetre."""
    mid, left, wl, wr, s = fine(road, raw=True)
    st = np.r_[np.arange(0., s[-1], HSTEP), s[-1]]
    m, l = fine_at(road, st)
    rows = []
    for a, mm, ll in zip(st, m, l):
        o = across(np.interp(a, s, wl), np.interp(a, s, wr))
        P = mm + np.outer(o, ll)
        y = np.asarray(height(P[:, 0], P[:, 1]), float)
        rows.append([round(float(v), 3) for v in y])
    return {'step': HSTEP, 'rows': rows}


def fine_at(road, at):
    """A road's middle (x, z) and the way to its left (unit) at how far along it (at)."""
    C, s, _ = centre(road)
    keep = np.r_[True, np.diff(s) > 1e-9]
    xz, s = C[keep, :2], s[keep]
    t = np.gradient(xz, s, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    mid = np.c_[np.interp(at, s, xz[:, 0]), np.interp(at, s, xz[:, 1])]
    tt = np.c_[np.interp(at, s, t[:, 0]), np.interp(at, s, t[:, 1])]
    tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-9)
    return mid, np.c_[-tt[:, 1], tt[:, 0]]


def rows_at(road, S):
    """A road's middle (x, z) and the way to its left (unit) at how far along it (S), as its mesh's rows lie
    (fine: its end rows turned to start_left / end_left), so what is laid along it meets what is laid along
    a road joined to it, point for point."""
    mid, left, wl, wr, at = fine(road)
    S = np.atleast_1d(np.asarray(S, float))
    m = np.c_[np.interp(S, at, mid[:, 0]), np.interp(S, at, mid[:, 1])]
    l = np.c_[np.interp(S, at, left[:, 0]), np.interp(S, at, left[:, 1])]
    return m, l / np.maximum(np.linalg.norm(l, axis=1, keepdims=True), 1e-9)


def line_strip(road, line, height=None):
    """A line of paint along a road's edge (line: its side, 'left' or 'right', from, to (how far along),
    pad (its middle that far in from the edge), w (its width)): points (x, y, z) a pair across it at
    every fine row it spans, faces (facing up). LINE_LIFT over the road."""
    mid, left, wl, wr, at = fine(road)
    a0, a1 = line.get('from') or 0., min(at[-1], line['to'] if line.get('to') is not None else at[-1])
    keep = (at > a0) & (at < a1)
    rows = np.r_[a0, at[keep], a1]
    m, l = rows_at(road, rows)
    w_side = np.interp(rows, at, wl if line['side'] == 'left' else wr)
    sgn = 1. if line['side'] == 'left' else -1.
    o_mid = sgn * (w_side - line.get('pad', LINE_PAD))
    P, F = [], []
    for k in range(len(rows)):
        for o in (o_mid[k] + line.get('w', LINE_W) / 2, o_mid[k] - line.get('w', LINE_W) / 2):
            q = m[k] + l[k] * o
            P.append((q[0], 0., q[1]))
        if k:
            a, b = 2 * (k - 1), 2 * k
            F += [(a, b, b + 1), (a, b + 1, a + 1)]
    P, F = np.array(P), np.array(F, int)
    offs = np.repeat(o_mid, 2) + np.tile([line.get('w', LINE_W) / 2, -line.get('w', LINE_W) / 2], len(rows))
    if height is not None:
        P[:, 1] = height(P[:, 0], P[:, 2]) + LINE_LIFT
    elif road.get('heights'):
        P[:, 1] = height_at(road, np.repeat(rows, 2), offs) + LINE_LIFT
    up = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)
    return P, F



PAINTS = {'white': ('LinePaint', 'LinePaint.dds'), 'yellow': ('PitLinePaint', 'PitLinePaint.dds')}


def paint_mesh(L, colour=None):
    """An empty mesh of the lines' model L in the paint of that colour (white unless 'yellow'), to lay
    a line in: a copy of a painted LINE_/GRID_ mesh, its material that paint's. The yellow, if L has none
    yet, is added: the white's material on source/textures/PitLinePaint.dds."""
    mat, tex = PAINTS['yellow' if colour == 'yellow' else 'white']
    names = [m['name'] for m in L['materials']]
    if mat not in names:
        import copy
        m = copy.deepcopy(L['materials'][names.index(PAINTS['white'][0])])
        m['name'] = mat
        for t in m['textures']:
            t['name'] = tex
        L['materials'].append(m)
        if not any(t['name'] == tex for t in L['textures']):
            L['textures'].append(dict(name=tex, active=1, data=open(os.path.join(bp.ROOT, 'source', 'textures', tex), 'rb').read()))
        names.append(mat)
    tpl = next(x for x, _ in bp.meshes(L) if x['name'].startswith(('LINE_', 'GRID_')) and len(x['faces']))
    return dict(tpl, name='', children=[], material=names.index(mat), verts=tpl['verts'][:0], faces=tpl['faces'][:0])


def along_width(desc, s):
    """A run-off's width at how far along its road (s): desc['profile'] ([[s, width], ...], eased from
    point to point, as at the nearest beyond the first and last), or desc['width'] all along; from
    nothing at its ends over desc['ease']."""
    al = desc['along']
    s = np.asarray(s, float)
    prof = sorted(desc.get('profile') or [])
    if prof:
        xs, ws = np.array([p[0] for p in prof], float), np.array([p[1] for p in prof], float)
        i = np.clip(np.searchsorted(xs, s) - 1, 0, max(len(xs) - 2, 0))
        if len(xs) == 1:
            w = np.full(s.shape, ws[0])
        else:
            t = np.clip((s - xs[i]) / np.maximum(xs[i + 1] - xs[i], 1e-9), 0., 1.)
            w = ws[i] + (ws[i + 1] - ws[i]) * t * t * (3 - 2 * t)
    else:
        w = np.full(s.shape, float(desc.get('width', 2.)))
    e = np.clip(np.minimum(s - al['from'], al['to'] - s) / max(desc.get('ease', 8.), 1e-6), 0., 1.)
    return w * e * e * (3 - 2 * e)


def along_road(desc, roads):
    """A run-off (desc['along']: the road it lies beside, its side, from, to (how far along that road);
    its width: along_width) as a road of its own for mesh(): its middle that road's edge (a point every
    metre, so it keeps to it), all its width outward, from nothing at its ends (it opens out along the
    road's curve)."""
    al = desc['along']
    parent = roads[al['road']]
    mid, left, wl, wr, at = fine(parent)
    sgn = 1. if al['side'] == 'left' else -1.
    s_ = np.r_[np.arange(al['from'], al['to'], 1.), al['to']]
    m, l = fine_at(parent, s_)
    w_side = np.interp(s_, at, wl if al['side'] == 'left' else wr)
    edge = m + l * (sgn * w_side)[:, None]
    w = along_width(desc, s_)
    pts = [[float(x), float(z), float(a_), float(b_)] for (x, z), a_, b_ in
           zip(edge, w if sgn > 0 else np.zeros_like(w), np.zeros_like(w) if sgn > 0 else w)]
    out = {'name': desc['name'], 'material': desc.get('material', parent.get('material')), 'map': 'flat', 'points': pts}
    if desc.get('heights'):
        out['heights'] = desc['heights']
    return out


def along_heights(desc, roads, ground):
    """A run-off's heights: its inner edge as high as the road it lies beside there, its outer edge as
    the ground (ground(x, z), plus ROAD) smoothed along it over RUNOFF_SMOOTH, straight across between."""
    from scipy.ndimage import gaussian_filter1d
    al = desc['along']
    parent = roads[al['road']]
    road = along_road(dict(desc, heights=None), roads)
    mid, left, wl, wr, s = fine(road, raw=True)
    st = np.r_[np.arange(0., s[-1], HSTEP), s[-1]]
    m, l = fine_at(road, st)
    pm, pl, pwl, pwr, pat = fine(parent)
    sgn = 1. if al['side'] == 'left' else -1.
    w_in = np.interp(st, s, wl if sgn > 0 else wr)
    # (where along the parent each row is: its middle on the parent's edge)
    from scipy.spatial import cKDTree
    edge = pm + pl * (sgn * (pwl if sgn > 0 else pwr))[:, None]
    j = cKDTree(edge).query(m)[1]
    inner = height_at(parent, pat[j], sgn * (pwl if sgn > 0 else pwr)[j])
    outer_pts = m + l * (sgn * w_in)[:, None]
    outer = gaussian_filter1d(np.asarray(ground(outer_pts[:, 0], outer_pts[:, 1]), float) + ROAD, RUNOFF_SMOOTH / HSTEP, mode='nearest')
    rows = []
    for a_, wi, yi, yo in zip(st, w_in, inner, outer):
        o = across(wi if sgn > 0 else 0., 0. if sgn > 0 else wi)
        f = np.abs(o) / max(wi, 1e-9)                          # (0 at the road's edge, 1 at its outer edge)
        rows.append([round(float(yi + (yo - yi) * ff), 3) for ff in f])
    return {'step': HSTEP, 'rows': rows}


def smooth_line(P, step=1., sigma=SMOOTH_DRAWN):
    """A drawn line (n, 2) as a smooth one: resampled every step, smoothed over sigma metres (its ends
    kept where they are)."""
    from scipy.ndimage import gaussian_filter1d
    P = np.asarray(P, float)
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.arange(0., d[-1], step)
    Q = np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])]
    S = np.c_[gaussian_filter1d(Q[:, 0], sigma / step, mode='nearest'), gaussian_filter1d(Q[:, 1], sigma / step, mode='nearest')]
    return S


def into_centre(C, parent, at_start, blend):
    """A centre line C (n, 2, from its start to its end) with one end (its start, or its end) on a road's
    centre line (parent: a road), tangent to it: its first (or last) blend metres replaced by a smooth
    curve from the nearest point of the parent's centre line, going the parent's way, to C there."""
    from scipy.spatial import cKDTree
    mid, left, wl, wr, at = fine(parent)
    tan = np.c_[left[:, 1], -left[:, 0]]
    C = C if at_start else C[::-1]
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    k = int(np.searchsorted(d, blend))
    k = min(max(k, 2), len(C) - 2)
    i = cKDTree(mid).query(C[0])[1]
    a, ta = mid[i], tan[i]
    tq = C[k + 1] - C[k - 1]
    tq /= max(np.linalg.norm(tq), 1e-9)
    if ta @ (C[k] - a) < 0:
        ta = -ta                                                # (the parent's way, as the branch goes)
    L = np.linalg.norm(C[k] - a)
    u = np.linspace(0., 1., max(int(L / .5), 4))[:, None]
    H = ((2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * L * ta
         + (-2 * u ** 3 + 3 * u ** 2) * C[k] + (u ** 3 - u ** 2) * L * tq)
    out = np.r_[H[:-1], C[k:]]
    return out if at_start else out[::-1]


def join_widths(parent, p, way):
    """A road's widths (left, right) at its centre line's point nearest p, seen going way (a branch
    leaving or joining it there: its left and right as the branch's, the other way round if it goes
    against the road)."""
    from scipy.spatial import cKDTree
    mid, left, wl, wr, at = fine(parent)
    i = cKDTree(mid).query(p)[1]
    tan = np.c_[left[:, 1], -left[:, 0]][i]
    return (wl[i], wr[i]) if tan @ way >= 0 else (wr[i], wl[i])


def branch_road(desc, roads):
    """A branch (desc['centre']: its drawn middle; desc['joins']: the roads it leaves and joins, at its
    start and its end, or None; desc['left'], desc['right']: its widths, or none to take its roads'
    where it leaves and joins them, eased from the one to the other along it; desc['blend']) as a road
    for mesh(): its middle smoothed, its ends into the roads' centre lines, tangent to them, as wide as
    they are there (so it forks off them and merges into them seamlessly); a point every POINT_STEP."""
    C = smooth_line(desc['centre'])
    j0, j1 = desc.get('joins', [None, None])
    blend = desc.get('blend', BLEND)
    if j0:
        C = into_centre(C, roads[j0], True, blend)
    if j1:
        C = into_centre(C, roads[j1], False, blend)
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    t = np.r_[np.arange(0., d[-1], POINT_STEP), d[-1]]
    P = np.c_[np.interp(t, d, C[:, 0]), np.interp(t, d, C[:, 1])]
    if desc.get('left') is not None:
        WL, WR = np.full(len(P), float(desc['left'])), np.full(len(P), float(desc['right']))
    else:
        a = join_widths(roads[j0], P[0], P[1] - P[0]) if j0 else None
        b = join_widths(roads[j1], P[-1], P[-1] - P[-2]) if j1 else None
        a, b = a or b, b or a
        u = t / max(t[-1], 1e-9)
        u = u * u * (3 - 2 * u)
        WL, WR = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
    out = {'name': desc['name'], 'material': desc.get('material', 'Tarmac Dark'), 'map': desc.get('map', 'flat'),
           'points': [[round(float(x), 3), round(float(z), 3), round(float(l), 3), round(float(r), 3)]
                      for (x, z), l, r in zip(P, WL, WR)]}
    for k_ in ('heights', 'lines', 'tex', 'joins'):
        if desc.get(k_):
            out[k_] = desc[k_]
    return out


def line_area(road, grow=0.):
    """The part of a road between its edge lines (each pad in from its edge), grown by grow metres
    (shrunk where less than 0): a shapely shape (x, z)."""
    from shapely.geometry import Polygon
    mid, left, wl, wr, at = fine(road)
    A = mid + left * (wl - LINE_PAD)[:, None]
    B = mid - left * (wr - LINE_PAD)[:, None]
    return Polygon(np.r_[A, B[::-1]]).buffer(0).buffer(grow)


def line_runs(road, line, keep):
    """A line (see line_strip) as the runs of it whose middles keep(x, z) says to keep (a bool each):
    lines as line_strip takes them, each from/to a fine row, none shorter than FINE * 3."""
    mid, left, wl, wr, at = fine(road)
    a0, a1 = line.get('from') or 0., line['to'] if line.get('to') is not None else at[-1]
    sel = (at >= a0 - 1e-6) & (at <= a1 + 1e-6)
    a = at[sel]
    if len(a) < 2:
        return []
    sgn = 1. if line['side'] == 'left' else -1.
    o = sgn * ((wl if sgn > 0 else wr)[sel] - line.get('pad', LINE_PAD))
    Q = mid[sel] + left[sel] * o[:, None]
    k = np.asarray(keep(Q[:, 0], Q[:, 1]), bool)
    if not k.all() and k.any():
        # (where two lines part at a shallow angle the test flickers: gaps shorter than GAP_FILL filled,
        # pieces shorter than MIN_RUN out, and each piece that gives way to another line on by OVERLAP)
        def runs(b):
            r_, i_ = [], 0
            while i_ < len(b):
                j_ = i_
                while j_ + 1 < len(b) and b[j_ + 1] == b[i_]:
                    j_ += 1
                r_.append((b[i_], i_, j_))
                i_ = j_ + 1
            return r_
        for v_, i_, j_ in runs(k):
            if not v_ and i_ > 0 and j_ < len(k) - 1 and a[j_] - a[i_] < GAP_FILL:
                k[i_:j_ + 1] = True
        for v_, i_, j_ in runs(k):
            if v_ and a[j_] - a[i_] < MIN_RUN and not (i_ == 0 and j_ == len(k) - 1):
                k[i_:j_ + 1] = False
        grow = int(round(OVERLAP / FINE))
        k0 = k.copy()
        for v_, i_, j_ in runs(k0):
            if v_:
                k[max(0, i_ - grow):i_] = True
                k[j_ + 1:j_ + 1 + grow] = True
    out, i = [], 0
    while i < len(a):
        if not k[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(a) and k[j + 1]:
            j += 1
        if a[j] - a[i] >= FINE * 3 - 1e-6:
            out.append(dict(line, **{'from': float(a[i]), 'to': float(a[j])}))
        i = j + 1
    return out


def priority(roads):
    """The roads' names in the order they are built in, each over those after it: as roads.json lists them
    (the road editor's order), a road the roads it joins or lies along always after them."""
    out, seen = [], set()

    def put(name, path=()):
        if name in seen or name not in roads or name in path:
            return
        r = roads[name]
        for x in list(r.get('joins') or []) + ([r['along']['road']] if 'along' in r else []):
            if x:
                put(x, path + (name,))
        seen.add(name)
        out.append(name)
    for name in roads:
        put(name)
    return out


def joined_to(road):
    """The roads joined end to end to a road's ends (its curve's nodes' joints)."""
    return {nd['joint']['road'] for nd in (road.get('curve') or []) if nd.get('joint')}


def box(road, roads):
    """A road's extent (x0, z0, x1, z1), with its widths."""
    r = along_road(road, roads) if 'along' in road else road
    C = centre(r)[0]
    w = float(np.max(C[:, 2:4])) + .5
    return C[:, 0].min() - w, C[:, 1].min() - w, C[:, 0].max() + w, C[:, 1].max() + w


def above(road, roads):
    """The roads built before a road (priority) whose extent meets its own, but those joined end to end to
    it: those it is cut by where it lies under them."""
    order = priority(roads)
    if road['name'] not in order:
        return []
    mine, ends = box(road, roads), joined_to(road)
    out = []
    for x in order[:order.index(road['name'])]:
        if x in ends or road['name'] in joined_to(roads[x]):
            continue
        b = box(roads[x], roads)
        if b[0] < mine[2] and b[2] > mine[0] and b[1] < mine[3] and b[3] > mine[1]:
            out.append(x)
    return out


def outline(road, roads=None):
    """A road's outline as built (x, z): its edges along its fine rows."""
    from shapely.geometry import Polygon
    r = along_road(road, roads) if 'along' in road else road
    mid, left, wl, wr, at = fine(r)
    return Polygon(np.r_[mid + left * wl[:, None], (mid - left * wr[:, None])[::-1]]).buffer(0)


HEIGHTS = os.path.join(bp.ROOT, 'build', 'cache', 'road_heights')   # (heights recorded for roads without their own)


def shape_key(road, roads):
    """A fingerprint of what a road's heights hang on: its shape, the shapes of the roads it joins or
    lies along, and the terrain (tools/terrain.py: an edit to it records them again)."""
    import hashlib
    import terrain
    keep = ('curve', 'points', 'start_left', 'end_left', 'along', 'joins', 'width', 'ease', 'profile')
    mine = {k: road[k] for k in keep if k in road}
    up = [x for x in (road.get('joins') or ()) if x] + ([road['along']['road']] if 'along' in road else []) + above(road, roads)
    mine['up'] = [shape_key(roads[x], roads) for x in sorted(set(up)) if x in roads]
    mine['ground'] = terrain.key()
    return hashlib.sha1(json.dumps(mine, sort_keys=True).encode()).hexdigest()[:16]


DASH, GAP = .5, .5    # metres: a dotted line's dashes and the gaps between them (line['dash'], line['gap'])


def dashes(line, runs):
    """A line's runs (see line_runs) as the dashes laid of them, if it is dotted (line['style']): a dash every
    dash + gap metres counted from where the line starts (so a line forked in runs keeps its rhythm), cut
    where a run ends; as they are if it is solid."""
    if line.get('style') != 'dotted':
        return runs
    dash, gap = max(line.get('dash') or DASH, .05), max(line.get('gap') or GAP, .05)
    a0, out = line.get('from') or 0., []
    for run in runs:
        k = np.floor((run['from'] - a0) / (dash + gap))
        a = a0 + k * (dash + gap)
        while a < run['to'] - 1e-6:
            lo, hi = max(a, run['from']), min(a + dash, run['to'])
            if hi - lo > .02:
                out.append(dict(run, **{'from': float(lo), 'to': float(hi)}))
            a += dash + gap
    return out


def load():
    """The roads described (tools/reference/roads.json), by name; a road with no heights of its own
    given those recorded for its shape (tools/build_roads.py records them), if any."""
    if not os.path.exists(ROADS):
        return {}
    roads = {r['name']: r for r in json.load(open(ROADS))['roads']}
    for r in roads.values():
        if not r.get('heights'):
            f = os.path.join(HEIGHTS, shape_key(r, roads) + '.json')
            if os.path.exists(f):
                r['heights'] = json.load(open(f))
    return roads
