"""The rumbles (kerbs) described with the roads in tools/reference/roads.json (road['kerbs'], placed in
the road editor (top-race-editor)) built along the roads' edges as the drape has laid them, in place of the old kerbs
along those roads.

    python tools/build_kerbs.py

Each rumble: its side of its road, from / to (metres along it), its kind, its width, how far inside the
road's edge it starts (inset), its ends square or tapering in (start / end: 'triangle', over taper
metres), and where it goes on along another road whose edge carries on from its road's (before / after:
{road, side, from, to, outer}). Built as the track's own kerbs are, by kind (PROFILES): its top across it
as high as the road's edge on the inside, as the ground on the outside, the kind's rise over that; its
texture along it a stripe pair every so many metres, across it the whole of the texture.
Behind it a back ramp (BACK_RAMP) down to its road's height, where its profile does not end that low;
its tapering ends rise from its road's height over the whole taper.

Every old kerb along a road that has rumbles described (its top, 1ROAD_KERB_<name>, within OLD_NEAR of
that road's edge, and the parts made with it: its side, its beds and their skirts) out; under each new
rumble the ground, the roads' edge ramps and any road (inset into it) cut away, a rumble over all of
them (the seal closes their seams).
"""
import os
import re

import numpy as np

import build_paths as bp
import gltf_source
import roads as rd

KERBS = os.path.join(bp.MODELS, 'velopark_kerbs.glb')
# a kind: its top's rise over its base across it ((across 0-1, metres up), ...), its texture across it
# (v at its inner side, at its outer: the painted part only, not the strip the old kerbs laid on the road),
# along it (u a metre), its material
PROFILES = {
    'ramp': (((0., 0.), (.15, .008), (.3, .026), (.45, .045), (.55, .052), (1., .05)), (0., .61), .15, 'KerbRamp'),
    'floor': (((0., .004), (.5, .004), (1., .004)), (0., 1.), .21, 'KerbFloor'),
    'concrete': (((0., 0.), (.08, .03), (.15, .05), (.9, .05), (1., .03)), (0., .97), .2, 'KerbConcrete'),
    'round': (((0., 0.), (.2, .025), (.4, .045), (.6, .04), (.8, .015), (1., -.01)), (0., .61), .2, 'KerbRound'),
}
BACK = .5            # metres of concrete behind a ramp + concrete rumble's ramp, unless it says
# a ramp + concrete rumble: one piece in KerbConcrete, as the old concrete kerbs are made: its ramp rising
# across its width (0-1: (across, metres up, v)), a lip at its top, then the concrete sloping back down
# to the ground across its back (0-1); the texture's painted part on the ramp, its concrete behind
RAMP_CONCRETE = (((0., 0., 0.), (.047, .0095, .014), (.185, .0142, .055), (.944, .0592, .283), (1., .0491, .3)),
                 ((.5, .0207, .63), (1., -.008, .967)), .2, 'KerbConcrete')
BACK_RAMP = .15      # metres behind a rumble's outer edge it ramps down to its road's height (a kart off it and back on
                     # rolls down, not off a step), unless it says (back_ramp); none where its profile ends that low already
ROW = .25            # metres between a rumble's rows along it
JOIN = 1.            # metres: a rumble's piece on one road and the next this near are one strip
ROAD_CLEAR = .02     # metres inside a rumble the roads under it are cut (its own road's edge left whole)
EDGE, RAMP = .02, 1.5   # metres from a rumble: the ground as high as it within EDGE, eased back to its own by RAMP
OTHER_EASE = .75     # metres from another road a rumble's tapering end comes down to the road's height
RAISED = .005        # metres: a rumble's part rising more than this over its road hides the paint under it
OLD_NEAR = 2.        # metres from a road's edge an old kerb's top lies, for it to go
NAME = '1ROAD_KERB_E_%s_%d%s'   # a new rumble's mesh: its road's number, which of its rumbles, its part


def kind_of(k):
    return 'round' if k.get('kind') == 'yellow' else k.get('kind') if k.get('kind') in PROFILES or k.get('kind') == 'ramp_concrete' else 'ramp'


def pieces(road, k, roads):
    """A rumble's pieces along roads (see the module's note): (road, side, s0, s1, taper at s0, at s1)."""
    out = []
    tri0, tri1 = k.get('start') == 'triangle', k.get('end') == 'triangle'

    def cont(c, start):
        r2 = roads.get(c['road'])
        if r2 is None:
            return None
        L = rd.fine(r2)[4][-1]
        a, b = max(0., c.get('from') or 0.), min(L, c['to'] if c.get('to') is not None else L)
        if b <= a:
            return None
        at0 = c.get('outer') != 'to'
        tri = tri0 if start else tri1
        return (r2, c['side'], a, b, at0 and tri, (not at0) and tri)
    if k.get('before'):
        p = cont(k['before'], True)
        if p:
            out.append(p)
    L = rd.fine(road)[4][-1]
    s0, s1 = max(0., k.get('from') or 0.), min(L, k['to'] if k.get('to') is not None else L)
    if s1 > s0:
        out.append((road, k['side'], s0, s1, not k.get('before') and tri0, not k.get('after') and tri1))
    if k.get('after'):
        p = cont(k['after'], False)
        if p:
            out.append(p)
    return out


def strip(road, side, s0, s1, t0, t1, k, a, b, top, ground):
    """A rumble's piece of one road, across it from a to b (0-1 of its whole width) of the given profile:
    points (x, y, z), across (0-1), along (metres from its start), faces."""
    mid, left, wl, wr, at = rd.fine(road)
    sgn = 1. if side == 'left' else -1.
    n = max(2, int(np.ceil((s1 - s0) / ROW)) + 1)
    S = np.linspace(s0, s1, n)
    m, l = rd.rows_at(road, S)
    edge = np.interp(S, at, wl if sgn > 0 else wr) - k.get('inset', 0.)
    tp = max(k.get('taper') or 2., 1e-6)
    f = np.ones_like(S)
    if t0:
        f = np.minimum(f, np.clip((S - s0) / tp, 0., 1.))
    if t1:
        f = np.minimum(f, np.clip((s1 - S) / tp, 0., 1.))
    return m, l, sgn, edge, f, S


def inside_reach(road, side, S, edge):
    """How far out past its road's edge (edge: its inner side's offset there) a rumble on that side may go at
    how far along (S): on the inside of a curve, no farther than INSIDE of the curve's radius from its middle
    (so it never folds over itself); elsewhere as far as it likes."""
    mid, left, wl, wr, at = rd.fine(road)
    tt = np.c_[left[:, 1], -left[:, 0]]
    h = np.unwrap(np.arctan2(tt[:, 1], tt[:, 0]))
    a = np.clip(np.searchsorted(at, at - rd.BASE, side='right') - 1, 0, len(at) - 1)
    b = np.clip(np.searchsorted(at, at + rd.BASE, side='left'), 0, len(at) - 1)
    k = (h[b] - h[a]) / np.maximum(at[b] - at[a], 1e-9)                 # (turning, + to its left)
    k = np.interp(S, at, k) * (1. if side == 'left' else -1.)          # (+ where this side is the inside)
    R = np.where(k > 1e-6, 1. / np.maximum(k, 1e-6), np.inf)
    return np.maximum(rd.INSIDE * R - edge, 0.)


def other_roads(k, road, roads):
    """The roads a rumble does not run along, as one shape (x, z): every road but its own and those its pieces go on
    along, and the run-offs beside them. Near one, a rumble's tapering end comes down to the road's height."""
    from shapely.ops import unary_union
    own = {r_['name'] for r_, *_ in pieces(road, k, roads)} | {road['name']}
    rest = [r for n, r in roads.items() if n not in own and not ('along' in r and r['along'].get('road') in own)]
    key = (id(roads), tuple(sorted(own)))
    if key not in _OTHERS:
        _OTHERS[key] = unary_union([rd.outline(r, roads) for r in rest]) if rest else None
    return _OTHERS[key]


_OTHERS = {}


def geometry(road, k, roads, top):
    """A rumble's surface, as triangles of its own: [(points (k, 3), texture places (k, 2), faces (m, 3),
    material, name's part)]: one (a ramp with concrete behind it too: one piece, as the old concrete kerbs). top(points (k, 2)): the road's
    top's height there (its edge's: the rumble level with it). What the build and the road editor's
    preview both make it of."""
    kind = kind_of(k)
    w_ramp = float(k.get('width', 1.))
    # (its profile across it in metres: (out from its inner edge, rise, texture v))
    if kind == 'ramp_concrete':
        # (one piece: its ramp and the concrete behind it, across the whole of both)
        back = float(k.get('back', BACK))
        ramp, conc, u_m_, mat_ = RAMP_CONCRETE
        pts = [(a * w_ramp, y, v) for a, y, v in ramp] + [(w_ramp + a * back, y, v) for a, y, v in conc]
    else:
        prof_, (v0_, v1_), u_m_, mat_ = PROFILES[kind]
        pts = [(a * w_ramp, y, v0_ + a * (v1_ - v0_)) for a, y in prof_]
    # (and behind it, its back ramp down to its road's height, the ground eased on from there; its texture
    # the rumble's own at its scale across, turned back from the outer edge into the painted part: its
    # stripes carried over the edge, not one line of it stretched)
    back_ramp = float(k.get('back_ramp', BACK_RAMP))                # (0: none)
    if pts[-1][1] > RAISED and back_ramp > 0:
        per_m = abs(pts[-1][2] - pts[0][2]) / max(pts[-1][0], 1e-9)
        pts.append((pts[-1][0] + back_ramp, 0., max(pts[-1][2] - back_ramp * per_m, 0.)))
    total = pts[-1][0]
    profile = (tuple((a / total, y) for a, y, v in pts), np.array([v for a, y, v in pts]), u_m_, mat_)
    parts = [(profile, 0., total, '')]
    import shapely
    others = other_roads(k, road, roads)
    out = []
    for (prof, vs, u_m, mat), w0, w1, suffix in parts:
        ts = np.array([p[0] for p in prof])
        runs = []                                                   # (each piece's rows: (rows, nc, 3))
        for r_, side, s0, s1, t0, t1 in pieces(road, k, roads):
            m, l, sgn, edge, f, S = strip(r_, side, s0, s1, t0, t1, k, 0, 1, None, None)
            rows = []
            reach = inside_reach(r_, side, S, edge)
            for j, (mm, ll, e, ff) in enumerate(zip(m, l, edge, f)):
                ff = min(ff, reach[j] / max(total, 1e-9))                  # (no wider than the curve lets it, inside it)
                o = sgn * (e + (w0 + ts * (w1 - w0)) * ff)                # (out from the road's middle)
                xz = mm + np.outer(o, ll)
                inner = mm + ll * sgn * e
                yi = top(np.array([inner]))[0]
                # (level with the road's edge, its rise over that: a tapering end's from nothing at its tip over the whole
                # taper; there, near another road (within OTHER_EASE of it), lower still, down to that road's height on
                # it, less so the farther in the taper, none past it: the rumble's own part as it is)
                fj = min(f[j], 1.)
                g = np.ones(len(ts))
                if others is not None and fj < 1.:
                    g = np.clip(shapely.distance(others, shapely.points(xz)) / OTHER_EASE, 0., 1.)
                y = yi + np.array([p[1] for p in prof]) * fj * g ** (1. - fj)
                rows.append(np.c_[xz[:, 0], y, xz[:, 1]])
            runs.append(np.array(rows))
        # (its pieces on one road and the next made one strip: each turned to go on from the one before, the
        # two rows where they meet (within JOIN) made one, halfway between; so no gap, no step between them)
        strips = []
        for R in runs:
            if not strips:
                strips.append(R)
                continue
            A = strips[-1]
            mid = lambda row: row[:, [0, 2]].mean(0)
            ends = [(np.linalg.norm(mid(A[-1]) - mid(R[0])), 'after', R), (np.linalg.norm(mid(A[-1]) - mid(R[-1])), 'after', R[::-1]),
                    (np.linalg.norm(mid(A[0]) - mid(R[-1])), 'before', R), (np.linalg.norm(mid(A[0]) - mid(R[0])), 'before', R[::-1])]
            d, where, R2 = min(ends, key=lambda e: e[0])
            if d > JOIN:
                strips.append(R)
            elif where == 'after':
                A = A.copy(); A[-1] = (A[-1] + R2[0]) / 2
                strips[-1] = np.concatenate([A, R2[1:]])
            else:
                A = A.copy(); A[0] = (A[0] + R2[-1]) / 2
                strips[-1] = np.concatenate([R2[:-1], A])
        nc = len(ts)
        Ps, Us, Fs, base, along0 = [], [], [], 0, 0.
        for R in strips:
            P = R.reshape(-1, 3)
            c = R[:, :, [0, 2]].mean(1)
            along = along0 + np.r_[0., np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
            U = np.c_[np.repeat(along * u_m, nc), np.tile(vs, len(R))]
            along0 = along[-1]
            F = []
            for j in range(len(R) - 1):
                for c_ in range(nc - 1):
                    a_, b_ = j * nc + c_, (j + 1) * nc + c_
                    F += [(a_, b_, b_ + 1), (a_, b_ + 1, a_ + 1)]
            F = np.array(F, int).reshape(-1, 3)
            Ps.append(P); Us.append(U); Fs.append(F + base); base += len(P)
        if not Ps:
            continue
        P, U, F = np.concatenate(Ps), np.concatenate(Us), np.concatenate(Fs)
        up = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        keep = np.abs(up[:, 1]) > 1e-9                                   # (none of no area: a triangle end's tip)
        F, up = F[keep], up[keep]
        F = np.where((up[:, 1] < 0)[:, None], F[:, ::-1], F)
        out.append((P, U, F, mat, suffix))
    return out


def build(road, k, roads, top, ground, kerbs, tpl):
    """A rumble's meshes (geometry's) added to the kerbs' model."""
    made = []
    for P, U, F, mat, suffix in geometry(road, k, roads, top):
        fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        N = np.zeros((len(P), 3))
        for i in range(3):
            np.add.at(N, F[:, i], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        mats = [x['name'] for x in kerbs['materials']]
        t = tpl.get(mat) or next(iter(tpl.values()))
        V = np.zeros((len(P), t['verts'].shape[1]))
        V[:, :3], V[:, 3:6], V[:, 6:8] = P, N, U
        if V.shape[1] > 8:
            V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
        num = re.sub(r'^1(ROAD|KERB)_', '', road['name'])
        node = dict(t, name=NAME % (num, (road.get('kerbs') or []).index(k), suffix), children=[],
                    material=mats.index(mat) if mat in mats else t['material'],
                    verts=V.astype('<f4'), faces=F.astype('<u2' if len(V) < 65536 else '<u4'))
        bp.sphere(node)
        kerbs['root']['children'].append(node)
        made.append(node)
    return made


def footprint(nodes):
    """The ground a rumble's meshes cover (x, z), each a hair smaller (its edges as they are)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    T = np.concatenate([n['verts'][:, [0, 2]].astype(float)[n['faces'].astype(int)] for n in nodes])
    return unary_union([Polygon(t).buffer(1e-4) for t in T if Polygon(t).area > 1e-8]).buffer(-.001)


def old_kerbs(kerbs, roads):
    """The old kerbs along the roads that have rumbles described: their names (each the base after
    1ROAD_KERB_), from their tops within OLD_NEAR of such a road's edge."""
    from shapely.geometry import Polygon, MultiPoint
    from shapely.ops import unary_union
    edges = []
    for r in roads.values():
        if not r.get('kerbs') or 'along' in r:
            continue
        mid, left, wl, wr, at = rd.fine(r)
        edges.append(Polygon(np.r_[mid + left * wl[:, None], (mid - left * wr[:, None])[::-1]]).buffer(0).boundary)
    if not edges:
        return set()
    E = unary_union(edges)
    import shapely
    out = set()
    for n, m in bp.meshes(kerbs):
        nm = n['name']
        if not nm.startswith('1ROAD_KERB_') or nm.startswith('1ROAD_KERB_E_') or nm.startswith('1ROAD_KERB_BED_') or not len(n['faces']):
            continue
        P = bp.triangles(n, m).reshape(-1, 3)[:, [0, 2]][::7]
        d = shapely.distance(E, shapely.points(P))
        if np.median(d) < OLD_NEAR:
            out.add(nm[len('1ROAD_KERB_'):])
    return out


def parts_of(base, name):
    """Whether a mesh is one of an old kerb's parts (base: its name after 1ROAD_KERB_)."""
    nm = name[len('SKIRT_'):] if name.startswith('SKIRT_') else name
    return (nm == '1ROAD_KERB_' + base or nm == 'KERB_%s_SIDE' % base or nm in ('1GRASS_KERB_BED_' + base, '1GRASS_KERB_FILL_' + base)
            or nm.startswith('KERB_BED_SKIRT_%s_' % base) and nm[len('KERB_BED_SKIRT_%s_' % base):] in ('GRASS', 'ASPHALT')
            or nm == '1ROAD_KERB_BED_' + base or nm.startswith('1ROAD_KERB_BED_%s_' % base))


def main():
    import restyle_paddock as rp
    roads = rd.load()
    with_kerbs = [r for r in roads.values() if r.get('kerbs') and 'along' not in r]
    if not with_kerbs or not os.path.exists(KERBS):
        print('no rumbles described')
        return
    track = gltf_source.load_kn5(bp.TRACK)
    kerbs = gltf_source.load_kn5(KERBS)
    # (the old kerbs along those roads out)
    gone = old_kerbs(kerbs, roads)
    out_ = 0
    for k_ in (kerbs, track):
        for n, _ in bp.meshes(k_):
            if len(n['faces']) and any(parts_of(b, n['name']) for b in gone):
                n['faces'] = n['faces'][:0]
                out_ += 1
    # (the roads' tops as the drape laid them, and the ground)
    road_T = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track) if n['name'] in roads and len(n['faces'])])
    ground_T = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                               if n['name'].startswith(('1GRASS', 'GRASS')) and not n['name'].startswith('1GRASS_EDGE_RAMP') and len(n['faces'])])
    Sr, Sg = bp.Surface(road_T), bp.Surface(ground_T)
    from scipy.spatial import cKDTree
    rv = road_T.reshape(-1, 3)
    rtree = cKDTree(rv[:, [0, 2]])

    def top(p):
        y = Sr.ys(p)
        gone_ = np.isnan(y)
        if gone_.any():
            y[gone_] = rv[rtree.query(p[gone_])[1], 1]
        return y

    def ground(p):
        return Sg.ys(p)
    tpl = {}
    for n, _ in bp.meshes(kerbs):
        mat = kerbs['materials'][n['material']]['name']
        if n['name'].startswith('1ROAD_KERB_') and mat not in tpl and n['verts'].shape[0]:
            tpl[mat] = n
    made, cover = [], []
    for r in with_kerbs:
        for k in r['kerbs']:
            nodes = build(r, k, roads, top, ground, kerbs, tpl)
            if nodes:
                made += nodes
                cover.append(footprint(nodes))
    # (under the rumbles: the ground, the roads' edge ramps, any road a rumble is set into, cut away)
    from shapely.ops import unary_union
    under = unary_union(cover)
    # (the roads under a rumble cut too (a paver strip or run-off beside it): a hair inside it, so its own
    # road's edge, where it starts, stays whole)
    over_roads = under.buffer(-ROAD_CLEAR)
    cut = 0
    for k_ in (track,):
        for n, m in list(bp.meshes(k_)):
            if not len(n['faces']) or not np.allclose(m, np.eye(4)) or not n['name'].startswith(('1GRASS', 'GRASS', '1ROAD')):
                continue

            before = len(n['faces'])
            res = rp.clip_out(n, over_roads if n['name'].startswith('1ROAD') else under)
            if res is None:
                continue
            P, Fc = res
            if len(Fc):
                node = dict(n, name=n['name'] + '_KCUT', children=[], verts=P.astype(n['verts'].dtype), faces=Fc.astype('<u2' if len(P) < 65536 else '<u4'))
                bp.sphere(node)
                k_['root']['children'].append(node)
            cut += before - len(n['faces'])
    # (the ground round the rumbles brought to them: its points on a rumble's edge as high as the rumble
    # there, those within RAMP of it eased from that back to their own, so the ground meets every rumble's
    # outer side, higher or lower than it was)
    from scipy.spatial import cKDTree
    K = np.concatenate([n['verts'][:, :3].astype(float) for n in made]) if made else np.zeros((0, 3))
    eased = 0
    if len(K):
        kt = cKDTree(K[:, [0, 2]])
        lo, hi = K[:, [0, 2]].min(0) - RAMP, K[:, [0, 2]].max(0) + RAMP
        for n, m in bp.meshes(track):
            if not len(n['faces']) or not np.allclose(m, np.eye(4)) or not n['name'].startswith(('1GRASS', 'GRASS')):
                continue
            V = n['verts']
            xz = V[:, [0, 2]].astype(float)
            box_ = (xz >= lo).all(1) & (xz <= hi).all(1)
            if not box_.any():
                continue
            d, j = kt.query(xz[box_])
            near = d < RAMP
            if not near.any():
                continue
            idx_ = np.nonzero(box_)[0][near]
            t = np.clip((d[near] - EDGE) / (RAMP - EDGE), 0., 1.)
            t = t * t * (3 - 2 * t)
            V[idx_, 1] = (K[j[near], 1] + (V[idx_, 1].astype(float) - K[j[near], 1]) * t).astype(V.dtype)
            bp.sphere(n)
            eased += len(idx_)
    # (the paint under the rumbles out: an old line along an old kerb, say)
    lines_path = os.path.join(bp.MODELS, 'velopark_lines.glb')
    painted_out = 0
    if os.path.exists(lines_path):
        L = gltf_source.load_kn5(lines_path)
        for n, m in bp.meshes(L):
            if len(n['faces']) and np.allclose(m, np.eye(4)) and n['name'].startswith(('LINE_', 'PAINT')):
                before = len(n['faces'])
                rp.clip_in(n, under.buffer(.01))
                painted_out += before - len(n['faces'])
        bp.save(L, lines_path)
    bp.save(kerbs, KERBS)
    bp.save(track, bp.TRACK)
    print('rumbles built: %d meshes for %d rumbles on %s; old kerbs out: %s (%d meshes); %d faces under them cut away; '
          '%d points of the ground round them brought to them; %d faces of paint under them out' % (
        len(made), sum(len(r['kerbs']) for r in with_kerbs), ', '.join(r['name'] for r in with_kerbs),
        ', '.join(sorted(gone)) or 'none', out_, cut, eased, painted_out))


if __name__ == '__main__':
    main()
