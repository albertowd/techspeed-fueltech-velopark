"""A road mesh of the track's read into a description (roads.json's, see ROADS.md) however it lets itself be:
by its rows, as a strip, or as a ring round an infield (two halves joined end to end). For the road
editor's Bring in (the road editor (top-race-editor)); kept apart from tools/describe_roads.py, which the build imports,
so the merge's core (tools/surface.py), which this uses, is not one of the build's roads step's tools.
"""
import numpy as np

import build_paths as bp
import roads as rd
from describe_roads import describe, describe_strip

FIT = .05                       # metres a fitted curve keeps to the road's middle (and widths) within


def describe_ring(n, m, material, step=.5):
    """A road mesh that is a ring (a loop round an infield: its outline one hole) described as two roads,
    its halves, joined end to end at both ends (roads.json has no closed road): its middle halfway between
    its outer edge and its inner one, every step metres; its widths to each; mapped flat from above.
    Its halves: n['name'] and n['name'] + 'B', from its point furthest north, the way the outline goes."""
    import shapely
    import surface as sf
    o = sf.outline(bp.triangles(n, m))
    parts = [p for p in shapely.get_parts(o) if p.area > 1.]
    if len(parts) != 1 or len(parts[0].interiors) != 1:
        raise SystemExit('%s: not a ring (%d pieces, %s holes)' % (n['name'], len(parts), [len(p.interiors) for p in parts]))
    outer, inner = parts[0].exterior, parts[0].interiors[0]
    L = outer.length
    k = max(16, int(L / step))
    Q = np.array([outer.interpolate(i * L / k).coords[0] for i in range(k)])
    J = np.array([inner.interpolate(inner.project(shapely.Point(q))).coords[0] for q in Q])
    mid = (Q + J) / 2
    # (smoothed a little along, the ring closed: the nearest inner point jumps where the inner edge bends)
    from scipy.ndimage import uniform_filter1d
    mid = uniform_filter1d(mid, 5, axis=0, mode='wrap')
    t = np.roll(mid, -1, 0) - np.roll(mid, 1, 0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    left = np.c_[-t[:, 1], t[:, 0]]
    do = shapely.distance(outer, shapely.points(mid))
    di = shapely.distance(inner, shapely.points(mid))
    pole = parts[0].representative_point()
    inside_left = (((np.array(pole.coords[0]) - mid) * left).sum(1)).mean() > 0      # (the infield on its left)
    wl, wr = (di, do) if inside_left else (do, di)
    s0 = int(np.argmax(mid[:, 1]))
    order = np.r_[np.arange(s0, k), np.arange(0, s0)]
    half = k // 2
    out = []
    for name, idx in ((n['name'], np.r_[order[:half + 1]]), (n['name'] + 'B', np.r_[order[half:], order[:1]])):
        out.append({'name': name, 'material': material, 'map': 'flat',
                    'points': [[round(float(mid[i, 0]), 3), round(float(mid[i, 1]), 3), round(float(wl[i]), 3), round(float(wr[i]), 3)] for i in idx]})
    return out


def describe_any(n, m, material):
    """A road mesh described however it lets itself be: by its rows (mapped along it, as the track's own
    roads), as a strip (its outline four-cornered), or as a ring (round an infield: two halves). A list of
    roads (dicts as roads.json), and how it was read."""
    V = n['verts'].astype(float)
    u = V[:, 6]
    if len(V) and abs(u.min() + 1) < 1e-3 and abs(u.max()) < 1e-3:
        return [describe(n, m, material)], 'rows'
    try:
        return [describe_strip(n, m, material)], 'strip'
    except SystemExit:
        pass
    return describe_ring(n, m, material), 'ring'


def fit_curve(road):
    """A road's middle as a curve of Bezier pieces (see roads.py): nodes where needed so it keeps within
    FIT of the road's middle and widths; each node's handles along the road's way there, a third of the
    way to the next node each."""
    C, s, _ = rd.centre(road)
    keep = np.r_[True, np.diff(s) > 1e-9]
    C, s = C[keep], s[keep]
    T = np.gradient(C[:, :2], s, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)

    def nodes_of(idx):
        out = []
        for j, i in enumerate(idx):
            d_in = s[i] - s[idx[j - 1]] if j else 0.
            d_out = s[idx[j + 1]] - s[i] if j < len(idx) - 1 else 0.
            out.append({'p': [round(float(C[i, 0]), 4), round(float(C[i, 1]), 4)],
                        'in': [round(float(-T[i, 0] * d_in / 3), 4), round(float(-T[i, 1] * d_in / 3), 4)],
                        'out': [round(float(T[i, 0] * d_out / 3), 4), round(float(T[i, 1] * d_out / 3), 4)],
                        'left': round(float(C[i, 2]), 4), 'right': round(float(C[i, 3]), 4)})
        return out

    from scipy.spatial import cKDTree
    idx = [0, len(C) - 1]
    for _ in range(400):
        nodes = nodes_of(idx)
        B, at = rd.bezier(nodes, .05)
        tree = cKDTree(B[:, :2])
        d, j = tree.query(C[:, :2])
        err = np.maximum(d, np.maximum(np.abs(B[j, 2] - C[:, 2]), np.abs(B[j, 3] - C[:, 3])))
        worst, k = 0., None
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a < 3:
                continue
            m = a + 1 + int(np.argmax(err[a + 1:b]))
            if err[m] > FIT and err[m] > worst:
                worst, k = err[m], (a, b)
        if k is None:
            break
        a, b = k
        mid = int(np.searchsorted(s, (s[a] + s[b]) / 2))     # (split where it is worst: at the middle of that piece)
        idx = sorted(set(idx + [min(max(mid, a + 1), b - 1)]))
    return nodes_of(idx)


def describe_axis(n, m, material, step=.5, flare=3., window=10.):
    """A road mesh described by its outline's middle (its medial axis): the longest line halfway between its
    edges, end to end; its width each side straight out to its edge on that side (a box's bay, a mouth where
    another road meets it, kept), a spike past flare times its width within window metres along cut back. For
    a road whose outline is no four-cornered strip: a pit lane with its boxes, a road with mouths. Mapped flat."""
    import shapely
    from scipy.spatial import Voronoi
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    import surface as sf
    o = sf.outline(bp.triangles(n, m))
    poly = max(shapely.get_parts(o), key=lambda p: p.area).simplify(.02)
    # (its edge in points every step/2: the middle between them, Voronoi's edges inside it)
    B = []
    for ring in [poly.exterior] + list(poly.interiors):
        L = ring.length
        B += [ring.interpolate(d).coords[0] for d in np.arange(0, L, step / 2)]
    B = np.array(B)
    vor = Voronoi(B)
    V = vor.vertices
    inside = shapely.contains_xy(poly, V[:, 0], V[:, 1])
    E = np.array([r for r in vor.ridge_vertices if -1 not in r and inside[r[0]] and inside[r[1]]])
    if not len(E):
        raise SystemExit('%s: no middle found' % n['name'])
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V), len(V))).tocsr()
    # (its two ends the farthest apart along it: from any point, the farthest; from that one, the farthest)
    start = E[0, 0]
    d0 = dijkstra(G, indices=start)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(G, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    P = V[path[::-1]]
    line = shapely.LineString(P)
    # (its ends go into its corners: cut back by its width there, then on straight to its edge)
    L = line.length
    S = np.arange(0, L, step)
    Q = np.array([line.interpolate(s).coords[0] for s in S])
    half = shapely.distance(poly.exterior, shapely.points(Q))
    cut0, cut1 = half[min(len(half) - 1, int(len(half) * .1))], half[max(0, int(len(half) * .9))]
    keep = (S >= cut0) & (S <= L - cut1)
    Q, half = Q[keep], half[keep]
    if len(Q) < 3:
        raise SystemExit('%s: too short a middle' % n['name'])
    for end in (0, -1):
        t = Q[end] - Q[end + (1 if end == 0 else -1)]
        t /= max(np.linalg.norm(t), 1e-9)
        ray = shapely.LineString([Q[end], Q[end] + t * 200])
        hit = ray.intersection(poly.exterior)
        if not hit.is_empty:
            pts = np.array([g.coords[0] for g in shapely.get_parts(hit)])
            e = pts[np.argmin(np.linalg.norm(pts - Q[end], axis=1))]
            extra = np.array([Q[end] + (e - Q[end]) * f for f in np.arange(step, 1., step / max(np.linalg.norm(e - Q[end]), 1e-9))] + [e])
            Q = np.r_[extra[::-1], Q] if end == 0 else np.r_[Q, extra]
            half = np.r_[np.full(len(extra), half[0]), half] if end == 0 else np.r_[half, np.full(len(extra), half[-1])]
    # (each side's width: straight out from the middle, across it, to its edge on that side)
    t = np.gradient(Q, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    left = np.c_[-t[:, 1], t[:, 0]]
    edge = poly.boundary
    wl, wr = np.zeros(len(Q)), np.zeros(len(Q))
    for i, (q, l) in enumerate(zip(Q, left)):
        for sgn, out in ((1., wl), (-1., wr)):
            hit = shapely.LineString([q, q + sgn * l * 200.]).intersection(edge)
            if hit.is_empty:
                out[i] = half[i]
                continue
            pts = shapely.get_coordinates(hit)
            out[i] = np.min(np.linalg.norm(pts - q, axis=1))
    # (a spike cut back: no wider than flare times the median within window along)
    k = max(1, int(window / step / 2))
    for w in (wl, wr):
        med = np.array([np.median(w[max(0, i - k):i + k + 1]) for i in range(len(w))])
        np.minimum(w, med * flare, out=w)
    return [{'name': n['name'], 'material': material, 'map': 'flat',
             'points': [[round(float(x), 3), round(float(z), 3), round(float(a), 3), round(float(b), 3)] for (x, z), a, b in zip(Q, wl, wr)]}]
