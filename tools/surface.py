"""The ground's surface built as one: every region on it (a road, a run-off, a rumble, a paved area, the
grass) meeting its neighbours edge to edge, point for point, at one height. A general tool: it knows
regions, not tracks.

A region (Region): its name, its place in the order (the first over all those after it), and either its
own triangles (a road's mesh, a rumble's, any floor already made), each corner with its texture's place
(u, v), or none (the ground: a height(x, z), a mapping uv(x, z), points every so many metres inside it).

build(regions, world):

1. Each region's outline (the edges of its triangles no other of its own shares), and the part of it no
   region before it covers (seen).
2. Every seen part's edges cut where they cross, one set of lines (noded), cut in pieces no longer than
   STEP: each piece of edge between two regions is the same points for both.
3. The faces those lines make, each its region's (the one whose seen part holds it).
4. Each face filled: its region's own triangles wholly inside it kept as they are (a road's rows and
   columns), the rest of it (the border by its edges, or all of it for the ground) filled with
   triangles on the face's edge points and its own (Delaunay, the edges kept, no point added on them).
5. One height for every point, whichever faces it is of: a region's own point as high as it is (the
   first region's where two have one); any other the height of the first region over it there; the
   ground's eased from the heights of what borders it to its own over RAMP.

check(meshes, world) proves it: every edge of every face shared by exactly two faces (but those on the
world's edge), one height at every point, the faces covering the world once (no gap, no overlap). It is
run on the faces a merge made (near=box), not only on the whole.
"""
import os

import numpy as np

GRID = 1e-4         # metres: every point put on this grid first, so a point two regions share is one
STEP = .5           # metres: the longest piece of a shared edge (both sides' points every so often)
SNAP = .01          # metres: an outline's point this near the edge of a region over it put on that edge
RAMP = 1.5          # metres over which the ground eases from what borders it to its own height
AREA = 4.           # square metres: the ground's largest triangle
VERBOSE = False     # (each stage of a build timed, printed)
WORKERS = 1         # processes the faces are filled and proved on (a build with many regions: the cores)
TCHECK_ALL = False
HOLES = True        # (the fill's outside told to Triangle as holes; False: by where each triangle's middle is)


class Region:
    """A region of the surface (see the module's note). tris: (n, 3, 3) its triangles (x, y, z), or None
    (the ground); uvs: (n, 3, 2) their corners' texture places; height, uv: functions of (x, z) arrays,
    for the ground (or a region whose points it is asked for between its own); spacing: metres between
    the ground's own points."""

    def __init__(self, name, tris=None, uvs=None, height=None, uv=None, spacing=2., ground=False, known=None):
        self.name, self.ground, self.spacing = name, ground, spacing
        self.tris = None if tris is None else np.asarray(tris, float).copy()
        self.uvs = None if uvs is None else np.asarray(uvs, float)
        self._height, self._uv = height, uv
        if self.tris is not None:
            self.tris[:, :, [0, 2]] = snap(self.tris[:, :, [0, 2]])
            # (none of no area, on the grid: three points in a line carry no surface, and share edges wrongly)
            T = self.tris[:, :, [0, 2]]
            a = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
            ok = np.abs(a) > 2e-10
            self.tris = self.tris[ok]
            if self.uvs is not None:
                self.uvs = self.uvs[ok]
            self._index()
            # (its outline, kept: the merge's; and its own triangles over others of its own, never kept:
            # none when they cover as much as its outline does (the most, at once), else each looked for.
            # known: both as worked out before for these same triangles (outline, folded))
            if known is not None and len(known[1]) == len(self.tris):
                self.outline_, self.folded = known
                return
            self.outline_ = outline(self.tris) if len(self.tris) else None
            T = self.tris[:, :, [0, 2]]
            tot = np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])).sum() / 2
            if self.outline_ is not None and abs(self.outline_.area - tot) <= 1e-8 + tot * 1e-13:     # (a clean mesh's: 1e-10)
                self.folded = np.zeros(len(self.tris), bool)
            else:
                self.folded = folded(self.tris)

    def _index(self):
        from scipy.spatial import cKDTree
        T = self.tris[:, :, [0, 2]]
        self._c = T.mean(1)
        self._tree = cKDTree(self._c)
        self._r = np.linalg.norm(T - self._c[:, None], axis=2).max()
        V = self.tris.reshape(-1, 3)
        self._vt = cKDTree(V[:, [0, 2]])

    def _locate(self, p):
        """For points (k, 2): the triangle each lies in and its barycentric weights; -1 where none."""
        T = self.tris[:, :, [0, 2]]
        p = np.asarray(p, float).reshape(-1, 2)
        idx = np.full(len(p), -1)
        W = np.zeros((len(p), 3))
        if not len(p):
            return idx, W
        # (the nearest triangles by their middles, all at once: the 12 nearest, then the 64 nearest for the
        # points none of those holds; a point no triangle holds after that lies outside them all (between
        # them, or past the region's edge), and its height is had otherwise)
        for k in (min(12, len(T)), min(64, len(T))):
            todo = np.nonzero(idx < 0)[0]
            if not len(todo):
                break
            _, near = self._tree.query(p[todo], k)
            near = near.reshape(len(todo), k)
            for j in range(k):
                left = idx[todo] < 0
                if not left.any():
                    break
                tj = todo[left]
                c = near[left, j]
                a, b, d = T[c, 0], T[c, 1], T[c, 2]
                e1, e2, q = b - a, d - a, p[tj] - a
                det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
                ok = np.abs(det) > 1e-14
                sd = np.where(ok, det, 1.)
                w1 = (q[:, 0] * e2[:, 1] - e2[:, 0] * q[:, 1]) / sd
                w2 = (e1[:, 0] * q[:, 1] - q[:, 0] * e1[:, 1]) / sd
                hit = ok & (w1 >= -1e-7) & (w2 >= -1e-7) & (w1 + w2 <= 1 + 1e-7)
                idx[tj[hit]] = c[hit]
                W[tj[hit]] = np.c_[1 - w1[hit] - w2[hit], w1[hit], w2[hit]]
        return idx, W

    def height(self, p):
        if self._height is not None and self.tris is None:
            return np.asarray(self._height(p[:, 0], p[:, 1]), float)
        idx, W = self._locate(p)
        y = np.full(len(p), np.nan)
        ok = idx >= 0
        y[ok] = (self.tris[idx[ok], :, 1] * W[ok]).sum(1)
        if (~ok).any():
            if self._height is not None:
                y[~ok] = self._height(p[~ok, 0], p[~ok, 1])
            else:
                y[~ok] = self._between(p[~ok])[:, 0]
        return y

    def _between(self, p):
        """Height and mapping (y, u, v) at points no triangle of its own holds: from its own points, linearly
        between the nearest of them (the nearest one's, past them all)."""
        if getattr(self, '_lin', None) is None:
            from scipy.interpolate import LinearNDInterpolator
            V = self.tris.reshape(-1, 3)
            Uv = self.uvs.reshape(-1, 2) if self.uvs is not None else np.zeros((len(V), 2))
            k_ = key(V[:, [0, 2]])
            _, first = np.unique(k_, return_index=True)
            self._lin = LinearNDInterpolator(V[first][:, [0, 2]], np.c_[V[first, 1], Uv[first]])
            self._linv = np.c_[V[:, 1], Uv]
        out = self._lin(p)
        bad = ~np.isfinite(out).all(1)
        if bad.any():
            out[bad] = self._linv[self._vt.query(p[bad])[1]]
        return out

    def uv(self, p):
        if self.tris is None or self.uvs is None:
            return np.asarray(self._uv(p[:, 0], p[:, 1]), float) if self._uv is not None else np.zeros((len(p), 2))
        idx, W = self._locate(p)
        out = np.zeros((len(p), 2))
        ok = idx >= 0
        out[ok] = (self.uvs[idx[ok]] * W[ok][:, :, None]).sum(1)
        if (~ok).any():
            out[~ok] = self._between(p[~ok])[:, 1:]
        return out


def folded(tris):
    """Which of a mesh's triangles overlap another of its own (a strip folded on a tight curve): (n,) bool."""
    import shapely
    T = tris[:, :, [0, 2]]
    polys = shapely.polygons(np.concatenate([T, T[:, :1]], axis=1))
    tree = shapely.STRtree(polys)
    a, b = tree.query(polys, predicate='intersects')
    out = np.zeros(len(T), bool)
    sel = a != b
    if sel.any():
        # (their insides meet: overlapping, or one inside another, or the same; not those only touching)
        hit = ~shapely.touches(polys[a[sel]], polys[b[sel]])
        out[a[sel][hit]] = True
        out[b[sel][hit]] = True
    return out


def areal(g):
    """Only the polygons of a shape: what a difference or the grid leaves of an edge or a point (a
    collection of them with its polygons) has no area, and a collection has no edge to take."""
    import shapely
    if g is None or g.is_empty or g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    from shapely.geometry import MultiPolygon, Polygon
    ps = [p for p in shapely.get_parts(g) if p.geom_type in ('Polygon', 'MultiPolygon')]
    ps = [q for p in ps for q in shapely.get_parts(p)]
    return MultiPolygon(ps) if len(ps) > 1 else ps[0] if ps else Polygon()


def snap(a):
    return np.round(np.asarray(a, float) / GRID) * GRID


def key(xz):
    k = np.round(np.asarray(xz) / GRID).astype(np.int64)
    return k[..., 0] * 100000007 + k[..., 1]


def outline(tris):
    """A mesh's outline (x, z): the union of its triangles (shapely)."""
    import shapely
    T = tris[:, :, [0, 2]]
    polys = shapely.polygons(np.concatenate([T, T[:, :1]], axis=1))
    polys = polys[shapely.area(polys) > 1e-12]
    try:
        out = shapely.coverage_union_all(polys)
    except Exception:
        out = shapely.union_all(polys)
    return shapely.set_precision(out.buffer(0), GRID)


def snap_to(f, over):
    """An outline with its points within SNAP of the edges of what is over it (and not on them) put on
    them: two edges made apart, a hair apart, one edge, no slivers between. Only those points moved, each
    straight onto the nearest edge; the outline as it was if that would change its area by more than the
    moves can (a twist)."""
    import shapely
    from shapely.geometry import Polygon, MultiPolygon
    from scipy.spatial import cKDTree
    B = over.boundary
    shapely.prepare(B)
    BV = shapely.get_coordinates(B)
    bt = cKDTree(BV)

    def ring(c):
        c = np.asarray(c, float)[:-1]
        d = shapely.distance(B, shapely.points(c))
        on = d < SNAP
        # (each such point onto the nearest of the edge's own points: on it exactly, as the grid has it)
        dv, jv = bt.query(c)
        move = on & (dv > GRID * .5)
        c[on] = BV[jv[on]]
        # (a segment between two points on that edge: the edge's own points between them taken too, so it
        # runs along it point for point, not as a chord across its bends)
        out, added = [], 0
        for i in range(len(c)):
            a_, b_ = c[i], c[(i + 1) % len(c)]
            out.append(a_)
            if not (on[i] and on[(i + 1) % len(c)]):
                continue
            ab = b_ - a_
            L2 = ab @ ab
            if L2 < 1e-12:
                continue
            cs = bt.query_ball_point((a_ + b_) / 2, np.sqrt(L2) / 2 + SNAP)
            if not cs:
                continue
            q = BV[cs]
            t = ((q - a_) @ ab) / L2
            off = np.abs((q[:, 0] - a_[0]) * ab[1] - (q[:, 1] - a_[1]) * ab[0]) / np.sqrt(L2)
            sel = (t > 1e-6) & (t < 1 - 1e-6) & (off < SNAP)
            if sel.any():
                o = np.argsort(t[sel])
                out += list(q[sel][o])
                added += int(sel.sum())
        out = snap(np.array(out))
        keep = np.r_[True, np.any(np.abs(np.diff(out, axis=0)) > GRID * .5, axis=1)]   # (two on one point: one)
        out = out[keep]
        if len(out) > 1 and np.all(np.abs(out[0] - out[-1]) < GRID * .5):
            out = out[:-1]
        return np.r_[out, out[:1]], int(move.sum()) + added
    parts, moved = [], 0
    for g in getattr(f, 'geoms', [f]):
        if g.geom_type != 'Polygon' or g.is_empty:
            continue
        ext, m0 = ring(g.exterior.coords)
        holes = [ring(h.coords) for h in g.interiors]
        moved += m0 + sum(m for _, m in holes)
        if len(ext) < 4:                           # (snapped to nothing: a sliver, gone onto that edge)
            moved += 1
            continue
        parts.append(Polygon(ext, [h for h, _ in holes if len(h) >= 4]))
    if not moved:
        return f
    if not parts:
        return f
    out = shapely.make_valid(MultiPolygon(parts) if len(parts) > 1 else parts[0])
    out = shapely.union_all([x for x in shapely.get_parts(out) if x.geom_type == 'Polygon'])
    if abs(out.area - f.area) > moved * SNAP * STEP + 1e-6:
        return f                                   # (twisted: as it was)
    return shapely.set_precision(out, GRID)


def build(regions, world, fixed=None):
    """The surface (see the module's note): {region name: (P (k, 3), UV (k, 2), F (m, 3))}, its faces and
    their regions. fixed: {key(x, z): height} the points kept as high whatever lies there (where the world's
    edge meets a mesh left as it was)."""
    import shapely
    from shapely.geometry import box as _box, Polygon
    W = shapely.set_precision(_box(*world) if isinstance(world, tuple) else world, GRID)
    import time
    t0 = [time.time()]

    def stage(what):
        if VERBOSE:
            print('  surface: %s %.0f s' % (what, time.time() - t0[0]), flush=True)
        t0[0] = time.time()
    # 1. outlines, and what of each is seen
    foot, seen, covered = {}, {}, None
    for r in regions:
        f = W if r.ground else (r.outline_ if getattr(r, 'outline_', None) is not None else outline(r.tris)).intersection(W)
        if covered is not None and not r.ground:
            f = shapely.set_precision(snap_to(f, covered).intersection(W), GRID)     # (snapped, and in the world still)
        f = areal(f)
        foot[r.name] = f
        s = f if covered is None else f.difference(covered)
        if not r.ground and f.area > 0 and covered is not None:
            lost = f.area - s.area
            under = f.intersection(covered).area
            if lost > under + 1e-3:
                c = f.representative_point()
                raise MergeError(r.name, [('%.2f m2 of it lost, more than the %.2f m2 under others' % (lost, under), c.x, c.y, lost - under)])
        seen[r.name] = areal(shapely.set_precision(s, GRID))
        covered = f if covered is None else shapely.set_precision(covered.union(f), GRID)
    stage('outlines, seen parts')
    # 2. one set of lines: every seen part's edges and the world's, cut where they cross, in short pieces
    # (joined with the grid in the overlay itself, so every ring stays closed)
    # (the world's own edge not cut: what lies past it, another mesh, meets its points as they are)
    # (each cut in pieces no longer than STEP first, then all noded together on the grid: every point of
    # every face on it, none moved after)
    Wb = W.boundary
    shapely.prepare(Wb)

    def cut(b):
        # (a seen part's edge in pieces no longer than STEP, but where it runs on the world's edge)
        out = []
        for g in shapely.get_parts(b):
            c = np.asarray(g.coords)
            if len(c) < 2:
                continue
            on = shapely.distance(Wb, shapely.points(c)) < GRID * 2
            mid = shapely.distance(Wb, shapely.points((c[1:] + c[:-1]) / 2)) < GRID * 2
            keep = on[1:] & on[:-1] & mid
            pts = [c[0]]
            for i in range(len(c) - 1):
                a, b_ = c[i], c[i + 1]
                L = np.hypot(*(b_ - a))
                n = 1 if keep[i] else max(1, int(np.ceil(L / STEP)))
                for k in range(1, n):
                    pts.append(a + (b_ - a) * (k / n))
                pts.append(b_)
            out.append(shapely.linestrings(np.array(pts)))
        return out
    lines = shapely.union_all([x for r in regions if not seen[r.name].is_empty for x in cut(shapely.boundary(seen[r.name]))]
                              + [Wb], grid_size=GRID)
    pieces = [g for g in getattr(lines, 'geoms', [lines]) if g.length > 0]
    stage('lines noded')
    # 3. the faces, each its region's
    faces = [f for f in shapely.get_parts(shapely.polygonize(pieces)) if f.area > 1e-12]     # (one a grid step across is still a face: without it, a hole)
    owner = []
    order = [r.name for r in regions]
    shapely.prepare(W)
    inside = [W.covers(f.representative_point()) for f in faces]
    faces = [f for f, i in zip(faces, inside) if i]          # (not a hole in the world: what lies there is not ours)
    for f in faces:
        p = f.representative_point()
        n_ = next((n for n in order if seen[n].covers(p)), None)
        if n_ is None:
            # (a sliver no region's seen part holds: the nearest one's, never none)
            n_ = min((n for n in order if not seen[n].is_empty), key=lambda n: seen[n].distance(p))
        owner.append(n_)
    # (every region's faces as much as its seen part: none lost to a ring that did not close)
    got = {}
    for f, n in zip(faces, owner):
        got[n] = got.get(n, 0.) + f.area
    for r in regions:
        want = seen[r.name].area
        if abs(got.get(r.name, 0.) - want) > max(1e-3, want * 1e-5):
            c = seen[r.name].representative_point()
            import os, pickle, tempfile
            pickle.dump(dict(name=r.name, seen=seen, foot=foot, W=W, faces=[f for f, n in zip(faces, owner) if n == r.name], all_faces=faces, owner=owner),
                        open(os.path.join(tempfile.gettempdir(), 'surface_owner_failed.pkl'), 'wb'))
            raise MergeError(r.name, [('faces %.3f m2 of its seen %.3f m2' % (got.get(r.name, 0.), want), c.x, c.y, got.get(r.name, 0.) - want)])
    faces, owner = fold_slivers(faces, owner)
    stage('faces (%d) and their regions' % len(faces))
    # 4. each face filled
    by = {r.name: r for r in regions}
    xyz = dict(fixed or {})                       # (point key) -> height: one height a point
    out = {r.name: [] for r in regions}
    mine = {r.name: [f for f, n in zip(faces, owner) if n == r.name] for r in regions}
    # (the regions' faces filled at once on every core, a big region's in a few lots; then each region
    # proved, so too)
    jobs = []
    for r in regions:
        fs = mine[r.name]
        k = max(1, min(WORKERS, len(fs) // 8))
        for i in range(k):
            if fs[i::k]:
                jobs.append((r, fs[i::k]))
    for r, parts in run_all(_fill_lot, jobs):
        out[r].extend(parts)
    stage('faces filled')
    # (each region proved as soon as it is filled, by itself: no open edge but on its outline, every point
    # of its outline one of its own, its faces as much as its seen part; the first to fail stops it all)
    # (an open edge a region finds inside it may be a seam with a sliver of another's, one the grid took out
    # of its outline: looked at across all of them, it is one where exactly two faces in all have it)
    inside = []
    for name, bad in run_all(_prove, [(r.name, out[r.name], mine[r.name], seen[r.name]) for r in regions]):
        other = [b for b in bad if b[0] != 'open edge inside']
        if other:
            raise MergeError(name, other)
        inside += [(name, b) for b in bad]
    if inside:
        ek, en = [], []
        for n, parts in out.items():
            for pts, idx in parts:
                k = key(pts)[idx]
                e = np.sort(np.r_[k[:, [0, 1]], k[:, [1, 2]], k[:, [2, 0]]], axis=1)
                ek.append(e)
                m = (pts[idx[:, [0, 1, 2]]] + pts[idx[:, [1, 2, 0]]]) / 2
                en.append(key(snap(m.reshape(-1, 2) if False else np.r_[m[:, 0], m[:, 1], m[:, 2]])))
        E = np.concatenate(ek)
        M = np.concatenate(en)
        u, cnt = np.unique(E, axis=0, return_counts=True)
        count = dict(zip(map(tuple, u.tolist()), cnt.tolist()))
        by_mid = {}
        for mk, e in zip(M.tolist(), map(tuple, E.tolist())):
            by_mid.setdefault(mk, e)
        left = {}
        for name, b in inside:
            e = by_mid.get(int(key(snap(np.array([[b[1], b[2]]])))[0]))
            if e is None or count.get(e) != 2:
                left.setdefault(name, []).append(b)
        for name, bad in left.items():
            raise MergeError(name, bad)
    stage('each region proved')
    # 5. heights: a region's own points first, by the order; then every other point
    for r in regions:
        if r.tris is None or r.ground:
            continue
        V = r.tris.reshape(-1, 3)
        for k_, y in zip(key(V[:, [0, 2]]).tolist(), V[:, 1].tolist()):
            xyz.setdefault(k_, y)
    result = {}
    allp = np.concatenate([p for n in out for p, _ in out[n]]) if any(out.values()) else np.zeros((0, 2))
    ak = key(allp)
    missing = np.array([k_ not in xyz for k_ in ak.tolist()], bool)
    if missing.any():
        # (the first region over each such point, as high as it is there; the ground's after the others)
        P = allp[missing]
        ys = np.full(len(P), np.nan)
        for r in regions:
            if r.ground:
                continue
            todo = np.isnan(ys) & shapely.contains_xy(foot[r.name].buffer(GRID * 2), P[:, 0], P[:, 1])
            if todo.any():
                ys[todo] = r.height(P[todo])
        for k_, y in zip(key(P).tolist(), ys.tolist()):
            if np.isfinite(y):
                xyz.setdefault(k_, y)
    stage('heights of the regions')
    ground = [r for r in regions if r.ground]
    if ground:
        g = ground[0]
        gp = np.concatenate([p for p, _ in out[g.name]]) if out[g.name] else np.zeros((0, 2))
        gp = gp[np.unique(key(gp), return_index=True)[1]]
        gk = key(gp)
        free = np.array([k_ not in xyz for k_ in gk.tolist()], bool)     # (its own points too: eased as the rest)
        if free.any():
            P = gp[free]
            own = g.height(P)
            # (eased from the heights of the points it shares with what borders it, over RAMP)
            edge = np.array([k_ in xyz for k_ in gk.tolist()], bool)
            E = gp[edge]
            if len(E):
                from scipy.spatial import cKDTree
                Ey = np.array([xyz[k_] for k_ in gk[edge].tolist()])
                Eo = g.height(E)
                d, j = cKDTree(E).query(P)
                t = np.clip(d / RAMP, 0., 1.)
                t = t * t * (3 - 2 * t)
                # (but on the world's edge: as high as its own there, as what lies past it meets it)
                t[shapely.distance(W.boundary, shapely.points(P)) < GRID * 2] = 1.
                own = own + (Ey[j] - Eo[j]) * (1 - t)
            for k_, y in zip(key(P).tolist(), own.tolist()):
                xyz.setdefault(k_, y)
    stage('heights of the ground')
    for r in regions:
        Ps, Fs, base = [], [], 0
        for pts, idx in out[r.name]:
            Ps.append(pts)
            Fs.append(idx + base)
            base += len(pts)
        if not Ps:
            continue
        P2 = np.concatenate(Ps)
        F = np.concatenate(Fs)
        y = np.array([xyz.get(k_, np.nan) for k_ in key(P2).tolist()])
        if np.isnan(y).any():
            y[np.isnan(y)] = r.height(P2[np.isnan(y)])
        P = np.c_[P2[:, 0], y, P2[:, 1]]
        UV = r.uv(P2)
        # (facing up)
        up = (P[F[:, 1], 0] - P[F[:, 0], 0]) * (P[F[:, 2], 2] - P[F[:, 0], 2]) - (P[F[:, 2], 0] - P[F[:, 0], 0]) * (P[F[:, 1], 2] - P[F[:, 0], 2])
        F = np.where((up > 0)[:, None], F[:, ::-1], F)
        result[r.name] = (P, UV, F)
    stage('meshes, their mapping')
    return result, faces, owner


class MergeError(ValueError):
    """A region whose faces did not come out whole: its name and its problems (what, x, z, how much)."""

    def __init__(self, name, issues):
        self.name, self.issues = name, issues
        super().__init__(name, issues)            # (its args as made: it comes back whole from another process)

    def __str__(self):
        return '%s: %d problems, e.g. %s' % (self.name, len(self.issues), ', '.join('%s at (%.2f, %.2f)' % tuple(i[:3]) for i in self.issues[:4]))


def _fill_lot(r, faces):
    return r.name, [fill(f, r) for f in faces]


def _prove(name, parts, faces, seen):
    return name, check_region(name, parts, faces, seen)


def run_all(fn, jobs):
    """fn(*job) for every job, on WORKERS processes (the biggest first) or here when one: their results."""
    if WORKERS <= 1 or len(jobs) < 2:
        return [fn(*j) for j in jobs]
    import pickle
    try:
        for r in {id(j[0]): j[0] for j in jobs if isinstance(j[0], Region)}.values():
            pickle.dumps((r._height, r._uv))
    except Exception:
        return [fn(*j) for j in jobs]               # (a region with a function no other process can have: here)
    from concurrent.futures import ProcessPoolExecutor

    def size(j):
        x = j[0]
        return len(x.tris) if isinstance(x, Region) and x.tris is not None else sum(len(p[1]) for p in j[1]) if isinstance(j[1], list) and j[1] and isinstance(j[1][0], tuple) else 0
    order = sorted(range(len(jobs)), key=lambda i: -size(jobs[i]))
    # (each worker's numerical libraries on one thread: they work in parallel already, and each thread of
    # each worker reserves its own memory, more than the machine has with many cores)
    for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')
    with ProcessPoolExecutor(min(WORKERS, len(jobs))) as ex:
        futs = {i: ex.submit(fn, *jobs[i]) for i in order}
        return [futs[i].result() for i in range(len(jobs))]


SLIVER = GRID * 20   # metres: a face thinner than this (its area over its half perimeter) is a sliver


def fold_slivers(faces, owner):
    """Each face thinner than SLIVER that is a hole in another face (a sliver the grid left inside it) made
    part of that one, of its region: the hole taken out of it, nothing else of either changed. A sliver
    beside others stays a face of its own (fill cuts it ear by ear): joined to one, its tip, within half a
    grid step of the edge across, would come into a third's edge, a T there. The faces and their regions."""
    import shapely
    from shapely.geometry import Polygon
    faces, owner = list(faces), list(owner)
    thin = [i for i, f in enumerate(faces) if not f.interiors and f.area < SLIVER * f.length / 2]
    if not thin:
        return faces, owner
    tree = shapely.STRtree(faces)
    gone = set()
    for i in thin:
        sk = set(key(snap(np.asarray(faces[i].exterior.coords)[:-1])).tolist())
        for j in tree.query(faces[i], predicate='intersects'):
            if j == i or j in gone or not faces[j].interiors:
                continue
            keep, hit = [], False
            for h in faces[j].interiors:
                if not hit and set(key(snap(np.asarray(h.coords)[:-1])).tolist()) == sk:
                    hit = True
                    continue
                keep.append(h)
            if hit:
                faces[j] = Polygon(faces[j].exterior, keep)
                gone.add(i)
                break
    return [f for i, f in enumerate(faces) if i not in gone], [o for i, o in enumerate(owner) if i not in gone]


def with_points(poly, pts):
    """A polygon with each of points pts that lies on its edge (within half a grid step) and is not one of its
    points put in that edge, in its order along it."""
    import shapely
    from shapely.geometry import Polygon
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)

    def ring(c):
        c = np.asarray(c)[:-1]
        out = []
        for a, b in zip(c, np.roll(c, -1, axis=0)):
            out.append(a)
            ab = b - a
            L2 = ab @ ab
            if L2 < 1e-16:
                continue
            cs = tree.query_ball_point((a + b) / 2, np.sqrt(L2) / 2 + GRID)
            if not cs:
                continue
            q = pts[cs]
            t = ((q - a) @ ab) / L2
            off = np.abs((q[:, 0] - a[0]) * ab[1] - (q[:, 1] - a[1]) * ab[0]) / np.sqrt(L2)
            sel = (t > 1e-9) & (t < 1 - 1e-9) & (off < GRID * .5)
            if sel.any():
                out += list(q[sel][np.argsort(t[sel])])
        out = np.array(out)
        return np.r_[out, out[:1]]
    return Polygon(ring(poly.exterior.coords), [ring(h.coords) for h in poly.interiors])


def check_region(name, parts, faces, seen):
    """One region's filled faces proved by themselves (problems, as check's): every edge of two of its
    faces, or one on its seen part's edge; every point of its faces' edges one of its points; no face of no
    area; its faces' area its seen part's."""
    import shapely
    if not parts:
        # (no faces left (all slivers, folded into those beside them): nothing to fill)
        want = sum(f.area for f in faces)
        return [] if want < 1e-6 else [('nothing filled', seen.centroid.x, seen.centroid.y, want)]
    Ps, Fs, base = [], [], 0
    for pts, idx in parts:
        Ps.append(pts)
        Fs.append(idx + base)
        base += len(pts)
    P, F = np.concatenate(Ps), np.concatenate(Fs)
    k = key(P)
    uk, inv = np.unique(k, return_inverse=True)
    G = inv.ravel()[F]
    issues = []
    e = np.sort(np.r_[G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]], axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    Q = np.zeros((len(uk), 2)); Q[inv.ravel()] = P
    one = u[cnt == 1]
    if len(one):
        m = (Q[one[:, 0]] + Q[one[:, 1]]) / 2
        # (on its seen part's edge, or on its faces' own: a sliver of a face the grid took from its seen part
        # is still its)
        off = shapely.distance(seen.boundary, shapely.points(m)) > GRID * 2
        if off.any() and faces:
            Bf = shapely.union_all([f.boundary for f in faces])
            off[off] = shapely.distance(Bf, shapely.points(m[off])) > GRID * 2
        issues += [('open edge inside', x, z, 0.) for x, z in m[off]]
    issues += [('edge of %d faces' % c, *((Q[a] + Q[b]) / 2), 0.) for (a, b), c in zip(u[cnt > 2], cnt[cnt > 2])]
    # (every ring point a corner of one of its triangles, not only among its points: one a triangle's edge
    # runs past is a T on the seam; all at once, the used points' keys sorted)
    rings = [np.asarray(ring.coords)[:-1] for f in faces for ring in [f.exterior] + list(f.interiors)]
    if rings:
        c = np.concatenate(rings)
        kc = key(snap(c))
        used = uk[np.unique(G)]
        i = np.clip(np.searchsorted(used, kc), 0, len(used) - 1)
        miss = used[i] != kc
        issues += [('edge point not kept', x, z, 0.) for x, z in c[miss]]
    T = Q[G]
    area = ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])) / 2
    # (a face of no area, three points in a line, is no fault: in a sliver it is the one using the middle
    # point, which the region beside it has on the seam; it closes the seam and draws nothing)
    # (its triangles cover its faces exactly: the faces themselves were proved against its seen part before a
    # sliver was folded into one beside it)
    tot = np.abs(area).sum()
    want = sum(f.area for f in faces) if faces else seen.area
    if abs(tot - want) > max(1e-4, want * 1e-6):
        issues.append(('area', seen.centroid.x, seen.centroid.y, tot - want))
    return issues


def ear_clip(ring):
    """A simple polygon's ring (k, 2), no point repeated, cut into triangles on its own points alone (ear by
    ear): (m, 3) indices. For a sliver, which a Delaunay fill cannot tell the sides of."""
    idx = list(range(len(ring)))
    if len(idx) < 3:
        return np.zeros((0, 3), int)
    area = sum(ring[i, 0] * ring[(i + 1) % len(ring), 1] - ring[(i + 1) % len(ring), 0] * ring[i, 1] for i in range(len(ring)))
    sign = 1. if area > 0 else -1.
    out = []
    guard = 0
    while len(idx) > 3 and guard < 10 * len(ring) ** 2:
        guard += 1
        best, bi = None, None
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            A, B, C = ring[a], ring[b], ring[c]
            cr = sign * ((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
            if cr <= 0:
                continue
            # (no other point of it inside this ear)
            inside = False
            for j in idx:
                if j in (a, b, c):
                    continue
                P = ring[j]
                d1 = (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])
                d2 = (C[0] - B[0]) * (P[1] - B[1]) - (C[1] - B[1]) * (P[0] - B[0])
                d3 = (A[0] - C[0]) * (P[1] - C[1]) - (A[1] - C[1]) * (P[0] - C[0])
                if sign * d1 >= 0 and sign * d2 >= 0 and sign * d3 >= 0:      # (on its edges too: in a sliver, points lie in line)
                    inside = True
                    break
            if not inside and (best is None or cr > best):
                best, bi = cr, k
        if bi is None:
            # (nothing left with area: the rest in a line, no surface)
            break
        a, b, c = idx[bi - 1], idx[bi], idx[(bi + 1) % len(idx)]
        out.append((a, b, c))
        idx.pop(bi)
    if len(idx) == 3:
        out.append(tuple(idx))
    return np.array(out, int).reshape(-1, 3)


def fill(face, r):
    """A face of a region filled with triangles: (points (k, 2), faces (m, 3)). Its region's own triangles
    wholly inside it kept; the rest filled on the face's edge points, the kept triangles' and (the
    ground) its own every spacing metres, no point added on an edge."""
    import shapely
    import triangle
    from shapely.geometry import Polygon
    if not face.interiors and face.area < SLIVER * face.length / 2:
        # (a sliver with no hole: its own points cut ear by ear, nothing added, nothing kept of its region's)
        ring = snap(np.asarray(face.exterior.coords)[:-1])
        k = key(ring)
        keep = np.r_[True, k[1:] != k[:-1]]
        ring = ring[keep]
        if len(ring) > 1 and key(ring[:1])[0] == key(ring[-1:])[0]:
            ring = ring[:-1]
        return ring, ear_clip(ring)
    rings = [np.asarray(face.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in face.interiors]
    pts, segs = [], []
    for ring in rings:
        b = len(pts)
        pts += [tuple(p) for p in ring]
        segs += [(b + i, b + (i + 1) % len(ring)) for i in range(len(ring))]
    kept = np.zeros((0, 3, 2))
    if r.tris is not None:
        T = r.tris[:, :, [0, 2]]
        lo, hi = np.array(face.bounds[:2]), np.array(face.bounds[2:])
        near = ((T.max(1) >= lo) & (T.min(1) <= hi)).all(1) & ~r.folded
        idx = np.nonzero(near)[0]
        if len(idx):
            polys = shapely.polygons(np.concatenate([T[idx], T[idx, :1]], axis=1))
            inner = face if HOLES else face.buffer(GRID * .5)   # (wholly in it: one reaching past its edge, by a hair, crosses it)
            shapely.prepare(inner)
            ok = shapely.within(polys, inner)
            kept = T[idx[ok]]
    if len(kept):
        # (not one with a point of the face's edge inside one of its own edges, as where another region's
        # edge crosses it, or a long edge was cut in pieces: it would leave a T; the fill takes it)
        # (nor one with a point of another kept one inside its edge: a T in the region's own mesh)
        # (only an edge no other kept one shares, on the kept ones' border, can have one: two that share an
        # edge meet point for point along it. Looked for again as long as one goes: its going makes others'
        # edges the border)
        from scipy.spatial import cKDTree
        R = snap(np.concatenate(rings))
        drop = np.zeros(len(kept), bool)
        for _ in range(8):
            live = np.nonzero(~drop)[0]
            K = kept[live]
            ea = np.concatenate([K[:, 0], K[:, 1], K[:, 2]])
            eb = np.concatenate([K[:, 1], K[:, 2], K[:, 0]])
            who = np.tile(np.arange(len(K)), 3)
            ka, kb = key(ea), key(eb)
            code = np.c_[np.minimum(ka, kb), np.maximum(ka, kb)]
            _, inv, cnt = np.unique(code, axis=0, return_inverse=True, return_counts=True)
            single = cnt[inv.ravel()] == 1
            if TCHECK_ALL:
                single = np.ones_like(single)
            if not single.any():
                break
            B = np.unique(np.r_[R, ea[single], eb[single]], axis=0)
            bt = cKDTree(B)
            a, b, w = ea[single], eb[single], who[single]
            mid, half = (a + b) / 2, np.linalg.norm(b - a, axis=1) / 2
            gone = False
            for i, cs in enumerate(bt.query_ball_point(mid, half + GRID)):
                if drop[live[w[i]]] or len(cs) <= 2:
                    continue
                q = B[cs]
                ab = b[i] - a[i]
                L2 = max(ab @ ab, 1e-18)
                t = ((q - a[i]) @ ab) / L2
                off = np.abs((q[:, 0] - a[i, 0]) * ab[1] - (q[:, 1] - a[i, 1]) * ab[0]) / np.sqrt(L2)
                if ((t > 1e-6) & (t < 1 - 1e-6) & (off < GRID)).any():
                    drop[live[w[i]]] = gone = True
            if not gone:
                break
        # (nor one with a corner on the face's edge that is not one of the edge's points: a T the other way)
        # (only a corner on the kept ones' border can be on the face's edge)
        import shapely as _sh
        bk = key(snap(np.concatenate(rings)))
        live = np.nonzero(~drop)[0]
        K = kept[live]
        ea = np.concatenate([K[:, 0], K[:, 1], K[:, 2]])
        eb = np.concatenate([K[:, 1], K[:, 2], K[:, 0]])
        ka, kb = key(ea), key(eb)
        _, inv, cnt = np.unique(np.c_[np.minimum(ka, kb), np.maximum(ka, kb)], axis=0, return_inverse=True, return_counts=True)
        single = cnt[inv.ravel()] == 1
        border = np.unique(np.r_[ka[single], kb[single]])
        border = border[~np.isin(border, bk)]                      # (its points on the edge are the edge's: fine)
        if len(border):
            BV = np.unique(np.r_[ea[single], eb[single]], axis=0)
            BV = BV[np.isin(key(BV), border)]
            on = _sh.distance(face.boundary, _sh.points(BV)) < GRID
            stray = key(BV[on])
            if len(stray):
                KV = key(K.reshape(-1, 2)).reshape(-1, 3)
                drop[live[np.isin(KV, stray).any(1)]] = True
        kept = kept[~drop]
    # (the kept triangles' own edges where the rest meets them: its edges too)
    if len(kept):
        A = np.concatenate([kept[:, [0, 1, 2]], kept[:, [1, 2, 0]]], axis=0)
        e = np.r_[np.c_[kept[:, 0], kept[:, 1]].reshape(-1, 2, 2), np.c_[kept[:, 1], kept[:, 2]].reshape(-1, 2, 2), np.c_[kept[:, 2], kept[:, 0]].reshape(-1, 2, 2)]
        ka, kb = key(e[:, 0]), key(e[:, 1])
        lo_, hi_ = np.minimum(ka, kb), np.maximum(ka, kb)
        _, inv, cnt = np.unique(np.c_[lo_, hi_], axis=0, return_inverse=True, return_counts=True)
        single = cnt[inv.ravel()] == 1
        for a, b in e[single]:
            i0 = len(pts)
            pts += [tuple(a), tuple(b)]
            segs.append((i0, i0 + 1))
    # (the ground's own points inside it, away from its edges)
    if r.ground and r.spacing and r.tris is None:
        x0, z0, x1, z1 = face.bounds
        gx, gz = np.meshgrid(np.arange(np.ceil(x0 / r.spacing) * r.spacing, x1, r.spacing),
                             np.arange(np.ceil(z0 / r.spacing) * r.spacing, z1, r.spacing))
        G = np.c_[gx.ravel(), gz.ravel()]
        if len(G):
            inside = shapely.contains_xy(face.buffer(-r.spacing * .3), G[:, 0], G[:, 1])
            pts += [tuple(p) for p in snap(G[inside])]
    P = snap(np.array(pts))
    # (one point a place: the segments on the merged points)
    k = key(P)
    uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
    P = P[first]
    S = inv.ravel()[np.array(segs, int)] if segs else np.zeros((0, 2), int)
    S = S[S[:, 0] != S[:, 1]]
    S = np.unique(np.sort(S, axis=1), axis=0)
    opts = 'pYYQ' + ('q25a%g' % AREA if r.ground else '')   # (YY: no point added on any edge, the outer ones or the kept triangles')
    # (what is outside the face, told to Triangle: a point in each of its holes; it takes away its triangles
    # out to the edges round each, and those outside its outer edge: by the edges themselves, not by where
    # a triangle's middle falls, which on a sliver thinner than the grid can be either side)
    # (a hole thinner than the grid (a sliver) not so: a point in it can fall either side of its edge; its
    # own triangles, on its ring's points alone, out after)
    H, thin = [], []
    for h in (face.interiors if HOLES else []):
        hp = Polygon(h)
        if hp.area / max(hp.length, 1e-12) > GRID * 20:
            H.append(np.asarray(hp.representative_point().coords)[0])
        else:
            thin.append(hp)
    tri_in = dict(vertices=P, segments=S)
    if H:
        tri_in['holes'] = np.array(H)
    try:
        t = triangle.triangulate(tri_in, opts)
    except RuntimeError:
        import pickle, tempfile, os
        path = os.path.join(tempfile.gettempdir(), 'surface_fill_failed.pkl')
        pickle.dump(dict(face=face, kept=kept, P=P, S=S, name=r.name), open(path, 'wb'))
        c = face.representative_point()
        raise MergeError(r.name, [('its fill failed (%d points, %d edges, %d kept; %s)' % (len(P), len(S), len(kept), path), c.x, c.y, face.area)])
    TV, TF = t['vertices'], t.get('triangles', np.zeros((0, 3), int))      # (none: a face of no area; its region's check says if it was not)
    # (its points as given; those it added inside, for the ground's shape, where it put them: on the grid
    # one could fall in line with two others, a face of no area. None is on an edge: none shared)
    TV = np.array(TV, float)
    TV[:len(P)] = P
    # (and not a kept triangle's place: those the kept ones themselves; by where its middle is, on a kept
    # triangle or its edge)
    c = TV[TF].mean(1)
    inside = np.ones(len(TF), bool) if HOLES else shapely.contains_xy(face, c[:, 0], c[:, 1])
    for hp in thin:
        hk = set(key(snap(np.asarray(hp.exterior.coords))).tolist())
        mine = np.isin(key(TV), list(hk))[TF].all(1) & shapely.contains_xy(hp, c[:, 0], c[:, 1])
        inside &= ~mine
    if len(kept):
        kp = shapely.polygons(np.concatenate([kept, kept[:, :1]], axis=1))
        hit = shapely.STRtree(kp).query(shapely.points(c), predicate='intersects')
        on_kept = np.zeros(len(c), bool)
        on_kept[hit[0]] = True
        inside &= ~on_kept
    TF = TF[inside]
    # (the kept triangles, on the same points)
    allP = np.r_[TV, kept.reshape(-1, 2)] if len(kept) else TV
    k = key(allP)
    uk, first, inv = np.unique(k, return_index=True, return_inverse=True)
    inv = inv.ravel()
    F = np.r_[inv[TF], inv[len(TV):].reshape(-1, 3)] if len(kept) else inv[TF]
    return allP[first], F


def check(meshes, world=None, near=None):
    """The surface proved (see the module's note): problems, each (what, x, z, how much). meshes: {name: (P,
    UV, F)}; world: its outline (x0, z0, x1, z1) or shapely; near: (x0, z0, x1, z1) to check only there."""
    import shapely
    from shapely.geometry import box as _box
    Ps, Fs, base, names = [], [], 0, []
    for n, (P, UV, F) in meshes.items():
        Ps.append(P)
        Fs.append(F + base)
        names += [n] * len(F)
        base += len(P)
    P = np.concatenate(Ps)
    F = np.concatenate(Fs)
    names = np.array(names)
    k = key(P[:, [0, 2]])
    uk, inv = np.unique(k, return_inverse=True)
    inv = inv.ravel()
    issues = []
    # (one height a point)
    ymin = np.full(len(uk), np.inf); ymax = np.full(len(uk), -np.inf)
    np.minimum.at(ymin, inv, P[:, 1]); np.maximum.at(ymax, inv, P[:, 1])
    bad = np.nonzero(ymax - ymin > 1e-6)[0]
    first = np.zeros(len(uk), int); first[inv[::-1]] = np.arange(len(inv))[::-1]
    for i in bad:
        p = P[first[i]]
        issues.append(('heights differ', p[0], p[2], ymax[i] - ymin[i]))
    # (every edge shared by exactly two faces, but the world's edge)
    G = inv[F]
    sel = np.ones(len(F), bool)
    if near is not None:
        c = P[F].mean(1)
        sel = (c[:, 0] >= near[0]) & (c[:, 0] <= near[2]) & (c[:, 2] >= near[1]) & (c[:, 2] <= near[3])
    # (all at once: every edge by its two points, how many faces have it; only those not two looked at)
    e = np.r_[G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]]
    es = np.sort(e, axis=1)
    code = es[:, 0].astype(np.int64) * (len(uk) + 1) + es[:, 1]
    u, first_e, cnt = np.unique(code, return_index=True, return_counts=True)
    fsel = np.r_[sel, sel, sel]
    looked = np.zeros(len(u), bool)
    looked[np.searchsorted(u, code[fsel])] = True
    bad = np.nonzero(looked & (cnt != 2))[0]
    Q = np.zeros((len(uk), 3)); Q[inv] = P
    if len(bad):
        ab = es[first_e[bad]]
        M = (Q[ab[:, 0]] + Q[ab[:, 1]]) / 2
        c_ = cnt[bad]
        on_edge = np.zeros(len(bad), bool)
        if world is not None:
            Wb = (_box(*world) if isinstance(world, tuple) else world).boundary
            one = c_ == 1
            on_edge[one] = shapely.distance(Wb, shapely.points(M[one][:, [0, 2]])) < GRID * 10
        for (a, b), m, k_ in zip(ab[~on_edge], M[~on_edge], c_[~on_edge]):
            issues.append(('open edge' if k_ == 1 else 'edge of %d faces' % k_, m[0], m[2], np.linalg.norm((Q[a] - Q[b])[[0, 2]])))
    # (the faces' area, for the cover below; a face of no area is no fault: see check_region)
    T = P[F]
    area = ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 2] - T[:, 0, 2]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 2] - T[:, 0, 2])) / 2
    # (the world covered once: the faces' area the world's)
    if world is not None and near is None:
        Wp = _box(*world) if isinstance(world, tuple) else world
        tot = np.abs(area).sum()
        if abs(tot - Wp.area) > max(1e-3, Wp.area * 1e-6):     # (the grid's rounding over many faces: a part in a million)
            issues.append(('cover', Wp.centroid.x, Wp.centroid.y, tot - Wp.area))
    return issues
