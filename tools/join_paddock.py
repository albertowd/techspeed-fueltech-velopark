"""The paddock's asphalt eased into the track's roads where it meets them, in the track's sources.

    python tools/join_paddock.py

The paddock's asphalt came with the 0.5.0 track, laid over the ground's slope beside roads
the track had made flat across: the car park climbs away from the pit lane by up to 4% where
the lane lies flat, and along their seam it stands a few centimetres over the lane's edge with
a dip behind. A kart crossing it tips over the seam, as over a kerb. This lays the paddock's
asphalt (source/models/velopark_paddock.glb, 1ROAD_PADDOCK) anew within about REACH of a road
of the track, its kerbs or the kids track: as a surface bending as little as it can between the
road at the seam and the paddock as it was past that band, meeting each in height and in slope,
and never under the ground it was laid over. Where the two differ at the seam by more than
STEP, as where the paddock stands on a level of its own, it stays as it was, and so do its
slivers, a few centimetres wide. Where it overlaps a road, it goes just under it. Its big faces
near the roads are cut by a grid of GRID metres first, so it has points to bend at, and the
paint lying on it moves with it. The paths of build_paths.py meet their roads as that lays
them, and are left as they are. Run again, it finds the seams met and changes nothing.
"""
import os, sys, urllib.parse

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label, uniform_filter
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve

import gltf_source
import kn5tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'source', 'models')
PADDOCK = os.path.join(MODELS, 'velopark_paddock.glb')
PAINT = os.path.join(MODELS, 'velopark_paint.glb')
ROADS = [os.path.join(MODELS, f) for f in ('velopark.glb', 'velopark_kerbs.glb', 'velopark_kids.glb')]
ASPHALT = '1ROAD_PADDOCK'     # and its parts, 1ROAD_PADDOCK_2 on, as this splits it
REACH = 5.           # metres from a road over which the paddock eases into it
STEP = .25           # the most the two may differ at a seam for it to be eased
SINK = .005          # how far under the road the paddock meets it
UNDER = .02          # how far under a road the paddock goes where it overlaps it
LIFT = .005          # how far over the ground under it the paddock stays at the least
GRID = 1.            # metres between the lines the faces near a road are cut along
R = .25              # metres a cell of the grids the surfaces are measured on
FIT = 1.             # metres round a point of a road's edge its slope is taken over
MAX_SLOPE = .12      # the most a road's slope is carried on at
DONE = .02           # metres; easing most of the paddock less than this, the seams are met
SMALL = .002         # metres; a cell eased less than this stays
MAX_VERTS = 65000    # the most points one mesh takes, a kn5 mesh numbering them in 16 bits


def meshes(k):
    return [(n, m) for n, m in kn5tree.walk(k['root']) if n['kind'] == 2]


def triangles(n, m):
    f = n['faces'].astype(int)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    p = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
    return p[f]


class Grid:
    """Cells of R metres over a box, (x, z) of each cell's middle."""

    def __init__(s, lo, hi):
        s.x0, s.z0 = lo
        s.nx, s.nz = (np.ceil((np.asarray(hi) - lo) / R)).astype(int) + 1
        s.X, s.Z = np.meshgrid(s.x0 + R * np.arange(s.nx), s.z0 + R * np.arange(s.nz))

    def top(s, tris):
        """The highest of the triangles over each cell; nan where none."""
        H = np.full((s.nz, s.nx), np.nan)
        for (ax, ay, az), (bx, by, bz), (cx, cy, cz) in tris:
            d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
            if abs(d) < 1e-9:
                continue
            i0 = max(0, int(np.ceil((min(ax, bx, cx) - s.x0) / R)))
            i1 = min(s.nx, int(np.floor((max(ax, bx, cx) - s.x0) / R)) + 1)
            j0 = max(0, int(np.ceil((min(az, bz, cz) - s.z0) / R)))
            j1 = min(s.nz, int(np.floor((max(az, bz, cz) - s.z0) / R)) + 1)
            if i1 <= i0 or j1 <= j0:
                continue
            X, Z = s.X[j0:j1, i0:i1], s.Z[j0:j1, i0:i1]
            l1 = ((bz - cz) * (X - cx) + (cx - bx) * (Z - cz)) / d
            l2 = ((cz - az) * (X - cx) + (ax - cx) * (Z - cz)) / d
            l3 = 1 - l1 - l2
            ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            h = l1 * ay + l2 * by + l3 * cy
            sub = H[j0:j1, i0:i1]
            up = ins & ~(sub >= h)
            sub[up] = h[up]
        return H

    def at(s, F, P):
        """The grid's values F at points P (x, z), between its cells' middles."""
        u = np.clip((P[:, 0] - s.x0) / R, 0, s.nx - 1.001)
        v = np.clip((P[:, 1] - s.z0) / R, 0, s.nz - 1.001)
        i, j = u.astype(int), v.astype(int)
        a, b = u - i, v - j
        return (F[j, i] * (1 - a) * (1 - b) + F[j, i + 1] * a * (1 - b)
                + F[j + 1, i] * (1 - a) * b + F[j + 1, i + 1] * a * b)


def road_planes(g, road):
    """Each road cell's plane, fitted to the road within FIT round it: (c, sx, sz) with the
    height c + sx x + sz z, x and z from the grid's corner."""
    m = (~np.isnan(road)).astype(float)
    h = np.nan_to_num(road)
    X, Z = g.X - g.x0, g.Z - g.z0
    k = 2 * int(round(FIT / R)) + 1
    S = lambda a: uniform_filter(a * m, k, mode='constant')
    n, sx, sz, sh = S(1.), S(X), S(Z), S(h)
    sxx, sxz, szz, sxh, szh = S(X * X), S(X * Z), S(Z * Z), S(X * h), S(Z * h)
    A = np.stack([np.stack([n, sx, sz], -1), np.stack([sx, sxx, sxz], -1), np.stack([sz, sxz, szz], -1)], -2)
    b = np.stack([sh, sxh, szh], -1)
    ok = m > 0
    sol = np.full(road.shape + (3,), np.nan)
    A = A[ok] + np.eye(3) * 1e-9
    sol[ok] = np.linalg.solve(A, b[ok][:, :, None])[:, :, 0]
    # a slope steeper than MAX_SLOPE is carried on at MAX_SLOPE, through the cell's own height
    c, gx, gz = sol[..., 0], sol[..., 1], sol[..., 2]
    sl = np.hypot(gx, gz)
    f = np.where(sl > MAX_SLOPE, MAX_SLOPE / np.maximum(sl, 1e-9), 1.)
    gx2, gz2 = gx * f, gz * f
    c2 = road - gx2 * X - gz2 * Z
    return np.where(ok, c2, np.nan), gx2, gz2


def easing(g, pad, road, ground):
    """Where each cell of the paddock's asphalt goes (nan where it stays), how far up that is
    (negative down, 0 where it stays), and how many road cells are seams."""
    on_road = ~np.isnan(road)
    d, (jq, iq) = distance_transform_edt(~on_road, sampling=R, return_indices=True)
    c, gx, gz = road_planes(g, road)
    X, Z = g.X - g.x0, g.Z - g.z0
    # the road carried on, eased where the nearest road's edge jumps from one cell to another
    T = gaussian_filter(c[jq, iq] + gx[jq, iq] * X + gz[jq, iq] * Z, .5 / R) - SINK
    has = ~np.isnan(pad)
    # the seams: a road's edge cell with the paddock's asphalt beside it, how far apart they
    # are there
    beside = has & ~on_road & (d <= .75)
    q = jq * g.nx + iq
    cnt = np.bincount(q[beside], minlength=g.nx * g.nz)
    gap = np.bincount(q[beside], weights=(pad - T)[beside], minlength=g.nx * g.nz) / np.maximum(cnt, 1)
    seam = (cnt > 0) & (np.abs(gap) <= STEP)
    u = np.clip(d / REACH, 0, 1)
    w = np.where(seam[q], 1 - u * u * (3 - 2 * u), 0.)
    w = gaussian_filter(w, .75 / R)
    off = has & ~on_road
    # the paddock's slivers, a cell wide, stay as they are: there is no surface to bend there
    thick = uniform_filter(has.astype(float), 3) * 9 > 4.5
    band = off & (w > 1e-3) & thick
    # the band filled in as smooth as it bends, from the road's surface at the seams to the
    # paddock's own past the band, meeting each in height and in slope; neither changes when
    # the band does, so run again this comes out the same
    known = np.where(off & ~band, pad, np.where(on_road, road - SINK, np.nan))
    # a piece of the band with nothing known round it stays as it is
    parts, count = label(band)
    touch = binary_dilation(~np.isnan(known), iterations=1) & band
    band &= np.isin(parts, np.unique(parts[touch]))
    aim = np.full(pad.shape, np.nan)
    aim[band] = fill(band, known, pad)[band]
    # never under the ground it was laid over
    laid = band & (ground <= pad + LIFT)
    aim[laid] = np.maximum(aim[laid], ground[laid] + LIFT)
    # over a road, or under it by less than STEP, just under it, so the paddock goes on
    # smooth from the seam under the road's edge
    over = has & on_road & (pad > road - STEP)
    aim[over] = (road - UNDER)[over]
    up = np.nan_to_num(aim - pad)
    aim[np.abs(up) < SMALL] = np.nan
    up[np.abs(up) < SMALL] = 0
    return aim, up, int(seam.sum())


def fill(unknown, known, prior):
    """Values for the unknown cells bending as little as they can between the known ones
    round them (nan where not known): least squares of the bend (each cell's height less its
    neighbours' mean) at the unknown cells and the known ones beside them, so the fill meets
    those in height and slope; held, just, to prior, so a piece of it with little known round
    it has an answer."""
    H, W = unknown.shape
    defined = unknown | ~np.isnan(known)
    near = binary_dilation(unknown) & defined
    J, I = np.nonzero(near)
    r = np.arange(len(J))
    rows, cols, vals = [], [], []
    deg = np.zeros(len(J))
    for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        j2, i2 = J + dj, I + di
        ok = (j2 >= 0) & (j2 < H) & (i2 >= 0) & (i2 < W)
        ok[ok] &= defined[j2[ok], i2[ok]]
        rows.append(r[ok]); cols.append(j2[ok] * W + i2[ok]); vals.append(-np.ones(ok.sum()))
        deg += ok
    rows.append(r); cols.append(J * W + I); vals.append(deg)
    L = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(J), H * W)).tocsc()
    u = np.nonzero(unknown.ravel())[0]
    k = np.nonzero((~np.isnan(known) & defined).ravel())[0]
    A, K = L[:, u], L[:, k]
    b = -(K @ known.ravel()[k])
    out = np.full(unknown.shape, np.nan)
    eps = 1e-6
    M = (A.T @ A + eps * identity(len(u))).tocsc()
    out.ravel()[u] = spsolve(M, A.T @ b + eps * prior.ravel()[u])
    return out


class Cutter:
    """Faces, by their points' rows of 11 numbers, cut along the grid's lines x and z every
    GRID, and faces beside them given the points the cuts make on the edges they share. A
    point on an edge is found from the edge's ends taken in one order, x's lines first, so
    faces sharing the edge share the point."""

    def __init__(s):
        s.pts, s.index, s.faces = [], {}, []

    @staticmethod
    def key(p):
        return tuple(np.round(p, 6))

    def add(s, p):
        k = s.key(p)
        if k not in s.index:
            s.index[k] = len(s.pts)
            s.pts.append(p)
        return s.index[k]

    @staticmethod
    def between(a, b, axis, at):
        if tuple(a[:3]) > tuple(b[:3]):
            a, b = b, a
        t = (at - a[axis]) / (b[axis] - a[axis])
        p = a + (b - a) * t
        p[axis] = at
        return p

    @staticmethod
    def lines(lo, hi):
        """The grid's lines strictly between lo and hi."""
        k = np.arange(np.ceil(min(lo, hi) / GRID), np.floor(max(lo, hi) / GRID) + 1) * GRID
        return [x for x in k if min(lo, hi) < x < max(lo, hi)]

    def split(s, poly, axis, at):
        lo, hi = [], []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            sa, sb = a[axis] - at, b[axis] - at
            if sa <= 0:
                lo.append(a)
            if sa >= 0:
                hi.append(a)
            if (sa < 0 < sb) or (sb < 0 < sa):
                p = s.between(a, b, axis, at)
                lo.append(p)
                hi.append(p)
        return lo, hi

    def fan(s, poly, centre=None):
        clean = []
        for p in poly:
            if not clean or s.key(p) != s.key(clean[-1]):
                clean.append(p)
        if len(clean) > 1 and s.key(clean[0]) == s.key(clean[-1]):
            clean.pop()
        if len(clean) < 3:
            return
        ids = [s.add(p) for p in clean]
        if centre is None:
            tris = [(ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1)]
        else:
            c = s.add(centre)
            tris = [(c, ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        s.faces += [t for t in tris if len(set(t)) == 3]

    def cut(s, face):
        polys = [list(face)]
        for axis in (0, 2):
            done = []
            for poly in polys:
                vals = [p[axis] for p in poly]
                for at in s.lines(min(vals) - 1, max(vals) + 1):
                    if not min(vals) < at < max(vals):
                        continue
                    lo, hi = s.split(poly, axis, at)
                    if len(lo) >= 3:
                        done.append(lo)
                    poly = hi
                    if len(poly) < 3:
                        break
                if len(poly) >= 3:
                    done.append(poly)
            polys = done
        for poly in polys:
            s.fan(poly)

    def along(s, a, b, axis):
        """The points one pass of cuts makes on the edge a to b, in order from a: as cut
        does, each from the last point and the edge's far end, the lines taken upwards."""
        lo, hi = (a, b) if a[axis] <= b[axis] else (b, a)
        pts, cur = [], lo
        for at in s.lines(lo[axis], hi[axis]):
            cur = s.between(cur, hi, axis, at)
            pts.append(cur)
        return pts if lo is a else pts[::-1]

    def on_edge(s, a, b):
        """The points the cuts make on the edge a to b, in order from a."""
        chain = [a] + s.along(a, b, 0) + [b]
        out = []
        for p, q in zip(chain, chain[1:]):
            if p is not a:
                out.append(p)
            out += s.along(p, q, 2)
        return out

    def patch(s, face, split):
        """A face given the points cut on those of its edges split marks (its edge i from its
        point i to the next), fanned from its middle."""
        poly = []
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            poly.append(a)
            if split[i]:
                poly += s.on_edge(a, b)
        if len(poly) == 3:
            s.fan(poly)
        else:
            s.fan(poly, sum(face) / 3)


def refine(V, F, moves):
    """A mesh's points V (rows of 11 numbers) and faces F with its faces bigger than GRID near
    where it moves (moves: whether each of some (x, y, z) points does) cut along the grid, and
    the faces sharing an edge with them given the points cut on it, the rest as they were.
    Returns the points, the faces, which points are new, which faces were cut or patched, and
    the faces made of those."""
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    t = V[F][:, :, :3]
    near = moves(t.reshape(-1, 3)).reshape(-1, 3).any(1) | moves(t.mean(1))
    span = np.maximum(np.ptp(t[:, :, 0], 1), np.ptp(t[:, :, 2], 1))
    big = near & (span > GRID)
    cutter = Cutter()
    pk = [tuple(np.round(p, 5)) for p in V[:, :3]]
    ek = lambda a, b: (pk[a], pk[b]) if pk[a] <= pk[b] else (pk[b], pk[a])
    shared = set()
    for f in F[big]:
        cutter.cut([V[i] for i in f])
        shared.update(ek(f[i], f[(i + 1) % 3]) for i in range(3))
    patched = np.zeros(len(F), bool)
    for k in np.nonzero(~big)[0]:
        f = F[k]
        if any(ek(f[i], f[(i + 1) % 3]) in shared for i in range(3)):
            patched[k] = True
            cutter.patch([V[i] for i in f], [ek(f[i], f[(i + 1) % 3]) in shared for i in range(3)])
    P, Fc = np.array(cutter.pts), np.array(cutter.faces, int)
    near = big | patched
    keep = F[~near]
    base = len(V)
    new = np.r_[np.zeros(len(V), bool), np.ones(len(P), bool)]
    V = np.r_[V, P] if len(P) else V
    F = np.r_[keep, Fc + base] if len(Fc) else keep
    used = np.unique(F)
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return V[used], remap[F], new[used], near, Fc


def chunks(V, F):
    """The faces in groups of at most MAX_VERTS points each, along x."""
    if len(np.unique(F)) <= MAX_VERTS:
        groups = [np.arange(len(F))]
    else:
        order = np.argsort(V[F].mean(1)[:, 0])
        groups, cur, seen = [], [], set()
        for k in order:
            new = [q for q in F[k] if q not in seen]
            if len(seen) + len(new) > MAX_VERTS:
                groups.append(cur)
                cur, seen, new = [], set(), list(F[k])
            cur.append(k)
            seen.update(new)
        groups.append(cur)
    out = []
    for gr in groups:
        f = F[gr]
        used = np.unique(f)
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        out.append((V[used], remap[f]))
    return out


def sphere(node):
    P = node['verts'][:, :3].astype(float)
    mid = (P.min(0) + P.max(0)) / 2
    node['sphere'] = [float(x) for x in mid] + [float(np.linalg.norm(P - mid, axis=1).max())]


def save(k, path):
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris[t['name']])
    gltf_source.write_glb(path, doc, binary)


def main():
    pad = gltf_source.load_kn5(PADDOCK)
    asphalt = [(n, m) for n, m in meshes(pad) if n['name'].startswith(ASPHALT)]
    for n, m in asphalt:
        if not np.allclose(m, np.eye(4)):
            sys.exit('%s is not at the model\'s origin' % n['name'])
    tris = np.concatenate([triangles(n, m) for n, m in asphalt])
    lo = tris[:, :, [0, 2]].min((0, 1)) - REACH - 2
    hi = tris[:, :, [0, 2]].max((0, 1)) + REACH + 2
    g = Grid(lo, hi)
    roads, ground = [], []
    for fn in ROADS:
        for n, m in meshes(gltf_source.load_kn5(fn)):
            if n['name'].startswith(('1ROAD', '1GRASS')):
                t = triangles(n, m)
                c = t[:, :, [0, 2]]
                (roads if n['name'].startswith('1ROAD') else ground).append(
                    t[(c.max(1) >= lo).all(1) & (c.min(1) <= hi).all(1)])
    road = g.top(np.concatenate(roads))
    top = g.top(tris)
    aim, up, seams = easing(g, top, road, g.top(np.concatenate(ground)))
    # met already when nine in ten of the cells it would ease in sight (not under a road) go
    # less than DONE: what is left is the faces' straight lines across the easing's curve
    seen = (up != 0) & np.isnan(road)
    most = np.percentile(np.abs(up[seen]), 90) if seen.any() else 0.
    if most < DONE:
        print('the paddock meets the roads already: nine in ten of its points would move less than %.3f m' % most)
        return
    print('%d m of seams; the paddock goes up to %.3f m up and %.3f m down'
          % (seams * R, up.max(), -up.min()))
    moving = binary_dilation(up != 0, iterations=int(round(1.5 / R)))
    # the surface each point goes to: where it is eased, the easing's; round that, the paddock
    # as it is; carried on past the paddock's edges, so a point at its edge, found between
    # cells, goes where the cells round it do
    goal = np.where(np.isnan(aim), top, aim)
    _, (jn, i_n) = distance_transform_edt(np.isnan(goal), return_indices=True)
    goal = goal[jn, i_n]
    eased = binary_dilation(up != 0, iterations=1)
    gx, gz = np.gradient(goal, R, axis=(1, 0))

    def cell(F, P):
        i = np.clip(np.round((P[:, 0] - g.x0) / R).astype(int), 0, g.nx - 1)
        j = np.clip(np.round((P[:, -1] - g.z0) / R).astype(int), 0, g.nz - 1)
        return F[j, i]

    moves = lambda P: cell(moving, P)
    for n, _ in asphalt:
        V, F, new, near, Fc = refine(n['verts'].astype(float), n['faces'].astype(int), moves)
        # each point where it is eased onto the easing's surface, one below the paddock's top
        # (as its skirts' feet) as far below it as it was; its normal as the surface slopes
        xz = V[:, [0, 2]]
        go = cell(eased, xz)
        to = g.at(goal, xz[go])
        below = np.minimum(V[go, 1] - cell(top, xz[go]), 0)
        below = np.where(np.isnan(below) | (below > -.03), 0, below)
        du = np.zeros(len(V))
        du[go] = to + below - V[go, 1]
        V[:, 1] += du
        flat = go & (V[:, 4] > .3)
        nn = np.c_[-g.at(gx, xz[flat]), np.ones(flat.sum()), -g.at(gz, xz[flat])]
        V[flat, 3:6] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
        redo = new | (du != 0)
        V[redo, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[redo]
        parts = chunks(V, F)
        name = n['name']
        root = pad['root']
        i = next(k for k, c in enumerate(root['children']) if c is n)
        nodes = []
        for k, (v, f) in enumerate(parts):
            node = dict(n)
            node.update(name=name if k == 0 else '%s_%d' % (name, k + 1), children=[],
                        verts=v.astype('<f4'), faces=f.astype('<u2'))
            sphere(node)
            nodes.append(node)
        root['children'][i:i + 1] = nodes
        print('%s: %d faces cut into %d; %d points' % (name, near.sum(), len(Fc), len(V)))
    save(pad, PADDOCK)
    # the paint on the paddock's asphalt with it
    paint = gltf_source.load_kn5(PAINT)
    lifted = 0
    for n, m in meshes(paint):
        if not np.allclose(m, np.eye(4)):
            continue
        V = n['verts']
        xz = V[:, [0, 2]].astype(float)
        under = cell(top, xz)
        k = (cell(eased, xz) & ~np.isnan(under) & ~(cell(road, xz) >= under)
             & (np.abs(V[:, 1] - under) < .05))
        if k.any():
            V[k, 1] += (g.at(goal, xz[k]) - under[k]).astype(V.dtype)
            sphere(n)
            lifted += k.sum()
    if lifted:
        save(paint, PAINT)
    print('the paddock eased into the roads in %s; %d points of paint with it'
          % (os.path.relpath(PADDOCK, ROOT), lifted))


if __name__ == '__main__':
    main()
