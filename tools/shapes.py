"""Shapes to make models of: boxes, beams, cylinders, roofs, walls along lines, strips over the
ground. Each is a list of parts (V, N, U, F) as tools/build_trees.py's are: points (n, 3) in AC's
x, y up, z; their normals; their uv; and the triangles (m, 3), each facing the way its normal does
(counter-clockwise seen from that side). The uv is in metres over `tile`, so a texture repeats
every `tile` metres, unless said otherwise.
"""
import math

import numpy as np


def _part(V, N, U, F):
    return (np.asarray(V, float), np.asarray(N, float), np.asarray(U, float), np.asarray(F, int))


def _facing(V, F, n):
    """Triangles turned so they face n."""
    F = np.array(F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    flip = (np.cross(b - a, c - a) @ n) < 0
    F[flip] = F[flip][:, ::-1]
    return F


def quad(p0, p1, p2, p3, tile=(1., 1.), uv0=(0., 0.), both=False):
    """A flat four sided face, p0 p1 p2 p3 round it; u along p0 to p1, v along p0 to p3, in metres
    over tile. Facing the way p0 p1 p2 turns (counter-clockwise seen from the front)."""
    P = np.array([p0, p1, p2, p3], float)
    n = np.cross(P[1] - P[0], P[3] - P[0])
    n /= np.linalg.norm(n) or 1
    eu = P[1] - P[0]
    lu = np.linalg.norm(eu) or 1
    eu /= lu
    ev = np.cross(n, eu)
    U = np.c_[(P - P[0]) @ eu / tile[0] + uv0[0], (P - P[0]) @ ev / tile[1] + uv0[1]]
    out = [_part(P, np.tile(n, (4, 1)), U, [(0, 1, 2), (0, 2, 3)])]
    if both:
        out.append(_part(P, np.tile(-n, (4, 1)), U, [(0, 2, 1), (0, 3, 2)]))
    return out


def poly(P, n=None, tile=(1., 1.), axes=None):
    """A flat polygon (convex or not, no holes), its points round it, facing n (from its winding
    when not given); uv from the plane's axes."""
    import triangle as tr
    P = np.asarray(P, float)
    if n is None:
        n = np.zeros(3)
        for i in range(len(P)):
            n += np.cross(P[i], P[(i + 1) % len(P)])
    n = n / np.linalg.norm(n)
    if axes is None:
        a = np.cross(n, [0, 0, 1]) if abs(n[2]) < .9 else np.cross(n, [1, 0, 0])
        a /= np.linalg.norm(a)
        axes = (a, np.cross(n, a))
    Q = np.c_[P @ axes[0], P @ axes[1]]
    seg = [(i, (i + 1) % len(P)) for i in range(len(P))]
    t = tr.triangulate(dict(vertices=Q, segments=seg), 'p')
    F = t['triangles']
    q = t['vertices']
    # back into the plane the polygon lies in
    V = q[:, :1] * axes[0] + q[:, 1:2] * axes[1] + (P[0] @ n) * n
    U = np.c_[q[:, 0] / tile[0], q[:, 1] / tile[1]]
    return [_part(V, np.tile(n, (len(V), 1)), U, _facing(V, F, n))]


def box(lo, hi, tile=(1., 1.), yaw=0., at=(0, 0, 0), faces='nsewtb'):
    """A box from lo to hi (x, y, z), turned by yaw about y round its origin, then moved to at. faces
    names the sides made: n (+z), s (-z), e (-x), w (+x), t (top), b (bottom)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    sides = {'s': (1, 0, 3, 2), 'n': (4, 5, 6, 7), 'w': (5, 1, 2, 6), 'e': (0, 4, 7, 3), 't': (3, 7, 6, 2), 'b': (0, 1, 5, 4)}
    out = []
    for k in faces:
        a, b, cc, d = (c[i] for i in sides[k])
        out += quad(a, b, cc, d, tile)
    return place(out, yaw, at)


def place(parts, yaw=0., at=(0, 0, 0), scale=1.):
    """Parts turned by yaw about y (from +x towards +z... as AC turns: x to -z), scaled, moved to at."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return [_part(V @ R.T * scale + at, N @ R.T, U, F) for V, N, U, F in parts]


def beam(a, b, w, h=None, tile=(1., 1.), up=(0, 1, 0), ends=True):
    """A beam of w by h (h = w when not given) from a to b: posts, rails, braces, legs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    h = w if h is None else h
    d = b - a
    L = np.linalg.norm(d)
    d /= L
    up = np.asarray(up, float)
    if abs(d @ up) > .95:
        up = np.array([1., 0, 0]) if abs(d[0]) < .9 else np.array([0, 0, 1.])
    s = np.cross(d, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, d)
    hw, hh = w / 2, h / 2
    ring = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    out = []
    for i in range(4):
        (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % 4]
        p0, p1 = a + s * x0 + u * y0, a + s * x1 + u * y1
        out += quad(p0, p1, p1 + d * L, p0 + d * L, tile)
        out[-1] = _part(out[-1][0], out[-1][1], out[-1][2][:, ::-1], out[-1][3])   # u across, v along
    if ends:
        for p, sign in ((a, -1), (b, 1)):
            cs = [p + s * x + u * y for x, y in ring]
            q = quad(*cs, tile)
            if (q[0][1][0] @ (d * sign)) < 0:
                q = quad(*cs[::-1], tile)
            out += q
    return out


def cylinder(c, r, h, seg=12, tile=(1., 1.), top=True, bottom=False, r_top=None):
    """An upright cylinder (or cone, r_top) with its foot's middle at c."""
    c = np.asarray(c, float)
    r_top = r if r_top is None else r_top
    a = np.linspace(0, 2 * np.pi, seg + 1)
    ca, sa = np.cos(a), np.sin(a)
    lo = c + np.c_[ca * r, np.zeros(seg + 1), sa * r]
    hi = c + np.c_[ca * r_top, np.full(seg + 1, h), sa * r_top]
    slope = (r - r_top) / h
    N = np.c_[ca, np.full(seg + 1, slope), sa]
    N /= np.linalg.norm(N, axis=1)[:, None]
    V = np.r_[lo, hi]
    U = np.r_[np.c_[a * r / tile[0], np.zeros(seg + 1)], np.c_[a * r / tile[0], np.full(seg + 1, h / tile[1])]]
    F = []
    for i in range(seg):
        F += [(i, seg + 1 + i + 1, i + 1), (i, seg + 1 + i, seg + 1 + i + 1)]
    out = [_part(V, np.r_[N, N], U, F)]
    out[0] = _part(V, out[0][1], U, _facing_out(V, np.array(F), c))
    for on, y, rr, n in ((top, h, r_top, 1), (bottom, 0, r, -1)):
        if on and rr > 0:
            P = c + np.c_[ca[:-1] * rr, np.full(seg, y), sa[:-1] * rr]
            out += poly(P, np.array([0, n, 0.]), tile, axes=(np.array([1., 0, 0]), np.array([0, 0, 1.])))
    return out


def _facing_out(V, F, c):
    m = V[F].mean(1)
    a, b, cc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = m - c
    out[:, 1] = 0
    flip = (np.cross(b - a, cc - a) * out).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    return F


def sphere(c, r, seg=10, rings=6):
    c = np.asarray(c, float)
    V, U = [], []
    for i in range(rings + 1):
        t = np.pi * i / rings
        for j in range(seg + 1):
            p = 2 * np.pi * j / seg
            V.append((math.sin(t) * math.cos(p), math.cos(t), math.sin(t) * math.sin(p)))
            U.append((j / seg, i / rings))
    V = np.array(V)
    F = []
    for i in range(rings):
        for j in range(seg):
            a, b = i * (seg + 1) + j, (i + 1) * (seg + 1) + j
            F += [(a, b, a + 1), (a + 1, b, b + 1)]
    F = np.array(F)
    P = c + V * r
    a, b, e = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    flip = (np.cross(b - a, e - a) * ((a + b + e) / 3 - c)).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    # the rings' first and last are points: drop the triangles they make with no area
    keep = np.linalg.norm(np.cross(b - a, e - a), axis=1) > 1e-12
    return [_part(P, V, U, F[keep])]


def gable_roof(x0, x1, z0, z1, eave, ridge, over=.4, tile=(1., 1.), thick=.08, along='x'):
    """A roof of two slopes over the rectangle x0..x1, z0..z1, its ridge along x (or z), from eave
    height up to ridge, reaching over the walls by over; with its edges' thickness."""
    if along == 'z':
        return [_part(V[:, [2, 1, 0]], N[:, [2, 1, 0]], U, F[:, ::-1]) for V, N, U, F in
                gable_roof(z0, z1, x0, x1, eave, ridge, over, tile, thick)]
    xa, xb = x0 - over, x1 + over
    zm = (z0 + z1) / 2
    half = (z1 - z0) / 2
    drop = (ridge - eave) / half * over
    za, zb = z0 - over, z1 + over
    out = []
    for zs in (za, zb):
        e = eave - drop
        p0, p1, p2, p3 = (xa, e, zs), (xb, e, zs), (xb, ridge, zm), (xa, ridge, zm)
        q = quad(p0, p1, p2, p3, tile)
        if q[0][1][0][1] < 0:
            q = quad(p3, p2, p1, p0, tile)
        out += q
        # under side
        q = quad((xa, e - thick, zs), (xb, e - thick, zs), (xb, ridge - thick, zm), (xa, ridge - thick, zm), tile)
        if q[0][1][0][1] > 0:
            q = quad((xa, ridge - thick, zm), (xb, ridge - thick, zm), (xb, e - thick, zs), (xa, e - thick, zs), tile)
        out += q
        # the edge along the eave
        out += quad((xa, e - thick, zs), (xb, e - thick, zs), (xb, e, zs), (xa, e, zs), tile, both=True)
    return out


def gable_ends(x0, x1, z0, z1, eave, ridge, tile=(1., 1.)):
    """The two triangles of wall under a gable roof whose ridge runs along x."""
    zm = (z0 + z1) / 2
    out = []
    for x, n in ((x0, -1), (x1, 1)):
        V = np.array([(x, eave, z0), (x, eave, z1), (x, ridge, zm)])
        U = np.c_[V[:, 2] / tile[0], V[:, 1] / tile[1]]
        out.append(_part(V, np.tile([n, 0, 0.], (3, 1)), U, _facing(V, [(0, 1, 2)], np.array([n, 0, 0.]))))
    return out


def hip_roof(x0, x1, z0, z1, eave, top, over=.5, tile=(1., 1.)):
    """A roof of four slopes over the rectangle, rising to a ridge (or a point) at top."""
    xa, xb, za, zb = x0 - over, x1 + over, z0 - over, z1 + over
    w, d = xb - xa, zb - za
    r = min(w, d) / 2
    k = (top - eave) / r
    e = eave - k * over
    if w >= d:
        r0, r1 = (xa + r, (za + zb) / 2), (xb - r, (za + zb) / 2)
    else:
        r0, r1 = ((xa + xb) / 2, za + r), ((xa + xb) / 2, zb - r)
    A, B, C, D = (xa, e, za), (xb, e, za), (xb, e, zb), (xa, e, zb)
    R0, R1 = (r0[0], top + k * 0, r0[1]), (r1[0], top, r1[1])
    faces = [(A, B, R1, R0), (B, C, R1), (C, D, R0, R1), (D, A, R0)] if w >= d else [(A, B, R0), (B, C, R1, R0), (C, D, R1), (D, A, R0, R1)]
    out = []
    for f in faces:
        V = np.array(f, float)
        # a square's ridge is a point: its four sided faces are three sided
        V = V[np.r_[True, np.linalg.norm(np.diff(V, axis=0), axis=1) > 1e-9]]
        if len(V) > 3 and np.linalg.norm(V[-1] - V[0]) < 1e-9:
            V = V[:-1]
        n = np.cross(V[1] - V[0], V[2] - V[0])
        if n[1] < 0:
            V = V[::-1]
            n = -n
        n /= np.linalg.norm(n)
        a = V[1] - V[0]
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        U = np.c_[(V - V[0]) @ a / tile[0], (V - V[0]) @ b / tile[1]]
        F = [(0, 1, 2)] if len(V) == 3 else [(0, 1, 2), (0, 2, 3)]
        out.append(_part(V, np.tile(n, (len(V), 1)), U, F))
    return out


def walls(P, y0, y1, tile=(1., 1.), outward=True, closed=True):
    """Upright walls along the points P (x, z) from y0 to y1 (each a number, or one a point),
    facing out of the polygon they go round (outward), or into it."""
    P = np.asarray(P, float)
    n = len(P)
    y0 = np.broadcast_to(np.asarray(y0, float), (n,))
    y1 = np.broadcast_to(np.asarray(y1, float), (n,))
    area = sum(P[i, 0] * P[(i + 1) % n, 1] - P[(i + 1) % n, 0] * P[i, 1] for i in range(n))
    out, u = [], 0.
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        a, b = P[i], P[j]
        L = np.linalg.norm(b - a)
        if L < 1e-6:
            continue
        p0, p1 = (a[0], y0[i], a[1]), (b[0], y0[j], b[1])
        p2, p3 = (b[0], y1[j], b[1]), (a[0], y1[i], a[1])
        q = quad(p0, p1, p2, p3, tile, uv0=(u, 0))
        nrm = q[0][1][0]
        mid = (a + b) / 2
        # out of the polygon: away from its inside, as its winding says
        d = np.array([b[1] - a[1], 0, -(b[0] - a[0])]) * (1 if area > 0 else -1)
        if (nrm @ d > 0) != outward:
            q = quad(p3, p2, p1, p0, tile, uv0=(u, 0))
        out += q
        u += L / tile[0]
    return out


def strip(line, width, ground, lift=.03, tile=(4., 4.), step=2., ends=0.):
    """A strip of width along the line (x, z points), laid over the ground (ground(x, z) -> y), lift
    over it: roads, paths. u across (0..width/tile), v along."""
    from shapely.geometry import LineString
    L = LineString(line)
    s = np.arange(0, L.length + step / 2, step)
    s[-1] = min(s[-1], L.length)
    C = np.array([L.interpolate(v).coords[0] for v in s])
    d = np.gradient(C, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    side = np.c_[-d[:, 1], d[:, 0]]
    cols = 5
    t = np.linspace(-.5, .5, cols)
    P = C[:, None, :] + side[:, None, :] * (t[None, :, None] * width)
    xz = P.reshape(-1, 2)
    y = ground(xz[:, 0], xz[:, 1]) + lift
    V = np.c_[xz[:, 0], y, xz[:, 1]]
    U = np.c_[np.tile((t + .5) * width / tile[0], len(s)), np.repeat(s / tile[1], cols)]
    F = []
    for i in range(len(s) - 1):
        for j in range(cols - 1):
            a, b = i * cols + j, i * cols + j + 1
            c, e = (i + 1) * cols + j, (i + 1) * cols + j + 1
            F += [(a, c, b), (b, c, e)]
    F = _facing(V, F, np.array([0, 1., 0]))
    N = _vertex_normals(V, F)
    return [_part(V, N, U, F)]


def _vertex_normals(V, F):
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1)[:, None], 1e-9)


def panels(line, ground, y0, y1, tile=(1., 1.), both=True, step=None):
    """An upright sheet along the line (x, z), from y0 to y1 over the ground under each point:
    fence mesh, sheet walls. Both sides when both. Points where the line has them, or every step."""
    from shapely.geometry import LineString
    line = np.asarray(line, float)
    if step:
        L = LineString(line)
        s = np.r_[np.arange(0, L.length, step), L.length]
        line = np.array([L.interpolate(v).coords[0] for v in s])
    g = ground(line[:, 0], line[:, 1])
    out, u = [], 0.
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        L = np.linalg.norm(b - a)
        if L < 1e-6:
            continue
        out += quad((a[0], g[i] + y0, a[1]), (b[0], g[i + 1] + y0, b[1]), (b[0], g[i + 1] + y1, b[1]), (a[0], g[i] + y1, a[1]),
                    tile, uv0=(u, y0 / tile[1]), both=both)
        u += L / tile[0]
    return out


def merge(parts):
    """Parts as one (V, N, U, F)."""
    Vs, Ns, Us, Fs, n = [], [], [], [], 0
    for V, N, U, F in parts:
        Vs.append(V); Ns.append(N); Us.append(U); Fs.append(np.asarray(F) + n); n += len(V)
    return np.concatenate(Vs), np.concatenate(Ns), np.concatenate(Us), np.concatenate(Fs)
