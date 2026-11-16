"""tools/surface.py proved on cases made up for it (no track's): a fork, two roads joined end to end, a
rumble on a road's edge and one set into it, a rumble going on across a joint, a run-off, a road turning
tighter than it is wide, each on rolling ground; a ground of its own mesh, its edge on the world's
kept point for point, and a world with a hole where the ground is left as it was; and old meshes as the
track's are: a T in a mesh's own edges, triangles of no area, a sliver of a hole, a tiny fold. Every case: the surface built, then checked (no open edge,
one height a point, the world covered once); a road's own rows and columns kept but by its edges.

    python tools/test_surface.py [CASE ...]
"""
import sys
import time

import numpy as np

import roads as rd
import surface as sf


def ground_h(x, z):
    return .02 * x + .01 * z + .15 * np.sin(x / 7.) * np.cos(z / 9.)


def road(name, curve, **kw):
    return dict(name=name, curve=[dict(p=p, **{'in': i, 'out': o}, left=l, right=r) for p, i, o, l, r in curve], **kw)


def mesh_region(r, lift=0.):
    P, U, F = rd.mesh(r, lambda x, z: ground_h(x, z) + lift)
    return sf.Region(r['name'], tris=P[F], uvs=U[F])


def kerb_region(name, r, side, s0, s1, width, inset=0., rise=.05):
    """A rumble along a road's edge: rows every .25 m, its profile across, level with the road's edge."""
    S = np.linspace(s0, s1, max(3, int((s1 - s0) / .25) + 1))
    m, l = rd.rows_at(r, S)
    mid, left, wl, wr, at = rd.fine(r)
    sgn = 1. if side == 'left' else -1.
    e = np.interp(S, at, wl if sgn > 0 else wr) - inset
    ts = np.array([0., .3, .55, 1.])
    dy = np.array([0., .026, .052, .05]) * rise / .05
    P, UV = [], []
    for mm, ll, ee, s in zip(m, l, e, S):
        xz = mm + np.outer(sgn * (ee + ts * width), ll)
        y0 = ground_h(*(mm + ll * sgn * ee)) + rd.ROAD
        P.append(np.c_[xz[:, 0], y0 + dy, xz[:, 1]])
        UV.append(np.c_[np.full(len(ts), s * .15), ts * .61])
    P, UV = np.concatenate(P), np.concatenate(UV)
    nc = len(ts)
    F = []
    for j in range(len(S) - 1):
        for c in range(nc - 1):
            a, b = j * nc + c, (j + 1) * nc + c
            F += [(a, b, b + 1), (a, b + 1, a + 1)]
    F = np.array(F)
    return sf.Region(name, tris=P[F], uvs=UV[F])


def ground_region():
    return sf.Region('ground', ground=True, height=ground_h, uv=lambda x, z: np.c_[x / 50., z / 50.], spacing=2.)


def mesh_ground(world, step=5.):
    """A ground of its own triangles (as a near ground already made is): a grid of them over the world."""
    x0, z0, x1, z1 = world
    xs, zs = np.arange(x0, x1 + 1e-9, step), np.arange(z0, z1 + 1e-9, step)
    X, Z = np.meshgrid(xs, zs)
    V = np.c_[X.ravel(), ground_h(X.ravel(), Z.ravel()), Z.ravel()]
    nx = len(xs)
    F = []
    for j in range(len(zs) - 1):
        for i in range(nx - 1):
            a = j * nx + i
            F += [(a, a + nx, a + 1), (a + 1, a + nx, a + nx + 1)]
    F = np.array(F)
    return sf.Region('ground', ground=True, tris=V[F], uvs=V[F][:, :, [0, 2]] / 50., height=ground_h)


def pad(name, x0, z0, x1, z1, lift=.03, t_at=None, flat=(), hole=None, fold=None):
    """A paved pad of its own triangles, as an old mesh is: a grid of 1 m squares; t_at: x where its left
    side has one long edge each row and its right side halves (a T in its own mesh); flat: triangles of no
    area added (three points in a line); hole: (x, z, w) a sliver of a triangle taken out of it, w wide."""
    T = []
    xs, zs = np.arange(x0, x1 + 1e-9, 1.), np.arange(z0, z1 + 1e-9, 1.)
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            a, b, c, d = (xs[i], zs[j]), (xs[i + 1], zs[j]), (xs[i + 1], zs[j + 1]), (xs[i], zs[j + 1])
            if t_at is not None and abs(xs[i + 1] - t_at) < 1e-9:
                m = (xs[i + 1], zs[j] + .5)
                T += [(a, b, m), (a, m, d), (d, m, c)]
            elif t_at is not None and abs(xs[i] - t_at) < 1e-9:
                m = (xs[i], zs[j] + .5)
                T += [(a, b, m), (m, b, c), (m, c, d)]
            else:
                T += [(a, b, c), (a, c, d)]
    if t_at is not None:
        # (the long edge on one side, the halves on the other: the right column's left edges split, the left
        # column's not)
        T2 = []
        for t in T:
            xs_ = [q[0] for q in t]
            if max(xs_) <= t_at + 1e-9 and any(abs(q[0] - t_at) < 1e-9 and abs(q[1] % 1 - .5) < 1e-9 for q in t):
                # (left of it: one triangle over the whole edge instead)
                continue
            T2.append(t)
        for j in range(len(zs) - 1):
            a, b, c, d = (t_at - 1, zs[j]), (t_at, zs[j]), (t_at, zs[j + 1]), (t_at - 1, zs[j + 1])
            T2 += [(a, b, c), (a, c, d)]
        T = T2
    for x, z in flat:
        T.append(((x, z), (x + .3, z), (x + .6, z)))
    if fold is not None:
        # (a tiny triangle over others of its own, as an old mesh's fold: 1.25e-5 square metres)
        x, z = fold
        T.append(((x, z), (x + .005, z), (x, z + .005)))
    T = np.array(T, float)
    if hole is not None:
        hx, hz, w = hole
        keep = ~((np.abs(T[:, :, 0].mean(1) - hx) < .6) & (np.abs(T[:, :, 1].mean(1) - hz) < .6))
        # (the squares round the hole out, laid again round a sliver triangle left empty)
        T = T[keep]
        import shapely
        from shapely.geometry import box as _b, Polygon
        ring = _b(np.floor(hx), np.floor(hz), np.floor(hx) + 1, np.floor(hz) + 1).difference(Polygon([(hx, hz), (hx + w, hz), (hx + w / 2, hz + .5)]))
        import triangle
        ext = np.asarray(ring.exterior.coords)[:-1]
        inn = np.asarray(ring.interiors[0].coords)[:-1]
        V = np.r_[ext, inn]
        S = [(i, (i + 1) % len(ext)) for i in range(len(ext))] + [(len(ext) + i, len(ext) + (i + 1) % len(inn)) for i in range(len(inn))]
        tr = triangle.triangulate(dict(vertices=V, segments=S, holes=[[hx + w / 2, hz + .2]]), 'pQ')
        T = np.r_[T, tr['vertices'][tr['triangles']]]
    P = np.c_[T.reshape(-1, 2)[:, 0], ground_h(*T.reshape(-1, 2).T) + lift, T.reshape(-1, 2)[:, 1]].reshape(-1, 3, 3)
    return sf.Region(name, tris=P, uvs=T / 10.)


A = road('A', [([0., 0.], [0, 0], [10., 0.], 5., 5.), ([60., 0.], [-10., 0.], [0, 0], 5., 5.)])
B = road('B', [([12., 0.], [0, 0], [8., 1.], 5., 5.), ([40., 14.], [-8., -6.], [0, 0], 5., 5.)], joins=['A', None])
C = road('C', [([60., 0.], [0, 0], [10., 0.], 5., 5.), ([90., 12.], [-8., -6.], [0, 0], 5., 5.)])
TIGHT = road('T', [([5., 0.], [0, 0], [0., 2.76], 8., 2.), ([0., 5.], [2.76, 0.], [-2.76, 0.], 8., 2.), ([-5., 0.], [0., 2.76], [0, 0], 8., 2.)])
RUN = dict(name='R', along={'road': 'A', 'side': 'right', 'from': 10., 'to': 40.}, width=2.5, ease=6.)


def cases():
    yield 'fork', [mesh_region(A), mesh_region(B, lift=.004)], (-10, -15, 75, 30)
    yield 'joint', [mesh_region(A), mesh_region(C, lift=.006)], (-10, -15, 100, 30)
    yield 'rumble', [mesh_region(A), kerb_region('K', A, 'left', 10., 30., 1.), kerb_region('KIN', A, 'right', 20., 35., 1., inset=.3)], (-10, -15, 75, 20)
    yield 'rumble across a joint', [mesh_region(A), mesh_region(C), kerb_region('K1', A, 'left', 50., 60., 1.), kerb_region('K2', C, 'left', 0., 10., 1.)], (-10, -15, 100, 30)
    yield 'rumble over its road', [kerb_region('KIN', A, 'right', 20., 35., 1., inset=.3), kerb_region('K', A, 'left', 10., 30., 1.), mesh_region(A)], (-10, -15, 75, 20)
    R = rd.along_road(RUN, {'A': A})
    yield 'run-off', [mesh_region(A), mesh_region(R)], (-10, -15, 75, 20)
    yield 'tight', [mesh_region(TIGHT)], (-20, -15, 20, 20)
    yield 'mesh ground', [mesh_region(A), kerb_region('K', A, 'left', 10., 30., 1.)], (-10, -15, 75, 20)
    yield 'own T', [pad('PAD', 0., 0., 8., 4., t_at=4.)], (-5, -5, 13, 9)
    yield 'no area', [pad('PAD', 0., 0., 8., 4., flat=[(2., 2.), (5., 1.)])], (-5, -5, 13, 9)
    yield 'sliver hole', [pad('PAD', 0., 0., 8., 4., hole=(4.3, 1.2, .0004))], (-5, -5, 13, 9)
    yield 'tiny fold', [pad('PAD', 0., 0., 8., 4., fold=(2.2, 1.3))], (-5, -5, 13, 9)
    yield 'world with a hole', [mesh_region(A)], 'hole'


def main():
    only = sys.argv[1:]
    failed = 0
    for name, regs, world in cases():
        if only and name not in only:
            continue
        t = time.time()
        if world == 'hole':
            # (the world with a hole where the ground is left as it was: its edge kept there too)
            from shapely.geometry import box as _b
            world = _b(-10, -15, 75, 20).difference(_b(20, 10, 40, 15))
            g = mesh_ground((-10, -15, 75, 20))
            keep = ~((g.tris[:, :, 0].mean(1) > 20) & (g.tris[:, :, 0].mean(1) < 40) & (g.tris[:, :, 2].mean(1) > 10) & (g.tris[:, :, 2].mean(1) < 15))
            g = sf.Region('ground', ground=True, tris=g.tris[keep], uvs=g.uvs[keep], height=ground_h)
            name = 'world with a hole'
        else:
            g = mesh_ground(world) if name == 'mesh ground' else ground_region()
        meshes, faces, owner = sf.build(regs + [g], world)
        issues = sf.check(meshes, world)
        if g.tris is not None:
            # (its edge on the world's as it was: no point added there, none moved, none at another height)
            P = meshes['ground'][0]
            import shapely
            from shapely.geometry import box as _b
            Wg = _b(*world) if isinstance(world, tuple) else world
            on = shapely.distance(Wg.boundary, shapely.points(P[:, [0, 2]])) < 1e-6
            V = g.tris.reshape(-1, 3)
            own = dict(zip(sf.key(V[:, [0, 2]]).tolist(), V[:, 1].tolist()))
            for p_ in P[on]:
                k_ = int(sf.key(p_[[0, 2]]))
                if k_ not in own or abs(own[k_] - p_[1]) > 1e-9:
                    issues.append(('world edge point not its own', p_[0], p_[2], 0.))
        kept = []
        for r in regs:
            P, UV, F = meshes[r.name]
            own = {tuple(sorted(sf.key(t_[:, [0, 2]]).tolist())) for t_ in r.tris}
            got = {tuple(sorted(sf.key(P[f][:, [0, 2]]).tolist())) for f in F}
            kept.append('%s %d%% of its own faces kept' % (r.name, 100 * len(own & got) / max(len(own), 1)))
        ok = not issues
        failed += not ok
        print('%-22s %s  (%.1fs; %s)' % (name, 'ok' if ok else '%d PROBLEMS' % len(issues), time.time() - t, '; '.join(kept)))
        for what, x, z, how in issues[:8]:
            print('    %s at (%.2f, %.2f): %.4f' % (what, x, z, how))
    print('all ok' if not failed else '%d cases failed' % failed)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
