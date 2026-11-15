"""Close the holes in the floor tools/check_holes.py finds: from every open edge of a floor mesh that
nothing meets (no floor at its height just past it, nothing over it, no other mesh sharing it), a
skirt straight down SKIRT, in the mesh's own material, facing out from it. A crack between two roads
shows their skirts, a kerb's open side is closed, never the sky.

    python tools/seal_holes.py

Each model's skirts are a mesh per floor mesh, SKIRT_<its name>, made again each time.

First the seams are welded: where a point of any floor lies on an open edge of another face (a
T: one side of a seam has a point where the other has none, as a floor kerb's grooves' ends along
its flat band, or a patch's edge along a road's), the edge is split there, its new point where that
point is, so both sides share it. Laid on the ground point by point, such an edge goes straight
past the point while the point goes up or down: a slit the sky shows through, which the probe past
the edge (there is floor just past it) does not find. ON (metres from the edge across, xz) and
UPDOWN (metres up or down from it) say what lies on an edge.
"""
import glob, os, re

import numpy as np

import build_paths as bp
import check_holes as ch
import gltf_source
import join_paddock as jp
import level_ground as lg

SKIRT = .25                  # metres a skirt goes down from its edge
ON, UPDOWN = .0015, .05      # metres: a point this near an edge across (x, z), and up or down, lies on it
GAP_WIN = 6.                 # metres round a gap the floors are looked at to find it
NARROW = (.04, .12)          # metres past a floor's open edge a slit narrower than check_holes's WIDE is looked for
SPLIT = '_W'                 # (the second part of a mesh welding has grown past the points one takes)


def fill_gaps(models):
    """The gaps in the floor (where no floor is at all, round which the floors' edges run: the ground
    cut away where a road was and is no more, a corner nothing covers) filled with ground, as high as
    the floors round them, in the near ground's material, as 1GRASS_FILL in the track's model. Found
    as tools/check_holes.py finds a hole: past an open edge of a floor, WIDE out, no floor. How many
    faces."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from scipy.spatial import cKDTree
    track = models[bp.TRACK]
    fl = [(f, n, m) for f, k in models.items() for n, m in bp.meshes(k)
          if len(n['faces']) and n['name'].upper().startswith(ch.FLOOR) and not n['name'].startswith('SKIRT_')]
    T = np.concatenate([bp.triangles(n, m) for _, n, m in fl])
    S = bp.Surface(T)
    lo, hi = T[:, :, [0, 2]].min((0, 1)) + 2., T[:, :, [0, 2]].max((0, 1)) - 2.
    pts = []
    done = ch.merged()
    for f_, n, m in fl:
        if (os.path.basename(f_), n['name']) in done:
            continue
        A, B, nrm = ch.open_edges(bp.triangles(n, m))
        if not len(A):
            continue
        P = (A + B) / 2
        # (just past it too: a slit narrower than WIDE between two floors, the probe WIDE out on the far one)
        for d in NARROW + (ch.WIDE,):
            Q = P[:, [0, 2]] + nrm * d
            bad = np.isnan(S.ys(Q)) & (Q >= lo).all(1) & (Q <= hi).all(1)
            pts.append(Q[bad])
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    if not len(pts):
        return 0
    c = T[:, :, [0, 2]].mean(1)
    tree = cKDTree(c)
    gaps = []
    done = np.zeros(len(pts), bool)
    for i, q in enumerate(pts):
        if done[i]:
            continue
        win = box(q[0] - GAP_WIN, q[1] - GAP_WIN, q[0] + GAP_WIN, q[1] + GAP_WIN)
        near = tree.query_ball_point(q, GAP_WIN * 1.5)
        U = unary_union([Polygon(t).buffer(1e-3) for t in T[near][:, :, [0, 2]] if Polygon(t).area > 1e-6])
        # (all of it with no floor, in the map: long slits run on past the window, the next one's takes them on)
        g = win.intersection(box(lo[0], lo[1], hi[0], hi[1])).difference(U)
        for part in getattr(g, 'geoms', [g]):
            if part.is_empty or part.area < 1e-4:
                continue
            gaps.append(part)
        done |= np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1]) < GAP_WIN * .5
    if not gaps:
        return 0
    G = unary_union(gaps).buffer(.01)
    import restyle_paddock as rp
    tris = rp.fill(G, most=.5)
    if not tris:
        return 0
    # (its faces sharing their points, as the triangulation made them)
    key = np.round(np.concatenate(tris) / 1e-5).astype(np.int64)
    _, first, inv = np.unique(key[:, 0] * 10 ** 9 + key[:, 1], return_index=True, return_inverse=True)
    P = np.concatenate(tris)[first]
    Fi = inv.reshape(-1, 3)
    V3 = T.reshape(-1, 3)
    vt = cKDTree(V3[:, [0, 2]])
    dd, ii = vt.query(P, k=6)
    w = 1. / np.maximum(dd, .02) ** 2
    y = (V3[ii, 1] * w).sum(1) / w.sum(1)
    g0 = next(n for n, _ in bp.meshes(track) if n['name'] == bp.GROUND)
    Vg = g0['verts'].astype(float)
    M, *_ = np.linalg.lstsq(np.c_[Vg[:, 0], Vg[:, 2], np.ones(len(Vg))], Vg[:, 6:8], rcond=None)
    V = np.zeros((len(P), g0['verts'].shape[1]))
    V[:, 0], V[:, 1], V[:, 2] = P[:, 0], y, P[:, 1]
    V[:, 4] = 1.
    V[:, 6:8] = np.c_[P, np.ones(len(P))] @ M
    F = Fi
    up = np.cross(V[F[:, 1], :3] - V[F[:, 0], :3], V[F[:, 2], :3] - V[F[:, 0], :3])[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)
    if V.shape[1] > 8:
        V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
    # (16-bit indices while it has no more points than they reach; the model's loader cuts a bigger one in pieces)
    node = dict(g0, name='1GRASS_FILL', children=[], verts=V.astype(g0['verts'].dtype), faces=F.astype('<u2' if len(V) < 65536 else '<u4'))
    bp.sphere(node)
    track['root']['children'] = [x for x in track['root']['children'] if x['name'] != '1GRASS_FILL'] + [node]
    return len(F)


def put(c, n, more):
    """Nodes put after node n wherever it is under c."""
    for j, ch_ in enumerate(c['children']):
        if ch_ is n:
            c['children'][j + 1:j + 1] = more
            return True
        if put(ch_, n, more):
            return True
    return False


def open_faces(F, P):
    """A mesh's faces' edges no other of its faces shares (by where their ends are, to a millimetre,
    x, z): which face, and which of its edges (i, from its point i to i + 1)."""
    key = np.round(P[:, [0, 2]] * 1000).astype(np.int64)
    k = key[:, 0] * 4000003 + key[:, 1]
    a = np.r_[k[F[:, 0]], k[F[:, 1]], k[F[:, 2]]]
    b = np.r_[k[F[:, 1]], k[F[:, 2]], k[F[:, 0]]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, inv, cnt = np.unique(np.c_[lo, hi], axis=0, return_inverse=True, return_counts=True)
    single = (cnt[inv.ravel()] == 1) & (lo != hi)
    n = len(F)
    return np.tile(np.arange(n), 3)[single], np.repeat(np.arange(3), n)[single]


def weld(floors, models):
    """The Ts in the floors' seams welded (see above): floors, (model, mesh) pairs, each mesh's faces
    with points lying on an edge of theirs no other of its faces shares split there; a mesh grown past
    the points one takes cut in two, the other part beside it (SPLIT and a number after its name, in
    the Custom Shaders Patch's grass too, if it is). How many points."""
    from scipy.spatial import cKDTree
    pool = np.unique(np.concatenate([n['verts'][:, :3] for _, n in floors]).astype(np.float32), axis=0).astype(float)
    tree = cKDTree(pool[:, [0, 2]])
    total = 0
    split = {}
    for f_, n in floors:
        V = n['verts'].astype(float)
        F = n['faces'].astype(int)
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        fi, ei = open_faces(F, V)
        if not len(fi):
            continue
        A, B = V[F[fi, ei], :3], V[F[fi, (ei + 1) % 3], :3]
        d = (B - A)[:, [0, 2]]
        L = np.linalg.norm(d, axis=1)
        mid = (A + B)[:, [0, 2]] / 2
        on = {}
        for j in np.where(L > .01)[0]:
            near = tree.query_ball_point(mid[j], L[j] / 2 + ON)
            if not near:
                continue
            Q = pool[near]
            t = ((Q[:, [0, 2]] - A[j, [0, 2]]) @ d[j]) / L[j] ** 2
            off = np.abs((Q[:, 0] - A[j, 0]) * d[j, 1] - (Q[:, 2] - A[j, 2]) * d[j, 0]) / L[j]
            dy = Q[:, 1] - (A[j, 1] + (B[j, 1] - A[j, 1]) * t)
            ok = (off < ON) & (t * L[j] > .003) & ((1 - t) * L[j] > .003) & (np.abs(dy) < UPDOWN)
            if ok.any():
                o = np.argsort(t[ok])
                on.setdefault(fi[j], {})[ei[j]] = (t[ok][o], Q[ok][o])
        if not on:
            continue
        V_, F_, keep = list(V), [], np.ones(len(F), bool)
        made = 0
        for f, edges in on.items():
            keep[f] = False
            ring = []
            for e in range(3):
                a, b = F[f, e], F[f, (e + 1) % 3]
                ring.append(a)
                for t, q in zip(*edges.get(e, ((), ()))):
                    v = V[a] + (V[b] - V[a]) * t
                    v[:3] = q
                    V_.append(v)
                    ring.append(len(V_) - 1)
                    made += 1
            if len(edges) == 1:
                # (all on one edge: a fan from the point across from it)
                e = next(iter(edges))
                r = ring.index(F[f, (e + 2) % 3])
                ring = ring[r:] + ring[:r]
                F_ += [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]
            else:
                c = V[F[f]].mean(0)
                V_.append(c)
                ci = len(V_) - 1
                F_ += [(ci, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        V2 = np.array(V_)
        V2[:, 3:6] /= np.maximum(np.linalg.norm(V2[:, 3:6], axis=1, keepdims=True), 1e-12)
        F2 = np.r_[F[keep], np.array(F_, int).reshape(-1, 3)]
        parts = jp.chunks(V2, F2)
        n['verts'], n['faces'] = parts[0][0].astype(n['verts'].dtype), parts[0][1].astype('<u2')
        bp.sphere(n)
        if len(parts) > 1:
            more = []
            for i, (v, f2) in enumerate(parts[1:]):
                node = dict(n, name='%s%s%d' % (n['name'], SPLIT, i + 1), children=[],
                            verts=v.astype(n['verts'].dtype), faces=f2.astype('<u2'))
                bp.sphere(node)
                more.append(node)
            put(models[f_]['root'], n, more)
            split[n['name']] = [x['name'] for x in more]
        total += made
    if split:
        text = open(lg.CSP, newline='').read()
        m_ = re.search(r'(?m)^GRASS_MESHES *= *(.*)$', text)
        have = [x.strip() for x in m_.group(1).split(',') if x.strip()]
        have += [x for a in have if a in split for x in split[a] if x not in have]
        open(lg.CSP, 'w', newline='').write(text[:m_.start(1)] + ', '.join(have) + text[m_.end(1):])
    return total


def main():
    models = {}
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        fn = os.path.basename(f)
        if fn in ch.SKIP or fn.startswith('vp_'):
            continue
        k = gltf_source.load_kn5(f)
        k['root']['children'] = [c for c in k['root']['children'] if not c['name'].startswith('SKIRT_')]
        models[f] = k
    done = ch.merged()                    # (the merge's own: proved there, their seams point for point)
    floors = [(f, n) for f, k in models.items() for n, m in bp.meshes(k)
              if len(n['faces']) and n['name'].upper().startswith(ch.FLOOR) and np.allclose(m, np.eye(4))
              and (os.path.basename(f), n['name']) not in done]
    welded = weld(floors, models)
    print('%d faces of ground over the gaps in the floor' % fill_gaps(models))
    for f in {f for f, _ in floors}:
        bp.save(models[f], f)
    print("%d points put in the floors' seams where the other side had one" % welded)
    S = bp.Surface(np.concatenate([t for _, _, t in ch.floors()]))
    shared = ch.all_edges()
    total = 0
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        fn = os.path.basename(f)
        if fn in ch.SKIP or fn.startswith('vp_'):
            continue
        k = gltf_source.load_kn5(f)
        k['root']['children'] = [c for c in k['root']['children'] if not c['name'].startswith('SKIRT_')]
        made = 0
        for n, m in list(bp.meshes(k)):
            if not len(n['faces']) or not n['name'].upper().startswith(ch.FLOOR) or not np.allclose(m, np.eye(4)):
                continue
            if (fn, n['name']) in done:
                continue
            T = bp.triangles(n, m)
            A, B, nrm = ch.open_edges(T)
            if not len(A):
                continue
            a, b = ch.edge_keys(A, B)
            closed = np.array([len(shared.get((x, y), ())) > 1 for x, y in zip(a.tolist(), b.tolist())], bool)
            bad = np.zeros(len(A), bool)
            for t in (.25, .5, .75):
                P = A + (B - A) * t
                Q = P[:, [0, 2]] + nrm * ch.STEP
                bad |= ~S.near(Q, P[:, 1] + (ch.ABOVE - ch.BELOW) / 2, (ch.ABOVE + ch.BELOW) / 2)
            bad &= ~closed
            if not bad.any():
                continue
            A, B, nrm = A[bad], B[bad], nrm[bad]
            N3 = np.c_[nrm[:, 0], np.zeros(len(nrm)), nrm[:, 1]]
            V = np.zeros((4 * len(A), n['verts'].shape[1]))
            down = np.array([0., SKIRT, 0.])
            V[0::4, :3], V[1::4, :3], V[2::4, :3], V[3::4, :3] = A, B, B - down, A - down
            for j in range(4):
                V[j::4, 3:6] = N3
            # (mapped as its edge is, top to bottom: it shows its floor's own colour there, not some
            # other part of its texture, as the ground's photo's far corner)
            from scipy.spatial import cKDTree
            own = cKDTree(np.asarray(T.reshape(-1, 3)))
            Vn = n['verts'].astype(float)
            uv = Vn[n['faces'].astype(int)[(n['faces'][:, 0] != n['faces'][:, 1]) & (n['faces'][:, 1] != n['faces'][:, 2])
                                         & (n['faces'][:, 0] != n['faces'][:, 2])].reshape(-1)][:, 6:8]
            ua, ub = uv[own.query(A)[1]], uv[own.query(B)[1]]
            V[0::4, 6:8], V[1::4, 6:8], V[2::4, 6:8], V[3::4, 6:8] = ua, ub, ub, ua
            i = np.arange(len(A)) * 4
            F = np.r_[np.c_[i, i + 1, i + 2], np.c_[i, i + 2, i + 3]]
            # (each facing out from the floor it hangs from)
            P = V[:, :3]
            fn_ = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
            out = (fn_ * V[F[:, 0], 3:6]).sum(1) < 0
            F[out] = F[out][:, ::-1]
            # (lit as its floor is, not as a wall facing out, which the sky's light shows pale blue)
            na, nb = Vn[n['faces'].astype(int)[(n['faces'][:, 0] != n['faces'][:, 1]) & (n['faces'][:, 1] != n['faces'][:, 2])
                                               & (n['faces'][:, 0] != n['faces'][:, 2])].reshape(-1)][:, 3:6][own.query(A)[1]], None
            nb = Vn[n['faces'].astype(int)[(n['faces'][:, 0] != n['faces'][:, 1]) & (n['faces'][:, 1] != n['faces'][:, 2])
                                           & (n['faces'][:, 0] != n['faces'][:, 2])].reshape(-1)][:, 3:6][own.query(B)[1]]
            V[0::4, 3:6], V[1::4, 3:6], V[2::4, 3:6], V[3::4, 3:6] = na, nb, nb, na
            if V.shape[1] > 8:
                V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
            for c0 in range(0, len(F), 20000):
                f_ = F[c0:c0 + 20000]
                used = np.unique(f_)
                remap = np.full(len(V), -1)
                remap[used] = np.arange(len(used))
                node = dict(n, name='SKIRT_%s%s' % (n['name'], '' if c0 == 0 else '_%d' % (c0 // 20000)), children=[],
                            verts=V[used].astype(n['verts'].dtype), faces=remap[f_].astype('<u2'))
                bp.sphere(node)
                k['root']['children'].append(node)
            made += len(A)
        if made:
            bp.save(k, f)
            print('%s: %d open edges skirted' % (fn, made))
        total += made
    print('%d open edges skirted' % total)


if __name__ == '__main__':
    main()
