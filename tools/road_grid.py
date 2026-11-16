"""The roads read as the grids they are made of: rows across them, one after another along them.

    python tools/road_grid.py [MODEL.glb ...]

The track's roads (as Race Track Builder made them) are grids of quads, each split in two along a
diagonal: so many points across, row after row along the road. Every change to a road keeps it so
(widened: its edge columns out; carried on: rows more; a fork: its end row along the other's edge,
sharing its points), so a road is always one even mesh, the paint along its columns.

grid(n, m) reads a mesh as such a grid: its points welded where they lie together (to WELD), its
triangles paired into quads across their longest edges, the quads walked row by row from one end.
It gives the grid's points (rows, columns, 3: x, y, z) and the mesh's point at each (for its
mapping and normals), or why the mesh is not a grid. Run, it lists every road of the models given
(the track's, the paddock's, the kerbs', the run-offs' by default) as a grid or not.
"""
import os, sys

import numpy as np

import build_paths as bp
import gltf_source

WELD = .002           # metres: points this near are one


def weld(P):
    """Points welded where they lie within WELD: each one's welded index, and the welded points."""
    key = np.round(P / WELD).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return inv.ravel(), P[first]


def quads(F, P):
    """Triangles F (welded indices) paired into quads, those making the squarest quads first (every
    corner nearest a right angle), each across the edge the two share: (q, 4) corners in turn, and
    the triangles left unpaired."""
    edges = {}
    for t, f in enumerate(F):
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edges.setdefault((min(a, b), max(a, b)), []).append(t)
    cand = []
    for (a, b), ts in edges.items():
        if len(ts) != 2:
            continue
        t, u = ts
        c = [v for v in F[t] if v not in (a, b)]
        d = [v for v in F[u] if v not in (a, b)]
        if len(c) != 1 or len(d) != 1:
            continue
        ring = [a, c[0], b, d[0]]
        X = P[ring][:, [0, 2]]
        dev = 0.
        convex = True
        sgn = None
        for i in range(4):
            e1, e2 = X[i - 1] - X[i], X[(i + 1) % 4] - X[i]
            cr = e1[0] * e2[1] - e1[1] * e2[0]
            sgn = sgn if sgn is not None else np.sign(cr)
            convex &= np.sign(cr) == sgn and cr != 0
            cos = (e1 @ e2) / max(np.linalg.norm(e1) * np.linalg.norm(e2), 1e-12)
            dev += abs(cos)
        if convex:
            cand.append((dev, t, u, ring))
    cand.sort(key=lambda x: x[0])
    used = np.zeros(len(F), bool)
    Q = []
    for dev, t, u, ring in cand:
        if used[t] or used[u]:
            continue
        used[t] = used[u] = True
        Q.append(ring)
    return np.array(Q, int).reshape(-1, 4), np.nonzero(~used)[0]


def outline(n, m):
    """Mesh n's (placed by m) outline as one ring of welded points (x, y, z), or (None, why)."""
    F = n['faces'].astype(int)
    W = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
    inv, P = weld(W)
    Fw = inv[F]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    # (its top only: not the skirts down its sides)
    nn = np.cross(P[Fw[:, 1]] - P[Fw[:, 0]], P[Fw[:, 2]] - P[Fw[:, 0]])
    Fw = Fw[np.abs(nn[:, 1]) > .5 * np.maximum(np.linalg.norm(nn, axis=1), 1e-12)]
    # (its outline as edges each face has on its own, the way the face runs round: followed from
    # each to the next out of where it ends, turning the least where more go on (a pinch), the
    # longest loop of them its outline)
    cnt = {}
    for f in Fw:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            e = (min(a, b), max(a, b))
            cnt[e] = cnt.get(e, 0) + 1
    out = {}
    for f in Fw:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if cnt[(min(a, b), max(a, b))] == 1:
                out.setdefault(a, []).append(b)
    if not out:
        return None, 'no outline (a ring of road)'
    X = P[:, [0, 2]]
    left = {(a, b) for a, bs in out.items() for b in bs}
    loops = []
    while left:
        a, b = next(iter(left))
        loop = [a]
        left.discard((a, b))
        while b != loop[0]:
            loop.append(b)
            nx = [c for c in out.get(b, []) if (b, c) in left]
            if not nx:
                break
            if len(nx) > 1:
                d0 = X[b] - X[loop[-2]]
                ang = lambda c: np.arctan2(*(lambda d1: (d0[0] * d1[1] - d0[1] * d1[0], d0 @ d1))(X[c] - X[b]))
                nx.sort(key=ang)
            left.discard((b, nx[0]))
            b = nx[0]
        loops.append(loop)
    ring = max(loops, key=lambda l: np.linalg.norm(np.diff(X[l + l[:1]], axis=0), axis=1).sum())
    return P[ring], None


def grid(n, m, corners=4, sharpest=False):
    """Mesh n (placed by m) as a strip: its outline split at its four sharpest corners into its two
    long sides and its two ends, its sides paired point by point into rows across it (resampled to
    as many points as the longer has, by how far along each, when they have not as many). (rows: (r,
    2, 3) its rows' ends, x, y, z, from one end to the other; exact: whether its sides had as many
    points), or (None, why)."""
    R, why = outline(n, m)
    if R is None:
        return None, why
    X = R[:, [0, 2]]
    k = len(X)
    turn = np.zeros(k)
    for i in range(k):
        a, b = X[i] - X[i - 1], X[(i + 1) % k] - X[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la > 1e-9 and lb > 1e-9:
            turn[i] = np.degrees(np.arccos(np.clip(a @ b / la / lb, -1., 1.)))
    # (its ends: the four corners, among its sharpest turns, that leave two short pieces of its
    # outline across it and two long ones along it, its sides, as many points on each the better)
    import itertools
    cand = [int(i) for i in np.argsort(-turn)[:12] if turn[i] >= 30.]
    if len(cand) < 4:
        return None, 'not four corners (its sharpest turns %s deg)' % np.round(np.sort(turn)[-5:], 0)
    seg = np.linalg.norm(np.diff(np.r_[X, X[:1]], axis=0), axis=1)
    cum = np.r_[0., np.cumsum(seg)]

    def span(i, j):
        return cum[j] - cum[i] if j >= i else cum[-1] - cum[i] + cum[j]
    best = None
    sharp4 = sorted(int(i) for i in np.argsort(-turn)[:4])
    if sharpest and turn[sharp4].min() >= 60.:
        cand = sharp4                                           # (its four corners plain to see: those)
    for cs in itertools.combinations(sorted(cand), 4):
        Ls = [span(cs[i], cs[(i + 1) % 4]) for i in range(4)]
        Ns = [(cs[(i + 1) % 4] - cs[i]) % k for i in range(4)]
        for o in (0, 1):
            ends, sides = Ls[o + 1] + Ls[(o + 3) % 4], Ls[o] + Ls[o + 2]
            score = ends / max(sides, 1e-9) + (0. if Ns[o] == Ns[o + 2] else .05)
            if best is None or score < best[0]:
                best = (score, cs, o)
    _, cs, o = best
    chains = [[c % k for c in range(cs[i], (cs[(i + 1) % 4] if cs[(i + 1) % 4] > cs[i] else cs[(i + 1) % 4] + k) + 1)] for i in range(4)]
    left, right = chains[o], chains[o + 2][::-1]
    A, B = R[left], R[right]
    exact = len(A) == len(B)
    if not exact:
        N = max(len(A), len(B))
        def resample(C):
            d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(C[:, [0, 2]], axis=0), axis=1))]
            t = np.linspace(0., d[-1], N)
            return np.c_[[np.interp(t, d, C[:, i]) for i in range(3)]].T
        A, B = resample(A), resample(B)
    return (np.stack([A, B], axis=1), exact), None


def main():
    files = sys.argv[1:] or [os.path.join(bp.MODELS, f) for f in ('velopark.glb', 'velopark_paddock.glb',
                                                                    'velopark_kerbs.glb', 'velopark_runoffs.glb')]
    for f in files:
        k = gltf_source.load_kn5(f)
        print(os.path.basename(f))
        for n, m in bp.meshes(k):
            if not n['name'].startswith(('1ROAD', '1KERB')) or not len(n['faces']):
                continue
            g, why = grid(n, m)
            if g is None:
                print('  %-40s not a strip: %s' % (n['name'], why))
            else:
                rows, exact = g
                w = np.linalg.norm(rows[:, 1, [0, 2]] - rows[:, 0, [0, 2]], axis=1)
                print('  %-40s strip: %d rows%s, %.1f to %.1f m wide' % (n['name'], len(rows), '' if exact else ' (resampled)', w.min(), w.max()))


if __name__ == '__main__':
    main()
