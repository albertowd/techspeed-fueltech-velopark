"""Find the holes in the floor: where the ground, the roads, the kerbs and the floors that are driven or
walked on leave a gap the sky shows through.

    python tools/check_holes.py [--near X Z R]

Every edge of a floor mesh that no other face of it shares is probed just past it (STEP out from it,
at three points along it): some floor (any of them) must be there, no more than BELOW under the
edge's height or ABOVE over it, or it is a hole or a crack (two meshes meeting a little apart, or one over the other, open between).
The floor's outer edge, past the far ground, is left out, as are edges against buildings' walls
(WALLS: the floors under them). Prints the holes found, by where they are; exits 1 while any is found.
"""
import glob, os, sys

import numpy as np

import build_paths as bp
import gltf_source

PAINT_NEAR = .03             # metres: paint within this of a road is on it
WIDE = .3                    # metres past an edge only a skirt closes some floor must be (else a hole, not a crack)
STEP, BELOW, ABOVE = .003, .015, .5  # metres past an edge probed; how far under its height, or over, a floor there may be
FLOOR = ('1ROAD', '1KERB', '1GRASS', 'GRASS', 'KERB', 'FAR_GROUND', 'GROUND')
SKIP = ('velopark_trees.glb', 'velopark_karts.glb', 'velopark_tyres.glb', 'velopark_lines.glb')
CELL = 5.                    # metres: holes counted by the square they are in


def all_edges():
    """Every edge of every mesh drawn (the floors' models' and the buildings'), by its ends to a
    centimetre: a floor's edge one of them shares is closed (a kerb's side, a step's face, a seam)."""
    keys = {}
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        if os.path.basename(f) in SKIP or os.path.basename(f).startswith('vp_'):
            continue
        k = gltf_source.load_kn5(f)
        for n, m in bp.meshes(k):
            if not len(n['faces']) or n['name'].startswith(('1WALL', 'AC_')):
                continue
            T = bp.triangles(n, m)
            for i in range(3):
                a, b = edge_keys(T[:, i], T[:, (i + 1) % 3])
                for x in set(zip(a.tolist(), b.tolist())):
                    keys.setdefault(x, set()).add((os.path.basename(f), n['name']))
    return keys


def edge_keys(A, B):
    """Edges' ends as keys, to a centimetre, the lower first."""
    ka = np.round(A * 100).astype(np.int64)
    kb = np.round(B * 100).astype(np.int64)
    ka = (ka[:, 0] * 2000003 + ka[:, 1]) * 2000003 + ka[:, 2]
    kb = (kb[:, 0] * 2000003 + kb[:, 1]) * 2000003 + kb[:, 2]
    return np.minimum(ka, kb), np.maximum(ka, kb)


def merged():
    """The floor meshes tools/merge_floor.py made and proved (model, name): their own edges not probed again
    (they meet what is round them point for point); still floor for the others' probes."""
    import json
    p = os.path.join(bp.MODELS, 'merged_floors.json')
    if not os.path.exists(p):
        return set()
    return {(f, n) for f, names in json.load(open(p)).items() for n in names}


def floors():
    """Every floor mesh's triangles, by model and name: the terrain's models only (the roads, rumbles and
    ground, tools/merge_floor.py's); what stands on it is not driven on, at most run into."""
    import merge_floor as mf
    out = []
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        if os.path.basename(f) in SKIP or os.path.basename(f).startswith('vp_') or os.path.basename(f) not in mf.MODELS:
            continue
        k = gltf_source.load_kn5(f)
        for n, m in bp.meshes(k):
            if len(n['faces']) and n['name'].upper().startswith(FLOOR) and n.get('renderable', 1) and n.get('visible', 1):
                out.append((os.path.basename(f), n['name'], bp.triangles(n, m)))
    return out


def open_edges(T):
    """A mesh's edges no other of its faces shares (by where their ends are, to a millimetre): their ends,
    and a direction away from their face, x, z."""
    key = np.round(T[:, :, [0, 2]] * 1000).astype(np.int64)
    k = key[..., 0] * 4000003 + key[..., 1]
    a = np.r_[k[:, 0], k[:, 1], k[:, 2]]
    b = np.r_[k[:, 1], k[:, 2], k[:, 0]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, inv, cnt = np.unique(np.c_[lo, hi], axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    single = (cnt[inv] == 1) & (lo != hi)
    n = len(T)
    i0 = np.r_[np.zeros(n, int), np.ones(n, int), np.full(n, 2)]
    i1 = (i0 + 1) % 3
    f = np.tile(np.arange(n), 3)
    A, B = T[f, i0], T[f, i1]
    C = T[f, (i0 + 2) % 3]
    A, B, C = A[single], B[single], C[single]
    d = (B - A)[:, [0, 2]]
    L = np.linalg.norm(d, axis=1)
    keep = L > .01
    A, B, C, d, L = A[keep], B[keep], C[keep], d[keep], L[keep]
    nrm = np.c_[d[:, 1], -d[:, 0]] / L[:, None]
    mid = (A + B)[:, [0, 2]] / 2
    flip = ((C[:, [0, 2]] - mid) * nrm).sum(1) > 0
    nrm[flip] *= -1
    return A, B, nrm


def holes(near=None):
    """The holes in the floor: points just past open edges with no floor there, (x, y, z), and the meshes
    the edges are of."""
    fl = floors()
    done = merged()
    if all((f, nm) in done for f, nm, _ in fl):
        return np.zeros((0, 3)), []                 # (all of it the merge's: proved there, seam by seam)
    allT = np.concatenate([t for _, _, t in fl])
    lo, hi = allT[:, :, [0, 2]].min((0, 1)) + 2., allT[:, :, [0, 2]].max((0, 1)) - 2.
    S = bp.Surface(allT)
    shared = all_edges()
    out, names = [], []
    done = merged()
    for f, nm, T in fl:
        if (f, nm) in done:
            continue
        A, B, nrm = open_edges(T)
        if not len(A):
            continue
        a, b = edge_keys(A, B)
        owners = [shared.get((x, y), {(f, nm)}) for x, y in zip(a.tolist(), b.tolist())]
        closed = np.array([len(o) > 1 or (f, nm) not in o for o in owners], bool)
        # (an edge closed only by a skirt (tools/seal_holes.py) closes a crack, not a hole: some floor
        # must be within WIDE past it, at any height near its own)
        skirted = np.array([c and all(o_[1].startswith('SKIRT_') for o_ in o if o_ != (f, nm)) for c, o in zip(closed, owners)], bool)
        if skirted.any():
            P = (A[skirted] + B[skirted]) / 2
            Q = P[:, [0, 2]] + nrm[skirted] * WIDE
            open_ = ~S.near(Q, P[:, 1], 1.) & (Q >= lo).all(1) & (Q <= hi).all(1)
            if near is not None:
                open_ &= np.hypot(Q[:, 0] - near[0], Q[:, 1] - near[1]) < near[2]
            out.append(np.c_[Q[open_, 0], P[open_, 1], Q[open_, 1]])
            names += ['%s:%s (wide)' % (f, nm)] * int(open_.sum())
        A, B, nrm = A[~closed], B[~closed], nrm[~closed]
        for t in (.25, .5, .75):
            P = A + (B - A) * t
            Q = P[:, [0, 2]] + nrm * STEP
            inside = (Q >= lo).all(1) & (Q <= hi).all(1)
            if near is not None:
                inside &= np.hypot(Q[:, 0] - near[0], Q[:, 1] - near[1]) < near[2]
            Q, Y, Pw = Q[inside], P[inside, 1], P[inside]
            if not len(Q):
                continue
            ok = S.near(Q, Y + (ABOVE - BELOW) / 2, (ABOVE + BELOW) / 2)
            bad = ~ok
            out.append(np.c_[Q[bad, 0], Y[bad], Q[bad, 1]])
            names += ['%s:%s' % (f, nm)] * int(bad.sum())
    return (np.concatenate(out) if out else np.zeros((0, 3))), names


_ROADS = []


def road_tree():
    """Every road's and kerb's faces (1ROAD, 1KERB, every model's), x, z, in a tree for the nearest."""
    if not _ROADS:
        import shapely
        from shapely.strtree import STRtree
        T = []
        for f in sorted(glob.glob(os.path.join(bp.MODELS, 'velopark*.glb'))):
            for n, m in bp.meshes(gltf_source.load_kn5(f)):
                if n['name'].startswith(('1ROAD', '1KERB')) and len(n['faces']):
                    T.append(bp.triangles(n, m)[:, :, [0, 2]])
        T = np.concatenate(T)
        polys = shapely.polygons(np.concatenate([T, T[:, :1]], axis=1))
        polys = polys[shapely.area(polys) > 1e-9]
        _ROADS.append((STRtree(polys), polys))
    return _ROADS[0]


def off_road(P):
    """Which of points P (x, z) are farther than PAINT_NEAR from every road and kerb."""
    import shapely
    tree, polys = road_tree()
    pts = shapely.points(P)
    _, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    return d > PAINT_NEAR


def paint_samples(T):
    """Where the paint of faces T (n, 3, 3) is looked at: their middles and their corners a fifth in,
    (4n, 3), face by face (the face of each: index % n)."""
    c = T.mean(1)
    return np.concatenate([c] + [T[:, i] + (c - T[:, i]) * .2 for i in range(3)])


def paint_off_road():
    """The paint (every mesh of the lines' model) not on a road or kerb (farther than PAINT_NEAR from
    every one): points (x, y, z) of it (see paint_samples), and the meshes they are of."""
    out, names = [], []
    for n, m in bp.meshes(gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_lines.glb'))):
        if not len(n['faces']):
            continue
        P = paint_samples(bp.triangles(n, m))
        off = off_road(P[:, [0, 2]])
        out.append(P[off])
        names += [n['name']] * int(off.sum())
    return (np.concatenate(out) if out else np.zeros((0, 3))), names


def main():
    args = sys.argv[1:]
    near = tuple(float(a) for a in args[args.index('--near') + 1:args.index('--near') + 4]) if '--near' in args else None
    import collections
    Q, qn = paint_off_road()
    if len(Q):
        print('%d points of paint off the road' % len(Q))
        print('  most, by mesh:', ', '.join('%s %d' % (n, c) for n, c in collections.Counter(qn).most_common(15)))
    P, names = holes(near)
    if not len(P):
        print('no holes in the floor')
        if len(Q):
            sys.exit(1)
        return
    import collections
    print('%d points of holes in the floor' % len(P))
    cells = collections.Counter((int(np.floor(x / CELL) * CELL), int(np.floor(z / CELL) * CELL)) for x, _, z in P)
    by = collections.Counter(names)
    print('  most, by where (x, z):', ', '.join('(%d, %d) %d' % (x, z, c) for (x, z), c in cells.most_common(25)))
    print('  most, by mesh:', ', '.join('%s %d' % (n, c) for n, c in by.most_common(15)))
    sys.exit(1)


if __name__ == '__main__':
    main()
