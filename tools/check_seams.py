"""The seams between the floors checked: wherever two floor meshes meet (road to road, road to rumble,
rumble to ground, road to ground), they must meet edge to edge, point for point, at one height.

    python tools/check_seams.py                 the models in source/models (as the build leaves them)
    python tools/check_seams.py --kn5 DIR       the kn5 files in DIR (a built or installed track)

An open edge of the floors (one no other face of any floor shares, its ends to a millimetre) with another
open edge NEAR it outward is a bad seam: a crack (the two apart by more than GAP across), a step (more
than RISE up or down where they are side by side), or two edges side by side not welded (one's points not
the other's: a T the seal should have closed). Each counted by where it is (CELL metres) and the meshes on
its two sides. The build fails on any (tools/check_holes.py runs this).
"""
import glob
import os
import sys

import numpy as np

import build_paths as bp
import check_holes as ch

NEAR = .03          # metres outward from an open edge another open edge is looked for (a seam, not the floor's end)
GAP = .002          # metres apart across: a crack
RISE = .004         # metres up or down: a step
CELL = 5.


def floors_glb():
    """The floors in source/models: (mesh name, triangles), a mesh's pieces together."""
    out = {}
    for f, name, T in ch.floors():
        out.setdefault((f, name), []).append(T)
    return [(f + ':' + n, np.concatenate(T)) for (f, n), T in out.items()]


def floors_kn5(folder):
    """The floors in a built track's kn5 files: (mesh name, triangles), a mesh's pieces together."""
    import kn5io
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, '*.kn5'))):
        fn = os.path.basename(f)
        if fn.replace('.kn5', '.glb') in ch.SKIP or fn.startswith('vp_'):
            continue
        mdl = kn5io.Model(f)
        for m in mdl.meshes:
            if m['name'].upper().startswith(ch.FLOOR) and m['visible'] and m['renderable'] and len(m['f']):
                out.setdefault((fn, m['name']), []).append(m['v'][:, :3].astype(float)[m['f'].astype(int)])
    return [(f + ':' + n, np.concatenate(T)) for (f, n), T in out.items()]


def open_edges_all(meshes):
    """The floors' open edges, all meshes together (an edge two faces share, of any meshes, is closed: its
    ends to a millimetre, heights too): their ends, outward (x, z), and the mesh each is of."""
    T = np.concatenate([t for _, t in meshes])
    owner = np.concatenate([np.full(len(t), i) for i, (_, t) in enumerate(meshes)])
    key = np.round(T * 1000).astype(np.int64)
    k = (key[..., 0] * 4000003 + key[..., 2]) * 40003 + key[..., 1]
    a = np.r_[k[:, 0], k[:, 1], k[:, 2]]
    b = np.r_[k[:, 1], k[:, 2], k[:, 0]]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, inv, cnt = np.unique(np.c_[lo, hi], axis=0, return_inverse=True, return_counts=True)
    single = (cnt[inv.ravel()] == 1) & (lo != hi)
    n = len(T)
    i0 = np.r_[np.zeros(n, int), np.ones(n, int), np.full(n, 2)]
    f = np.tile(np.arange(n), 3)
    A, B, C = T[f, i0], T[f, (i0 + 1) % 3], T[f, (i0 + 2) % 3]
    A, B, C, who = A[single], B[single], C[single], owner[f][single]
    d = (B - A)[:, [0, 2]]
    L = np.linalg.norm(d, axis=1)
    keep = L > .005
    A, B, C, d, L, who = A[keep], B[keep], C[keep], d[keep], L[keep], who[keep]
    nrm = np.c_[d[:, 1], -d[:, 0]] / L[:, None]
    mid = (A + B)[:, [0, 2]] / 2
    nrm[((C[:, [0, 2]] - mid) * nrm).sum(1) > 0] *= -1
    return A, B, nrm, who


def bad_seams(meshes):
    """The bad seams: (x, y, z) where, what (crack, step, unwelded), the two meshes' names, how far (m)."""
    from scipy.spatial import cKDTree
    A, B, nrm, who = open_edges_all(meshes)
    if not len(A):
        return []
    mid = (A + B) / 2
    L = np.linalg.norm((B - A)[:, [0, 2]], axis=1)
    tree = cKDTree(mid[:, [0, 2]])
    reach = L.max() / 2 + NEAR
    out = []
    for i in range(len(A)):
        p = mid[i, [0, 2]]
        cand = tree.query_ball_point(p, L[i] / 2 + reach)
        best = None
        for j in cand:
            if j == i:
                continue
            a, b = A[j, [0, 2]], B[j, [0, 2]]
            ab = b - a
            t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0., 1.)
            q = a + ab * t
            off = q - p
            dist = np.linalg.norm(off)
            if dist > NEAR or (dist > 1e-4 and off @ nrm[i] < 0):       # (only outward: across its seam)
                continue
            # (side by side: the other edge runs along this one, not across it)
            if abs(ab @ (B[i, [0, 2]] - A[i, [0, 2]])) < .5 * np.linalg.norm(ab) * L[i]:
                continue
            y = A[j, 1] + (B[j, 1] - A[j, 1]) * t
            if best is None or dist < best[0]:
                best = (dist, abs(y - mid[i, 1]), j)
        if best is None:
            continue
        dist, dy, j = best
        what = 'crack' if dist > GAP else 'step' if dy > RISE else 'unwelded'
        out.append((mid[i], what, meshes[who[i]][0], meshes[who[j]][0], max(dist, dy)))
    return out


def report(bad, top=12):
    """A summary of the bad seams: how many of each kind, where most, between which meshes most."""
    from collections import Counter
    if not bad:
        return 'no bad seams'
    kinds = Counter(w for _, w, _, _, _ in bad)
    where = Counter((int(np.floor(p[0] / CELL) * CELL), int(np.floor(p[2] / CELL) * CELL)) for p, _, _, _, _ in bad)
    pairs = Counter(' / '.join(sorted((a.split(':')[1], b.split(':')[1]))) for _, _, a, b, _ in bad)
    worst = max(bad, key=lambda r: r[4])
    lines = ['%d bad seam edges: %s' % (len(bad), ', '.join('%d %ss' % (c, k) for k, c in kinds.most_common())),
             '  worst: %s %.3f m at (%.1f, %.1f) between %s and %s' % (worst[1], worst[4], worst[0][0], worst[0][2], worst[2], worst[3]),
             '  most, by where (x, z): ' + ', '.join('%s %d' % (k, c) for k, c in where.most_common(top)),
             '  most, by meshes: ' + ', '.join('%s %d' % (k, c) for k, c in pairs.most_common(top))]
    return '\n'.join(lines)


def main():
    if '--kn5' in sys.argv:
        meshes = floors_kn5(sys.argv[sys.argv.index('--kn5') + 1])
    else:
        meshes = floors_glb()
    bad = bad_seams(meshes)
    print(report(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
