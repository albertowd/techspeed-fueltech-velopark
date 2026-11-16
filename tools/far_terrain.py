"""The far ground (GRASS_1 of source/models/velopark.glb) made again at the real ground's heights.

    python tools/far_terrain.py [--dry-run]

The far ground was a flat ring of 76 points round the near ground (1GRASS_0, a triangle: its
north edge along z = 433 from x = -500 to 500, its sides meeting at x = 0, z = -433). This makes it again from the near ground's edge out to the
satellite pictures' edges (tools/reference/surroundings/surroundings.json's area), a point every
STEP metres, at the real ground's heights: tools/terrain.py's filtered heights, the tops of trees and
buildings left out by its filters (no edits on them: they are the near ground's).

Along the near ground's edge (its outline traced from its mesh) its points are the near ground's
own (a point every EDGE metres on its straight edges, at their heights), so the two meet with no gap; what the real ground differs
from them there fades out over EASE metres. Mapped onto FarGround.dds as tools/far_ground.py maps
it (which must have run). Run again, it makes it again, the same.
"""
import json, os, sys

import numpy as np
from shapely import contains_xy
import triangle as tr

import build_paths as bp
import gltf_source
import shapes as sh

SURROUNDINGS = os.path.join(bp.ROOT, 'tools', 'reference', 'surroundings')
FAR, NEAR = 'GRASS_1', '1GRASS_0'
STEP = 20.                  # metres between the far ground's points
EDGE = 10.                  # metres between its points along the near ground's edge
EASE = 120.                 # metres out from the near ground the difference at its edge fades over


def real_ground(info=None, order=1):
    """The real ground's heights as the track's y, as a function of x, z: tools/terrain.py's filtered
    heights (no edits on them), between its grid's points linear (order 1), or a cubic spline (order 3)."""
    import terrain
    return terrain.filtered(order)


def near_edge(k):
    """The near ground's outer edge, its points in order round it: the longest loop of the edges
    only one of its triangles has."""
    from collections import Counter, defaultdict
    # all of it: it is cut into meshes of fewer than 65536 points, 1GRASS_0, 1GRASS_0_2 and on
    import re
    parts = [(n_, m_) for n_, m_ in bp.meshes(k) if re.fullmatch(re.escape(NEAR) + r'(_\d+)*', n_['name']) and len(n_['faces'])]
    V = np.concatenate([n_['verts'][:, :3].astype(float) + m_[3, :3] for n_, m_ in parts])
    off = np.cumsum([0] + [len(n_['verts']) for n_, _ in parts])
    faces = np.concatenate([n_['faces'].astype(int) + o for (n_, _), o in zip(parts, off)])
    # its points welded where they lie together (each split by its normals and uv)
    key, idx = {}, np.empty(len(V), int)
    for i, v in enumerate(np.round(V, 3)):
        idx[i] = key.setdefault(tuple(v), len(key))
    W = np.zeros((len(key), 3))
    W[idx] = V
    F = idx[faces]
    count = Counter(tuple(sorted(e)) for a, b, c in F for e in ((a, b), (b, c), (c, a)))
    nxt = defaultdict(list)
    for (a, b), c in count.items():
        if c == 1:
            nxt[a].append(b)
            nxt[b].append(a)
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            step = [q for q in nxt[cur] if q != prev]
            if not step or step[0] == start:
                break
            prev, cur = cur, step[0]
            if cur in seen:
                break
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    area = lambda L: abs(np.sum(W[L, 0] * np.roll(W[L, 2], -1) - np.roll(W[L, 0], -1) * W[L, 2])) / 2
    return W[max(loops, key=area)]


def build(k, info, real):
    (xa, xb), (za, zb) = info['area_x_z']
    ring = near_edge(k)
    # its points every EDGE metres, on the near ground's straight edges between its own
    inner = []
    for p, q in zip(ring, np.roll(ring, -1, 0)):
        L = np.hypot(q[0] - p[0], q[2] - p[2])
        for t in np.arange(0, L, EDGE) / L:
            inner.append(p + (q - p) * t)
    inner = np.array(inner)
    outer = []
    for (p, q) in (((xa, za), (xb, za)), ((xb, za), (xb, zb)), ((xb, zb), (xa, zb)), ((xa, zb), (xa, za))):
        p, q = np.array(p), np.array(q)
        L = np.linalg.norm(q - p)
        outer += [p + (q - p) * t for t in np.arange(0, L, STEP) / L]
    outer = np.array(outer)
    from shapely.geometry import Polygon
    hole = Polygon(ring[:, [0, 2]])
    gx, gz = np.meshgrid(np.arange(xa + STEP, xb - STEP / 2, STEP), np.arange(za + STEP, zb - STEP / 2, STEP))
    G = np.c_[gx.ravel(), gz.ravel()]
    G = G[~contains_xy(hole.buffer(STEP / 2), G[:, 0], G[:, 1])]
    pts = np.r_[inner[:, [0, 2]], outer, G]
    ni, no = len(inner), len(outer)
    seg = [(i, (i + 1) % ni) for i in range(ni)] + [(ni + i, ni + (i + 1) % no) for i in range(no)]
    inside = hole.representative_point()
    T = tr.triangulate(dict(vertices=pts, segments=seg, holes=[(inside.x, inside.y)]), 'p')
    P2, F = T['vertices'], T['triangles']
    y = real(P2[:, 0], P2[:, 1])
    # the near ground's own heights on its edge, and the difference there fading out
    from scipy.spatial import cKDTree
    t_in = cKDTree(inner[:, [0, 2]])
    d, i = t_in.query(P2)
    diff = inner[i, 1] - real(inner[i, 0], inner[i, 2])
    s = np.clip(1 - d / EASE, 0, 1)
    y = y + diff * s * s * (3 - 2 * s)
    y[:ni] = inner[:, 1]
    V = np.c_[P2[:, 0], y, P2[:, 1]]
    # each triangle facing up
    n_ = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F[n_[:, 1] < 0] = F[n_[:, 1] < 0][:, ::-1]
    N = sh._vertex_normals(V, F)
    U = np.c_[(xb - V[:, 0]) / (xb - xa), (zb - V[:, 2]) / (zb - za)]
    return V, N, U, F, diff


def main():
    k = gltf_source.load_kn5(bp.TRACK)
    if not any(m['name'] == 'BackgroundFar' for m in k['materials']):
        sys.exit('run tools/far_ground.py first')
    info = json.load(open(os.path.join(SURROUNDINGS, 'surroundings.json')))
    real = real_ground(info)
    V, N, U, F, diff = build(k, info, real)
    print('  %d points, %d triangles; %.1f to %.1f m; at the near ground\'s edge the real ground differs by %+.2f to %+.2f m' % (
        len(V), len(F), V[:, 1].min(), V[:, 1].max(), diff.min(), diff.max()))
    if len(V) > 65535:
        sys.exit('too many points for one mesh')
    if '--dry-run' in sys.argv:
        return
    node, m = next((n_, m_) for n_, m_ in bp.meshes(k) if n_['name'] == FAR)
    Vl = V - m[3, :3]
    Tg = gltf_source.tangents(Vl, N, U, F)
    node['verts'] = np.c_[Vl, N, U, Tg].astype('<f4')
    node['faces'] = F.astype('<u2')
    bp.sphere(node)
    doc, _ = gltf_source.read_glb(bp.TRACK)
    import urllib.parse
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(bp.TRACK, doc, binary)
    print('  %s made again in %s' % (FAR, os.path.relpath(bp.TRACK, bp.ROOT)))


if __name__ == '__main__':
    main()
