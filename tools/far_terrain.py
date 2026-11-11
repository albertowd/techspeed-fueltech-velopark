"""The far ground (GRASS_1 of source/models/velopark.glb) made again at the real ground's heights.

    python tools/far_terrain.py [--dry-run]

The far ground was a flat ring of 76 points round the near ground (1GRASS_0, a triangle: its
north edge along z = 433 from x = -500 to 500, its sides meeting at x = 0, z = -433). This makes it again from the near ground's edge out to the
satellite pictures' edges (tools/reference/surroundings/surroundings.json's area), a point every
STEP metres, at the real ground's heights: every height there is (tools/reference/elevation's
google_*.json and google_wide_*.json, tools/reference/surroundings' google_heights_*.json), less
the levelling's offset, but where they are the tops of trees or buildings, as tools/level_ground.py
leaves them out: where the satellite pictures show trees round them (dark and uneven both), or
the ground photo does (as tools/level_ground.py finds them there), within level_ground.BUILDING metres of
OpenStreetMap's buildings, or over the heights' opening round them; the ground round them fills
in, smoothed over FILLS metres, the finest where there are points enough.

Along the near ground's edge (its outline traced from its mesh) its points are the near ground's
own (a point every EDGE metres on its straight edges, at their heights), so the two meet with no gap; what the real ground differs
from them there fades out over EASE metres. Mapped onto FarGround.dds as tools/far_ground.py maps
it (which must have run). Run again, it makes it again, the same.
"""
import glob, json, os, sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_opening, map_coordinates, uniform_filter
from shapely import contains_xy
import triangle as tr

import build_paths as bp
import elevation as ev
import gltf_source
import level_ground as lg
import shapes as sh

SURROUNDINGS = os.path.join(bp.ROOT, 'tools', 'reference', 'surroundings')
FAR, NEAR = 'GRASS_1', '1GRASS_0'
STEP = 20.                  # metres between the far ground's points
EDGE = 10.                  # metres between its points along the near ground's edge
EASE = 120.                 # metres out from the near ground the difference at its edge fades over
CELL = 10.                  # metres, the real ground's grid
FILLS = (20., 40., 80., 160., 320.)   # metres, the smoothing that fills where no ground is, finest first
# trees in the satellite pictures: darker than TREE_LUM and more uneven than TREE_SD over 9 m, both
# (the pictures are sharper than the ground photo, so either alone takes in fields' and roads' edges)
TREE_LUM, TREE_SD = 75., 12.


def heights():
    rows = []
    for f in (glob.glob(os.path.join(ev.OUT, 'google_2*.json')) + glob.glob(os.path.join(ev.OUT, 'google_wide_*.json'))
              + glob.glob(os.path.join(SURROUNDINGS, 'google_heights_*.json'))):
        rows += json.load(open(f))['rows']
    R = np.array(rows, float)
    return R[:, 0], R[:, 1], R[:, 4]


def tree_share(info):
    """The share of trees round each metre of the satellite pictures: where they are dark and
    uneven both, within level_ground.TREE_NEAR metres."""
    img = Image.open(os.path.join(SURROUNDINGS, info['image'])).convert('L')
    lum = np.asarray(img, float)
    mean = uniform_filter(lum, 9)
    spread = np.sqrt(np.maximum(uniform_filter(lum ** 2, 9) - mean ** 2, 0))
    return uniform_filter(((spread > TREE_SD) & (mean < TREE_LUM)).astype(float), int(2 * lg.TREE_NEAR) + 1)


def buildings(loc):
    from shapely.geometry import Polygon
    out = []
    for f in [lg.latest('osm_property_*.json')] + sorted(glob.glob(os.path.join(SURROUNDINGS, 'osm_*.json'))):
        for e in json.load(open(f))['elements']:
            if e['type'] == 'way' and 'building' in e.get('tags', {}) and 'geometry' in e:
                g = e['geometry']
                out.append(Polygon(ev.to_ac(loc, ev.enu([p['lat'] for p in g], [p['lon'] for p in g]))).buffer(lg.BUILDING))
    return out


def real_ground(info):
    """The real ground's heights as the track's y on a CELL metre grid over the pictures' area."""
    loc = json.load(open(ev.LOCATION))
    offset = json.load(open(lg.DONE))['offset_m']
    x, z, h = heights()
    h = h - offset
    (xa, xb), (za, zb) = info['area_x_z']
    X, Z = np.arange(xa - 100, xb + 100.1, CELL), np.arange(za - 100, zb + 100.1, CELL)
    # not trees: by the pictures, where they are, else by the ground photo
    trees = tree_share(info)
    px = info['metres_a_pixel']
    col, row = ((xb - x) / px).astype(int), ((zb - z) / px).astype(int)
    inside = (col >= 0) & (col < trees.shape[1]) & (row >= 0) & (row < trees.shape[0])
    share = np.ones(len(x))
    share[inside] = trees[row[inside], col[inside]]
    # and in the ground photo, its trees as tools/level_ground.py finds them there, which the
    # pictures' rule misses in the photo's colours
    import gltf_source
    in_photo = (np.abs(x) < lg.PHOTO_SIZE / 2) & (np.abs(z) < lg.PHOTO_SIZE / 2)
    photo = lg.tree_share(gltf_source.load_kn5(bp.TRACK))
    share[in_photo] = np.maximum(share[in_photo], lg.photo_at(photo, x[in_photo], z[in_photo]))
    ground = share < lg.TREE_SHARE
    for b in buildings(loc):
        ground &= ~contains_xy(b, x, z)

    def splat(xx, zz, vv):
        A, W = np.zeros((len(Z), len(X))), np.zeros((len(Z), len(X)))
        i, j = np.round((zz - Z[0]) / CELL).astype(int), np.round((xx - X[0]) / CELL).astype(int)
        ok = (i >= 0) & (i < len(Z)) & (j >= 0) & (j < len(X))
        np.add.at(A, (i[ok], j[ok]), vv[ok])
        np.add.at(W, (i[ok], j[ok]), 1.)
        return A, W
    A, W = splat(x, z, h)
    tops = gaussian_filter(A, 1.) / np.maximum(gaussian_filter(W, 1.), 1e-9)
    at = lambda G: map_coordinates(G, [(z - Z[0]) / CELL, (x - X[0]) / CELL], order=1)
    ground &= h - at(gaussian_filter(grey_opening(tops, size=int(lg.OPENING / CELL) | 1), 1.)) < lg.OVER
    ground &= h - at(gaussian_filter(grey_opening(tops, size=int(lg.WIDE_OPENING / CELL) | 1), 2.)) < lg.WIDE_OVER
    A, W = splat(x[ground], z[ground], h[ground])
    real = np.full(A.shape, float(np.median(h[ground])))
    for s in FILLS[::-1]:
        a, w = gaussian_filter(A, s / CELL), gaussian_filter(W, s / CELL)
        t = np.clip(w / (lg.ENOUGH * (CELL / lg.CELL) ** 2), 0, 1)
        real += t * t * (3 - 2 * t) * (a / np.maximum(w, 1e-9) - real)
    print('  %d heights, %d of them the ground; %.1f to %.1f m' % (len(h), ground.sum(), real.min(), real.max()))
    return lambda xx, zz: map_coordinates(real, [(np.asarray(zz) - Z[0]) / CELL, (np.asarray(xx) - X[0]) / CELL], order=1, mode='nearest')


def near_edge(k):
    """The near ground's outer edge, its points in order round it: the longest loop of the edges
    only one of its triangles has."""
    from collections import Counter, defaultdict
    n, m = next((n_, m_) for n_, m_ in bp.meshes(k) if n_['name'] == NEAR)
    V = n['verts'][:, :3].astype(float) + m[3, :3]
    # its points welded where they lie together (each split by its normals and uv)
    key, idx = {}, np.empty(len(V), int)
    for i, v in enumerate(np.round(V, 3)):
        idx[i] = key.setdefault(tuple(v), len(key))
    W = np.zeros((len(key), 3))
    W[idx] = V
    F = idx[n['faces'].astype(int)]
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
