"""The track's old walls out of the paddock, in the track's sources.

    python tools/clear_walls.py

The track came with invisible walls along where its asphalt ended. The paddock's asphalt was
laid out past them later, so two of them stand across it: between the pit building and the car
park and round the road by the buildings to the west, and along the strip north of those
buildings. A kart drives into them as into a wall nobody sees, in AC and in BeamNG, which makes
them solid too. Others like them were sunk deep under the ground before, where nothing reaches.

This takes out of source/models/velopark.glb every face of a wall whose foot runs inside the
asphalt, the roads, kerbs and paddock together, at least INSIDE in from its edge, and that
stands across the asphalt's height there. A wall at the asphalt's edge stays, and so do the
sunk ones and the buildings' walls, which are the paddock's. A mesh left with no faces goes.
Run again, it finds none.
"""
import os, sys, urllib.parse

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString
from shapely.ops import unary_union

import gltf_source
import kn5tree
import track_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'source', 'models')
TRACK = os.path.join(MODELS, 'velopark.glb')
PADDOCK = os.path.join(MODELS, 'velopark_paddock.glb')
KERBS = os.path.join(MODELS, 'velopark_kerbs.glb')
INSIDE = 1.          # metres in from the asphalt's edge a wall's foot must run
REACH = .5           # metres of a wall's foot inside, for its face to go
# how far under and over the asphalt a face must reach to stand in a kart's way
UNDER, OVER = .5, .05


def meshes(k):
    """The mesh nodes with the matrix to the model's space."""
    return [(n, m) for n, m in kn5tree.walk(k['root']) if n['kind'] == 2]


def faces(n):
    f = n['faces'].astype(int)
    return f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]


def world(n, m):
    p = n['verts'][:, :3].astype(float)
    return (np.c_[p, np.ones(len(p))] @ m)[:, :3]


class Surface:
    """The top of a mesh's triangles over a point, found through a grid of 4 m cells."""
    CELL = 4.

    def __init__(s, tris):
        s.t = tris
        lo = np.floor(tris[:, :, [0, 2]].min(1) / s.CELL).astype(int)
        hi = np.floor(tris[:, :, [0, 2]].max(1) / s.CELL).astype(int)
        s.grid = {}
        for k in range(len(tris)):
            for gx in range(lo[k, 0], hi[k, 0] + 1):
                for gz in range(lo[k, 1], hi[k, 1] + 1):
                    s.grid.setdefault((gx, gz), []).append(k)

    def y(s, x, z):
        best = None
        for k in s.grid.get((int(np.floor(x / s.CELL)), int(np.floor(z / s.CELL))), []):
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = s.t[k]
            d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
            if abs(d) < 1e-12:
                continue
            l1 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d
            l2 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / d
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-6:
                h = l1 * ay + l2 * by + l3 * cy
                best = h if best is None else max(best, h)
        return best


def in_the_way(tri, inner, ground):
    """Whether a wall's face stands across the asphalt, well inside its edge."""
    xz = tri[:, [0, 2]]
    length, a, b = max((np.linalg.norm(xz[i] - xz[j]), i, j) for i, j in ((0, 1), (1, 2), (0, 2)))
    if length < 1e-3:
        return False
    part = LineString([xz[a], xz[b]]).intersection(inner)
    if part.length < REACH:
        return False
    # the asphalt's height along the part of the foot inside it
    pts = [part.interpolate(t, normalized=True) for t in np.linspace(0, 1, 5)]
    hs = [h for h in (ground.y(p.x, p.y) for p in pts if contains_xy(inner, p.x, p.y)) if h is not None]
    if not hs:
        return False
    return tri[:, 1].min() < min(hs) + UNDER and tri[:, 1].max() > max(hs) + OVER


def save(k, path):
    """The model back over its .glb, pointing at the textures it pointed at."""
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris[t['name']])
    gltf_source.write_glb(path, doc, binary)


def main():
    track = gltf_source.load_kn5(TRACK)
    tris, shapes = [], []
    for k in (track, gltf_source.load_kn5(PADDOCK), gltf_source.load_kn5(KERBS)):
        for n, m in meshes(k):
            if n['name'].startswith('1ROAD') and len(faces(n)):
                t = world(n, m)[faces(n)]
                tris.append(t)
                shapes.append(track_map.shape(t[:, :, [0, 2]]))
    ground = Surface(np.concatenate(tris))
    inner = unary_union(shapes).buffer(-INSIDE)
    gone = []
    for n, m in meshes(track):
        name = n['name']
        if not (name[:1].isdigit() and 'WALL' in name):
            continue
        f = faces(n)
        p = world(n, m)
        out = np.array([in_the_way(p[x], inner, ground) for x in f], bool)
        if not out.any():
            continue
        keep = f[~out]
        # the vertices the faces left still use, in their order
        used = np.unique(keep)
        index = np.full(len(n['verts']), -1)
        index[used] = np.arange(len(used))
        n['verts'] = n['verts'][used]
        n['faces'] = index[keep].astype(n['faces'].dtype)
        if len(used):
            v = n['verts'][:, :3].astype(float)
            mid = (v.min(0) + v.max(0)) / 2
            n['sphere'] = [float(x) for x in mid] + [float(np.linalg.norm(v - mid, axis=1).max())]
        gone.append((name, int(out.sum()), len(keep)))
    if not gone:
        print('no wall stands across the asphalt in %s' % os.path.relpath(TRACK, ROOT))
        return
    # meshes left with no faces go
    empty = {name for name, _, left in gone if not left}

    def prune(node):
        node['children'] = [c for c in node['children'] if not (c['kind'] == 2 and c['name'] in empty)]
        for c in node['children']:
            prune(c)
    prune(track['root'])
    save(track, TRACK)
    for name, out, left in gone:
        print('%s: %d faces out%s' % (name, out, ', the mesh gone' if not left else ', %d left' % left))


if __name__ == '__main__':
    sys.exit(main())
