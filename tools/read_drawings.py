"""Lines drawn over the road maps (build/road_maps/*.png, made from the committed track) read back into
the map's metres: each map's blue strokes (a road's middle) and yellow strokes (an edge drawn), as
polylines (x, z), into tools/reference/road_drawings.json.

    python tools/read_drawings.py

Each map's frame is made again as it was drawn: the road's bounds in the committed track (its meshes'
names below), MARGIN round them, the ground photo's pixels over them scaled to SIDE on the longer side.
"""
import json, os, subprocess

import numpy as np
from PIL import Image

import build_paths as bp
import gltf_source

MAPS = os.path.join(bp.ROOT, 'build', 'road_maps')
OUT = os.path.join(bp.ROOT, 'tools', 'reference', 'road_drawings.json')
BASE = '6928794'
MARGIN, SIDE, K = 10., 1800, 8192 / 1060
MIN_STROKE = 10.      # metres: a stroke shorter than this is no line (the grid's labels)
ROADS = {'R6': ['1ROAD_6'], 'R7': ['1ROAD_7'], 'R8': ['1ROAD_8'], 'R9': ['1ROAD_9'], 'R10': ['1ROAD_10'], 'R12': ['1ROAD_12'],
         'R19': ['1ROAD_19'], 'R29': ['1KERB_29'], 'R30': ['1KERB_30'], 'R43_R44': ['1KERB_43', '1KERB_44']}
INKS = {'blue': lambda R, G, B: (B > 150) & (B - R > 90) & (B - G > 70) & (R < 120),
        'yellow': lambda R, G, B: (R > 200) & (G > 140) & (G < 225) & (B < 80)}


def frame(track, names):
    """A map's frame: the photo's crop box (pixels) and its scale."""
    T = np.concatenate([bp.triangles(n, m)[:, :, [0, 2]].reshape(-1, 2) for n, m in bp.meshes(track) if n['name'] in names and len(n['faces'])])
    (a, b), (c, d) = T.min(0) - MARGIN, T.max(0) + MARGIN
    x0, x1, z0, z1 = a, c, b, d
    box = (int((530 - x1) * K), int((530 - z1) * K), int((530 - x0) * K), int((530 - z0) * K))
    sc = SIDE / max(box[2] - box[0], box[3] - box[1])
    return box, sc


def strokes(mask, box, sc, cell=6):
    """A mask's strokes as polylines in metres (x, z): each stroke's pixels gathered into cells of
    cell pixels, the cells' middles joined by the shortest tree through them and the longest path
    along that tree its line, longest first."""
    from scipy import ndimage
    from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
    from scipy.spatial.distance import cdist as distance_matrix
    lab, k = ndimage.label(ndimage.binary_dilation(mask, iterations=2), structure=np.ones((3, 3)))
    out = []
    for i in range(1, k + 1):
        P = np.argwhere((lab == i) & mask)
        if len(P) < 40:
            continue
        key = P // cell
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        C = np.array([P[inv == j].mean(0) for j in range(inv.max() + 1)])
        D = distance_matrix(C, C)
        D[D > cell * 2.5] = 0
        T = minimum_spanning_tree(D)
        d0 = shortest_path(T, directed=False, indices=0)
        a = int(np.nanargmax(np.where(np.isinf(d0), -1, d0)))
        da, pred = shortest_path(T, directed=False, indices=a, return_predecessors=True)
        b = int(np.nanargmax(np.where(np.isinf(da), -1, da)))
        path = [b]
        while path[-1] != a and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        Q = C[path]
        px, py = Q[:, 1] / sc + box[0], Q[:, 0] / sc + box[1]
        out.append(np.c_[530 - px / K, 530 - py / K])
    return sorted(out, key=lambda q: -len(q))


def main():
    path = os.path.join(bp.ROOT, 'build', 'tmpcmp', 'models', 'base_velopark.glb')
    track = gltf_source.load_kn5(path)
    res = {}
    for label, names in ROADS.items():
        f = os.path.join(MAPS, label + '.png')
        if not os.path.exists(f):
            continue
        box, sc = frame(track, names)
        im = np.asarray(Image.open(f).convert('RGB')).astype(int)
        R, G, B = im[..., 0], im[..., 1], im[..., 2]
        res[label] = {}
        for ink, test in INKS.items():
            ss = [q for q in strokes(test(R, G, B), box, sc) if np.linalg.norm(np.diff(q, axis=0), axis=1).sum() > MIN_STROKE]
            if ss:
                res[label][ink] = [[[round(float(x), 2), round(float(z), 2)] for x, z in q] for q in ss]
        print('%s: %s' % (label, ', '.join('%s %s' % (ink, ' + '.join('%.0f m' % np.linalg.norm(np.diff(np.array(q), axis=0), axis=1).sum() for q in v)) for ink, v in res[label].items())))
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
