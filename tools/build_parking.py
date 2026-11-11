"""The car park south of the pit building, as asphalt with marked bays, in the track's sources.

    python tools/build_parking.py

The ground photo shows it dirty, as it was taken after a flood, so the paddock laid a sandy
dirt over its asphalt there and over the pits beside it. It is asphalt, the track's own, with
bays marked as in the car park by the kids track. This takes the dirt out of
source/models/velopark_paddock.glb, so the paddock's asphalt under it shows, and paints the
bays' lines on it as PARKING_LINES in source/models/velopark_paint.glb, in the paint's own
material. Run again, it lays the lines again and finds the dirt gone.

The bays are laid in rows along the pit building, measured from the west end of its south
face: ALONG it eastwards, ACROSS southwards away from it. The strip along the building, where
the teams' trucks stand, is left clear. Two rows stand back to back, a line between them, with
an aisle between each pair; each row runs between the ends the photo leaves clear of grass,
trees and the buildings.
"""
import os, sys, urllib.parse

import numpy as np

import gltf_source
import kn5tree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'source', 'models')
PADDOCK = os.path.join(MODELS, 'velopark_paddock.glb')
PAINT = os.path.join(MODELS, 'velopark_paint.glb')
CONFIG = os.path.join(ROOT, 'source', 'track', 'extension', 'ext_config.ini')

DIRT_MESH, DIRT_MATERIAL = 'PADDOCK_DIRT', 'PaddockDirt'
NAME = 'PARKING_LINES'
# the pit building's south face: its west end, and a point further along it
FACE = np.array([(119.2, -20.8), (17.5, -43.3)])
BAY_W, BAY_D = 2.5, 5.       # a bay's width along the row and its depth
LINE_WIDTH = .12             # as the paint's own lines
LIFT = .012                  # over the asphalt, as the paint's lines lie over the road
STEP = 1.                    # metres between the points a line is laid on the asphalt at
# the rows: where a row's bays start across (they reach BAY_D further), and the runs of bays
# along it, from where to where along
ROWS = ((22., ((-50., -22.5), (5., 87.5))),
        (27., ((-50., -22.5), (5., 87.5))),
        (38., ((-47.5, -22.5), (12.5, 90.))),
        (43., ((-47.5, -22.5), (12.5, 27.5), (40., 90.))))


def frame():
    along = FACE[1] - FACE[0]
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    if across[1] > 0:       # south, z down
        across = -across
    return FACE[0], along, across


def strips():
    """The lines, each as its two ends in (along, across): a line between every two bays and at
    each end of a run, across the row, and the line along the row's back."""
    out, backs = [], {}
    starts = [c for c, _ in ROWS]
    for c, runs in ROWS:
        # the back, the side away from the aisle the bays open onto: the line a pair shares
        back = c + BAY_D if any(abs(c + BAY_D - d) < 1e-6 for d in starts) else c
        for a0, a1 in runs:
            for a in np.arange(a0, a1 + 1e-6, BAY_W):
                out.append(((a, c), (a, c + BAY_D)))
            backs.setdefault(round(back, 3), []).append((a0, a1))
    # the two rows' backs as one line where they overlap, as paint laid twice shows whiter
    for back, runs in backs.items():
        merged = []
        for a0, a1 in sorted(runs):
            if merged and a0 <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], a1)
            else:
                merged.append([a0, a1])
        out += [((a0, back), (a1, back)) for a0, a1 in merged]
    return out


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
        ks = s.grid.get((int(np.floor(x / s.CELL)), int(np.floor(z / s.CELL))), [])
        best = None
        for k in ks:
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


def meshes(k):
    return [n for n, _ in kn5tree.walk(k['root']) if n['kind'] == 2]


def triangles(node):
    v, f = node['verts'], node['faces'].astype(int)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v[f][:, :, :3].astype(float)


def save(k, path):
    """The model back over its .glb, pointing at the textures it pointed at."""
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris[t['name']])
    gltf_source.write_glb(path, doc, binary)


def drop_dirt(pad):
    """The dirt mesh and its material out of the paddock; whether there was any."""
    root = pad['root']
    if not any(c['name'] == DIRT_MESH for c in root['children']):
        return False
    root['children'] = [c for c in root['children'] if c['name'] != DIRT_MESH]
    i = next((j for j, m in enumerate(pad['materials']) if m['name'] == DIRT_MATERIAL), None)
    if i is not None and not any(n['material'] == i for n in meshes(pad)):
        texs = {t['name'] for t in pad['materials'][i]['textures']}
        del pad['materials'][i]
        for n in meshes(pad):
            if n['material'] > i:
                n['material'] -= 1
        used = {t['name'] for m in pad['materials'] for t in m['textures']}
        pad['textures'] = [t for t in pad['textures'] if t['name'] not in texs or t['name'] in used]
    return True


def lines(paint, ground):
    """The bays' lines as one mesh, a copy of the paint's own lines' node."""
    mat = next(j for j, m in enumerate(paint['materials']) if m['name'] == 'LinePaint')
    template = next(n for n in meshes(paint) if n['name'].startswith('LINE') and n['material'] == mat)
    # the paint's own winding: its triangles face up
    t = triangles(template)[0]
    up = np.cross(t[1] - t[0], t[2] - t[0])[1] > 0
    o, along, across = frame()
    P, F = [], []
    for (a0, c0), (a1, c1) in strips():
        p, q = o + along * a0 + across * c0, o + along * a1 + across * c1
        d = q - p
        side = np.array([-d[1], d[0]]) / np.linalg.norm(d) * LINE_WIDTH / 2
        # each end of the line, and points between, so it follows the asphalt
        n = max(1, int(np.ceil(np.linalg.norm(d) / STEP)))
        for s in np.linspace(0, 1, n + 1):
            m = p + d * s
            for e in (m - side, m + side):
                h = ground.y(*e)
                if h is None:
                    sys.exit('the asphalt does not reach the line at x %.1f, z %.1f' % tuple(e))
                P.append((e[0], h + LIFT, e[1]))
        base = len(P) - 2 * (n + 1)
        for j in range(n):
            a, b, c, e = base + 2 * j, base + 2 * j + 1, base + 2 * j + 2, base + 2 * j + 3
            for tri in ((a, b, e), (a, e, c)):
                pa, pb, pc = (np.array(P[x]) for x in tri)
                if (np.cross(pb - pa, pc - pa)[1] > 0) != up:
                    tri = tri[::-1]
                F.append(tri)
    P = np.array(P, float)
    F = np.array(F, int)
    if len(P) > 65535:
        sys.exit('the lines take %d vertices, more than a kn5 mesh holds' % len(P))
    N = np.tile([0, 1., 0], (len(P), 1))
    U = np.c_[P[:, 0] / 10, P[:, 2] / 20]
    T = gltf_source.tangents(P, N, U, F)
    node = dict(template)
    node.update(name=NAME, children=[], verts=np.c_[P, N, U, T].astype('<f4'), faces=F.astype('<u2'))
    mid = (P.min(0) + P.max(0)) / 2
    node['sphere'] = [float(x) for x in mid] + [float(np.linalg.norm(P - mid, axis=1).max())]
    return node, len(strips())


def main():
    pad = gltf_source.load_kn5(PADDOCK)
    paint = gltf_source.load_kn5(PAINT)
    asphalt = next(n for n in meshes(pad) if n['name'] == '1ROAD_PADDOCK')
    ground = Surface(triangles(asphalt))
    if drop_dirt(pad):
        save(pad, PADDOCK)
        print('the dirt is out of %s' % os.path.relpath(PADDOCK, ROOT))
    node, count = lines(paint, ground)
    root = paint['root']
    root['children'] = [c for c in root['children'] if c['name'] != NAME] + [node]
    save(paint, PAINT)
    print('%s: %d lines, %d triangles in %s' % (NAME, count, len(node['faces']), os.path.relpath(PAINT, ROOT)))
    # the grass config named the dirt among what keeps grass off
    text = open(CONFIG, encoding='utf-8', newline='').read()
    new = text.replace(', ' + DIRT_MATERIAL, '')
    if new != text:
        open(CONFIG, 'w', encoding='utf-8', newline='').write(new)
        print('%s no longer names %s' % (os.path.relpath(CONFIG, ROOT), DIRT_MATERIAL))


if __name__ == '__main__':
    main()
