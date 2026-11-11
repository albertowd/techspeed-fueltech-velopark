"""The track's heights brought to the real ground's, from Google's (tools/reference/elevation), in every source.

    python tools/level_ground.py [--dry-run]

The track's ground was made from a few points of the old elevation data, smoothed, and has
swells the real ground has not: it climbs 3 to 5 m north towards the plantation (the old data's
tops of its trees), has a hollow in the vp_1000's middle a metre under the ground round it, and
a hill 5 m high in the south west where the ground is flat. This finds how much higher or lower
than the track's the real ground is, smoothly (the field), and moves everything by it: every
point of every model (the ground, the roads, kerbs, paint, walls, buildings, poles, trees), every
AC_ node with its meshes as one, every layout's AI lines, the Custom Shaders Patch's lights and
the trees BeamNG's forest stands. So nothing comes apart, the roads' and kerbs' shapes and the
paint on them are kept, and only their heights over the long run change, as gently as the field.

The field: Google's heights (google_*.json and google_wide_*.json), less the median difference on
the roads (compare_*.json's offset_m, so the track keeps its height on the whole), where they
are the ground: not where the ground photo shows trees (TREE_LUM darker, or TREE_SD more varied,
within TREE_NEAR metres), nor within BUILDING metres of OpenStreetMap's buildings
(osm_property_*.json), nor more than OVER over the opening of the heights round them (or
WIDE_OVER over a WIDE_OPENING one, for woods wider than OPENING), as a surface model's tops of
trees and roofs stand. Where none is left, under the plantation, the
woods and the buildings, the ground round it is taken as the ground under it. That, smoothed
over SIGMA metres, less the track's top surface smoothed the same way, on a CELL metre grid;
fading out over FADE metres to the edge of the near ground, so the far ground still meets it.
It is saved in tools/reference/elevation/level_ground.npz; a summary in level_ground.json,
which marks it done: run again, it finds the ground levelled and changes nothing.
"""
import datetime, glob, io, json, os, re, struct, sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_opening, map_coordinates, uniform_filter
from shapely import contains_xy
from shapely.geometry import Polygon

import build_paths as bp
import elevation as ev
import gltf_source
import kn5tree

ROOT = bp.ROOT
TRACK = os.path.join(ROOT, 'source', 'track')
FIELD = os.path.join(ev.OUT, 'level_ground.npz')
DONE = os.path.join(ev.OUT, 'level_ground.json')
FOREST = os.path.join(bp.MODELS, 'velopark_extra_trees.json')
CSP = os.path.join(TRACK, 'extension', 'ext_config.ini')
PHOTO = 'GoogleMap.jpg'
PHOTO_SIZE = 1060.           # metres the ground photo spans, its middle at AC's origin
TREE_LUM, TREE_SD = 84., 13.  # the photo's brightness under which, or its spread over 9 m over which, trees
TREE_NEAR, TREE_SHARE = 7., .3    # trees on more than this share within this many metres: no ground
BUILDING = 10.               # metres round OpenStreetMap's buildings: no ground
OVER, OPENING = 1.5, 55.     # metres over the heights' opening over this many metres: tops, no ground
WIDE_OVER, WIDE_OPENING = 4., 250.   # and over a wide one, for woods too wide for the first to see
CELL = 5.                    # metres, the field's grid
SIGMA = 15.                  # metres the field is smoothed over
FILLS = (15., 30., 60., 125.)    # metres, the smoothing that fills where no ground is, finest first
ENOUGH = .04                 # points a cell, smoothed, a smoothing needs to be taken alone
EXTENT = ((-640., 640.), (-560., 560.))  # AC's x and z the field covers: the near ground and the far's first ring
FADE = 60.                   # metres it fades out over to its edge


# ---- the field ------------------------------------------------------------------------------
def latest(pattern):
    files = sorted(glob.glob(os.path.join(ev.OUT, pattern)))
    if not files:
        sys.exit('no %s in %s: run tools/elevation.py first' % (pattern, os.path.relpath(ev.OUT, ROOT)))
    return files[-1]


def tree_share(track):
    """The share of trees round each metre of the ground photo, as an image a metre a pixel."""
    data = next(t['data'] for t in track['textures'] if t['name'] == PHOTO)
    n = int(PHOTO_SIZE)
    lum = np.asarray(Image.open(io.BytesIO(data)).convert('L').resize((n, n), Image.BOX), float)
    mean = uniform_filter(lum, 9)
    spread = np.sqrt(np.maximum(uniform_filter(lum ** 2, 9) - mean ** 2, 0))
    return uniform_filter(((spread > TREE_SD) | (mean < TREE_LUM)).astype(float), int(2 * TREE_NEAR) + 1)


def photo_at(img, x, z):
    """The photo's pixel over AC's x, z: its u runs against x and its v against z."""
    h = PHOTO_SIZE / 2
    return img[np.clip((h - np.asarray(z)).astype(int), 0, img.shape[0] - 1),
               np.clip((h - np.asarray(x)).astype(int), 0, img.shape[1] - 1)]


def buildings(loc):
    osm = json.load(open(latest('osm_property_*.json')))
    out = []
    for e in osm['elements']:
        if e['type'] == 'way' and 'building' in e.get('tags', {}):
            g = e['geometry']
            out.append(Polygon(ev.to_ac(loc, ev.enu([p['lat'] for p in g], [p['lon'] for p in g]))).buffer(BUILDING))
    return out


def splat(X, Z, x, z, v):
    """Values summed into the grid's cells, and how many in each."""
    A, W = np.zeros((len(Z), len(X))), np.zeros((len(Z), len(X)))
    i, j = np.round((z - Z[0]) / CELL).astype(int), np.round((x - X[0]) / CELL).astype(int)
    ok = (i >= 0) & (i < len(Z)) & (j >= 0) & (j < len(X))
    np.add.at(A, (i[ok], j[ok]), v[ok])
    np.add.at(W, (i[ok], j[ok]), 1.)
    return A, W


def field():
    loc = json.load(open(ev.LOCATION))
    offset = json.load(open(latest('compare_2*.json')))['offset_m']
    rows = np.array(json.load(open(latest('google_2*.json')))['rows'] + json.load(open(latest('google_wide_*.json')))['rows'], float)
    x, z, h = rows[:, 0], rows[:, 1], rows[:, 4] - offset
    track = gltf_source.load_kn5(bp.TRACK)
    (xa, xb), (za, zb) = EXTENT
    X, Z = np.arange(xa, xb + .1, CELL), np.arange(za, zb + .1, CELL)
    # the ground: not trees, not buildings, not tops over the opening round them
    ground = photo_at(tree_share(track), x, z) < TREE_SHARE
    for b in buildings(loc):
        ground &= ~contains_xy(b, x, z)
    A, W = splat(X, Z, x, z, h)
    tops = gaussian_filter(A, 2.) / np.maximum(gaussian_filter(W, 2.), 1e-9)
    opening = gaussian_filter(grey_opening(tops, size=int(OPENING / CELL) | 1), 2.)
    ground &= h - map_coordinates(opening, [(z - Z[0]) / CELL, (x - X[0]) / CELL], order=1) < OVER
    wide = gaussian_filter(grey_opening(tops, size=int(WIDE_OPENING / CELL) | 1), 4.)
    ground &= h - map_coordinates(wide, [(z - Z[0]) / CELL, (x - X[0]) / CELL], order=1) < WIDE_OVER
    # the real ground, filled from round it where none is
    A, W = splat(X, Z, x[ground], z[ground], h[ground])
    # from the coarsest smoothing to the finest, each where it has points enough, eased into the
    # coarser where it has fewer, so no seam shows where one hands over to the next
    real = np.full(A.shape, float(np.mean(h[ground])))
    for s in FILLS[::-1]:
        a, w = gaussian_filter(A, s / CELL), gaussian_filter(W, s / CELL)
        t = np.clip(w / ENOUGH, 0, 1)
        real += t * t * (3 - 2 * t) * (a / np.maximum(w, 1e-9) - real)
    # the track's top surface, smoothed the same
    XX, ZZ = np.meshgrid(X, Z)
    y = surface().ys(np.c_[XX.ravel(), ZZ.ravel()]).reshape(XX.shape)
    has = ~np.isnan(y)
    mine = gaussian_filter(np.where(has, y, 0), SIGMA / CELL) / np.maximum(gaussian_filter(has.astype(float), SIGMA / CELL), 1e-9)
    dy = np.where(gaussian_filter(has.astype(float), SIGMA / CELL) > .05, real - mine, 0.)
    # faded out to the edge
    edge = np.minimum.reduce([XX - xa, xb - XX, ZZ - za, zb - ZZ])
    t = np.clip(edge / FADE, 0, 1)
    dy *= t * t * (3 - 2 * t)
    return X, Z, dy, dict(offset_m=offset, google_points=int(len(x)), ground_points=int(ground.sum()))


def surface():
    """The track's top surface as tools/elevation.py takes it, and the far ground round the near."""
    tris = []
    for path in (bp.TRACK, bp.KERBS, bp.PADDOCK):
        for n, m in bp.meshes(gltf_source.load_kn5(path)):
            if n['name'].startswith(ev.SURFACE + ('GRASS_1',)) and len(n['faces']):
                tris.append(bp.triangles(n, m))
    return bp.Surface(np.concatenate(tris))


def at(F, x, z):
    """The field at x, z: F the grid (X, Z, dy), or a function of x, z (tools/flatten_roads.py's)."""
    if callable(F):
        return F(np.asarray(x, float), np.asarray(z, float))
    X, Z, dy = F
    return map_coordinates(dy, [(np.asarray(z, float) - Z[0]) / CELL, (np.asarray(x, float) - X[0]) / CELL], order=1, cval=0.)


# ---- moving everything by it ----------------------------------------------------------------
def move_model(k, F):
    """Every point by the field where it stands; an AC_ node, or any node turned or moved, with
    its meshes as one, by the field at its origin. Points moved."""
    moved = [0]

    def walk(n, parent):
        if n['kind'] == 1:
            M = np.array(n['matrix'], float).reshape(4, 4)
            m = M @ parent
            if not np.allclose(M, np.eye(4)):
                o = m[3, :3]
                d = float(at(F, [o[0]], [o[2]])[0])
                # its translation, in its parent's space, by d up in the model's
                up = np.array([0., d, 0.]) @ np.linalg.inv(parent[:3, :3])
                M[3, :3] += up
                n['matrix'] = list(M.ravel())
                moved[0] += 1
                return
            for c in n['children']:
                walk(c, m)
        elif n['kind'] == 2 and len(n['verts']):
            V = n['verts']
            P = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ parent)[:, :3]
            P[:, 1] += at(F, P[:, 0], P[:, 2])
            V[:, :3] = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(parent))[:, :3].astype(V.dtype)
            bp.sphere(n)
            moved[0] += len(V)
            for c in n['children']:
                walk(c, parent)
    walk(k['root'], np.eye(4))
    return moved[0]


def move_ai(path, F):
    b = bytearray(open(path, 'rb').read())
    _, n, _, _ = struct.unpack_from('<4i', b, 0)
    P = np.frombuffer(bytes(b[16:16 + n * 20]), '<f4').reshape(-1, 5).copy()
    P[:, 1] += at(F, P[:, 0], P[:, 2]).astype('<f4')
    b[16:16 + n * 20] = P.astype('<f4').tobytes()
    open(path, 'wb').write(bytes(b))
    return n


def move_csp(path, F):
    text = open(path, newline='').read()
    count = [0]

    def one(mt):
        x, y, z = (float(v) for v in mt.group(2, 3, 4))
        count[0] += 1
        return '%s%s, %s, %s' % (mt.group(1), mt.group(2), ('%.3f' % (y + float(at(F, [x], [z])[0]))).rstrip('0').rstrip('.'), mt.group(4))
    text = re.sub(r'(?m)^(POSITION(?:_\d+)? *= *)(-?[\d.]+), *(-?[\d.]+), *(-?[\d.]+)', one, text)
    open(path, 'w', newline='').write(text)
    return count[0]


def move_forest(path, F):
    d = json.load(open(path))
    t = d['trees']
    dy = at(F, [s[1] for s in t], [s[3] for s in t])
    for s, v in zip(t, dy):
        s[2] += float(v)
    json.dump(d, open(path, 'w'), indent=0)
    return len(t)


def main():
    dry = '--dry-run' in sys.argv
    if os.path.exists(DONE):
        print('the ground is levelled (%s); to level it again, restore the sources and delete it' % os.path.relpath(DONE, ROOT))
        return
    X, Z, dy, info = field()
    F = (X, Z, dy)
    near = (np.abs(X)[None, :] < 260) & (np.abs(Z - 130)[:, None] < 230)
    gy, gx = np.gradient(dy, CELL)
    print('the field: %+.2f to %+.2f m over the track, steepest %.2f%%' % (dy[near].min(), dy[near].max(), np.hypot(gx, gy)[near].max() * 100))
    if dry:
        return
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        k = gltf_source.load_kn5(f)
        n = move_model(k, F)
        bp.save(k, f)
        print('  %-30s %7d points' % (os.path.basename(f), n))
    ai = sorted(glob.glob(os.path.join(TRACK, '*', 'ai', '*.ai')) + glob.glob(os.path.join(TRACK, '*', 'data', '*.ai')))
    for f in ai:
        move_ai(f, F)
    print('  %d AI lines' % len(ai))
    print('  %d lights in %s' % (move_csp(CSP, F), os.path.basename(CSP)))
    print('  %d trees of the forest' % move_forest(FOREST, F))
    np.savez_compressed(FIELD, x=X, z=Z, dy=dy)
    json.dump(dict(info, levelled=datetime.date.today().isoformat(), cell_m=CELL, sigma_m=SIGMA,
                   track_min_m=round(float(dy[near].min()), 3), track_max_m=round(float(dy[near].max()), 3),
                   field=os.path.basename(FIELD)), open(DONE, 'w'), indent=1)


if __name__ == '__main__':
    main()
