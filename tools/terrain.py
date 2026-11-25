"""The terrain: the ground a track lies on, made first, before its roads and what stands on the map.

    python tools/terrain.py build        # the ground made from tools/reference/terrain (cached), its pictures and points written

A terrain is a folder (TERRAIN) the road editor (top-race-editor) makes for a new track (where it lies, its
picture, its heights) and edits after. Its terrain.json says:

- place: where the track lies on the Earth: the file that turns AC's x and z to latitude and longitude;
- image: the picture of the ground round it (file, area_x_z it covers, metres_a_pixel) and photo, the
  finer one in the middle, if there is one (file, size_m, its middle at AC's origin, u against x, v
  against z);
- heights: the files of the real ground's heights (each row x, z, latitude, longitude, height, resolution),
  offset_m from them to the track's y; buildings: OpenStreetMap's files whose buildings are not ground;
- grid: area_x_z and cell_m, the grid the ground is worked out on;
- filters: what among the heights is not the ground (the tops of trees and of buildings), as each rule
  finds it (see filter_points), and areas drawn (kind 'ground' or 'not_ground', ring [[x, z], ...]) that
  say so whatever the rules find;
- edits: changes to the ground, in order, each over the ground as those before it left it:
  raster (file: an .npz of x0, z0, cell, d: metres added on that grid, linear between its points, 0 off
  it; the editor's brushes write them), offset (ring, by metres, ease_m out of it), flat (ring, height:
  the track's y, or at [x, z] and plus metres over the ground there; slope [per metre along x, along z]
  from at; ease_m), smooth (ring, sigma_m, ease_m). An edit with "off": true is left out.

The original heights are every height there is; the filtered, those the filters keep, the ground round
them filling in where they took some out (FILLS); the ground, the filtered with the edits on them. build
writes them (original.png, filtered.png, ground.png: 16 bit, lo + value * (hi - lo) / 65535 metres, see
heights.json) and points.json (each height, and why a filter took it out: WHY) into the folder, for the
editor and for the steps after (tools/build_models.py's first step: a change to the terrain runs every
step, the roads' heights taken again from it). ground() is the ground as a function for the tools: the
roads' heights (tools/describe_roads.py), the drape (tools/drape_track.py), the far ground.
"""
import glob, hashlib, json, os, sys

import numpy as np

import build_paths as bp

TERRAIN = os.path.join(bp.ROOT, 'tools', 'reference', 'terrain')
DOC = os.path.join(TERRAIN, 'terrain.json')
TRACK_TERRAIN = TERRAIN     # (the track's own: its description where tools/track_data.py finds it, track.json or here)
CACHE = os.path.join(bp.ROOT, 'build', 'cache', 'terrain')
GROUND_CELL = 2.        # metres a pixel of ground.png
# why a height is not the ground (points.json's why): 0 it is
WHY = {0: 'ground', 1: 'trees in the picture', 2: 'trees in the photo', 3: 'a building', 4: 'over the ground round it',
       5: 'over the wide ground round it', 6: 'an area drawn as not ground'}
SMOOTH_CELL = .5        # metres, the grid a smooth edit is worked out on
VERSION = 1             # the ground's sums: one more when a change to this code changes the ground it makes


def rel(p):
    return os.path.relpath(p, bp.ROOT).replace(os.sep, '/')


def at(folder, cache=None):
    """Work on the terrain in folder (a new track's: the road editor's wizard), its cache in cache (else
    build/cache/terrain under bp.ROOT as it is now)."""
    global TERRAIN, DOC, CACHE
    TERRAIN = os.path.abspath(folder)
    DOC = os.path.join(TERRAIN, 'terrain.json')
    CACHE = cache or os.path.join(bp.ROOT, 'build', 'cache', 'terrain')
    _FP.clear()
    _GROUND.clear()


def _own():
    return os.path.abspath(TERRAIN) == os.path.abspath(TRACK_TERRAIN)


def load(path=None):
    """The terrain's description: path's, or the track's (track.json's terrain, or terrain.json while it is not in it),
    or a new track's in its folder (at())."""
    if path:
        return json.load(open(path, encoding='utf-8'))
    if _own():
        import track_data
        return track_data.part('terrain')[0]
    return json.load(open(DOC, encoding='utf-8'))


def exists():
    if _own():
        import track_data
        return track_data.part('terrain')[0] is not None
    return os.path.exists(DOC)


def edit_file(e):
    """An edit's file: named from the folder of the file the description is in (the track's root in track.json)."""
    if _own():
        import track_data
        return os.path.join(track_data.part('terrain')[1] or TERRAIN, e['file'])
    return os.path.join(TERRAIN, e['file'])


def doc_files():
    if _own():
        import track_data
        return track_data.files('terrain')
    return [DOC] if os.path.exists(DOC) else []


def _files(patterns):
    out = []
    for p in patterns:
        out += sorted(glob.glob(os.path.join(bp.ROOT, p)))
    return out


def fingerprint(doc=None):
    """What the ground hangs on: the terrain's description, its edits' files, the heights' and pictures'
    files (by name and size), the version of the sums (VERSION)."""
    doc = doc or load()
    h = hashlib.sha1(json.dumps(doc, sort_keys=True).encode())
    h.update(b'terrain %d' % VERSION)
    files = _files(doc['heights']['files']) + _files(doc.get('buildings', {}).get('files', []))
    files += [os.path.join(bp.ROOT, d['file']) for d in (doc['image'], doc.get('photo')) if d]
    files += [edit_file(e) for e in doc.get('edits', []) if e.get('file')]
    for f in files:
        h.update(('%s %d' % (rel(f), os.path.getsize(f) if os.path.exists(f) else -1)).encode())
    for e in doc.get('edits', []):
        if e.get('file') and os.path.exists(edit_file(e)):
            h.update(open(edit_file(e), 'rb').read())
    return h.hexdigest()[:16]


_FP = {}


def key():
    """fingerprint() of the terrain as it is on disk, worked out once a process (each time its files change)."""
    if not exists():
        return 'none'
    doc = load()
    st = tuple((f, os.path.getmtime(f)) for f in doc_files() + [edit_file(e) for e in doc.get('edits', []) if e.get('file')]
               if os.path.exists(f))
    if _FP.get('st') != st:
        _FP.update(st=st, key=fingerprint(doc))
    return _FP['key']


# ---- the heights and their filters -------------------------------------------------------------

def points(doc):
    """Every height: x, z, the track's y."""
    rows = []
    for f in _files(doc['heights']['files']):
        rows += json.load(open(f))['rows']
    R = np.array(rows, float)
    return R[:, 0], R[:, 1], R[:, 4] - doc['heights']['offset_m']


def _share(lum, both, lum_max, sd_max, near):
    """The share of trees round each pixel of a picture (a metre a pixel): darker than lum_max and more uneven
    over 9 m than sd_max, both or either, within near metres."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(lum, 9)
    spread = np.sqrt(np.maximum(uniform_filter(lum ** 2, 9) - mean ** 2, 0))
    tree = (spread > sd_max) & (mean < lum_max) if both else (spread > sd_max) | (mean < lum_max)
    return uniform_filter(tree.astype(float), int(2 * near) + 1)


def _buildings(doc):
    import elevation as ev
    from shapely.geometry import Polygon
    loc = json.load(open(os.path.join(bp.ROOT, doc['place']['location'])))
    b = doc.get('buildings', {})
    out = []
    for f in _files(b.get('files', [])):
        for e in json.load(open(f))['elements']:
            if e['type'] == 'way' and 'building' in e.get('tags', {}) and 'geometry' in e:
                g = e['geometry']
                out.append(Polygon(ev.to_ac(loc, ev.enu([p['lat'] for p in g], [p['lon'] for p in g]))).buffer(b.get('buffer_m', 10.)))
    return out


def grid(doc):
    (xa, xb), (za, zb) = doc['grid']['area_x_z']
    c = doc['grid']['cell_m']
    return np.arange(xa, xb + c * .01, c), np.arange(za, zb + c * .01, c), c


def _splat(X, Z, c, x, z, v):
    A, W = np.zeros((len(Z), len(X))), np.zeros((len(Z), len(X)))
    i, j = np.round((z - Z[0]) / c).astype(int), np.round((x - X[0]) / c).astype(int)
    ok = (i >= 0) & (i < len(Z)) & (j >= 0) & (j < len(X))
    np.add.at(A, (i[ok], j[ok]), v[ok])
    np.add.at(W, (i[ok], j[ok]), 1.)
    return A, W


def filter_points(doc, x, z, h):
    """Why each height is not the ground (WHY's codes, 0 where it is): the trees in the picture (darker and
    more uneven than its rule, both), in the photo (darker or more uneven than its: its colours differ), on
    more than tree_share of the ground within tree_near metres; within a building's buffer; more than over_m
    over the heights' opening (their lowest round, over opening_m) or wide_over_m over a wide one (for woods
    too wide for the first); then the areas drawn, which say ground or not whatever those find."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter, grey_opening, map_coordinates
    from shapely import contains_xy
    from shapely.geometry import Polygon
    Image.MAX_IMAGE_PIXELS = None
    f = doc['filters']
    why = np.zeros(len(x), int)

    def mark(code, sel):
        why[(why == 0) & sel] = code
    near = f['tree_near_m']
    im = doc['image']
    (xa, xb), (za, zb) = im['area_x_z']
    px = im['metres_a_pixel']
    lum = np.asarray(Image.open(os.path.join(bp.ROOT, im['file'])).convert('L'), float)
    sat = _share(lum, True, f['picture_trees']['lum'], f['picture_trees']['sd'], near)
    col, row = ((xb - x) / px).astype(int), ((zb - z) / px).astype(int)
    inside = (col >= 0) & (col < sat.shape[1]) & (row >= 0) & (row < sat.shape[0])
    share = np.ones(len(x))
    share[inside] = sat[row[inside], col[inside]]
    mark(1, share >= f['tree_share'])
    ph = doc.get('photo')
    sel = np.zeros(len(x), bool)
    if ph:
        n = int(ph['size_m'])
        lum = np.asarray(Image.open(os.path.join(bp.ROOT, ph['file'])).convert('L').resize((n, n), Image.BOX), float)
        photo = _share(lum, False, f['photo_trees']['lum'], f['photo_trees']['sd'], near)
        hs = ph['size_m'] / 2
        in_photo = (np.abs(x) < hs) & (np.abs(z) < hs)
        at = photo[np.clip((hs - z[in_photo]).astype(int), 0, photo.shape[0] - 1), np.clip((hs - x[in_photo]).astype(int), 0, photo.shape[1] - 1)]
        sel[in_photo] = at >= f['tree_share']
    mark(2, sel)
    for b in _buildings(doc):
        mark(3, contains_xy(b, x, z))
    X, Z, c = grid(doc)
    A, W = _splat(X, Z, c, x, z, h)
    tops = gaussian_filter(A, 1.) / np.maximum(gaussian_filter(W, 1.), 1e-9)
    at = lambda G: map_coordinates(G, [(z - Z[0]) / c, (x - X[0]) / c], order=1)
    mark(4, h - at(gaussian_filter(grey_opening(tops, size=int(f['opening_m'] / c) | 1), 1.)) >= f['over_m'])
    mark(5, h - at(gaussian_filter(grey_opening(tops, size=int(f['wide_opening_m'] / c) | 1), 2.)) >= f['wide_over_m'])
    for a in f.get('areas', []):
        inside = contains_xy(Polygon(a['ring']).buffer(0), x, z)
        if a['kind'] == 'ground':
            why[inside] = 0
        else:
            why[inside & (why == 0)] = 6
    return why


def fill(doc, x, z, h):
    """The heights given on the grid: where there are points enough, theirs smoothed over the finest of
    FILLS that has enough, the coarser filling in between them."""
    from scipy.ndimage import gaussian_filter
    X, Z, c = grid(doc)
    A, W = _splat(X, Z, c, x, z, h)
    G = np.full(A.shape, float(np.median(h)))
    enough = doc['filters']['enough_a_cell']
    for s in doc['filters']['fills_m'][::-1]:
        a, w = gaussian_filter(A, s / c), gaussian_filter(W, s / c)
        t = np.clip(w / enough, 0, 1)
        G += t * t * (3 - 2 * t) * (a / np.maximum(w, 1e-9) - G)
    return G


def layers(doc=None, verbose=True):
    """The terrain worked out (cached by its fingerprint in CACHE): {'original', 'filtered': grids on grid(doc),
    'x', 'z', 'h', 'why': the heights}."""
    doc = doc or load()
    fp = fingerprint(doc)
    path = os.path.join(CACHE, 'layers_%s.npz' % fp)
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    x, z, h = points(doc)
    why = filter_points(doc, x, z, h)
    g = why == 0
    out = dict(x=x, z=z, h=h, why=why, original=fill(doc, x, z, h), filtered=fill(doc, x[g], z[g], h[g]))
    if verbose:
        print('  %d heights, %d of them the ground; %.1f to %.1f m' % (len(h), g.sum(), out['filtered'].min(), out['filtered'].max()))
    os.makedirs(CACHE, exist_ok=True)
    for old in glob.glob(os.path.join(CACHE, 'layers_*.npz')):
        os.remove(old)
    np.savez_compressed(path, **out)
    return out


def sampler(G, doc, order=3):
    """A grid on grid(doc) as a function of x, z: cubic between its points (order 3), or linear."""
    from scipy.ndimage import map_coordinates
    X, Z, c = grid(doc)
    return lambda xx, zz: map_coordinates(G, [(np.asarray(zz, float) - Z[0]) / c, (np.asarray(xx, float) - X[0]) / c],
                                          order=order, mode='nearest')


def filtered(order=3, doc=None):
    """The filtered heights, no edits on them: height(x, z)."""
    doc = doc or load()
    return sampler(layers(doc)['filtered'], doc, order)


# ---- the edits ---------------------------------------------------------------------------------

def _smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3 - 2 * t)


def _weight(ring, ease, x, z):
    """1 inside the ring, easing to 0 over ease metres out of it."""
    import shapely
    from shapely.geometry import Polygon
    P = Polygon(ring).buffer(0)
    pts = shapely.points(np.c_[np.ravel(x), np.ravel(z)])
    d = shapely.distance(P, pts).reshape(np.shape(x))
    return 1. - _smoothstep(d / ease) if ease > 0 else (d <= 0).astype(float)


def _raster(path):
    d = np.load(path)
    return float(d['x0']), float(d['z0']), float(d['cell']), d['d'].astype(float)


def _on(R, xx, zz):
    from scipy.ndimage import map_coordinates
    x0, z0, c, D = R
    return map_coordinates(D, [(zz - z0) / c, (xx - x0) / c], order=1, cval=0.)


def edit(prev, e):
    """The ground with one edit on it (see the module's note): height(x, z) of the ground before it."""
    t = e['type']
    if t == 'raster':
        R = _raster(edit_file(e))
        return lambda xx, zz: prev(xx, zz) + _on(R, np.asarray(xx, float), np.asarray(zz, float))
    ring, ease = e['ring'], float(e.get('ease_m', 0.))
    if t == 'offset':
        return lambda xx, zz: prev(xx, zz) + float(e['by']) * _weight(ring, ease, xx, zz)
    if t == 'flat':
        if 'at' in e:
            a = np.array(e['at'], float)
            h0 = float(prev(a[:1], a[1:])[0]) + float(e.get('plus', 0.))
        else:
            a, h0 = np.zeros(2), float(e['height'])
        gx, gz = e.get('slope', (0., 0.))

        def flat(xx, zz):
            xx, zz = np.asarray(xx, float), np.asarray(zz, float)
            y = prev(xx, zz)
            return y + (h0 + gx * (xx - a[0]) + gz * (zz - a[1]) - y) * _weight(ring, ease, xx, zz)
        return flat
    if t == 'smooth':
        from scipy.ndimage import gaussian_filter
        r = np.asarray(ring, float)
        s = float(e['sigma_m'])
        lo, hi = r.min(0) - ease - 3 * s, r.max(0) + ease + 3 * s
        X, Z = np.arange(lo[0], hi[0] + SMOOTH_CELL, SMOOTH_CELL), np.arange(lo[1], hi[1] + SMOOTH_CELL, SMOOTH_CELL)
        XX, ZZ = np.meshgrid(X, Z)
        Y = prev(XX, ZZ)
        D = (gaussian_filter(Y, s / SMOOTH_CELL, mode='nearest') - Y) * _weight(ring, ease, XX, ZZ)
        R = (X[0], Z[0], SMOOTH_CELL, D)
        return lambda xx, zz: prev(xx, zz) + _on(R, np.asarray(xx, float), np.asarray(zz, float))
    raise ValueError('an edit of no type known: %r' % t)


_GROUND = {}


def ground(order=3, doc=None):
    """The ground: the filtered heights (cubic between their points, or linear: order) with every edit on
    them, in order: height(x, z), the track's y."""
    doc = doc or load()
    fp = (fingerprint(doc), order)
    if fp not in _GROUND:
        g = filtered(order, doc)
        for e in doc.get('edits', []):
            if not e.get('off'):                       # (an edit turned off, to compare: as if not there)
                g = edit(g, e)
        _GROUND.clear()
        _GROUND[fp] = g
    return _GROUND[fp]


# ---- for the editor: the layers as pictures, the heights as points --------------------------------

def _png16(G, path):
    from PIL import Image
    lo, hi = float(np.nanmin(G)), float(np.nanmax(G))
    v = np.round((G - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
    Image.fromarray(v[::-1, ::-1]).save(path)       # (north up, west on the left: x grows west, z north)
    return lo, hi


def write(doc=None):
    """original.png, filtered.png, ground.png, heights.json and points.json into the terrain's folder."""
    doc = doc or load()
    L = layers(doc)
    X, Z, c = grid(doc)
    meta = {'about': "16 bit heightmaps, north up and west on the left (a pixel's x falls left to right, its z top to "
                     "bottom): a pixel's value v is the track's y lo + v * (hi - lo) / 65535 metres; the first pixel "
                     "(top left) at x_max, z_max", 'fingerprint': fingerprint(doc), 'layers': {}}
    for name in ('original', 'filtered'):
        lo, hi = _png16(L[name], os.path.join(TERRAIN, name + '.png'))
        meta['layers'][name] = dict(file=name + '.png', x_max=float(X[-1]), z_max=float(Z[-1]), cell_m=c,
                                    size=[len(X), len(Z)], lo=lo, hi=hi)
    g = ground(3, doc)
    Xg, Zg = np.arange(X[0], X[-1] + 1e-6, GROUND_CELL), np.arange(Z[0], Z[-1] + 1e-6, GROUND_CELL)
    XX, ZZ = np.meshgrid(Xg, Zg)
    G = g(XX.ravel(), ZZ.ravel()).reshape(XX.shape)
    lo, hi = _png16(G, os.path.join(TERRAIN, 'ground.png'))
    meta['layers']['ground'] = dict(file='ground.png', x_max=float(Xg[-1]), z_max=float(Zg[-1]), cell_m=GROUND_CELL,
                                    size=[len(Xg), len(Zg)], lo=lo, hi=hi)
    why = L['why']
    meta['heights'] = {'count': int(len(why)), 'why': {WHY[k]: int((why == k).sum()) for k in WHY}}
    json.dump(meta, open(os.path.join(TERRAIN, 'heights.json'), 'w'), indent=1)
    rows = [[round(float(a), 2), round(float(b), 2), round(float(h), 3), int(w)] for a, b, h, w in zip(L['x'], L['z'], L['h'], why)]
    json.dump({'columns': ['x', 'z', 'y', 'why'], 'why': WHY, 'rows': rows}, open(os.path.join(TERRAIN, 'points.json'), 'w'),
              separators=(',', ':'))
    print('  the ground %.2f to %.2f m; %d heights, %s' % (lo, hi, len(why), ', '.join('%d %s' % (v, k) for k, v in meta['heights']['why'].items() if v)))
    return meta


# ---- a new terrain (the editor's wizard for a new track) -------------------------------------------

# the filters' rules, as they found the ground at the Velopark (Google's heights there, its satellite
# picture's and ground photo's colours): a new track's to start from
FILTERS = {'picture_trees': {'lum': 75., 'sd': 12.}, 'photo_trees': {'lum': 84., 'sd': 13.}, 'tree_near_m': 7., 'tree_share': .3,
           'over_m': 1.5, 'opening_m': 55., 'wide_over_m': 4., 'wide_opening_m': 250.,
           'fills_m': [20., 40., 80., 160., 320.], 'enough_a_cell': .16, 'areas': []}


def new(location, image, heights, offset_m=0., photo=None, buildings=(), cell_m=10., margin_m=100.):
    """A terrain's description for a new track, no edits yet (paths from ROOT): location, the file that turns
    AC's x, z to latitude and longitude; image, {file, area_x_z, metres_a_pixel, source}; heights, the height
    files' globs; offset_m from their heights to the track's y; photo, {file, size_m} (the picture itself
    where there is no finer one); buildings, OpenStreetMap's files. Its grid the picture's area and margin_m."""
    (xa, xb), (za, zb) = image['area_x_z']
    return {'version': 1, 'place': {'location': location}, 'image': dict(image),
            'photo': dict(photo) if photo else None,
            'heights': {'files': list(heights), 'offset_m': offset_m},
            'buildings': {'files': list(buildings), 'buffer_m': 10.},
            'grid': {'area_x_z': [[xa - margin_m, xb + margin_m], [za - margin_m, zb + margin_m]], 'cell_m': cell_m},
            'filters': json.loads(json.dumps(FILTERS)), 'edits': []}


def main():
    a = sys.argv[1:] or ['build']
    if a[0] == 'build':
        if not exists():
            print('no terrain (%s): the ground as the models have it' % rel(DOC))
            return
        write()
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
