"""What lies round the track, 3 km by 3 km, for reference: what could be added round it (trees,
roads, fields, water), until the far terrain is made. In tools/reference/surroundings.

    python tools/surroundings.py heights     # Google's heights, where tools/elevation.py has none
    python tools/surroundings.py photo       # Google's satellite pictures, round the ground photo
    python tools/surroundings.py osm         # OpenStreetMap's roads, land use, woods and water
    python tools/surroundings.py relief      # all the heights as one picture, with contours

AREA is AC's x and z (x runs west, z north), round the origin as the ground photo is; the pictures
cover PHOTO_AREA, which reaches as far west as the track's far ground (GRASS_1) does.

heights: Google's Elevation API every GRID metres over AREA, but where tools/elevation.py's
google_*.json or google_wide_*.json already has them; saved as google_heights_<date>.json, as
those are (x, z, lat, lon, elevation, resolution).

photo: Google's Static Maps satellite pictures, at zoom ZOOM with scale 2 (about half a metre a
pixel), each with its bottom LOGO pixels (Google's logo and credits) cut off, laid edge to edge
with a little overlap; none where the track's ground photo (GoogleMap.jpg, 1060 m round the
origin) already shows it. Laid onto AC's x and z, north up and x falling to the right as the
ground photo is, PIXEL metres a pixel, over the ground photo (which shows only where no picture
was fetched, in the middle: its edges are smeared): surroundings.jpg,
and surroundings.json, where it lies. Its pictures are kept in tiles/, so it lays them again
without fetching them.

relief: every height there is (tools/elevation.py's and these) on one GRID metre grid over AREA,
coloured from low to high with a contour every CONTOUR metres: relief.png, PIXEL metres a pixel as
surroundings.jpg is, to lay over it.

heights and photo need GOOGLE_API_KEY in .env (as tools/elevation.py does), with the Elevation and the Maps
Static APIs on.
"""
import datetime, glob, io, json, math, os, sys, urllib.parse, urllib.request

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

import build_paths as bp
import elevation as ev

OUT = os.path.join(bp.ROOT, 'tools', 'reference', 'surroundings')
TILES = os.path.join(OUT, 'tiles')
AREA = ((-1500., 1500.), (-1500., 1500.))   # AC's x and z
PHOTO_AREA = ((-1500., 1800.), (-1500., 1500.))   # and the pictures', as far west as the far ground reaches
GRID = 20.                  # metres between the heights
ZOOM = 17                   # the pictures' zoom; with scale 2, as fine as zoom 18
SIZE, LOGO = 1280, 60       # a picture's pixels a side (640 at scale 2), and the bottom ones cut off
STEP = (1240, 1160)         # pixels from one picture's middle to the next, across and down
PIXEL = 1.                  # metres a pixel of surroundings.jpg
PHOTO, PHOTO_SIZE = 'GoogleMap.jpg', 1060.  # the track's ground photo, and the metres it spans
CONTOUR = 2.                # metres between relief.png's contours


def world(lat, lon, zoom=ZOOM + 1):
    """Web Mercator's world pixels at zoom: x east, y south."""
    n = 256 * 2 ** zoom
    lat = np.radians(np.clip(lat, -85, 85))
    return (np.asarray(lon) + 180) / 360 * n, (1 - np.log(np.tan(lat) + 1 / np.cos(lat)) / math.pi) / 2 * n


def unworld(px, py, zoom=ZOOM + 1):
    n = 256 * 2 ** zoom
    lon = np.asarray(px) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * np.asarray(py) / n))))
    return lat, lon


def have_heights():
    """Where tools/elevation.py's heights are: each file's area."""
    out = []
    for f in glob.glob(os.path.join(ev.OUT, 'google_2*.json')) + glob.glob(os.path.join(ev.OUT, 'google_wide_*.json')):
        (xa, xb), (za, zb) = json.load(open(f))['area_x_z']
        out.append((xa, xb, za, zb))
    return out


def heights():
    loc = json.load(open(ev.LOCATION))
    x, z, shape = ev.grid(AREA, GRID)
    new = np.ones(len(x), bool)
    for xa, xb, za, zb in have_heights():
        new &= ~((x >= xa) & (x <= xb) & (z >= za) & (z <= zb))
    x, z = x[new], z[new]
    lat, lon = ev.to_latlon(loc, x, z)
    k, rows = ev.key(), []
    print('%d points to fetch, %d requests' % (len(x), -(-len(x) // ev.BATCH)))
    for i in range(0, len(x), ev.BATCH):
        where = '|'.join('%.7f,%.7f' % p for p in zip(lat[i:i + ev.BATCH], lon[i:i + ev.BATCH]))
        url = 'https://maps.googleapis.com/maps/api/elevation/json?' + urllib.parse.urlencode({'locations': where, 'key': k})
        got = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': ev.UA}), timeout=60))
        if got.get('status') != 'OK':
            raise SystemExit('Google: %s %s' % (got.get('status'), got.get('error_message', '')))
        for j, r in enumerate(got['results']):
            rows.append([float(x[i + j]), float(z[i + j]), round(lat[i + j], 7), round(lon[i + j], 7),
                         round(r['elevation'], 3), round(r['resolution'], 2)])
        print('  %d of %d' % (len(rows), len(x)))
    os.makedirs(OUT, exist_ok=True)
    day = datetime.date.today().isoformat()
    path = os.path.join(OUT, 'google_heights_%s.json' % day)
    json.dump(dict(source='Google Maps Elevation API', fetched=day, grid_m=GRID, area_x_z=AREA,
                   left_out='the areas of tools/reference/elevation/google_*.json and google_wide_*.json',
                   location='../elevation/location.json', columns=['x', 'z', 'lat', 'lon', 'elevation', 'resolution'],
                   rows=rows), open(path, 'w'), separators=(',', ':'))
    e = np.array([r[4] for r in rows])
    print('saved %s: %.1f to %.1f m' % (os.path.relpath(path, bp.ROOT), e.min(), e.max()))


def tile_grid(loc):
    """The pictures' middles in world pixels covering PHOTO_AREA, and their first ones' numbers: the
    grid is laid from AREA's corner, so the pictures it had keep their numbers when it grows."""
    def corners(area):
        (xa, xb), (za, zb) = area
        lat, lon = ev.to_latlon(loc, [xa, xa, xb, xb], [za, zb, za, zb])
        return world(lat, lon)
    w, h = SIZE, SIZE - LOGO
    px, py = corners(AREA)
    x0, y0 = px.min() + w / 2 - 20, py.min() + h / 2 - 20
    qx, qy = corners(PHOTO_AREA)
    j0 = int(np.floor((qx.min() + w / 2 - x0) / STEP[0]))
    j1 = int(np.ceil((qx.max() - w / 2 - x0) / STEP[0])) + 1
    i0 = int(np.floor((qy.min() + h / 2 - y0) / STEP[1]))
    i1 = int(np.ceil((qy.max() - h / 2 - y0) / STEP[1])) + 1
    return x0 + STEP[0] * np.arange(j0, j1 + 1), y0 + STEP[1] * np.arange(i0, i1 + 1), (i0, j0)


def in_photo(loc, cx, cy):
    """Whether the picture round cx, cy shows only what the ground photo does."""
    w, h = SIZE / 2, (SIZE - LOGO) / 2
    lat, lon = unworld(np.array([cx - w, cx + w, cx - w, cx + w]), np.array([cy - h, cy - h, cy + h, cy + h]))
    P = ev.to_ac(loc, ev.enu(lat, lon))
    return bool(np.all(np.abs(P) < PHOTO_SIZE / 2 - 10))


def photo():
    loc = json.load(open(ev.LOCATION))
    os.makedirs(TILES, exist_ok=True)
    xs, ys, (i0, j0) = tile_grid(loc)
    k = None
    fetched = kept = skipped = 0
    for i, cy in enumerate(ys, i0):
        for j, cx in enumerate(xs, j0):
            path = os.path.join(TILES, '%d_%d.png' % (i, j))
            if in_photo(loc, cx, cy):
                skipped += 1
                continue
            if os.path.exists(path):
                kept += 1
                continue
            k = k or ev.key()
            # the picture's middle: its bottom LOGO pixels are cut off, so its kept part is centred
            # LOGO / 2 pixels above it
            lat, lon = unworld(cx, cy + LOGO / 2)
            url = 'https://maps.googleapis.com/maps/api/staticmap?' + urllib.parse.urlencode(dict(
                center='%.7f,%.7f' % (lat, lon), zoom=ZOOM, size='%dx%d' % (SIZE // 2, SIZE // 2), scale=2,
                maptype='satellite', key=k))
            data = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': ev.UA}), timeout=60).read()
            Image.open(io.BytesIO(data)).convert('RGB').crop((0, 0, SIZE, SIZE - LOGO)).save(path)
            fetched += 1
            print('  picture %d, %d' % (i, j))
    print('%d pictures fetched, %d had already, %d in the ground photo' % (fetched, kept, skipped))
    lay(loc, xs, ys, (i0, j0))


def lay(loc, xs, ys, first):
    """The pictures and the ground photo laid onto AC's x and z: surroundings.jpg."""
    (xa, xb), (za, zb) = PHOTO_AREA
    W, H = int((xb - xa) / PIXEL), int((zb - za) / PIXEL)
    out = np.zeros((H, W, 3), np.uint8)
    # the track's ground photo first, under the pictures: its edges are smeared, and the pictures
    # cover all of it but where none was fetched, its middle
    import gltf_source
    track = gltf_source.load_kn5(bp.TRACK)
    data = next(t['data'] for t in track['textures'] if t['name'] == PHOTO)
    n = int(PHOTO_SIZE / PIXEL)
    ph = np.asarray(Image.open(io.BytesIO(data)).convert('RGB').resize((n, n), Image.LANCZOS))
    c0, r0 = int((xb - PHOTO_SIZE / 2) / PIXEL), int((zb - PHOTO_SIZE / 2) / PIXEL)
    out[r0:r0 + n, c0:c0 + n] = ph
    for r0 in range(0, H, 500):
        rows = np.arange(r0, min(r0 + 500, H))
        cols = np.arange(W)
        C, R = np.meshgrid(cols, rows)
        x, z = xb - (C + .5) * PIXEL, zb - (R + .5) * PIXEL
        lat, lon = ev.to_latlon(loc, x.ravel(), z.ravel())
        px, py = world(lat, lon)
        # the picture nearest each point's middle
        i = np.clip(np.round((py - ys[0]) / STEP[1]).astype(int), 0, len(ys) - 1)
        j = np.clip(np.round((px - xs[0]) / STEP[0]).astype(int), 0, len(xs) - 1)
        block = out[rows[0]:rows[-1] + 1].reshape(-1, 3).copy()
        for key in set(zip(i.tolist(), j.tolist())):
            path = os.path.join(TILES, '%d_%d.png' % (key[0] + first[0], key[1] + first[1]))
            if not os.path.exists(path):
                continue
            sel = (i == key[0]) & (j == key[1])
            img = np.asarray(Image.open(path), float)
            u = px[sel] - (xs[key[1]] - SIZE / 2)
            v = py[sel] - (ys[key[0]] - (SIZE - LOGO) / 2)
            block[sel] = np.stack([map_coordinates(img[..., c], [v, u], order=1, mode='nearest') for c in range(3)], -1).astype(np.uint8)
        out[rows[0]:rows[-1] + 1] = block.reshape(len(rows), W, 3)
    Image.fromarray(out).save(os.path.join(OUT, 'surroundings.jpg'), quality=88)
    json.dump(dict(image='surroundings.jpg', metres_a_pixel=PIXEL, area_x_z=PHOTO_AREA,
                   pixel='column = (%g - x) / %g, row = (%g - z) / %g' % (xb, PIXEL, zb, PIXEL),
                   sources=['Google Maps Static API, satellite, zoom %d scale 2, fetched %s' % (ZOOM, datetime.date.today().isoformat()),
                            'the track\'s ground photo (%s) in the middle, %g m square' % (PHOTO, PHOTO_SIZE)]),
              open(os.path.join(OUT, 'surroundings.json'), 'w'), indent=1)
    print('laid %s, %d by %d' % (os.path.relpath(os.path.join(OUT, 'surroundings.jpg'), bp.ROOT), W, H))


def osm():
    loc = json.load(open(ev.LOCATION))
    (xa, xb), (za, zb) = AREA
    lat, lon = ev.to_latlon(loc, [xa, xa, xb, xb], [za, zb, za, zb])
    box = '%.6f,%.6f,%.6f,%.6f' % (lat.min(), lon.min(), lat.max(), lon.max())
    q = ('[out:json][timeout:180];(way["highway"](%s);way["landuse"](%s);way["natural"](%s);way["waterway"](%s);'
         'way["building"](%s);relation["landuse"](%s);relation["natural"](%s););out geom;' % ((box,) * 7))
    got = ev.overpass(q)
    os.makedirs(OUT, exist_ok=True)
    day = datetime.date.today().isoformat()
    path = os.path.join(OUT, 'osm_%s.json' % day)
    json.dump(dict(got, area_x_z=AREA, fetched=day), open(path, 'w'), separators=(',', ':'))
    kinds = {}
    for e in got['elements']:
        t = e.get('tags', {})
        key = next((k for k in ('highway', 'landuse', 'natural', 'waterway', 'building') if k in t), 'other')
        kinds[key] = kinds.get(key, 0) + 1
    print('saved %s: %s' % (os.path.relpath(path, bp.ROOT), ', '.join('%d %s' % (v, k) for k, v in sorted(kinds.items()))))


def relief():
    from scipy.interpolate import griddata
    rows = []
    for f in (glob.glob(os.path.join(ev.OUT, 'google_2*.json')) + glob.glob(os.path.join(ev.OUT, 'google_wide_*.json'))
              + glob.glob(os.path.join(OUT, 'google_heights_*.json'))):
        rows += json.load(open(f))['rows']
    R = np.array(rows, float)
    (xa, xb), (za, zb) = AREA
    W, H = int((xb - xa) / PIXEL), int((zb - za) / PIXEL)
    small = 4                   # worked out at a quarter of the size, then enlarged
    C, Rw = np.meshgrid(np.arange(0, W, small), np.arange(0, H, small))
    x, z = xb - (C + .5) * PIXEL, zb - (Rw + .5) * PIXEL
    h = griddata(R[:, :2], R[:, 4], (x, z), method='linear')
    lo, hi = np.nanpercentile(h, 1), np.nanpercentile(h, 99)
    t = np.clip((h - lo) / (hi - lo), 0, 1)
    stops = np.array([[40, 90, 160], [70, 150, 90], [200, 190, 110], [150, 100, 60], [245, 245, 245]], float)
    f = t * (len(stops) - 1)
    k = np.clip(f.astype(int), 0, len(stops) - 2)
    rgb = stops[k] + (stops[k + 1] - stops[k]) * (f - k)[..., None]
    band = np.floor(np.nan_to_num(h) / CONTOUR)
    edge = (band != np.roll(band, 1, 0)) | (band != np.roll(band, 1, 1))
    rgb[edge] *= .55
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    img.save(os.path.join(OUT, 'relief.png'))
    print('laid %s: %.1f to %.1f m above the sea, a contour every %g m' % (
        os.path.relpath(os.path.join(OUT, 'relief.png'), bp.ROOT), np.nanmin(h), np.nanmax(h), CONTOUR))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    {'heights': heights, 'photo': photo, 'osm': osm, 'relief': relief}.get(cmd, lambda: sys.exit(__doc__))()
