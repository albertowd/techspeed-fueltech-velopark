"""The real ground's heights, from Google's Elevation API, against the track's, in tools/reference/elevation.

    python tools/elevation.py locate     # where the track lies on the Earth, from OpenStreetMap
    python tools/elevation.py fetch      # Google's heights over it, saved as google_<date>.json
    python tools/elevation.py fetch wide # over the whole near ground, as google_wide_<date>.json
    python tools/elevation.py compare [google_<date>.json]   # against the track's ground
    python tools/elevation.py methods [google_<date>.json]   # which interpolation comes nearest it

locate fits the roads of source/models/velopark.glb, from above, to the kart track's raceways in
OpenStreetMap round the track's geotag (GEOTAG, as in the layouts' ui_track.json): turned and
moved, not scaled, so as many of OpenStreetMap's points along them as can lie on the roads. It
saves the fit in location.json, which the other two read: how AC's x and z map to latitude and
longitude, and how much of OpenStreetMap's lines the roads hold.

fetch asks Google for the height of every point of a GRID metre grid over the track (AREA, in AC's
x and z), BATCH points a request, with the key GOOGLE_API_KEY in .env, and saves each point's x,
z, latitude, longitude, height and the resolution Google gives it, dated, so a later fetch is
kept beside it to compare with.

compare takes the track's top surface (its ground, roads, kerbs, paths and the paddock's asphalt)
at the same points, and the heights of a fetch (the latest where none is given) less their median
difference from it, AC's y being only relative: and saves both, the difference and the plane
that best fits it (a tilt of the whole ground) in compare_<date>.csv, a summary in
compare_<date>.json, and a picture of the three in compare_<date>.png, roads outlined. Where
the eucalyptus plantation stands (build_trees.band) Google's heights are its tops, so the
differences are given apart for the roads, the open ground and the plantation, and the offset and
tilt are the roads'.

methods interpolates a fetch's points the ways terrain is commonly made from them, from points
further and further apart, and saves how near each comes to the track's ground in
methods_<date>.json: whether the way the ground was interpolated, or the data it was made from,
is what makes it differ from Google's.
"""
import datetime, glob, io, json, math, os, sys, urllib.parse, urllib.request

import numpy as np

import build_paths as bp
import build_trees
import gltf_source

ROOT = bp.ROOT
OUT = os.path.join(ROOT, 'tools', 'reference', 'elevation')
LOCATION = os.path.join(OUT, 'location.json')
GEOTAG = (-29.821383, -51.324252)        # ui_track.json's, the origin of the fit's east and north
EARTH = 6378137.                         # metres, WGS84's equator radius
AREA = ((-260., 300.), (-140., 380.))    # AC's x and z the heights are fetched over
# fetch wide: the whole of the track's near ground (1GRASS_0, 1000 by 866 m), more sparsely
WIDE, WIDE_GRID = ((-500., 500.), (-440., 440.)), 20.
GRID = 10.                               # metres apart; Google's data is some 9.5 m apart here
BATCH = 400                              # points a request: Google takes 512, URLs to 16 kB
SKIP = (412134875, 95507048)             # OpenStreetMap's car circuit and drag strip, not the kart track's
OVERPASS = ('https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter')
SURFACE = ('1ROAD_', '1GRASS_0', '1GRASS_PATHS', '1KERB_')   # the meshes that are the ground's top
UA = 'techspeed-fueltech-velopark-elevation'
SPACINGS = (10, 20, 40, 60, 80, 120)    # metres apart, the points methods interpolates from


# ---- where the track lies -------------------------------------------------------------------
def enu(lat, lon):
    """Metres east and north of GEOTAG."""
    lat0, lon0 = GEOTAG
    return np.c_[np.radians(np.asarray(lon, float) - lon0) * EARTH * math.cos(math.radians(lat0)),
                 np.radians(np.asarray(lat, float) - lat0) * EARTH]


def to_latlon(loc, x, z):
    """AC's x and z to latitude and longitude, by the fit."""
    c, s = math.cos(loc['theta']), math.sin(loc['theta'])
    u, v = -np.asarray(x, float), np.asarray(z, float)          # AC's x runs west, z north
    e, n = c * u - s * v + loc['east'], s * u + c * v + loc['north']
    lat0, lon0 = GEOTAG
    return lat0 + np.degrees(n / EARTH), lon0 + np.degrees(e / (EARTH * math.cos(math.radians(lat0))))


def to_ac(loc, P):
    """Metres east and north of GEOTAG to AC's x and z, by the fit."""
    c, s = math.cos(loc['theta']), math.sin(loc['theta'])
    Q = P - [loc['east'], loc['north']]
    return np.c_[-(c * Q[:, 0] + s * Q[:, 1]), -s * Q[:, 0] + c * Q[:, 1]]


def overpass(query):
    for url in OVERPASS:
        try:
            req = urllib.request.Request(url, urllib.parse.urlencode({'data': query}).encode(), {'User-Agent': UA})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print('  %s: %s' % (url, e))
    raise SystemExit('no Overpass server answered')


def locate():
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt, map_coordinates
    from scipy.optimize import minimize
    from shapely.geometry import LineString
    lat0, lon0 = GEOTAG
    osm = overpass('[out:json][timeout:25];way(around:600,%f,%f)["highway"="raceway"];out geom tags;' % (lat0, lon0))
    ways = [e for e in osm['elements'] if e['id'] not in SKIP]
    pts = []
    for e in ways:
        line = LineString(enu([p['lat'] for p in e['geometry']], [p['lon'] for p in e['geometry']]))
        pts += [line.interpolate(s).coords[0] for s in np.arange(0, line.length, 2.)]
    pts = np.array(pts)
    # the roads from above, and how far from them every half metre is
    res, x0, z0, nx, nz = .5, -300., -200., 1400, 1200
    img = Image.new('L', (nx, nz), 0)
    draw = ImageDraw.Draw(img)
    for n, m in bp.meshes(gltf_source.load_kn5(bp.TRACK)):
        if n['name'].startswith('1ROAD_') and len(n['faces']):
            for t in bp.triangles(n, m):
                draw.polygon([((p[0] - x0) / res, (p[2] - z0) / res) for p in t], fill=255)
    road = np.asarray(img) > 0
    far = distance_transform_edt(~road) * res

    def off(p):
        A = to_ac(dict(theta=p[0], east=p[1], north=p[2]), pts)
        return map_coordinates(far, [(A[:, 1] - z0) / res, (A[:, 0] - x0) / res], order=1, cval=60.)

    def cost(p):
        return np.mean(np.minimum(off(p), 15.) ** 2)
    zs, xs = np.nonzero(road)
    cx, cz = xs.mean() * res + x0, zs.mean() * res + z0
    best = None
    for deg in range(-180, 180, 5):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        t0 = pts.mean(0) - [-c * cx - s * cz, -s * cx + c * cz]
        for de in (-80, -40, 0, 40, 80):
            for dn in (-80, -40, 0, 40, 80):
                r = minimize(cost, [th, t0[0] + de, t0[1] + dn], method='Nelder-Mead',
                             options=dict(xatol=1e-5, fatol=1e-5, maxiter=800))
                if best is None or r.fun < best.fun:
                    best = r
    d = off(best.x)
    loc = dict(theta=float(best.x[0]), east=float(best.x[1]), north=float(best.x[2]))
    lat, lon = to_latlon(loc, [0.], [0.])
    out = dict(
        about="AC's (x, z) to the Earth: east, north = R(theta) (-x, z) + (east, north), in metres from geotag; "
              "lat, lon from them on a sphere of radius %d m" % EARTH,
        made=datetime.date.today().isoformat(), geotag=list(GEOTAG),
        theta=loc['theta'], theta_degrees=round(math.degrees(loc['theta']), 4),
        east=round(loc['east'], 3), north=round(loc['north'], 3),
        origin_latlon=[round(float(lat[0]), 7), round(float(lon[0]), 7)],
        osm_ways=[e['id'] for e in ways], osm_points=len(pts),
        on_road=round(float((d == 0).mean()), 4), off_road_p95_m=round(float(np.percentile(d, 95)), 2))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(LOCATION, 'w'), indent=1)
    print('turned %.3f deg, AC origin %.1f m east %.1f m north of the geotag, at %.7f, %.7f'
          % (out['theta_degrees'], out['east'], out['north'], *out['origin_latlon']))
    print('%.1f%% of OpenStreetMap\'s points on the roads, 95%% within %.2f m' % (out['on_road'] * 100, out['off_road_p95_m']))


# ---- Google's heights -----------------------------------------------------------------------
def key():
    for line in open(os.path.join(ROOT, '.env')):
        k, _, v = line.strip().partition('=')
        if k == 'GOOGLE_API_KEY':
            return v.strip().strip('"\'')
    raise SystemExit('no GOOGLE_API_KEY in .env')


def grid(area=AREA, step=GRID):
    (xa, xb), (za, zb) = area
    x, z = np.meshgrid(np.arange(xa, xb + .1, step), np.arange(za, zb + .1, step))
    return x.ravel(), z.ravel(), x.shape


def fetch(wide=False):
    area, step = (WIDE, WIDE_GRID) if wide else (AREA, GRID)
    loc = json.load(open(LOCATION))
    x, z, shape = grid(area, step)
    lat, lon = to_latlon(loc, x, z)
    k, rows = key(), []
    for i in range(0, len(x), BATCH):
        where = '|'.join('%.7f,%.7f' % p for p in zip(lat[i:i + BATCH], lon[i:i + BATCH]))
        url = 'https://maps.googleapis.com/maps/api/elevation/json?' + urllib.parse.urlencode({'locations': where, 'key': k})
        got = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=60))
        if got.get('status') != 'OK':
            raise SystemExit('Google: %s %s' % (got.get('status'), got.get('error_message', '')))
        for j, r in enumerate(got['results']):
            rows.append([float(x[i + j]), float(z[i + j]), round(lat[i + j], 7), round(lon[i + j], 7),
                         round(r['elevation'], 3), round(r['resolution'], 2)])
        print('  %d of %d' % (len(rows), len(x)))
    day = datetime.date.today().isoformat()
    out = dict(source='Google Maps Elevation API', fetched=day, grid_m=step, area_x_z=area, shape=list(shape),
               location=os.path.basename(LOCATION), columns=['x', 'z', 'lat', 'lon', 'elevation', 'resolution'], rows=rows)
    path = os.path.join(OUT, ('google_wide_%s.json' if wide else 'google_%s.json') % day)
    json.dump(out, open(path, 'w'), separators=(',', ':'))
    e = np.array([r[4] for r in rows])
    print('saved %s: %.2f to %.2f m, resolution %s m' % (os.path.relpath(path, ROOT), e.min(), e.max(),
                                                       sorted({r[5] for r in rows})))


# ---- against the track ----------------------------------------------------------------------
def track_surface():
    tris = []
    for path in (bp.TRACK, bp.KERBS, bp.PADDOCK):
        for n, m in bp.meshes(gltf_source.load_kn5(path)):
            if n['name'].startswith(SURFACE) and len(n['faces']):
                tris.append(bp.triangles(n, m))
    return bp.Surface(np.concatenate(tris))


def roads_outline():
    segs = []
    for n, m in bp.meshes(gltf_source.load_kn5(bp.TRACK)):
        if n['name'].startswith('1ROAD_') and len(n['faces']):
            segs += list(bp.triangles(n, m)[:, :, [0, 2]])
    return segs


def ramp(v, lo, hi, cols):
    """Values to colours along cols, from lo to hi."""
    t = np.nan_to_num(np.clip((v - lo) / (hi - lo), 0, 1)) * (len(cols) - 1)
    i = np.minimum(t.astype(int), len(cols) - 2)
    f = (t - i)[..., None]
    c = np.array(cols, float)
    return c[i] * (1 - f) + c[i + 1] * f


def picture(path, shape, model, google, diff, title, scale):
    from PIL import Image, ImageDraw
    terrain = [(40, 70, 140), (60, 140, 170), (120, 180, 120), (220, 210, 120), (190, 110, 60), (250, 245, 240)]
    div = [(40, 80, 170), (130, 170, 220), (245, 245, 245), (230, 140, 110), (170, 40, 40)]
    # the heights' colours span the open ground's, so the plantation's tops do not wash them out
    lo, hi = np.nanpercentile(np.r_[model[scale], google[scale]], [1, 99])
    lim = max(1., np.nanpercentile(np.abs(diff[scale]), 98))
    px, gap, head = 6, 20, 40
    h, w = shape
    W, H = 3 * w * px + 4 * gap, h * px + head + 60
    img = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((gap, 10), title, fill=(0, 0, 0))
    (xa, xb), (za, zb) = AREA
    segs = roads_outline()
    for k, (vals, lab, cols, a, b) in enumerate(((model, 'track y + offset (m)', terrain, lo, hi),
                                                 (google, 'Google (m)', terrain, lo, hi),
                                                 (diff, 'Google - track (m)', div, -lim, lim))):
        rgb = ramp(vals.reshape(shape), a, b, cols)
        rgb[np.isnan(vals.reshape(shape))] = 200
        # x runs west, so east (lower x) is drawn on the right; north (higher z) up
        im = Image.fromarray(rgb[::-1, ::-1].astype(np.uint8)).resize((w * px, h * px), Image.NEAREST)
        ox = gap + k * (w * px + gap)
        img.paste(im, (ox, head))
        layer = ImageDraw.Draw(img)
        for t in segs[::3]:
            q = [(ox + (xb - p[0]) / GRID * px + px / 2, head + (zb - p[1]) / GRID * px + px / 2) for p in t]
            layer.line(q + q[:1], fill=(30, 30, 30))
        d.text((ox, head + h * px + 6), '%s: %.1f to %.1f' % (lab, a, b), fill=(0, 0, 0))
        bar = ramp(np.linspace(a, b, w * px), a, b, cols).astype(np.uint8)[None].repeat(10, 0)
        img.paste(Image.fromarray(bar), (ox, head + h * px + 24))
    d.text((gap, H - 18), 'north up, east right; roads of velopark.glb outlined', fill=(80, 80, 80))
    img.save(path)


def compare(name=None):
    files = sorted(glob.glob(os.path.join(OUT, 'google_2*.json')))
    path = os.path.join(OUT, name) if name else files[-1]
    g = json.load(open(path))
    rows = np.array(g['rows'], float)
    x, z, elev = rows[:, 0], rows[:, 1], rows[:, 4]
    surf = track_surface()
    y = surf.ys(np.c_[x, z])
    ok = ~np.isnan(y)
    road = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(gltf_source.load_kn5(bp.TRACK))
                                      if n['name'].startswith('1ROAD_') and len(n['faces'])]))
    on = np.zeros(len(x), bool)
    on[ok] = ~np.isnan(road.ys(np.c_[x[ok], z[ok]]))
    # where the plantation stands, Google's heights are its tops, not the ground
    trees = ok & ~on & build_trees.band(x, z)
    ground = ok & ~on & ~trees
    region = np.where(on, 'road', np.where(trees, 'trees', np.where(ground, 'ground', '')))
    # the roads are the ground for sure: the offset and the tilt are theirs
    offset = float(np.median(elev[on] - y[on]))
    model = y + offset
    diff = elev - model
    A = np.c_[x[on], z[on], np.ones(on.sum())]
    (ax, az, a0), *_ = np.linalg.lstsq(A, diff[on], rcond=None)
    rest = diff - (ax * x + az * z + a0)

    def stats(v, where):
        v = v[where]
        return dict(points=int(len(v)), mean=round(float(v.mean()), 3), rms=round(float(np.sqrt((v ** 2).mean())), 3),
                    min=round(float(v.min()), 3), max=round(float(v.max()), 3), p05=round(float(np.percentile(v, 5)), 3),
                    p95=round(float(np.percentile(v, 95)), 3))
    day = g['fetched']
    summary = dict(
        google_file=os.path.basename(path), compared=datetime.date.today().isoformat(),
        offset_m=round(offset, 3),
        note='track y + offset_m is the height Google gives, at the median on the roads; difference is Google less that; '
             'regions: road (1ROAD_ of velopark.glb), trees (build_trees.band, off the roads), ground (the rest)',
        track_y=dict(min=round(float(np.nanmin(y)), 3), max=round(float(np.nanmax(y)), 3)),
        google=dict(min=round(float(elev.min()), 3), max=round(float(elev.max()), 3),
                    resolution_m=sorted({float(r) for r in rows[:, 5]})),
        difference={k: stats(diff, w) for k, w in (('road', on), ('ground', ground), ('trees', trees))},
        # the plane through the roads' difference, a tilt of the whole track against Google's: east
        # is -x and north +z, so Google's ground falls along (ax, -az) in (east, north) against it
        tilt=dict(per_100m_x=round(float(ax * 100), 3), per_100m_z=round(float(az * 100), 3),
                  degrees=round(math.degrees(math.atan(math.hypot(ax, az))), 4),
                  falls_toward_bearing=round((math.degrees(math.atan2(ax, -az)) + 360) % 360, 1)),
        difference_less_tilt={k: stats(rest, w) for k, w in (('road', on), ('ground', ground))},
        correlation={k: round(float(np.corrcoef(y[w], elev[w])[0, 1]), 4) for k, w in (('road', on), ('ground', ground))})
    base = os.path.join(OUT, 'compare_%s' % day)
    json.dump(summary, open(base + '.json', 'w'), indent=1)
    with open(base + '.csv', 'w') as f:
        f.write('x,z,lat,lon,region,track_y,track_y_offset,google,google_resolution,difference,difference_less_tilt\n')
        for i in range(len(x)):
            f.write('%.1f,%.1f,%.7f,%.7f,%s,%s\n' % (x[i], z[i], rows[i, 2], rows[i, 3], region[i], ','.join(
                ['%.3f' % y[i] if ok[i] else '', '%.3f' % model[i] if ok[i] else '', '%.3f' % elev[i], '%.2f' % rows[i, 5],
                 '%.3f' % diff[i] if ok[i] else '', '%.3f' % rest[i] if ok[i] else ''])))
    picture(base + '.png', tuple(g['shape']), np.where(ok, model, np.nan), elev, np.where(ok, diff, np.nan),
            'Google %s against the track, offset %.2f m (median on the roads); colours span the roads and '
            'ground, the plantation beyond them' % (day, offset), on | ground)
    print(json.dumps(summary, indent=1))


def methods(name=None):
    """How near Google's heights come to the track's ground, interpolated each of the ways a
    terrain is commonly made from points, from points SPACINGS apart: so whether the way the
    points were interpolated, or how far apart they were, is what the track's ground has from
    them. The points in the plantation are left out, being its tops; each way is measured
    against the track's top surface half a grid cell off the points, on the roads and the open
    ground, less the median difference."""
    from scipy.interpolate import RBFInterpolator, griddata
    from scipy.spatial import cKDTree
    files = sorted(glob.glob(os.path.join(OUT, 'google_2*.json')))
    path = os.path.join(OUT, name) if name else files[-1]
    g = json.load(open(path))
    rows = np.array(g['rows'], float)
    x, z, elev = rows[:, 0], rows[:, 1], rows[:, 4]
    h, w = g['shape']
    keep = ~build_trees.band(x, z)
    (xa, xb), (za, zb) = AREA
    qx, qz = np.meshgrid(np.arange(xa + GRID / 2, xb, GRID), np.arange(za + GRID / 2, zb, GRID))
    qx, qz = qx.ravel(), qz.ravel()
    qy = track_surface().ys(np.c_[qx, qz])
    ok = ~np.isnan(qy) & ~build_trees.band(qx, qz)
    Q, qy = np.c_[qx[ok], qz[ok]], qy[ok]

    def idw(P, v, k=8, p=2.):
        d, i = cKDTree(P).query(Q, k)
        wt = 1 / np.maximum(d, 1e-6) ** p
        return (wt * v[i]).sum(1) / wt.sum(1)
    ways = {'nearest': lambda P, v: griddata(P, v, Q, 'nearest'),
            'linear (Delaunay triangles)': lambda P, v: griddata(P, v, Q, 'linear'),
            'cubic (Clough-Tocher)': lambda P, v: griddata(P, v, Q, 'cubic'),
            'inverse distance (power 2, 8 nearest)': idw,
            'thin plate spline': lambda P, v: RBFInterpolator(P, v, kernel='thin_plate_spline', neighbors=80)(Q),
            'thin plate spline, smoothed': lambda P, v: RBFInterpolator(P, v, kernel='thin_plate_spline', neighbors=80,
                                                                        smoothing=len(v) * .5)(Q)}
    out = []
    for step in SPACINGS:
        n = int(round(step / GRID))
        sub = np.zeros((h, w), bool)
        sub[::n, ::n] = True
        sub = sub.ravel() & keep
        P, v = np.c_[x[sub], z[sub]], elev[sub]
        for way, f in ways.items():
            got = f(P, v)
            m = ~np.isnan(got)
            d = got[m] - qy[m]
            d -= np.median(d)
            out.append(dict(spacing_m=step, way=way, points_in=int(sub.sum()), measured=int(m.sum()),
                            rms_m=round(float(np.sqrt((d ** 2).mean())), 3),
                            p95_abs_m=round(float(np.percentile(np.abs(d), 95)), 3),
                            correlation=round(float(np.corrcoef(got[m], qy[m])[0, 1]), 4),
                            relief_m=round(float(np.percentile(got[m], 99) - np.percentile(got[m], 1)), 3)))
            print('%4d m  %-40s rms %.2f  p95 %.2f  corr %.3f  relief %.2f' % (
                step, way, out[-1]['rms_m'], out[-1]['p95_abs_m'], out[-1]['correlation'], out[-1]['relief_m']))
    track_relief = float(np.percentile(qy, 99) - np.percentile(qy, 1))
    print('track relief %.2f m' % track_relief)
    res = dict(google_file=os.path.basename(path), compared=datetime.date.today().isoformat(),
               note=' '.join(methods.__doc__.split()),
               measured_points=int(len(qy)), track_relief_m=round(track_relief, 3),
               relief='the 1st to 99th percentile of the heights at the points measured', results=out)
    json.dump(res, open(os.path.join(OUT, 'methods_%s.json' % g['fetched']), 'w'), indent=1)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what == 'locate':
        locate()
    elif what == 'fetch':
        fetch(sys.argv[2:3] == ['wide'])
    elif what == 'methods':
        methods(sys.argv[2] if len(sys.argv) > 2 else None)
    elif what == 'compare':
        compare(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
