"""The track's ground photo (source/textures/GoogleMap.jpg: 1060 m round the origin, north up, x falling to the
right) made again where Google's satellite pictures are better: where the photo is smeared (its top and bottom
bands, and a strip down its west edge, are its edge rows and columns repeated, not a picture), where the ground has changed since, and where they are sharper.

    python tools/ground_photo.py fetch      # Google's pictures over the photo, kept in tools/reference/ground_photo/tiles
    python tools/ground_photo.py compare    # where the photo would change, and why: build/cache/ground_photo/compare.jpg
    python tools/ground_photo.py make       # the photo made again (the one before kept in build/cache)

fetch: Google's Static Maps satellite pictures at ZOOM with scale 2, as fine as the photo is (0.13 m a pixel
here), each with its bottom LOGO pixels (the logo and credits) cut off, laid edge to edge with a little overlap
over AREA. Needs GOOGLE_API_KEY in .env (as tools/elevation.py does), with the Maps Static API on.

compare and make: the pictures laid onto the photo's own pixels by tools/elevation.py's fit of AC's x and z to
the Earth, then moved by however far the photo and they are apart (found by correlating them where both are
pictures). Square BLOCKs of the photo are then taken from the pictures where the photo's is smeared (it has
hardly any detail one way and some the other), where they differ (CHANGED: how little their detail matches,
or, where neither has much, how far apart their colours are), or where the pictures' is SHARPER, those last two
grown into whole places (GROW, SPECK, HULL: the circuit is taken whole, not block by block); the pictures
coloured as the photo is where both show the same, and their blocks blended in over FEATHER blocks. What it did
is kept in tools/reference/ground_photo/ground_photo.json.
"""
import datetime, io, json, math, os, shutil, sys, urllib.parse, urllib.request

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label, map_coordinates, zoom as zoom_array

import build_paths as bp
import elevation as ev
from surroundings import unworld, world

Image.MAX_IMAGE_PIXELS = None
PHOTO = os.path.join(bp.ROOT, 'source', 'textures', 'GoogleMap.jpg')
SPAN = 1060.                # metres the photo spans, round the origin
OUT = os.path.join(bp.ROOT, 'tools', 'reference', 'ground_photo')
TILES = os.path.join(OUT, 'tiles')
WORK = os.path.join(bp.ROOT, 'build', 'cache', 'ground_photo')
BEFORE = os.path.join(bp.ROOT, 'build', 'cache', 'GoogleMap_before_ground_photo.jpg')
ZOOM = 19                   # with scale 2: world pixels of zoom 20, 0.13 m here, as the photo's
SIZE, LOGO = 1280, 60       # a picture's pixels a side (640 at scale 2), and the bottom ones cut off
STEP = (1200, 1120)         # pixels from one picture's middle to the next, across and down
MARGIN = 20.                # metres the pictures reach past the photo
BLOCK = 64                  # the photo's pixels a block side (8.3 m)
FEATHER = 1.                # blocks a taken block is blended in over
SMEAR = .12                 # a block whose detail one way is under this of the other's: smeared
PAD_FLAT, PAD_MARGIN = 1., 32   # rows (columns) changing less than this from one to the next: padding; pixels more
CHANGED = .35               # blocks whose detail matches less than this (correlation): changed
FLAT, COLOUR = 6., 18.      # detail (grey's spread) under which a block is flat; colours further apart: changed
SHARPER = 1.6               # the pictures' block this many times as sharp: taken
GROW, SPECK = 6, 6          # blocks between changed ones closed (50 m), and the fewest a changed place has
HULL = 100                  # blocks a changed place has from which all within its outline is taken
QUALITY = 95                # the photo's JPEG quality, as it was


def photo_px():
    with Image.open(PHOTO) as im:
        return im.size[0]


def pixel_xz(n, rows, cols):
    """AC's x and z of photo pixels' middles."""
    k = n / SPAN
    return SPAN / 2 - (cols + .5) / k, SPAN / 2 - (rows + .5) / k


def tile_grid(loc):
    """The pictures' middles in world pixels covering the photo and MARGIN round it."""
    a = SPAN / 2 + MARGIN
    lat, lon = ev.to_latlon(loc, [-a, -a, a, a], [-a, a, -a, a])
    px, py = world(lat, lon, ZOOM + 1)
    w, h = SIZE, SIZE - LOGO
    xs = np.arange(px.min() + w / 2 - 20, px.max() - w / 2 + STEP[0], STEP[0])
    ys = np.arange(py.min() + h / 2 - 20, py.max() - h / 2 + STEP[1], STEP[1])
    return xs, ys


def fetch():
    loc = json.load(open(ev.LOCATION))
    os.makedirs(TILES, exist_ok=True)
    xs, ys = tile_grid(loc)
    k = None
    got = kept = 0
    for i, cy in enumerate(ys):
        for j, cx in enumerate(xs):
            path = os.path.join(TILES, '%d_%d.png' % (i, j))
            if os.path.exists(path):
                kept += 1
                continue
            k = k or ev.key()
            # (its kept part, the bottom LOGO pixels cut off, is centred LOGO / 2 pixels above its middle)
            lat, lon = unworld(cx, cy + LOGO / 2, ZOOM + 1)
            url = 'https://maps.googleapis.com/maps/api/staticmap?' + urllib.parse.urlencode(dict(
                center='%.7f,%.7f' % (lat, lon), zoom=ZOOM, size='%dx%d' % (SIZE // 2, SIZE // 2), scale=2,
                maptype='satellite', key=k))
            data = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': ev.UA}), timeout=60).read()
            Image.open(io.BytesIO(data)).convert('RGB').crop((0, 0, SIZE, SIZE - LOGO)).save(path)
            got += 1
            print('  picture %d, %d' % (i, j), flush=True)
    print('%d pictures fetched, %d had already (%d by %d)' % (got, kept, len(ys), len(xs)))


def lay(n, fit=None, step=1):
    """The pictures laid onto the photo's pixels (every step-th), each photo pixel (column, row) taken from where
    fit (2 by 3: to the pixel the Earth's fit puts there) sends it: n / step pixels a side, and where none was."""
    loc = json.load(open(ev.LOCATION))
    xs, ys = tile_grid(loc)
    m = n // step
    out = np.zeros((m, m, 3), np.uint8)
    have = np.zeros((m, m), bool)
    tiles = {}
    cols = np.arange(m) * step + (step - 1) / 2
    for r0 in range(0, m, max(1, 512 // step)):
        rows = np.arange(r0, min(r0 + max(1, 512 // step), m))
        C, R = np.meshgrid(cols, rows * step + (step - 1) / 2)
        if fit is not None:
            C, R = fit[0, 0] * C + fit[0, 1] * R + fit[0, 2], fit[1, 0] * C + fit[1, 1] * R + fit[1, 2]
        x, z = pixel_xz(n, R, C)
        lat, lon = ev.to_latlon(loc, x.ravel(), z.ravel())
        px, py = world(lat, lon, ZOOM + 1)
        i = np.clip(np.round((py - ys[0]) / STEP[1]).astype(int), 0, len(ys) - 1)
        j = np.clip(np.round((px - xs[0]) / STEP[0]).astype(int), 0, len(xs) - 1)
        block = np.zeros((len(px), 3), np.uint8)
        got = np.zeros(len(px), bool)
        for key in set(zip(i.tolist(), j.tolist())):
            path = os.path.join(TILES, '%d_%d.png' % key)
            if not os.path.exists(path):
                continue
            if key not in tiles:
                tiles[key] = np.asarray(Image.open(path))
            img = tiles[key]
            sel = (i == key[0]) & (j == key[1])
            u = px[sel] - (xs[key[1]] - SIZE / 2)
            v = py[sel] - (ys[key[0]] - (SIZE - LOGO) / 2)
            inside = (u >= 0) & (u <= SIZE - 1) & (v >= 0) & (v <= SIZE - LOGO - 1)
            block[sel] = np.stack([map_coordinates(img[..., c].astype(np.float32), [v, u], order=1, mode='nearest')
                                   for c in range(3)], -1).clip(0, 255).astype(np.uint8)
            got[np.flatnonzero(sel)[inside]] = True
        out[rows[0]:rows[-1] + 1] = block.reshape(len(rows), m, 3)
        have[rows[0]:rows[-1] + 1] = got.reshape(len(rows), m)
    return out, have


def grey(a):
    return a[..., :3].astype(np.float32) @ np.array([.299, .587, .114], np.float32)


def blocks(a, b=BLOCK):
    """An image's BLOCKs: (rows, cols, b, b[, ...])."""
    h, w = a.shape[0] // b, a.shape[1] // b
    return a[:h * b, :w * b].reshape(h, b, w, b, *a.shape[2:]).swapaxes(1, 2)


def smeared(g):
    """Each block's: whether it is smeared (its detail along one way hardly any, the other's some)."""
    gx = np.abs(np.diff(g, axis=1, append=g[:, -1:]))
    gy = np.abs(np.diff(g, axis=0, append=g[-1:, :]))
    bx = blocks(gx).mean((2, 3))
    by = blocks(gy).mean((2, 3))
    lo, hi = np.minimum(bx, by), np.maximum(bx, by)
    return (lo < SMEAR * hi) & (hi > .5)


def correlate(a, b):
    """How far b's content is from a's (pixels, x and y: a(q) = b(q - shift)), by phase correlation; and how
    strongly they match (the peak)."""
    size = a.shape[0]
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = A * np.conj(B)
    r = np.fft.ifft2(R / np.maximum(np.abs(R), 1e-9)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)

    def sub(c, m1, p1):
        d = m1 - 2 * c + p1
        return 0. if abs(d) < 1e-12 else .5 * (m1 - p1) / d
    dy = y + sub(r[y, x], r[y - 1, x], r[(y + 1) % size, x])
    dx = x + sub(r[y, x], r[y, x - 1], r[y, (x + 1) % size])
    return (dx - size if dx > size / 2 else dx), (dy - size if dy > size / 2 else dy), float(r.max())


def padded(g):
    """The photo's padding: the rows and columns from its edges that are the edge's repeated (hardly any change
    from one to the next), and PAD_MARGIN more pixels in, where it was blended into the picture: a mask."""
    out = np.zeros(g.shape, bool)
    for axis in (0, 1):
        d = np.abs(np.diff(g, axis=axis)).mean(1 - axis)
        flat = d < PAD_FLAT
        a = int(np.argmax(~flat)) if flat[0] else 0
        b = len(d) - int(np.argmax(~flat[::-1])) if flat[-1] else len(d) + 1
        lo, hi = (a + PAD_MARGIN if a else 0), (b - PAD_MARGIN if b <= len(d) else g.shape[axis])
        if axis == 0:
            out[:lo] = out[hi:] = True
        else:
            out[:, :lo] = out[:, hi:] = True
    return out


def offsets(o, p, have, n, size=1024, grid=6):
    """Where the pictures' content is for patches of the photo across its middle: (column, row, column',
    row', peak) each, the photo's pixel at (column, row) showing what the pictures' does at (column', row')."""
    go, gp = grey(o), grey(p)
    out = []
    for cy in np.linspace(n * .17, n * .83, grid).astype(int):
        for cx in np.linspace(n * .17, n * .83, grid).astype(int):
            sl = (slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2))
            if not have[sl].all():
                continue
            dx, dy, peak = correlate(go[sl], gp[sl])
            out.append((cx, cy, cx - dx, cy - dy, peak))
    return np.array(out)


def fit_affine(S, keep=3.):
    """The 2 by 3 map from the photo's pixels to the pictures' (as laid) that best fits the patches, those off it
    by more than keep pixels left out (again until none is); and how far off each patch is."""
    use = S[:, 4] > .02
    for _ in range(10):
        X = np.c_[S[use, 0], S[use, 1], np.ones(use.sum())]
        M = np.linalg.lstsq(X, S[use, 2:4], rcond=None)[0].T
        off = np.hypot(*(np.c_[S[:, 0], S[:, 1], np.ones(len(S))] @ M.T - S[:, 2:4]).T)
        now = (S[:, 4] > .02) & (off < keep)
        if (now == use).all():
            break
        use = now
    return M, off, use


def tone(p, o, use):
    """The pictures coloured as the photo is: each channel's mean and spread matched over the blocks used."""
    P, O = blocks(p).astype(np.float32), blocks(o).astype(np.float32)
    out = p.astype(np.float32)
    for c in range(3):
        a, b = P[use][..., c], O[use][..., c]
        out[..., c] = (out[..., c] - a.mean()) * (b.std() / max(a.std(), 1e-6)) + b.mean()
    return out.clip(0, 255)


def judge(o, p, have):
    """Each block's reason to be taken from the pictures (0 keep the photo's, 1 smeared, 2 changed, 3 sharper)."""
    go, gp = grey(o), grey(p)
    Bo, Bp = blocks(go), blocks(gp)
    so, sp = Bo.std((2, 3)), Bp.std((2, 3))
    ao, ap = Bo - Bo.mean((2, 3), keepdims=True), Bp - Bp.mean((2, 3), keepdims=True)
    ncc = (ao * ap).mean((2, 3)) / np.maximum(so * sp, 1e-6)
    lap = lambda g: np.abs(g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])  # noqa: E731
    lo_, lp_ = np.pad(lap(go), 1), np.pad(lap(gp), 1)
    sharp_o, sharp_p = blocks(lo_).mean((2, 3)), blocks(lp_).mean((2, 3))
    full = blocks(have).all((2, 3))
    smear = (smeared(go) | blocks(padded(go)).any((2, 3)))[:full.shape[0], :full.shape[1]]
    reason = np.zeros(full.shape, np.uint8)
    flat = (so < FLAT) & (sp < FLAT)
    co = blocks(o.astype(np.float32)).mean((2, 3))
    cp = blocks(p.astype(np.float32)).mean((2, 3))
    # (colours compared after the pictures are toned as the photo, over the blocks neither smeared nor flat)
    same = full & ~smear & (ncc > .6)
    k = np.array([cp[same][:, c].std() and co[same][:, c].std() / cp[same][:, c].std() for c in range(3)])
    cp_t = (cp - cp[same].mean(0)) * k + co[same].mean(0)
    apart = np.linalg.norm(cp_t - co, axis=-1)
    changed = np.where(flat, apart > COLOUR, ncc < CHANGED)
    # (changed or sharper blocks made whole areas: gaps up to GROW blocks closed, holes filled, specks under
    # SPECK blocks left out, so a changed place, the circuit, is taken whole, not block by block)
    better = full & ~smear & (changed | ((sharp_p > SHARPER * sharp_o) & ~flat))
    disk = np.hypot(*np.mgrid[-GROW:GROW + 1, -GROW:GROW + 1]) <= GROW
    grown = binary_fill_holes(binary_closing(np.pad(better, GROW), disk)[GROW:-GROW, GROW:-GROW])
    lab, k = label(grown)
    size = np.bincount(lab.ravel(), minlength=k + 1)
    grown &= size[lab] >= SPECK
    # (a place of HULL blocks or more: all within its outline, its middle too)
    from scipy.spatial import Delaunay
    rr, cc = np.mgrid[:grown.shape[0], :grown.shape[1]]
    for j in np.flatnonzero(size >= HULL):
        if j == 0:
            continue
        pts = np.c_[rr[lab == j], cc[lab == j]]
        grown |= Delaunay(pts).find_simplex(np.c_[rr.ravel(), cc.ravel()]).reshape(grown.shape) >= 0
    reason[full & grown & ~changed] = 3
    reason[full & grown & changed] = 2
    reason[full & smear] = 1
    return reason, same, dict(ncc=ncc, sharp_o=sharp_o, sharp_p=sharp_p)


def compose(o, p, reason, same):
    """The photo with the blocks to take from the (toned) pictures blended in."""
    n = o.shape[0]
    pt = tone(p, o, same)
    m = gaussian_filter((reason > 0).astype(np.float32), FEATHER)
    m = np.maximum(m, (reason > 0) * .999)
    M = zoom_array(m, n / m.shape[0], order=1)[:n, :n, None]
    return (o.astype(np.float32) * (1 - M) + pt * M).clip(0, 255).astype(np.uint8)


def work():
    """The photo, the pictures laid on it (fitted onto it), each block's reason, and what was found."""
    o = np.asarray(Image.open(PHOTO).convert('RGB'))
    n = o.shape[0]
    print('laying the pictures on the photo (%d pixels a side)...' % n, flush=True)
    p, have = lay(n)
    S = offsets(o, p, have, n)
    M, off, use = fit_affine(S)
    a, b = M[:, 0], M[:, 1]
    turn = math.degrees(math.atan2(a[1], a[0]))
    scale = math.sqrt(abs(np.linalg.det(M[:, :2])))
    print('  fitted from %d of %d patches: turned %.3f degrees, scaled %.4f, moved %.1f, %.1f pixels; off it by %.2f '
          'pixels at most' % (use.sum(), len(S), turn, scale, M[0, 2], M[1, 2], off[use].max()), flush=True)
    del p, have
    p, have = lay(n, M)
    R = offsets(o, p, have, n)
    left = np.hypot(R[:, 0] - R[:, 2], R[:, 1] - R[:, 3])
    print('  laid on it: patches off by %.2f pixels (median), %.2f at most' % (np.median(left), left.max()), flush=True)
    reason, same, _ = judge(o, p, have)
    return o, p, have, reason, same, dict(turn_degrees=round(turn, 4), scale=round(scale, 5),
                                          move_px=[round(float(M[0, 2]), 2), round(float(M[1, 2]), 2)],
                                          left_px=round(float(np.median(left)), 2))


def counts(reason):
    t = reason.size
    return {w: round(float((reason == k).sum()) / t, 4) for k, w in ((1, 'smeared'), (2, 'changed'), (3, 'sharper'))}


def compare():
    o, p, have, reason, same, shift = work()
    os.makedirs(WORK, exist_ok=True)
    small = 1024
    so = np.asarray(Image.fromarray(o).resize((small, small), Image.LANCZOS))
    sp = np.asarray(Image.fromarray(p).resize((small, small), Image.LANCZOS))
    tint = np.array([[0, 0, 0], [255, 60, 60], [255, 200, 0], [60, 160, 255]], np.float32)[reason]
    T = zoom_array(tint, (small / reason.shape[0], small / reason.shape[1], 1), order=0)[:small, :small]
    over = (so * .55 + T * .45 * (T.sum(-1, keepdims=True) > 0) + so * .45 * (T.sum(-1, keepdims=True) == 0)).clip(0, 255)
    made = np.asarray(Image.fromarray(compose(o, p, reason, same)).resize((small, small), Image.LANCZOS))
    Image.fromarray(np.concatenate([np.concatenate([so, sp], 1), np.concatenate([over.astype(np.uint8), made], 1)], 0)) \
        .save(os.path.join(WORK, 'compare.jpg'), quality=90)
    print('blocks to take: %s; fit %s' % (counts(reason), shift))
    print('saved %s: the photo, the pictures; where (red smeared, yellow changed, blue sharper), the photo made'
          % os.path.relpath(os.path.join(WORK, 'compare.jpg'), bp.ROOT))


def make():
    o, p, have, reason, same, shift = work()
    out = compose(o, p, reason, same)
    if not os.path.exists(BEFORE):
        os.makedirs(os.path.dirname(BEFORE), exist_ok=True)
        shutil.copy2(PHOTO, BEFORE)
    Image.fromarray(out).save(PHOTO, quality=QUALITY, subsampling=0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(photo='source/textures/GoogleMap.jpg', made=datetime.date.today().isoformat(),
                   source='Google Maps Static API, satellite, zoom %d scale 2' % ZOOM,
                   fit=shift, block_px=BLOCK, taken=counts(reason),
                   before=os.path.relpath(BEFORE, bp.ROOT).replace(os.sep, '/')),
              open(os.path.join(OUT, 'ground_photo.json'), 'w'), indent=1)
    print('made %s: %s of its blocks taken from the pictures (%s); the one before is %s'
          % (os.path.relpath(PHOTO, bp.ROOT), '%.0f%%' % (100 * float((reason > 0).mean())), counts(reason),
             os.path.relpath(BEFORE, bp.ROOT)))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    {'fetch': fetch, 'compare': compare, 'make': make}.get(cmd, lambda: sys.exit(__doc__))()
