"""Grass for the BeamNG.drive level: clumps of blades standing on the ground where Assetto Corsa
grows its Custom Shaders Patch grass, as BeamNG has no grass of its own on meshes.

Where: on the meshes the track's GRASS_FX config names, where no mesh of its occluding
materials (roads, kerbs, tyres, buildings) lies over them, and where the ground photo is green,
as that config's colour mask keeps grass off what is not; within BAND metres of the layout's
racing line, as further out it would only cost frames. Each clump is two crossed quads with a
texture of blades made here, a little turned, sized and tilted at random; the clumps are cut
in tiles of TILE metres, a shape each, so BeamNG draws only those in view.
"""
import io, math, os, re

import numpy as np
from PIL import Image, ImageDraw

BAND = 18.          # metres either side of the racing line
STEP = .6           # metres between clumps, jittered
TILE = 64.          # metres a side of a shape
HEIGHT = (.18, .34)  # metres, the clumps' height range
WIDTH = .55         # metres across a clump
GREEN = .015        # how much greener than red and blue the photo must be
SEED = 7


def texture(path):
    """The blades: a clump of thin tapered blades in the photo's grass greens, clear around."""
    rng = np.random.default_rng(SEED)
    w, h = 256, 256
    img = Image.new('RGBA', (w, h), (70, 90, 45, 0))
    dr = ImageDraw.Draw(img)
    for _ in range(70):
        x = rng.uniform(.1, .9) * w
        top = rng.uniform(.0, .45) * h
        lean = rng.normal(0, .12) * w
        half = rng.uniform(1.5, 3.5)
        g = rng.uniform(.75, 1.15)
        col = tuple(int(min(255, c * g)) for c in (72, 98, 44)) + (255,)
        dr.polygon([(x - half, h), (x + half, h), (x + lean + .4, top), (x + lean - .4, top)], fill=col)
    # what shows through takes the blades' own colour at its edges, not black
    a = np.asarray(img).copy()
    clear = a[..., 3] == 0
    a[clear, :3] = (70, 90, 45)
    Image.fromarray(a).save(path)
    Image.fromarray(a[..., 3]).save(os.path.splitext(path)[0] + '_opacity.png')


def _config(track):
    path = os.path.join(track, 'extension', 'ext_config.ini')
    text = open(path, encoding='utf-8', errors='replace').read() if os.path.exists(path) else ''
    body = re.search(r'\[GRASS_FX\]\n(.*?)(?=\n\[|\Z)', text, re.S)
    body = body.group(1) if body else ''
    get = lambda k: [x.strip() for x in (re.search(r'^%s\s*=(.*)$' % k, body, re.M) or [None, ''])[1].split(',') if x.strip()]
    return set(get('GRASS_MESHES')), set(get('OCCLUDING_MATERIALS'))


def places(track, models, line):
    """Where clumps stand, in AC's axes: (n, 3). models: {file: kn5io.Model}; line: (n, 2) the
    racing line's x, z."""
    grass_meshes, occluding = _config(track)
    if not grass_meshes:
        return np.zeros((0, 3))
    tris, uvs, kind, photo = [], [], [], []
    photos = {}
    for fn, m in models.items():
        for x in m.meshes:
            mat = m.materials[x['mat']]
            is_grass = x['name'] in grass_meshes
            if not (is_grass or mat['name'] in occluding) or not (x['visible'] and x['renderable']):
                continue
            f = x['f'][(x['f'][:, 0] != x['f'][:, 1]) & (x['f'][:, 1] != x['f'][:, 2]) & (x['f'][:, 0] != x['f'][:, 2])]
            if not len(f):
                continue
            pi = -1
            if is_grass:
                tex = mat['tex'].get('txDiffuse')
                if (fn, tex) not in photos and tex in m.textures:
                    blk = m.textures[tex]
                    n = int.from_bytes(blk[4:8], 'little')
                    Image.MAX_IMAGE_PIXELS = None
                    photos[(fn, tex)] = np.asarray(Image.open(io.BytesIO(blk[8 + n + 4:])).convert('RGB'))
                pi = list(photos).index((fn, tex)) if (fn, tex) in photos else -1
            tris.append(x['v'][f][:, :, :3])
            uvs.append(x['v'][f][:, :, 6:8])
            kind.append(np.full(len(f), is_grass))
            photo.append(np.full(len(f), pi))
    tris, uvs = np.concatenate(tris), np.concatenate(uvs)
    kind, photo = np.concatenate(kind), np.concatenate(photo)
    photos = list(photos.values())
    # the triangles by 4 m cells
    cell = 4.
    lo = np.floor(tris[:, :, [0, 2]].min(1) / cell).astype(int)
    hi = np.floor(tris[:, :, [0, 2]].max(1) / cell).astype(int)
    grid = {}
    for k in range(len(tris)):
        for gx in range(lo[k, 0], hi[k, 0] + 1):
            for gz in range(lo[k, 1], hi[k, 1] + 1):
                grid.setdefault((gx, gz), []).append(k)
    grid = {c: np.array(v) for c, v in grid.items()}
    # candidates: a jittered grid over the band round the line
    rng = np.random.default_rng(SEED)
    x0, z0 = line.min(0) - BAND
    x1, z1 = line.max(0) + BAND
    gx, gz = np.meshgrid(np.arange(x0, x1, STEP), np.arange(z0, z1, STEP))
    pts = np.c_[gx.ravel(), gz.ravel()] + rng.uniform(-STEP / 2, STEP / 2, (gx.size, 2))
    near = np.zeros(len(pts), bool)
    for i in range(0, len(line), 4):
        near |= np.linalg.norm(pts - line[i], axis=1) < BAND
    pts = pts[near]
    out = []
    for p in pts:
        ks = grid.get((int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell))))
        if ks is None:
            continue
        t = tris[ks]
        a, b, c = t[:, 0, [0, 2]], t[:, 1, [0, 2]], t[:, 2, [0, 2]]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-9
        d = np.where(ok, d, 1.)
        l1 = ((b[:, 1] - c[:, 1]) * (p[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[1] - c[:, 1])) / d
        l2 = ((c[:, 1] - a[:, 1]) * (p[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[1] - c[:, 1])) / d
        l3 = 1 - l1 - l2
        hit = np.flatnonzero(ok & (l1 >= 0) & (l2 >= 0) & (l3 >= 0))
        if not len(hit):
            continue
        y = l1[hit] * t[hit, 0, 1] + l2[hit] * t[hit, 1, 1] + l3[hit] * t[hit, 2, 1]
        top = hit[y.argmax()]
        k = ks[top]
        if not kind[k] or photo[k] < 0:
            continue
        # anything that keeps grass off lying over it, even a little below the grass's top
        if (~kind[ks[hit]] & (y > y.max() - .05)).any():
            continue
        uv = l1[top] * uvs[k, 0] + l2[top] * uvs[k, 1] + l3[top] * uvs[k, 2]
        img = photos[photo[k]]
        r, g, bl = img[int(uv[1] % 1 * img.shape[0]) % img.shape[0], int(uv[0] % 1 * img.shape[1]) % img.shape[1]] / 255.
        if g - max(r, bl) < GREEN:
            continue
        out.append((p[0], y.max(), p[1]))
    return np.array(out).reshape(-1, 3)


def clumps(at, to_beamng):
    """The clumps standing at at (n, 3, AC's axes), by tile: {(i, j): (P, N, U, F)} in BeamNG's."""
    rng = np.random.default_rng(SEED + 1)
    tiles = {}
    for p in at:
        key = (int(p[0] // TILE), int(p[2] // TILE))
        tiles.setdefault(key, []).append(p)
    out = {}
    for key, ps in tiles.items():
        ps = np.array(ps)
        n = len(ps)
        h = rng.uniform(*HEIGHT, n)
        w = WIDTH * rng.uniform(.8, 1.2, n)
        turn = rng.uniform(0, math.pi, n)
        tilt = rng.normal(0, .06, (n, 2))
        P, U = [], []
        for q in range(2):
            a = turn + q * math.pi / 2
            dx, dz = np.cos(a) * w / 2, np.sin(a) * w / 2
            base = ps[:, None, :].repeat(4, 1).copy()
            base[:, 0, [0, 2]] -= np.c_[dx, dz]
            base[:, 1, [0, 2]] += np.c_[dx, dz]
            base[:, 2] = base[:, 1]
            base[:, 3] = base[:, 0]
            for c in (2, 3):
                base[:, c, 1] += h
                base[:, c, [0, 2]] += tilt * h[:, None]
            # a little into the ground, so no clump floats over a dip
            base[:, :2, 1] -= .03
            P.append(base)
            U.append(np.tile([[0, 1], [1, 1], [1, 0], [0, 0]], (n, 1, 1)))
        P = np.concatenate(P).reshape(-1, 3)
        U = np.concatenate(U).reshape(-1, 2).astype(float)
        quads = np.arange(len(P)).reshape(-1, 4)
        F = np.r_[quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]
        # lit as the ground under them is, from above
        N = np.tile([0, 1., 0], (len(P), 1))
        out[key] = (to_beamng(P), to_beamng(N), np.c_[U[:, 0], 1 - U[:, 1]], F)
    return out
