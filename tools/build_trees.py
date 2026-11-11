"""The eucalyptus plantation along the north and the west of the track, in source/models/velopark_extra.glb.

    python tools/build_trees.py

The 0.5.0 track stands broad leaf and ash trees there, boards 21 m high some 10 m apart. The real
track has a plantation of eucalyptus: rows of them 3 m apart, 27 to 37 m high, their trunks
pale, straight and bare up to the small clumps of leaves of their crowns, with a dark, ragged
mass of shrubs and young trees at their feet. This takes the old trees out of the band (BAND)
and plants it again:

- the trees, made here in VARIANTS of their own, turned and sized at random: three dimensional,
  a trunk and branches of bark and their leaves on crossed cards, DENSE apart in the DEPTH of
  the band nearest the track and SPARSE apart behind;
- the understory, small clumps of leaves on crossed cards through a mass 3.5 to 8 m high, from
  FRONT metres before the trees to UNDER metres in, and young eucalyptus among it;
- an unseen wall round each trunk up to TRUNK metres, that cars hit.

But for the north west of it (NW), round the L1500's top hairpin, which is no plantation: broad
leaf trees there, BROAD high, round crowns of leaf clumps down low on a short trunk, BROAD_APART
apart wherever the ground's photo shows the crowns of trees (darker than DARK), with the meadow
of grass round them as the photo has it, in place of the eucalyptus and their understory.

Where the band was is taken from where its old trees stood, its front edge and its back. Nothing
is planted within CLEAR of a road, a building, a kerb or a tyre wall. The leaves' cards have
normals as if they were round, out from the middle of their clump, so a clump is lit as one.
Their textures are made here: source/textures/euca_leaves.dds, euca_bark.dds, euca_shrub.dds,
broad_leaves.dds.
The meshes are cut in TILE metre squares. The trees as made, and where each stands, turned and
sized, go in source/models/velopark_extra_trees.json too, from which tools/export_beamng.py makes
them BeamNG's forest. Run again, it finds the plantation planted.
"""
import glob, io, json, math, os, urllib.parse

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, median_filter, uniform_filter

import build_paths as bp
import gltf_source
import kn5io
import kn5tree
from plant_trees import pieces

ROOT = bp.ROOT
EXTRA = os.path.join(bp.MODELS, 'velopark_extra.glb')
# the trees as made and where each stands, for BeamNG's forest (tools/export_beamng.py)
FOREST = os.path.join(bp.MODELS, 'velopark_extra_trees.json')
TEX = os.path.join(ROOT, 'source', 'textures')
PREFIX = 'EUCA_'
HEIGHT = (27., 37.)      # metres, the trees
YOUNG = (6., 13.)        # and the young ones among the understory
VARIANTS = 12            # trees made, each turned and sized at random where it stands
YOUNG_VARIANTS = 6
DENSE, SPARSE = 3.2, 6.  # metres apart, near the front of the band and behind
DEPTH = 14.              # metres of the band planted DENSE
FRONT, UNDER = 4., 9.    # the understory, metres before the band's front edge and into it
CLUMPS = .9              # of its leaf clumps a square metre
CLEAR = 3.               # metres from a road, building, kerb or tyre wall
TRUNK = 3.5              # metres up a trunk's wall goes
TILE = 50.               # metres, the squares the meshes are cut in
NW_X, NW_Z = 100., 180.  # the north west, broad leaf trees: x and z over these
BROAD = (9., 16.)        # metres, its trees
BROAD_VARIANTS = 8
BROAD_APART = 4.5        # metres apart
DARK = 78.               # the ground photo's brightness, of 255, under which it shows trees
CROWNS = .55             # of the photo round a spot, within CROWN metres, showing trees for one to stand there
CROWN = 3.
PHOTO = 'GoogleMap.jpg'  # the ground's photo
# the textures of the trees' parts, made here, and BeamNG's forest's (tools/export_beamng.py)
TEXTURES = {'bark': 'euca_bark.dds', 'leaves': 'euca_leaves.dds', 'broad': 'broad_leaves.dds',
            'palm': 'palm_fronds.dds', 'bamboo': 'bamboo_leaves.dds', 'culm': 'bamboo_culm.dds'}
R = 1.                   # metres, the grid the band and what stands in it are found on


def north_west(x, z):
    return (x >= NW_X) & (z >= NW_Z)


def band(x, z):
    """The north and the west of the track, where the plantation is."""
    return (z >= 280) | ((x >= 160) & (z >= -45))


# ---- textures ----------------------------------------------------------------------------
def leaf_texture(seed, size=512, ss=4):
    """A clump of drooping eucalyptus leaves on thin twigs, with alpha."""
    rng = np.random.default_rng(seed)
    S = size * ss
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tips = []
    for k in range(5):
        a = math.radians(rng.uniform(-60, 60))
        x0, y0 = S * .5, S * .98
        L = S * rng.uniform(.45, .8)
        pts = [(x0, y0)]
        for i in range(1, 9):
            t = i / 8
            pts.append((x0 + math.sin(a) * L * t + rng.normal(0, S * .01), y0 - math.cos(a) * L * t + S * .05 * t * t))
        d.line(pts, fill=(88, 70, 55, 255), width=int(S * .009))
        tips += pts[3:]
    for _ in range(420):
        cx, cy = tips[rng.integers(len(tips))]
        x, y = cx + rng.normal(0, S * .07), cy + rng.normal(0, S * .06)
        if not (S * .05 < x < S * .95 and S * .03 < y < S * .92):
            continue
        ang = math.radians(rng.normal(90, 35))
        L, W = S * rng.uniform(.045, .085), S * rng.uniform(.008, .014)
        dx, dy = math.cos(ang), math.sin(ang)
        nx, ny = -dy, dx
        poly = [(x, y), (x + dx * L * .35 + nx * W, y + dy * L * .35 + ny * W), (x + dx * L, y + dy * L + L * .1),
                (x + dx * L * .35 - nx * W, y + dy * L * .35 - ny * W)]
        g = rng.uniform(.75, 1.15)
        d.polygon(poly, fill=(int(72 * g), int(92 * g), int(58 * g * rng.uniform(.85, 1.1)), 255))
    return unpremultiplied(img.resize((size, size), Image.LANCZOS), (70., 88., 56.))


def broad_texture(seed, size=512, ss=3):
    """A clump of broad leaves: scattered round a few middles, thinning out to its edges
    raggedly, lit at its top and darker inside and below."""
    rng = np.random.default_rng(seed)
    S = size * ss
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    middles = [(rng.uniform(.3, .7), rng.uniform(.3, .7), rng.uniform(.09, .16)) for _ in range(rng.integers(5, 8))]
    patch = np.asarray(Image.fromarray(rng.normal(128, 70, (6, 6)).clip(0, 255).astype(np.uint8)).resize((S, S), Image.BICUBIC), float) / 255
    for _ in range(7000):
        cx, cy, sd = middles[rng.integers(len(middles))]
        x, y = cx + rng.normal(0, sd), cy + rng.normal(0, sd)
        if not (.03 < x < .97 and .03 < y < .97):
            continue
        r = math.hypot(x - cx, y - cy) / sd
        X, Y = x * S, y * S
        L, W = S * rng.uniform(.018, .03), S * rng.uniform(.01, .016)
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        poly = [(X - dx * L, Y - dy * L), (X - dy * W, Y + dx * W), (X + dx * L, Y + dy * L), (X + dy * W, Y - dx * W)]
        g = rng.uniform(.75, 1.05) * (.6 + .45 * patch[int(min(Y, S - 1)), int(X)]) * (1.2 - .55 * y) * (.8 + .12 * min(r, 2))
        d.polygon(poly, fill=(int(64 * g), int(90 * g), int(44 * g), 255))
    return unpremultiplied(img.resize((size, size), Image.LANCZOS), (56., 78., 40.))


def unpremultiplied(img, fill):
    """Colour under the edges of what shows, so its fringe does not go dark when filtered."""
    a = np.asarray(img).astype(float)
    al = a[..., 3:] / 255
    rgb = np.where(al > .01, a[..., :3] / np.maximum(al, .01), fill)
    return Image.fromarray(np.dstack([rgb.clip(0, 255), a[..., 3]]).astype(np.uint8), 'RGBA')


def bark_texture(size=256, seed=2):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (size, size // 8))
    n = np.asarray(Image.fromarray(((n * 30) + 128).clip(0, 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), float) - 128
    patches = np.asarray(Image.fromarray(rng.normal(128, 60, (6, 3)).clip(0, 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), float) - 128
    img = np.array([150., 140., 126.]) * (1 + .08 * n[..., None] / 30) * (1 - .25 * (patches[..., None] > 30))
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


# ---- shapes -------------------------------------------------------------------------------
def tube(a, b, r0, r1, seg, v0, v1):
    ax = (b - a) / np.linalg.norm(b - a)
    u = np.cross(ax, [0, 0, 1] if abs(ax[2]) < .9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    ang = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
    V = np.r_[a + r0 * ring, b + r1 * ring]
    N = np.r_[ring, ring]
    U = np.c_[np.r_[ang, ang] / (2 * np.pi), np.r_[np.full(seg + 1, v0), np.full(seg + 1, v1)]]
    F = [(i, i + 1, seg + 2 + i) for i in range(seg)] + [(i, seg + 2 + i, seg + 1 + i) for i in range(seg)]
    return V, N, U, np.array(F)


def round_normals(V, centre, up):
    n = V - centre
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    n = n + [0, up, 0]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def cards(c, w, h, yaw, tilt, centre, up):
    """Two crossed quads of leaves hanging from c, seen from both sides, lit as a round mass."""
    out = []
    for extra in (0, np.pi / 2):
        a = yaw + extra
        d = np.array([math.cos(a), 0, math.sin(a)])
        n = np.cross(d, [0, 1, 0])
        top, bot = c + [0, h * .25, 0], c - [0, h * .75, 0] + n * tilt
        V = np.array([top - d * w / 2, top + d * w / 2, bot + d * w / 2, bot - d * w / 2])
        U = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        out.append((V, round_normals(V, centre, up), U, np.array([(0, 1, 2), (0, 2, 3), (0, 2, 1), (0, 3, 2)])))
    return out


def tree(height, seed, young=False):
    """A eucalyptus standing at the origin: {'bark': parts, 'leaves': parts}, and its trunk's radius."""
    rng = np.random.default_rng(seed)
    lean = rng.normal(0, .005, 2) * height
    top = np.array([lean[0], height, lean[1]])
    base = np.array([0., -.3, 0.])
    r = .09 + .004 * height * rng.uniform(.8, 1.2)
    out = {'bark': [], 'leaves': []}
    mid1 = base + (top - base) * .45 + [rng.normal(0, .15), 0, rng.normal(0, .15)]
    mid2 = base + (top - base) * .75 + [rng.normal(0, .2), 0, rng.normal(0, .2)]
    for a, b, ra, rb, va, vb in ((base, mid1, r, r * .75, 0, 4), (mid1, mid2, r * .75, r * .5, 4, 7), (mid2, top, r * .5, r * .15, 7, 9)):
        out['bark'].append(tube(a, b, ra, rb, 6, va, vb))
    crown0 = rng.uniform(.35, .5) if young else rng.uniform(.72, .82)
    clumps = []
    for k in range(rng.integers(3, 6)):
        t = rng.uniform(crown0, .92)
        p = base + (top - base) * t
        a = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(1.2, 3.2) * (1.15 - t) * 1.6 * (.5 if young else 1)
        e = p + [math.cos(a) * L, L * rng.uniform(.6, 1.2), math.sin(a) * L]
        out['bark'].append(tube(p, e, .05, .02, 3, 0, 1))
        clumps.append(e)
    for t in np.linspace(crown0 + .08, 1., 3):
        clumps.append(base + (top - base) * t + [rng.normal(0, .6), 0, rng.normal(0, .6)])
    crown = np.mean(clumps, axis=0)
    for c in clumps:
        s = rng.uniform(2.0, 3.6) * (.7 if young else 1)
        out['leaves'] += cards(c + [0, s * .45, 0], s * rng.uniform(.9, 1.2), s, rng.uniform(0, np.pi), rng.normal(0, .2), crown - [0, 2, 0], .6)
    return out, r


def broad(height, seed, clumps=None):
    """A broad leaf tree standing at the origin: {'bark': parts, 'broad': parts}, and its trunk's
    radius: a short trunk forking into a few branches, under a round crown of leaf clumps (as
    many as clumps says, for trees seen only from afar; else 44 to 59)."""
    rng = np.random.default_rng(seed)
    r = .12 + .012 * height * rng.uniform(.85, 1.15)
    base = np.array([0., -.3, 0.])
    fork = np.array([rng.normal(0, .2), height * rng.uniform(.28, .4), rng.normal(0, .2)])
    out = {'bark': [tube(base, fork, r, r * .8, 6, 0, 3)], 'broad': []}
    rx, ry = height * rng.uniform(.32, .42), height * rng.uniform(.36, .44)
    centre = np.array([rng.normal(0, .3), height - ry, rng.normal(0, .3)])
    for k in range(rng.integers(4, 7)):
        a = rng.uniform(0, 2 * np.pi)
        e = centre + [math.cos(a) * rx * .55, rng.uniform(-.1, .4) * ry, math.sin(a) * rx * .55]
        out['bark'].append(tube(fork, e, r * .5, r * .15, 4, 0, 2))
    n = rng.integers(44, 60)
    for k in range(clumps or n):
        u = rng.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        if u[1] < -.55:
            u[1] = -u[1] * .4
        f = math.sqrt(rng.uniform(.45, 1.))
        p = centre + u * [rx, ry, rx] * f
        s = rng.uniform(1.8, 2.8) * height / 12
        out['broad'] += cards(p + [0, s * .45, 0], s * rng.uniform(.9, 1.15), s, rng.uniform(0, np.pi), rng.normal(0, .2),
                              centre - [0, ry * .3, 0], .5)
    return out, r


def palm(height, seed):
    """A palm standing at the origin: {'bark': parts, 'palm': parts}, and its trunk's radius: a
    slender trunk, leaning and bending a little, under a crown of arching fronds."""
    rng = np.random.default_rng(seed)
    r = .14 + .006 * height
    lean = rng.normal(0, .06, 2) * height
    pts = [np.array([lean[0] * t * t, -.3 + (height + .3) * t, lean[1] * t * t]) for t in np.linspace(0, 1, 5)]
    out = {'bark': [tube(a, b, r * (1 - .25 * i / 4), r * (1 - .25 * (i + 1) / 4), 7, i * 2, i * 2 + 2)
                    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:]))], 'palm': []}
    top = pts[-1]
    for k in range(rng.integers(11, 15)):
        a = 2 * np.pi * k / 13 + rng.normal(0, .2)
        rise = rng.uniform(-.2, .9)
        L = rng.uniform(2.4, 3.4) * height / 8
        d = np.array([math.cos(a), 0, math.sin(a)])
        side = np.cross(d, [0, 1, 0])
        # the frond's spine, rising and then drooping under its weight
        spine = [top + d * L * t + np.array([0, (rise * t - 1.3 * t * t) * L * .55, 0]) for t in np.linspace(0, 1, 6)]
        w = rng.uniform(.8, 1.1) * L / 3
        V, U = [], []
        for i, c in enumerate(spine):
            t = i / 5
            ww = w * (.35 + .65 * math.sin(math.pi * min(t * 1.1, 1)))
            droop = np.array([0, -ww * .25, 0])
            V += [c - side * ww / 2 + droop, c + side * ww / 2 + droop]
            U += [(0, t), (1, t)]
        V = np.array(V)
        F = []
        for i in range(5):
            q = (2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2)
            F += [(q[0], q[1], q[2]), (q[0], q[2], q[3]), (q[0], q[2], q[1]), (q[0], q[3], q[2])]
        out['palm'].append((V, round_normals(V, top - [0, 1., 0], .8), np.array(U, float), np.array(F)))
    return out, r


def bamboo(height, seed):
    """A clump of bamboo at the origin: {'culm': parts, 'bamboo': parts}, and the clump's radius:
    many thin culms rising from a tight foot and arching out, leafy along their upper halves."""
    rng = np.random.default_rng(seed)
    out = {'culm': [], 'bamboo': []}
    foot = 1.6
    for k in range(rng.integers(28, 40)):
        a, rr = rng.uniform(0, 2 * np.pi), foot * math.sqrt(rng.uniform(0, 1))
        b = np.array([math.cos(a) * rr, -.3, math.sin(a) * rr])
        h = height * rng.uniform(.7, 1.05)
        out_d = np.array([math.cos(a), 0, math.sin(a)]) * rng.uniform(.15, .35) * h
        pts = [b + out_d * t * t + np.array([0, (h + .3) * t, 0]) for t in np.linspace(0, 1, 5)]
        cr = rng.uniform(.035, .06)
        for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
            out['culm'].append(tube(p0, p1, cr * (1 - .2 * i / 4), cr * (1 - .2 * (i + 1) / 4), 4, i * 2, i * 2 + 2))
        for t in rng.uniform(.45, 1., 4):
            i = min(int(t * 4), 3)
            c = pts[i] + (pts[i + 1] - pts[i]) * (t * 4 - i)
            s = rng.uniform(1.4, 2.4)
            out['bamboo'] += cards(c + [0, s * .3, 0], s * rng.uniform(.8, 1.1), s, rng.uniform(0, np.pi), rng.normal(0, .3),
                                   np.array([0, h * .6, 0]), .5)
    return out, foot


def palm_texture(size=512, seed=21):
    """A palm frond seen from above, along v: its rib, and narrow leaflets out to both sides."""
    rng = np.random.default_rng(seed)
    S = size * 2
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(70):
        t = .02 + .96 * i / 70
        y = t * S
        for side in (-1, 1):
            L = S * .48 * math.sin(math.pi * min(t * 1.05, 1)) * rng.uniform(.85, 1.05)
            ang = math.radians(rng.normal(70, 6))
            x1 = S / 2 + side * math.sin(ang) * L
            y1 = y + math.cos(ang) * L * .5
            g = rng.uniform(.8, 1.1)
            d.line([(S / 2, y), (x1, y1)], fill=(int(96 * g), int(122 * g), int(54 * g), 255), width=int(S * .012))
    d.line([(S / 2, 0), (S / 2, S)], fill=(150, 140, 90, 255), width=int(S * .012))
    return unpremultiplied(img.resize((size, size), Image.LANCZOS), (96., 122., 54.))


def bamboo_texture(size=512, seed=23):
    """A spray of narrow bamboo leaves, with alpha."""
    rng = np.random.default_rng(seed)
    S = size * 2
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for _ in range(260):
        x, y = rng.uniform(.1, .9) * S, rng.uniform(.08, .9) * S
        ang = math.radians(rng.normal(100, 30))
        L, W = S * rng.uniform(.08, .15), S * rng.uniform(.008, .014)
        dx, dy = math.cos(ang), math.sin(ang)
        nx, ny = -dy, dx
        poly = [(x, y), (x + dx * L * .3 + nx * W, y + dy * L * .3 + ny * W), (x + dx * L, y + dy * L),
                (x + dx * L * .3 - nx * W, y + dy * L * .3 - ny * W)]
        g = rng.uniform(.8, 1.15)
        d.polygon(poly, fill=(int(98 * g), int(128 * g), int(58 * g), 255))
    return unpremultiplied(img.resize((size, size), Image.LANCZOS), (98., 128., 58.))


def culm_texture(size=256):
    """Bamboo's culm: yellow green, a node every half of the texture."""
    y = np.arange(size)[:, None] / size
    x = np.arange(size)[None, :] / size
    node = np.exp(-(((y * 2) % 1 - .02) / .015) ** 2)
    shade = .85 + .15 * np.sin(x * 2 * np.pi)
    img = np.array([118., 140., 70.]) * (shade * (1 - .35 * node))[..., None] + node[..., None] * np.array([20., 10., 0.])
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def shape_of(v):
    """The tree a variant of the forest's data says, as tree(), broad(), palm() or bamboo() make it."""
    kind = v.get('kind', 'young' if v.get('young') else 'euca')
    if kind == 'broad':
        return broad(v['height'], v['seed'], v.get('clumps'))
    if kind == 'palm':
        return palm(v['height'], v['seed'])
    if kind == 'bamboo':
        return bamboo(v['height'], v['seed'])
    return tree(v['height'], v['seed'], kind == 'young')


def placed(parts, yaw, scale, at):
    """Parts turned by yaw, sized by scale and moved to at."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rm = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return [(V @ Rm.T * scale + at, N @ Rm.T, U, F) for V, N, U, F in parts]


# ---- where --------------------------------------------------------------------------------
class Grid:
    def __init__(s, lo, hi):
        s.lo = np.floor(lo)
        s.shape = tuple((np.ceil(hi - s.lo) / R).astype(int) + 1)

    def cells(s, P):
        return np.clip(np.round((P - s.lo) / R).astype(int), 0, np.array(s.shape) - 1)

    def mark(s, tris):
        """Cells under any of the triangles (n, 3, 3)."""
        m = np.zeros(s.shape, bool)
        # small ones by their corners and middle, big ones cell by cell
        span = np.ptp(tris[:, :, [0, 2]], axis=1).max(1)
        small = tris[span <= 2 * R]
        for P in list(small[:, :, [0, 2]].transpose(1, 0, 2)) + [small[:, :, [0, 2]].mean(1)]:
            m[tuple(s.cells(P).T)] = True
        for t in tris[span > 2 * R]:
            lo = s.cells(t[:, [0, 2]].min(0))
            hi = s.cells(t[:, [0, 2]].max(0))
            xs, zs = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
            P = np.c_[xs.ravel(), zs.ravel()] * R + s.lo
            a, b, c = t[:, [0, 2]]
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-9:
                continue
            l1 = ((b[1] - c[1]) * (P[:, 0] - c[0]) + (c[0] - b[0]) * (P[:, 1] - c[1])) / d
            l2 = ((c[1] - a[1]) * (P[:, 0] - c[0]) + (a[0] - c[0]) * (P[:, 1] - c[1])) / d
            inside = (l1 >= -.3) & (l2 >= -.3) & (1 - l1 - l2 >= -.3)
            m[xs.ravel()[inside], zs.ravel()[inside]] = True
        return m


def photo_trees(track, grid):
    """Where the ground's photo shows trees, on the grid: its colour under each cell, from the
    ground meshes' mapping of it, darker than DARK."""
    tex = {t['name']: t['data'] for t in track['textures']}
    img = np.asarray(Image.open(io.BytesIO(tex[PHOTO])).convert('L')).astype(float)
    H, W = img.shape
    nx, nz = grid.shape
    uv = np.full((nx, nz, 2), np.nan)
    for n, m in bp.meshes(track):
        mat = track['materials'][n['material']]
        if 'GRASS' not in n['name'] or not len(n['faces']) or \
                next((t['name'] for t in mat['textures'] if t['sampler'] == 'txDiffuse'), None) != PHOTO:
            continue
        V = n['verts'].astype(float)
        F = n['faces'].astype(int)
        P = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
        for t, u in zip(P[F], V[F][:, :, 6:8]):
            lo = grid.cells(t[:, [0, 2]].min(0))
            hi = grid.cells(t[:, [0, 2]].max(0))
            if (t[:, [0, 2]].max(0) < grid.lo).any() or (t[:, [0, 2]].min(0) > grid.lo + np.array(grid.shape) * R).any():
                continue
            xs, zs = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
            Pp = np.c_[xs.ravel(), zs.ravel()] * R + grid.lo
            a, b, c = t[:, [0, 2]]
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-9:
                continue
            l1 = ((b[1] - c[1]) * (Pp[:, 0] - c[0]) + (c[0] - b[0]) * (Pp[:, 1] - c[1])) / d
            l2 = ((c[1] - a[1]) * (Pp[:, 0] - c[0]) + (a[0] - c[0]) * (Pp[:, 1] - c[1])) / d
            l3 = 1 - l1 - l2
            ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            uv[xs.ravel()[ins], zs.ravel()[ins]] = l1[ins, None] * u[0] + l2[ins, None] * u[1] + l3[ins, None] * u[2]
    ok = ~np.isnan(uv[..., 0])
    lum = np.full((nx, nz), 255.)
    lum[ok] = img[np.clip((uv[ok, 1] % 1) * H, 0, H - 1).astype(int), np.clip((uv[ok, 0] % 1) * W, 0, W - 1).astype(int)]
    return lum < DARK


def edges(old):
    """The band's front edge and back, from where its old trees stood: for the north, z by x;
    for the west, x by z; each as (bins, front, back)."""
    out = {}
    for part, sel, along, across, sign in (('north', (old[:, 1] >= 280) & (old[:, 0] < 160), 0, 1, 1),
                                           ('west', old[:, 0] >= 160, 1, 0, 1)):
        p = old[sel]
        bins = np.arange(np.floor(p[:, along].min() / 10) * 10, p[:, along].max() + 10, 10.)
        front, back = [], []
        for b in bins:
            q = p[(p[:, along] >= b - 10) & (p[:, along] < b + 20), across]
            front.append(np.percentile(q, 10) if len(q) else np.nan)
            back.append(q.max() if len(q) else np.nan)
        front, back = np.array(front), np.array(back)
        ok = ~np.isnan(front)
        front = np.interp(bins, bins[ok], front[ok])
        back = np.interp(bins, bins[ok], back[ok])
        out[part] = (bins + 5, median_filter(front, 5, mode='nearest'), median_filter(back, 5, mode='nearest'))
    return out


def depth_in(edge, P):
    """How far into the band each point is from its front edge, and how far its back is."""
    out = np.full(len(P), -np.inf)
    room = np.full(len(P), -np.inf)
    for part, (bins, front, back) in edge.items():
        if part == 'north':
            sel = (P[:, 0] < 160) & (P[:, 0] >= bins[0] - 10) & (P[:, 0] <= bins[-1] + 10)
            d = P[:, 1] - np.interp(P[:, 0], bins, front)
            rm = np.interp(P[:, 0], bins, back) - np.interp(P[:, 0], bins, front)
        else:
            sel = (P[:, 1] >= bins[0] - 10) & (P[:, 1] <= bins[-1] + 10)
            d = P[:, 0] - np.interp(P[:, 1], bins, front)
            rm = np.interp(P[:, 1], bins, back) - np.interp(P[:, 1], bins, front)
        better = sel & (d > out)
        out[better], room[better] = d[better], rm[better]
    return out, room


def jittered(lo, hi, step, rng):
    xs, zs = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step), indexing='ij')
    P = np.c_[xs.ravel(), zs.ravel()]
    P[1::2] += [0, step / 2]
    return P + rng.normal(0, step * .12, P.shape)


# ---- meshes -------------------------------------------------------------------------------
def node(template, name, parts, mat, flags=None):
    Vs, Ns, Us, Fs, n = [], [], [], [], 0
    for V, N, U, F in parts:
        Vs.append(V); Ns.append(N); Us.append(U); Fs.append(F + n); n += len(V)
    P, N, U, F = np.concatenate(Vs), np.concatenate(Ns), np.concatenate(Us), np.concatenate(Fs)
    T = gltf_source.tangents(P, N, U, F)
    out = dict(template)
    out.update(name=name, children=[], material=mat, verts=np.c_[P, N, U, T].astype('<f4'), faces=F.astype('<u2'))
    out.update(flags or {})
    bp.sphere(out)
    return out


def chunks(parts, limit=60000):
    """Parts in runs of fewer vertices than a kn5 mesh can number."""
    run, n = [], 0
    for p in parts:
        if n + len(p[0]) > limit and run:
            yield run
            run, n = [], 0
        run.append(p)
        n += len(p[0])
    if run:
        yield run


def save(k, path):
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)


def main():
    extra = gltf_source.load_kn5(EXTRA)
    if any(n['name'].startswith(PREFIX) for n, m in bp.meshes(extra)):
        print('the plantation is planted already')
        return
    track = gltf_source.load_kn5(bp.TRACK)
    rng = np.random.default_rng(1)

    # the old trees in the band out, and where they stood
    old = []
    for k, path in ((track, bp.TRACK), (extra, EXTRA)):
        took = 0
        emptied = []
        for n, m in bp.meshes(k):
            if k['materials'][n['material']].get('shader') != 'ksTree' or not len(n['faces']):
                continue
            V = n['verts'][:, :3].astype(float) + m[3, :3]
            F = n['faces'].astype(int)
            lab = pieces(F, len(V))
            gone = np.zeros(len(F), bool)
            for t in np.unique(lab[F[:, 0]]):
                sel = lab == t
                p = V[sel]
                c = (p[:, [0, 2]].min(0) + p[:, [0, 2]].max(0)) / 2
                if band(c[0], c[1]):
                    gone |= sel[F].all(1)
                    if np.ptp(p[:, 1]) > 1:
                        old.append(c)
                        took += 1
            if gone.all():
                emptied.append(n)
            elif gone.any():
                bp.split(n, gone)
        # meshes whose trees all went, out of the model
        drop = {id(x) for x in emptied}
        for parent, _ in kn5tree.walk(k['root']):
            parent['children'] = [c for c in parent['children'] if id(c) not in drop]
        print('  %-22s %4d old trees out' % (os.path.basename(path), took))
    old = np.unique(np.round(np.array(old), 0), axis=0)
    edge = edges(old)

    # the ground, and what nothing is planted near
    others = [gltf_source.load_kn5(os.path.join(bp.MODELS, f + '.glb')) for f in
              ('velopark_paddock', 'velopark_kerbs', 'velopark_tyres', 'velopark_paint', 'velopark_poles', 'velopark_kids')]
    ground, blocked = [], []
    for k in [track, extra] + others:
        for n, m in bp.meshes(k):
            if not len(n['faces']) or k['materials'][n['material']].get('shader') == 'ksTree':
                continue
            t = bp.triangles(n, m)
            if not len(t):
                continue
            name = n['name']
            if 'GRASS' in name[:7]:
                ground.append(t)
            elif not name.startswith('1WALL') and (n['visible'] and n['renderable'] or name[:1].isdigit()):
                blocked.append(t)
    ground = bp.Surface(np.concatenate(ground))
    lo = np.array([old[:, 0].min(), old[:, 1].min()]) - 30
    hi = np.array([old[:, 0].max(), old[:, 1].max()]) + 30
    grid = Grid(lo, hi)
    near = np.concatenate(blocked)
    mid = near[:, :, [0, 2]].mean(1)
    near = near[((mid > lo - 10) & (mid < hi + 10)).all(1)]
    mask = binary_dilation(grid.mark(near), iterations=int(round(CLEAR / R)))
    free = lambda P: ~mask[tuple(grid.cells(P).T)]

    # where the trees stand
    spots = []
    for step, lo_d, hi_d in ((DENSE, 0., DEPTH), (SPARSE, DEPTH, np.inf)):
        P = jittered(lo, hi, step, rng)
        d, room = depth_in(edge, P)
        keep = (d >= lo_d) & (d < hi_d) & (d <= room + 6) & free(P)
        spots.append(P[keep])
    spots = np.concatenate(spots)
    spots = spots[~north_west(*spots.T)]
    # the north west's broad leaf trees: where the ground's photo shows trees
    crowns = uniform_filter(photo_trees(track, grid).astype(float), size=int(2 * CROWN / R) + 1)
    B = jittered(lo, hi, BROAD_APART, rng)
    broad_spots = B[north_west(*B.T) & (crowns[tuple(grid.cells(B).T)] > CROWNS) & free(B)]
    # the understory, and the young trees in it
    C = rng.uniform(lo, hi, (int(np.prod(hi - lo) * CLUMPS), 2))
    d, room = depth_in(edge, C)
    keep = (d >= -FRONT) & (d < UNDER) & free(C) & ~north_west(*C.T)
    C, dC = C[keep], d[keep]
    Y = rng.uniform(lo, hi, (int(np.prod(hi - lo) / 40), 2))
    dy, _ = depth_in(edge, Y)
    Y = Y[(dy >= -1) & (dy < UNDER) & free(Y) & ~north_west(*Y.T)]

    # the trees made, and put where they stand
    made = {}
    for i in range(VARIANTS):
        made['euca_%d' % i] = dict(kind='euca', height=rng.uniform(*HEIGHT), seed=100 + i)
    for i in range(YOUNG_VARIANTS):
        made['euca_young_%d' % i] = dict(kind='young', height=rng.uniform(*YOUNG), seed=200 + i)
    for i in range(BROAD_VARIANTS):
        made['broad_%d' % i] = dict(kind='broad', height=rng.uniform(*BROAD), seed=300 + i)
    shapes = {name: shape_of(v) for name, v in made.items()}
    tall = [n for n in made if made[n]['kind'] == 'euca']
    small = [n for n in made if made[n]['kind'] == 'young']
    wide = [n for n in made if made[n]['kind'] == 'broad']
    stands = []
    tiles = {}
    walls = {}

    def add(kind, key, parts):
        tiles.setdefault(key, {}).setdefault(kind, []).extend(parts)

    def plant(kinds, P, sizes):
        n = 0
        for (x, z) in P:
            kind = kinds[rng.integers(len(kinds))]
            shape, r = shapes[kind]
            g = ground.ys(np.array([[x, z], [x + .4, z], [x - .4, z], [x, z + .4], [x, z - .4]]))
            if np.isnan(g).all():
                continue
            y = np.nanmin(g)
            scale = rng.uniform(*sizes)
            yaw = rng.uniform(0, 2 * np.pi)
            key = tuple(np.floor(np.array([x, z]) / TILE).astype(int))
            for part in shape:
                add(part, key, placed(shape[part], yaw, scale, np.array([x, y, z])))
            stands.append([kind, float(x), float(y), float(z), float(yaw), float(scale)])
            # the wall round its trunk
            a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            ring = np.c_[np.cos(a), np.zeros(6), np.sin(a)] * r * scale * 1.1
            Vw = np.r_[ring + [x, y - .3, z], ring + [x, y + TRUNK, z]]
            Fw = [(i, (i + 1) % 6, 6 + (i + 1) % 6) for i in range(6)] + [(i, 6 + (i + 1) % 6, 6 + i) for i in range(6)]
            walls.setdefault(key, []).append((Vw, np.r_[ring, ring] / np.linalg.norm(ring[0]), np.zeros((12, 2)), np.array(Fw)))
            n += 1
        return n

    trees_n = plant(tall, spots, (.92, 1.08))
    young_n = plant(small, Y, (.85, 1.15))
    broad_n = plant(wide, broad_spots, (.8, 1.2))
    # the understory's clumps: an uneven top over its strip, denser low down
    knots = rng.uniform(3.5, 8., 400)
    for (x, z), dd in zip(C, dC):
        g = ground.y(x, z)
        if np.isnan(g):
            continue
        H = np.interp((x + z) / 9 % 400, np.arange(400), knots) * (.75 + .35 * (dd + FRONT) / (FRONT + UNDER))
        y = H * rng.uniform(0, 1) ** 1.2
        size = rng.uniform(1.6, 3.0) * (1 + .3 * y / H)
        c = np.array([x, g + y, z])
        key = tuple(np.floor(np.array([x, z]) / TILE).astype(int))
        add('shrub', key, cards(c + [0, size * .25, 0], size, size, rng.uniform(0, np.pi), rng.normal(0, .3), c - [0, size * .3, 0], .8))

    # the materials and textures
    for name, img, fmt in (('euca_leaves.dds', leaf_texture(1), 'DXT5'), ('euca_shrub.dds', leaf_texture(7), 'DXT5'),
                           ('euca_bark.dds', bark_texture(), 'DXT1'), ('broad_leaves.dds', broad_texture(11), 'DXT5')):
        data = kn5io.dds(img, fmt)
        open(os.path.join(TEX, name), 'wb').write(data)
        extra['textures'].append(dict(name=name, active=1, data=data))
    like = next(m for m in extra['materials'] if m['shader'] == 'ksTree')
    mats = {}
    for kind, tex, shader, at in (('bark', 'euca_bark.dds', 'ksPerPixel', 0), ('leaves', 'euca_leaves.dds', 'ksPerPixelAT', 1),
                                  ('shrub', 'euca_shrub.dds', 'ksPerPixelAT', 1), ('broad', 'broad_leaves.dds', 'ksPerPixelAT', 1)):
        extra['materials'].append(dict(like, name='Euca' + kind.title(), shader=shader, alpha_tested=at,
                                       props=[dict(p) for p in like['props']], textures=[dict(sampler='txDiffuse', slot=0, name=tex)]))
        mats[kind] = len(extra['materials']) - 1
    template = next(n for n, m in bp.meshes(extra) if n['name'].startswith('TREE_'))
    root = extra['root']['children']
    meshes_n = tris = 0
    for key in sorted(tiles):
        for kind, parts in tiles[key].items():
            for i, run in enumerate(chunks(parts)):
                nd = node(template, '%s%s_%d_%d_%d' % (PREFIX, kind.upper(), key[0], key[1], i), run, mats[kind])
                root.append(nd)
                meshes_n += 1
                tris += len(nd['faces'])
    for key in sorted(walls):
        for i, run in enumerate(chunks(walls[key])):
            root.append(node(template, '1WALL_%sTRUNKS_%d_%d_%d' % (PREFIX, key[0], key[1], i), run, mats['bark'],
                             dict(renderable=0, cast_shadows=0)))
    for k, path in ((track, bp.TRACK), (extra, EXTRA)):
        save(k, path)
    json.dump({'variants': {n: dict(v, radius=shapes[n][1]) for n, v in made.items()}, 'trees': stands},
              open(FOREST, 'w'), indent=0)
    print('%d eucalyptus, %d young, %d broad leaf, %d clumps of understory: %d meshes, %d triangles, in %s' % (
        trees_n, young_n, broad_n, len(C), meshes_n, tris, os.path.relpath(EXTRA, ROOT)))


if __name__ == '__main__':
    main()
