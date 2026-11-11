"""What stands round the track, outside it, in source/models/velopark_outside.glb.

    python tools/build_outside.py

Made from the ground photo, OpenStreetMap (tools/reference/elevation/osm_property_*.json, placed
by location.json) and screenshots of the vp_1000 driven from K1_S_LEFT to K1_T1:

- the dirt roads round the property: Estrada Porto da Figueira along the north, the road down
  the west side and the service road down the east, DIRT wide, over the ground;
- the property's fence, between those roads (the town's) and its trees; and the fences that part the track
  from the parking: the track's
  fence round the vp_1000's east, TRACK_FENCE metres out from its roads and tyre walls, with
  globe lamps where it runs by the south east's wood; and the south west's fence, made again as
  the others: concrete posts POST_APART apart, welded mesh between them (with walls that cars
  hit, 1WALL_FENCE_);
- the shed beyond K1_T5, its wall of light blue sheets and its open roof on timber posts; the
  yard north of the pond: a house, sixteen round bins, a barn, a low building and a white tent;
  the pond;
- the judges' tower on its stilts on the bank east of the vp_1000; the flag marshals' tents by
  the vp_1000's and the vp_1500's starts, with their chairs and cones; the vp_1500's flagpoles;
- the south fence on thin steel posts, open at the pit exit (R27), its sliding gate slid aside;
- the Velogrill and the two houses in the south east's wood, and the sports court between them;
- the sheds along the east road, beyond it;
- by the vp_1500's start and south, the race control tower and the fuel station in the R27 loop,
  as the paddock's buildings are (tools/restyle_paddock.py takes the old ones out), and the
  Techspeed banners on the stand by R26 and on the south fence west of R15.

Everything stands on the ground as it is (after tools/level_ground.py). Its textures are made
here, into source/textures (OUT_*.dds), but the roofs' sheets, which are BuildingRoof.dds. Run
again, it makes the model again, the same.
"""
import glob, io, json, math, os, sys

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import build_paths as bp
import build_trees
import elevation as ev
import gltf_source
import kn5io
import level_ground
import shapes as sh

ROOT = bp.ROOT
OUT = os.path.join(bp.MODELS, 'velopark_outside.glb')
TEX = os.path.join(ROOT, 'source', 'textures')
LAYOUTS = os.path.join(ROOT, 'source', 'track')
DIRT = 6.                # metres, the dirt roads' width
FENCE_IN = 4.5           # metres from a dirt road's middle to the property's fence
TRACK_FENCE = 12.        # metres out from the vp_1000's roads and tyre walls to its fence
# the middle of the tyre wall the vp_1000's fence follows close behind (TYRES_51 to TYRES_54: no way
# in behind them), and how far out
TYRE_WALL = [(-141, 62), (-142, 58), (-142, 54), (-141, 51), (-140, 47), (-139, 44), (-137, 40), (-135, 38), (-133, 34), (-131, 31), (-129, 29), (-125, 28), (-122, 26), (-117, 24), (-112, 23), (-107, 22), (-101, 22), (-96, 23), (-91, 26), (-86, 29), (-82, 32),
             (-78, 36), (-75, 40), (-72, 45), (-71, 50), (-69, 56), (-69, 61), (-69, 66), (-68, 72), (-68, 79), (-68, 85)]
FENCE_OFF = 1.5
# the vp_1000's east fence where the ground photo shows its posts (their shadows): from where it
# leaves its line round the track (TRACE_FROM_Z), behind TYRES_44 and bending in past TYRES_45,
# then down to where a path through the wood starts; along that path (EAST_PATH) to TYRES_52,
# behind which it goes on (TYRE_WALL from TYRES_52); and its branch carrying straight on from the
# bend to the parking lot's edge by the wood (PARKING_BRANCH)
TRACE_FROM_Z = 210.
EAST_TRACE = [(-145, 208), (-147.5, 204), (-151, 200), (-154, 196), (-156, 192), (-156.5, 188), (-156.6, 180),
              (-157.5, 172), (-158.1, 164), (-158.9, 156), (-159.5, 148), (-159.5, 140), (-158, 132), (-157, 128),
              (-155, 124), (-153.5, 120), (-151, 116), (-150.5, 112), (-149.5, 108), (-149.2, 104), (-148.6, 100),
              (-148.2, 96), (-148, 88), (-147.8, 80), (-148, 74), (-149.3, 68), (-150.3, 62)]
EAST_PATH = [(-149.5, 56), (-146, 47), (-141, 39), (-135, 32), (-129, 27)]
PARKING_BRANCH = [(-157, 128), (-159.5, 120), (-162, 108), (-160, 100), (-159, 92), (-158.3, 84), (-157.6, 76),
                  (-157.3, 68), (-156.4, 60), (-156.2, 54)]
POST_APART = 2.6         # metres between the fence's posts
POST, POST_H = .12, 2.1  # the posts' side and height over the ground
MESH_LO, MESH_HI = .05, 1.9   # the mesh's foot and top over the ground
LAMP_APART = 12.         # metres between the globe lamps
SEED = 7


# ---- where, from OpenStreetMap -----------------------------------------------------------------
def osm_lines():
    loc = json.load(open(ev.LOCATION))
    osm = json.load(open(sorted(glob.glob(os.path.join(ev.OUT, 'osm_property_*.json')))[-1]))
    out = {}
    for e in osm['elements']:
        if e['type'] == 'way':
            g = e['geometry']
            out[e['id']] = ev.to_ac(loc, ev.enu([p['lat'] for p in g], [p['lon'] for p in g]))
    return out


def clip(P, lo=-560., hi=560.):
    """A line's run within the near ground and the far ground's first ring."""
    keep = (np.abs(P[:, 0]) < hi) & (P[:, 1] > lo) & (P[:, 1] < hi)
    idx = np.nonzero(keep)[0]
    return P[max(idx[0] - 1, 0):idx[-1] + 2] if len(idx) else P[:0]


def offset(P, d):
    """The line d metres to its left (going along it), as shapely offsets it."""
    L = LineString(P).offset_curve(d, join_style=2, mitre_limit=3.)
    return np.array(L.coords)


# ---- the ground ----------------------------------------------------------------------------------
class Ground:
    def __init__(s):
        s.surf = level_ground.surface()

    def __call__(s, x, z):
        x, z = np.atleast_1d(x), np.atleast_1d(z)
        y = s.surf.ys(np.c_[x, z])
        if np.isnan(y).any():
            # off the ground (a hole under a road): the nearest ground round it
            for i in np.nonzero(np.isnan(y))[0]:
                for r in (1., 2., 4., 8.):
                    a = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                    g = s.surf.ys(np.c_[x[i] + r * np.cos(a), z[i] + r * np.sin(a)])
                    if not np.isnan(g).all():
                        y[i] = np.nanmin(g)
                        break
        return y

    def low(s, P):
        """The lowest ground under the points: where a building's floor goes."""
        P = np.asarray(P, float)
        return float(np.min(s(P[:, 0], P[:, 1])))


# ---- textures --------------------------------------------------------------------------------------
def noise(n, scale, seed, octaves=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp = 1.
    for o in range(octaves):
        k = max(2, int(scale * 2 ** o))
        g = rng.random((k, k))
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float) / 255
        out += g * amp
        amp *= .5
    out -= out.min()
    return out / out.max()


def rgb(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def textures():
    n = 512
    T = {}
    y, x = np.mgrid[0:n, 0:n] / n
    # welded mesh, a metre a tile: 6 squares across, 7 up, galvanised, its wires 3 px
    wire = np.zeros((n, n, 4))
    on = (np.abs(((x * 6) % 1) - .5) > .5 - 1.6 / n * 6) | (np.abs(((y * 7) % 1) - .5) > .5 - 1.6 / n * 7)
    shade = 150 + 40 * noise(n, 6, 1)
    wire[..., 0], wire[..., 1], wire[..., 2] = shade * .98, shade, shade * .96
    rust = noise(n, 10, 2) > .78
    wire[rust, 0] *= 1.15
    wire[rust, 2] *= .7
    wire[..., 3] = on * 255
    T['OUT_wire.dds'] = (Image.fromarray(wire.astype(np.uint8), 'RGBA'), 'DXT5')
    # concrete, stained darker low down (v from the foot)
    c = 168 + 30 * (noise(n, 8, 3) - .5) + 18 * (noise(n, 40, 4) - .5)
    c = c[..., None] * np.array([1., .985, .95]) - (1 - y[..., None]) * 0
    T['OUT_concrete.dds'] = (rgb(c), 'DXT1')
    # dirt road: u across (6 m), v along; two worn tracks, gravel
    base = noise(n, 6, 5)
    tracks = np.exp(-((x - .3) / .07) ** 2) + np.exp(-((x - .7) / .07) ** 2)
    grav = (noise(n, 90, 6) > .7) * 25
    d = 128 + 40 * (base - .5) + 22 * tracks + grav
    T['OUT_dirt.dds'] = (rgb(d[..., None] * np.array([1.14, .97, .78])), 'DXT1')
    # light blue corrugated sheet, a metre a tile: 13 ridges, streaks down it
    rid = .5 + .5 * np.cos(x * 2 * np.pi * 13)
    streak = noise(n, 3, 7)[:, :1].repeat(n, 1).T * 0 + noise(n, 30, 8) * .3
    b = (np.array([142, 186, 212]) * (.82 + .2 * rid[..., None])) * (1 - .18 * streak[..., None]) - (y[..., None] > .9) * 25
    T['OUT_metal_blue.dds'] = (rgb(b), 'DXT1')
    # weathered timber, grain along v
    g = noise(n, 4, 9)
    grain = np.sin(x * 2 * np.pi * 9 + 5 * noise(n, 3, 10)) * .5 + .5
    w = (np.array([128, 104, 80]) * (.75 + .3 * grain[..., None] * .5 + .25 * g[..., None]))
    T['OUT_wood.dds'] = (rgb(w), 'DXT1')
    # white painted panels: seams every half tile, a little dirt
    seam = (np.abs((x * 2) % 1 - .5) > .49) * 30
    wh = 226 - 18 * noise(n, 10, 11) - seam - (1 - y) * 0
    T['OUT_white.dds'] = (rgb(wh[..., None] * np.array([1, 1, .98])), 'DXT1')
    # cream plaster
    pl = 205 + 22 * (noise(n, 12, 12) - .5)
    T['OUT_plaster.dds'] = (rgb(pl[..., None] * np.array([1., .93, .8])), 'DXT1')
    # clay tiles, rows a tile a quarter metre; and dark grey ones
    row = ((y * 4) % 1)
    col = ((x * 5 + (np.floor(y * 4) % 2) * .5) % 1)
    t = .75 + .25 * row - .25 * (np.abs(col - .5) > .46) - .2 * (row < .08)
    v = noise(n, 8, 13)
    T['OUT_tiles.dds'] = (rgb((np.array([168, 88, 58]) * (t * (.85 + .3 * v))[..., None])), 'DXT1')
    T['OUT_tiles_dark.dds'] = (rgb((np.array([78, 78, 80]) * (t * (.85 + .3 * v))[..., None])), 'DXT1')
    # a window in the left half (frame, dark glass, sky in it) and a door in the right
    win = np.zeros((n, n, 3)) + 235
    gl = (x > .04) & (x < .46) & (y > .06) & (y < .94) & ~((np.abs(x - .25) < .012) | (np.abs(y - .5) < .012))
    sky = 40 + 60 * np.clip(1.4 - y * 1.6 - x, 0, 1)
    win[gl] = np.c_[sky[gl] * .8, sky[gl] * .9, sky[gl] * 1.1]
    door = (x > .54) & (x < .96) & (y > .02)
    win[door] = (np.array([96, 70, 50]) * (.85 + .15 * noise(n, 6, 14)[door][:, None]))
    win[door & (np.abs(x - .9) < .012) & (np.abs(y - .55) < .03)] = 200
    T['OUT_openings.dds'] = (rgb(win), 'DXT1')
    # galvanised steel
    st = 170 + 25 * (noise(n, 6, 15) - .5) + 10 * np.sin(y * 2 * np.pi * 40)
    T['OUT_steel.dds'] = (rgb(st[..., None] * np.array([.98, 1, 1.02])), 'DXT1')
    # water: dark, greenish, faint ripples
    rp = noise(n, 20, 16)
    T['OUT_water.dds'] = (rgb(np.array([46, 62, 52]) * (.8 + .4 * rp[..., None])), 'DXT1')
    # white fabric
    T['OUT_fabric.dds'] = (rgb((236 - 14 * noise(n, 5, 17))[..., None] * np.array([1, 1, .97])), 'DXT1')
    # artificial turf over the whole court, its lines in it
    tf = np.array([62, 128, 66]) * (.85 + .2 * noise(n, 40, 18)[..., None])
    ln = ((np.abs(x - .5) < .004) | (x < .015) | (x > .985) | (y < .02) | (y > .98) |
          (np.abs(np.hypot((x - .5) * 1.6, y - .5) - .14) < .006))
    tf[ln] = 230
    T['OUT_turf.dds'] = (rgb(tf), 'DXT1')
    # painted concrete, black and white in blocks: the vp_1000's walls
    T['OUT_globe.dds'] = (rgb(np.full((n, n, 3), 245.)), 'DXT1')
    # a traffic cone, v up it: orange, a white band round its middle
    cone = np.zeros((n, n, 3)) + np.array([236., 96., 30.]) * (.9 + .1 * noise(n, 8, 19)[..., None])
    cone[(y > .42) & (y < .62)] = 238
    T['OUT_cone.dds'] = (rgb(cone), 'DXT1')
    # the race control tower's upper floor, a side of it: three panes across, a transom, dark
    # frames; the glass light, so it lights up when the patch makes it glow at night
    tg = np.zeros((n, n, 3)) + np.array([70., 72., 76.])
    pane = (((x * 3) % 1 > .04) & ((x * 3) % 1 < .96) & (y > .04) & (y < .96) & (np.abs(y - .62) > .02))
    sky = (np.array([150., 170., 190.]) * (.85 + .25 * (1 - y))[..., None]) * (.9 + .1 * noise(n, 4, 21)[..., None])
    tg[pane] = sky[pane]
    T['OUT_tower_glass.dds'] = (rgb(tg), 'DXT1')
    # the fuel station canopy's fascia: dark grey, a light line along it
    fa = np.zeros((n, n, 3)) + 70 + 10 * noise(n, 6, 20)[..., None]
    fa[(y > .7) & (y < .76)] = (200, 205, 210)
    T['OUT_fascia.dds'] = (rgb(fa), 'DXT1')
    return {k: (img if img.size == (n, n) else img.resize((n, n)), fmt) for k, (img, fmt) in T.items()}


def banner_texture():
    """The Techspeed banner, 8 wide to 1 high: two sponsors' boards and the Techspeed logo on navy."""
    from PIL import ImageDraw, ImageFont
    W, H = 2048, 256
    img = Image.new('RGB', (W, H), (22, 30, 58))
    d = ImageDraw.Draw(img)

    def font(size):
        for f in ('arialbd.ttf', 'C:/Windows/Fonts/arialbd.ttf', 'DejaVuSans-Bold.ttf'):
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                pass
        return ImageFont.load_default()
    # the sponsors: white boards
    d.rectangle((20, 24, 300, 232), fill=(246, 246, 246))
    d.text((160, 128), 'TJBT', fill=(196, 30, 40), font=font(110), anchor='mm')
    d.rectangle((320, 24, 600, 232), fill=(246, 246, 246))
    d.ellipse((400, 40, 520, 216), fill=(245, 196, 30))
    d.text((460, 128), 'K', fill=(40, 120, 60), font=font(130), anchor='mm')
    # the logo: an orange and red swoosh either side, the name between
    for x0, flip in ((640, 1), (1980, -1)):
        pts = [(x0, 200), (x0 + flip * 240, 60), (x0 + flip * 300, 70), (x0 + flip * 70, 205)]
        d.polygon(pts, fill=(226, 92, 38))
        d.polygon([(x0 + flip * 30, 215), (x0 + flip * 300, 92), (x0 + flip * 320, 100), (x0 + flip * 80, 222)], fill=(170, 30, 30))
    d.text((1310, 112), 'TECHSPEED', fill=(250, 250, 250), font=font(150), anchor='mm')
    d.text((1310, 212), 'CIRCUITOS DE KART', fill=(220, 220, 225), font=font(40), anchor='mm')
    return img


def pylon_texture():
    """The fuel station's sign: a silver V on white over a navy band, 1 wide to 2 high."""
    from PIL import ImageDraw
    W, H = 256, 512
    img = Image.new('RGB', (W, H), (240, 241, 243))
    d = ImageDraw.Draw(img)
    d.polygon([(40, 60), (128, 300), (216, 60), (180, 60), (128, 220), (76, 60)], fill=(150, 156, 166))
    d.rectangle((0, 340, W, H), fill=(28, 36, 70))
    for i, w in enumerate((160, 110, 130)):
        d.rectangle((48, 370 + i * 40, 48 + w, 382 + i * 40), fill=(225, 228, 235))
    return img




# ---- the model -------------------------------------------------------------------------------------
PROPS = ('ksAmbient', 'ksDiffuse', 'ksSpecular', 'ksSpecularEXP', 'ksEmissive', 'ksAlphaRef')


def material(name, tex, amb=.3, dif=.6, spec=.08, exp=12., at=False, emissive=0.):
    vals = (amb, dif, spec, exp, emissive, 1.)
    return dict(name=name, shader='ksPerPixelAT' if at else 'ksPerPixel', blend=0, alpha_tested=int(at), depth_mode=0,
                props=[dict(name=p, value=float(v), rest=b'\0' * 36) for p, v in zip(PROPS, vals)],
                textures=[dict(sampler='txDiffuse', slot=0, name=tex)])


MATERIALS = {
    'Wire': ('OUT_wire.dds', dict(at=True, spec=.15, exp=20.)),
    'Concrete': ('OUT_concrete.dds', {}),
    'Dirt': ('OUT_dirt.dds', dict(spec=.03, exp=4.)),
    'MetalBlue': ('OUT_metal_blue.dds', dict(spec=.2, exp=25.)),
    'Roof': ('BuildingRoof.dds', dict(spec=.15, exp=20.)),
    'Wood': ('OUT_wood.dds', dict(spec=.03, exp=6.)),
    'White': ('OUT_white.dds', {}),
    'Plaster': ('OUT_plaster.dds', {}),
    'Tiles': ('OUT_tiles.dds', dict(spec=.05)),
    'TilesDark': ('OUT_tiles_dark.dds', dict(spec=.08)),
    'Openings': ('OUT_openings.dds', dict(spec=.3, exp=40.)),
    'Steel': ('OUT_steel.dds', dict(spec=.3, exp=30.)),
    'Water': ('OUT_water.dds', dict(spec=.6, exp=60., dif=.4)),
    'Fabric': ('OUT_fabric.dds', {}),
    'Turf': ('OUT_turf.dds', dict(spec=.03)),
    'Globe': ('OUT_globe.dds', dict(amb=.6, dif=.6, emissive=.0)),
    'Cone': ('OUT_cone.dds', dict(spec=.2, exp=20.)),
    'TowerGlass': ('OUT_tower_glass.dds', dict(spec=.4, exp=40.)),
    'GreyWall': ('BuildingWallGrey.dds', {}),
    'Pillar': ('BuildingPillar.dds', {}),
    'Fascia': ('OUT_fascia.dds', dict(spec=.2, exp=20.)),
    'Banner': ('OUT_banner.dds', dict(spec=.1, amb=.35, dif=.65)),
    'Pylon': ('OUT_pylon.dds', dict(spec=.2, exp=20., amb=.35, dif=.65)),
}


class Model:
    def __init__(s):
        s.parts = {}          # (name, material) -> parts

    def add(s, name, mat, parts):
        s.parts.setdefault((name, mat), []).extend(parts)


def mesh_node(name, parts, mat, flags=None):
    V, N, U, F = sh.merge(parts)
    T = gltf_source.tangents(V, N, U, F)
    out = dict(kind=2, name=name, active=1, children=[], material=mat, cast_shadows=1, visible=1, transparent=0,
               layer=0, lod_in=0., lod_out=0., renderable=1,
               verts=np.c_[V, N, U, T].astype('<f4'), faces=F.astype('<u2'))
    out.update(flags or {})
    bp.sphere(out)
    return out


def write(model, tex):
    used = sorted({m for (_, m), parts in model.parts.items() if parts})
    mats, names = [], {}
    for i, k in enumerate(used):
        t, kw = MATERIALS[k]
        mats.append(material('Out' + k, t, **kw))
        names[k] = i
    textures = []
    for k in used:
        t = MATERIALS[k][0]
        if t not in [x['name'] for x in textures]:
            data = tex[t] if t in tex else open(os.path.join(TEX, t), 'rb').read()
            textures.append(dict(name=t, active=1, data=data))
    children = []
    for (name, mat), parts in sorted(model.parts.items()):
        if not parts:
            continue
        flags = None
        if name.startswith('1WALL_'):
            flags = dict(renderable=0, cast_shadows=0)
        elif name.startswith(('FENCE_MESH', 'LAMPS')):
            flags = dict(lod_out=450.)
        for i, run in enumerate(build_trees.chunks(parts)):
            children.append(mesh_node(name if i == 0 else '%s_%d' % (name, i), run, names[mat], flags))
    k = dict(version=6, extra=0, textures=textures, materials=mats,
             root=dict(kind=1, name='velopark_outside', active=1, matrix=list(np.eye(4).ravel()), children=children))
    doc, binary = gltf_source.to_gltf(k, lambda t: '../textures/' + t['name'])
    gltf_source.write_glb(OUT, doc, binary)
    return sum(len(c['faces']) for c in children), len(children)


# ---- fences --------------------------------------------------------------------------------------
def fence(model, name, line, ground, lamps=False, metal=False):
    """Posts along the line POST_APART apart (and at its corners), the mesh between them on its
    one side, a wall to hit along it; and globe lamps on poles every LAMP_APART when lamps. The
    posts concrete, or thin round steel ones when metal."""
    L = LineString(line)
    if L.length < 1:
        return 0
    s = np.r_[np.arange(0, L.length, POST_APART), L.length]
    P = np.array([L.interpolate(v).coords[0] for v in s])
    g = ground(P[:, 0], P[:, 1])
    posts = []
    for (x, z), y in zip(P, g):
        if metal:
            posts += sh.cylinder((x, y - .2, z), .03, POST_H + .2, seg=6, tile=(.2, 1.))
        else:
            posts += sh.beam((x, y - .2, z), (x, y + POST_H, z), POST, tile=(.5, 1.))
    if metal:
        # rails along the top and the foot
        for h in (MESH_LO, MESH_HI):
            for a, b, ya, yb in zip(P[:-1], P[1:], g[:-1], g[1:]):
                posts += sh.beam((a[0], ya + h, a[1]), (b[0], yb + h, b[1]), .03, tile=(1., .2))
    model.add('FENCE_POSTS_' + name, 'Steel' if metal else 'Concrete', posts)
    model.add('FENCE_MESH_' + name, 'Wire', sh.panels(P, ground, MESH_LO, MESH_HI, tile=(1., 1.)))
    model.add('1WALL_FENCE_' + name, 'Concrete', sh.panels(P, ground, -.3, 2.2, both=True))
    if lamps:
        sl = np.arange(LAMP_APART / 2, L.length, LAMP_APART)
        out = []
        for v in sl:
            x, z = L.interpolate(v).coords[0]
            # a step into the track's side of the fence
            d = np.array(L.interpolate(min(v + .5, L.length)).coords[0]) - np.array(L.interpolate(max(v - .5, 0)).coords[0])
            d /= np.linalg.norm(d)
            x, z = x - d[1] * .35, z + d[0] * .35
            y = float(ground(x, z)[0])
            out += sh.cylinder((x, y - .1, z), .045, 2.55, seg=8, tile=(1, 1))
            out += sh.sphere((x, y + 2.68, z), .17)
        model.add('LAMPS_' + name, 'Globe', out)
    return len(P)


def crossing(a, b):
    x = a.intersection(b)
    if x.is_empty:
        return None
    return a.project(x if x.geom_type == 'Point' else list(x.geoms)[0])


def run(L, d0, d1, step=2.):
    d0, d1 = sorted((d0, d1))
    return np.array([L.interpolate(t).coords[0] for t in np.r_[np.arange(d0, d1, step), d1]])


def between(L, a, b):
    """L's run between where it crosses a and b."""
    return run(L, crossing(L, a) or 0., crossing(L, b) or L.length)


def down_from(L, top, z_min):
    """L's run from where it crosses top on to where it goes under z_min."""
    d0 = crossing(L, top) or 0.
    P = run(L, d0, L.length, 1.)
    below = np.nonzero(P[:, 1] < z_min)[0]
    return P[:below[0]] if len(below) else P


def sliding_gate(model, ground, a, b, side=1.):
    """The gate of the south fence's opening, a and b its ends: a mesh panel in a steel frame on
    wheels, slid open beside the opening along the fence, on its rail."""
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    w = np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]]) * .35 * side
    p0, p1 = b + n + d * .3, b + n + d * (w + .3)
    g0, g1 = float(ground(*p0)[0]), float(ground(*p1)[0])
    steel = []
    y0, h = .12, 1.9
    corners = [(p0, g0 + y0), (p1, g1 + y0), (p1, g1 + h), (p0, g0 + h)]
    for (pa, ya), (pb, yb) in zip(corners, corners[1:] + corners[:1]):
        steel += sh.beam((pa[0], ya, pa[1]), (pb[0], yb, pb[1]), .06)
    for t in np.linspace(0, 1, 6)[1:-1]:
        p = p0 + (p1 - p0) * t
        y = g0 + (g1 - g0) * t
        steel += sh.beam((p[0], y + y0, p[1]), (p[0], y + h, p[1]), .04)
    for t in (.08, .92):
        p = p0 + (p1 - p0) * t
        y = g0 + (g1 - g0) * t
        steel += sh.cylinder((p[0] - n[0] * .3, y + .02, p[1] - n[1] * .3), .09, .06, seg=10)
    # the rail it runs on, from the opening's far side along beside the fence
    r0, r1 = a + n, p1 + d * .5
    steel += sh.beam((r0[0], float(ground(*r0)[0]) + .01, r0[1]), (r1[0], g1 + .01, r1[1]), .05, .03)
    model.add('GATE_SOUTH', 'Steel', steel)
    model.add('GATE_SOUTH_MESH', 'Wire', sh.panels([p0, p1], lambda x, z: np.array([g0, g1])[:np.size(x)], y0, h, tile=(1., 1.)))


def track_fence_line():
    """The vp_1000's fence: TRACK_FENCE out from the track's roads and tyre walls, round its east
    from the north east corner beyond K1_T5 to the paddock south of K1_T8, smoothed."""
    polys = []
    from shapely.geometry import Polygon as Pg
    for f in (bp.TRACK, bp.PADDOCK, bp.KERBS, bp.TYRES):
        for n, m in bp.meshes(gltf_source.load_kn5(f)):
            if (n['name'].startswith('1ROAD_') or 'TYRE' in n['name']) and len(n['faces']):
                t = bp.triangles(n, m)
                if not len(t) or np.abs(t[:, :, 1]).max() > 30:
                    continue
                polys += [Pg(x[:, [0, 2]]).buffer(.05) for x in t if Pg(x[:, [0, 2]]).area > .01]
    B = unary_union(polys).buffer(TRACK_FENCE + 10).buffer(-10)
    ring = np.array(max(getattr(B, 'geoms', [B]), key=lambda g: g.area).exterior.coords)
    # the run from the north east (x < -160, z > 290) round the east and the south to x > -65
    keep = (ring[:, 0] < -60) & (ring[:, 1] > 15) & ~((ring[:, 1] > 300) & (ring[:, 0] > -160))
    idx = np.nonzero(keep)[0]
    # the longest run of kept points, going round the ring
    runs, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(ring) - 1:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    run = ring[max(runs, key=len)]
    L = LineString(run).simplify(.8)
    return np.array(L.coords)


# ---- buildings -----------------------------------------------------------------------------------
def rect(cx, cz, w, d, yaw):
    """The corners of a w (along x) by d rectangle round cx, cz, turned by yaw (radians, x to z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.array([(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)])
    return P @ np.array([[c, s], [-s, c]]) + (cx, cz)


def building(model, name, cx, cz, w, d, yaw, ground, wall_h, roof, mat_wall='Plaster', mat_roof='Tiles',
             ridge=2.5, windows=(), doors=(), over=.5):
    """A house: walls w by d round cx, cz turned by yaw, on its floor at the lowest ground under
    it (a plinth down to the ground round it), a roof of roof kind ('gable', 'hip', 'flat'), the
    windows (side, at along it, sill, w, h) and doors (side, at, w, h) on its walls."""
    corners = rect(cx, cz, w, d, yaw)
    y0 = ground.low(corners)
    # walls and plinth, built at the origin and turned and moved after
    parts = sh.walls([(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)], 0, wall_h, tile=(2., 2.))
    plinth = sh.walls([(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)], -1.5, 0, tile=(2., 2.))
    if roof == 'gable':
        top = sh.gable_roof(-w / 2, w / 2, -d / 2, d / 2, wall_h, wall_h + ridge, over=over, tile=(2., 2.))
        parts += sh.gable_ends(-w / 2, w / 2, -d / 2, d / 2, wall_h, wall_h + ridge, tile=(2., 2.))
    elif roof == 'hip':
        top = sh.hip_roof(-w / 2, w / 2, -d / 2, d / 2, wall_h, wall_h + ridge, over=over, tile=(2., 2.))
    else:
        top = sh.box((-w / 2 - over, wall_h, -d / 2 - over), (w / 2 + over, wall_h + .25, d / 2 + over), tile=(2., 2.))
    ops = []
    sides = {'s': ((-w / 2, -d / 2), (w / 2, -d / 2), (0, -1)), 'n': ((w / 2, d / 2), (-w / 2, d / 2), (0, 1)),
             'w': ((w / 2, -d / 2), (w / 2, d / 2), (1, 0)), 'e': ((-w / 2, d / 2), (-w / 2, -d / 2), (-1, 0))}
    for side, at, sill, ww, hh in list(windows) + [(s_, a, 0., ww, hh) for s_, a, ww, hh in doors]:
        a, b, nrm = sides[side]
        a, b = np.array(a, float), np.array(b, float)
        dirv = (b - a) / np.linalg.norm(b - a)
        p = a + dirv * at
        o = np.array(nrm) * .02
        p0 = p - dirv * ww / 2 + o
        p1 = p + dirv * ww / 2 + o
        # uv: windows the left half of the texture, doors the right
        door = (side, at, ww, hh) in doors
        u0, u1 = (.5, 1.) if door else (0., .5)
        V = np.array([(p0[0], sill, p0[1]), (p1[0], sill, p1[1]), (p1[0], sill + hh, p1[1]), (p0[0], sill + hh, p0[1])])
        U = np.array([(u0, 0), (u1, 0), (u1, 1), (u0, 1)])
        Nn = np.tile([nrm[0], 0, nrm[1]], (4, 1))
        F = sh._facing(V, [(0, 1, 2), (0, 2, 3)], np.array([nrm[0], 0, nrm[1]]))
        ops.append(sh._part(V, Nn, U, F))
    model.add(name + '_WALLS', mat_wall, sh.place(parts + plinth, yaw, (cx, y0, cz)))
    model.add(name + '_ROOF', mat_roof, sh.place(top, yaw, (cx, y0, cz)))
    if ops:
        model.add(name + '_OPENINGS', 'Openings', sh.place(ops, yaw, (cx, y0, cz)))
    return y0


def blue_shed(model, ground):
    """Beyond K1_T5: a wall of light blue sheets along its track side and its south end, and over
    it an open roof of sheets on timber posts and trusses."""
    x0, x1, z0, z1 = -187.5, -180.5, 285.5, 306.
    y = ground.low([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
    wall = []
    # the blue wall, its sheets on rails, standing just inside the posts
    wall += sh.panels([(x1 - .1, z1), (x1 - .1, z0), (x0 + 1., z0 + .1)], lambda a, b: np.full(np.size(a), y), -.1, 2.4, tile=(1., 2.4))
    model.add('SHED_BLUE_WALL', 'MetalBlue', wall)
    timber = []
    eave, ridge = 3.6, 5.0
    zs = np.linspace(z0, z1, 7)
    for zz in zs:
        for xx in (x0, x1):
            timber += sh.beam((xx, y - .2, zz), (xx, y + eave, zz), .15, tile=(.5, 1.))
        # the truss: tie beam, two rafters, a king post, two braces
        a, b, top = (x0 - .4, y + eave - .3, zz), (x1 + .4, y + eave - .3, zz), ((x0 + x1) / 2, y + ridge - .1, zz)
        timber += sh.beam(a, b, .1, .14, tile=(.5, 1.))
        timber += sh.beam(a, top, .1, .14, tile=(.5, 1.))
        timber += sh.beam(b, top, .1, .14, tile=(.5, 1.))
        timber += sh.beam(((x0 + x1) / 2, y + eave - .3, zz), top, .1, tile=(.5, 1.))
        for xx in (x0, x1):
            timber += sh.beam((xx, y + eave - 1.2, zz), ((x0 + x1) / 2 + (xx - (x0 + x1) / 2) * .45, y + eave - .3, zz), .08, tile=(.5, 1.))
    for xx in (x0, x1, (x0 + x1) / 2):
        timber += sh.beam((xx, y + (eave - .1 if xx != (x0 + x1) / 2 else ridge - .15), z0 - .4),
                          (xx, y + (eave - .1 if xx != (x0 + x1) / 2 else ridge - .15), z1 + .4), .1, .14, tile=(.5, 1.))
    model.add('SHED_TIMBER', 'Wood', timber)
    model.add('SHED_ROOF', 'Roof', sh.gable_roof(x0, x1, z0, z1, y + eave, y + ridge, over=.6, tile=(1., 1.), along='z'))


def bins(model, ground):
    """Sixteen round bins north of the pond, four by four, on a slab."""
    out, roofs = [], []
    xs = (-219., -222.7, -226.2, -229.5)
    zs = (305.8, 308.9, 312., 315.2)
    y = ground.low([(xs[0], zs[0]), (xs[-1], zs[0]), (xs[-1], zs[-1]), (xs[0], zs[-1])])
    for x in xs:
        for z in zs:
            out += sh.cylinder((x, y, z), 1.45, 4.2, seg=16, tile=(1., 1.), top=False)
            roofs += sh.cylinder((x, y + 4.2, z), 1.5, .9, seg=16, tile=(1., 1.), r_top=.25)
    slab = sh.box((xs[-1] - 2.3, y - .5, zs[0] - 2.3), (xs[0] + 2.3, y + .15, zs[-1] + 2.3), tile=(2., 2.), faces='nsewt')
    model.add('BINS', 'Steel', out + roofs)
    model.add('BINS_SLAB', 'Concrete', slab)


def tent(model, ground, cx, cz, w):
    y = ground.low(rect(cx, cz, w, w, 0))
    h = 2.6
    out = []
    for x, z in rect(cx, cz, w, w, 0):
        out += sh.beam((x, y, z), (x, y + h, z), .08, tile=(.5, 1.))
    model.add('TENT_POLES', 'Steel', out)
    model.add('TENT_TOP', 'Fabric', sh.hip_roof(cx - w / 2, cx + w / 2, cz - w / 2, cz + w / 2, y + h, y + h + 2.4, over=.2, tile=(2., 2.)))
    # a skirt of fabric along its top
    model.add('TENT_TOP', 'Fabric', sh.walls(rect(cx, cz, w + .4, w + .4, 0), y + h - .35, y + h, tile=(2., 2.)))


def tower(model, ground, cx, cz):
    """The judges' tower: a white cabin with windows all round on four legs, its stairs down."""
    y = ground.low(rect(cx, cz, 4, 4, 0))
    floor, cab, s = 4.6, 2.5, 2.6
    legs = []
    for x, z in rect(cx, cz, s - .2, s - .2, 0):
        legs += sh.beam((x, y - .3, z), (x, y + floor, z), .12, tile=(.5, 1.))
    # cross braces between the legs
    c = rect(cx, cz, s - .2, s - .2, 0)
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        legs += sh.beam((a[0], y + .3, a[1]), (b[0], y + floor - .4, b[1]), .05, tile=(.5, 1.))
        legs += sh.beam((b[0], y + .3, b[1]), (a[0], y + floor - .4, a[1]), .05, tile=(.5, 1.))
    # the stairs: a flight down from the cabin's north side towards the north east, its rails
    top = np.array([cx - s / 2 - .1, y + floor, cz + s / 2 - .5])
    foot = np.array([cx - s / 2 - .1 - 5.2, y, cz + s / 2 - .5])
    steps = 16
    across = np.array([0, 0, 1.])
    tread = []
    for i in range(steps):
        p = top + (foot - top) * (i + .5) / steps
        tread += sh.box((p[0] - .15, p[1] - .03, p[2] - .45), (p[0] + .15, p[1] + .02, p[2] + .45), tile=(.5, .5))
    for side in (-.47, .47):
        a, b = top + across * side, foot + across * side
        legs += sh.beam(a, b, .05, .2, tile=(.5, 1.))
        legs += sh.beam(a + (0, 1., 0), b + (0, 1., 0), .04, tile=(.5, 1.))
        for t in np.linspace(0, 1, 5):
            p = a + (b - a) * t
            legs += sh.beam(p, p + (0, 1., 0), .035, tile=(.5, 1.))
    legs += sh.beam(foot + (0, -.3, -.5), foot + (0, 1.2, -.5), .06) + sh.beam(foot + (0, -.3, .5), foot + (0, 1.2, .5), .06)
    model.add('TOWER_STEEL', 'Steel', legs + tread)
    # the floor slab, the cabin's walls with a band of windows, its roof reaching over
    model.add('TOWER_CABIN', 'White', sh.box((cx - s / 2, y + floor - .15, cz - s / 2), (cx + s / 2, y + floor, cz + s / 2), tile=(2., 2.)))
    model.add('TOWER_CABIN', 'White', sh.walls(rect(cx, cz, s, s, 0), y + floor, y + floor + 1.0, tile=(2., 2.)))
    model.add('TOWER_CABIN', 'White', sh.walls(rect(cx, cz, s, s, 0), y + floor + 2.1, y + floor + cab, tile=(2., 2.)))
    win = sh.walls(rect(cx, cz, s - .02, s - .02, 0), y + floor + 1.0, y + floor + 2.1, tile=(1., 1.1))
    # each side the window's half of the texture across it
    win = [sh._part(V, N, np.c_[.02 + .44 * (U[:, 0] - U[:, 0].min()) / np.ptp(U[:, 0]), (U[:, 1] - U[:, 1].min()) / np.ptp(U[:, 1])], F)
           for V, N, U, F in win]
    model.add('TOWER_WINDOWS', 'Openings', win)
    model.add('TOWER_ROOF', 'Roof', sh.box((cx - s / 2 - .4, y + floor + cab, cz - s / 2 - .4), (cx + s / 2 + .4, y + floor + cab + .12, cz + s / 2 + .4), tile=(1., 1.)))


def marshal(model, ground, cx, cz, side=1., tag='', w=3.):
    """A flag marshal's post by a start: a white canopy on four poles, braced by two struts from
    its back, a vent box on its top; a chair, and two cones on the track's side of it (side, +1
    the track towards +x, -1 towards -x)."""
    y = ground.low(rect(cx, cz, w, w, 0))
    h = 2.3
    poles = []
    for x, z in rect(cx, cz, w, w, 0):
        poles += sh.beam((x, y, z), (x, y + h, z), .04)
    # the chair: seat, back, legs
    seat = sh.box((cx - .25, y + .42, cz - .22), (cx + .25, y + .46, cz + .22))
    seat += sh.box((cx + .21, y + .46, cz - .22), (cx + .25, y + .9, cz + .22))
    for dx in (-.22, .22):
        for dz in (-.19, .19):
            seat += sh.beam((cx + dx, y, cz + dz), (cx + dx, y + .42, cz + dz), .025)
    # two struts down from the back legs' tops, outwards
    back = cx - w / 2 * side
    for dz in (-w / 2, w / 2):
        poles += sh.beam((back, y + h - .1, cz + dz), (back - 1.6 * side, y - .05, cz + dz * 1.15), .035)
    model.add('MARSHAL_POLES' + tag, 'Steel', poles + seat)
    top = sh.hip_roof(cx - w / 2, cx + w / 2, cz - w / 2, cz + w / 2, y + h, y + h + .3, over=.05, tile=(2., 2.))
    top += sh.walls(rect(cx, cz, w + .1, w + .1, 0), y + h - .25, y + h, tile=(2., 2.))
    top += sh.box((cx - .3, y + h + .2, cz - .3), (cx + .3, y + h + .5, cz + .3), tile=(1., 1.), faces='nsewt')
    top += sh.box((cx - .38, y + h + .5, cz - .38), (cx + .38, y + h + .56, cz + .38), tile=(1., 1.))
    model.add('MARSHAL_TOP' + tag, 'Fabric', top)
    cones = []
    for dz in (-.5, .5):
        x, z = cx + (w / 2 + 1.4) * side, cz - w / 2 + dz
        g = float(ground(x, z)[0])
        cones += sh.box((x - .2, g, z - .2), (x + .2, g + .03, z + .2), tile=(.4, .4))
        cones += sh.cylinder((x, g + .03, z), .14, .67, seg=12, tile=(1., .7), r_top=.025)
    model.add('MARSHAL_CONES' + tag, 'Cone', cones)


def control_tower(model, ground, cx, cz, w=5.6, d=5.6, front='w'):
    """The race control tower by the vp_1500's start: a ground floor of grey block with its
    windows, an upper floor glazed all round, salmon red pillars at its corners, a roof slab
    reaching over it."""
    corners = rect(cx, cz, w, d, 0)
    y = ground.low(corners)
    grey = sh.walls(corners, y - .3, y + 3.6, tile=(2., 2.)) + sh.walls(corners, y + 3.6, y + 4.1, tile=(2., 2.))
    grey += sh.walls(corners, y + 7.7, y + 8.1, tile=(2., 2.))
    grey += sh.box((cx - w / 2, y + 3.55, cz - d / 2), (cx + w / 2, y + 3.65, cz + d / 2), faces='b')
    model.add('TOWER_CONTROL_WALLS', 'GreyWall', grey)
    # the upper floor's glass, of a material of its own, which the Custom Shaders Patch lights at night
    upper = sh.walls(rect(cx, cz, w + .02, d + .02, 0), y + 4.1, y + 7.7, tile=(1., 1.))
    upper = [sh._part(V, N, np.c_[(U[:, 0] - U[:, 0].min()) / np.ptp(U[:, 0]), (U[:, 1] - U[:, 1].min()) / np.ptp(U[:, 1])], F)
             for V, N, U, F in upper]
    model.add('TOWER_CONTROL_UPPER_GLASS', 'TowerGlass', upper)
    win = []
    # the ground floor's windows on its front and its door at the back
    ops = []
    side = {'w': (1, 0), 'e': (-1, 0), 'n': (0, 1), 's': (0, -1)}[front]
    for t in (-1.2, 1.2):
        if side[0]:
            x = cx + side[0] * (w / 2 + .02)
            ops += sh.quad((x, y + 1., cz + t - .7 * side[0]), (x, y + 1., cz + t + .7 * side[0]), (x, y + 2.4, cz + t + .7 * side[0]), (x, y + 2.4, cz + t - .7 * side[0]))
    for V, N, U, F in ops:
        win.append(sh._part(V, N, np.c_[.02 + .44 * (U[:, 0] - U[:, 0].min()) / max(np.ptp(U[:, 0]), 1e-6), (U[:, 1] - U[:, 1].min()) / max(np.ptp(U[:, 1]), 1e-6)], F))
    model.add('TOWER_CONTROL_GLASS', 'Openings', win)
    pillars = []
    for x, z in rect(cx, cz, w + .1, d + .1, 0):
        pillars += sh.beam((x, y - .3, z), (x, y + 8.15, z), .36, tile=(.5, 1.))
    model.add('TOWER_CONTROL_PILLARS', 'Pillar', pillars)
    # the roof: a slab sloping down to the back, reaching over by most at the front
    o = .9
    xa, xb, za, zb = cx - w / 2 - o, cx + w / 2 + o, cz - d / 2 - o, cz + d / 2 + o
    hi, lo = y + 8.55, y + 8.15
    top = sh.quad((xa, lo, za), (xa, lo, zb), (xb, hi, zb), (xb, hi, za), tile=(2., 2.)) if side[0] > 0 else \
        sh.quad((xa, hi, za), (xa, hi, zb), (xb, lo, zb), (xb, lo, za), tile=(2., 2.))
    if top[0][1][0][1] < 0:
        top = [sh._part(V[::-1], -N, U[::-1], F) for V, N, U, F in top]
    under = [sh._part(V - [0, .15, 0], -N, U, F[:, ::-1]) for V, N, U, F in top]
    model.add('TOWER_CONTROL_ROOF', 'Roof', top + under)
    model.add('1WALL_TOWER_CONTROL', 'GreyWall', sh.walls(corners, y - .3, y + 3., tile=(2., 2.)))


def fuel_station(model, ground):
    """In the R27 loop south of T3: a canopy on steel posts over two islands of pumps, the shop
    and a small building beside it in grey block with salmon red pillars, and its sign."""
    x0, x1, z0, z1 = 199.5, 213., -18.5, -7.
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    y = ground.low([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
    h = 4.6
    posts = []
    for x in (x0 + 2.5, x1 - 2.5):
        for z in (z0 + 2.2, z1 - 2.2):
            posts += sh.cylinder((x, y - .2, z), .16, h + .3, seg=10)
    # the islands: a kerb of concrete, two pumps on each
    islands, pumps, tops = [], [], []
    for z in (cz - 2.6, cz + 2.6):
        islands += sh.box((cx - 4.2, y, z - .6), (cx + 4.2, y + .18, z + .6), tile=(1., 1.))
        for x in (cx - 2., cx + 2.):
            pumps += sh.box((x - .45, y + .18, z - .28), (x + .45, y + 1.9, z + .28), tile=(1., 2.))
            tops += sh.box((x - .5, y + 1.9, z - .32), (x + .5, y + 2.05, z + .32), tile=(1., 1.))
    model.add('FUEL_POSTS', 'Steel', posts + tops)
    model.add('FUEL_ISLANDS', 'Concrete', islands)
    model.add('FUEL_PUMPS', 'White', pumps)
    model.add('FUEL_CANOPY', 'Fascia', sh.box((x0, y + h, z0), (x1, y + h + .8, z1), tile=(14., .8), faces='nsew'))
    model.add('FUEL_CANOPY', 'Roof', sh.box((x0, y + h + .75, z0), (x1, y + h + .8, z1), tile=(2., 2.), faces='t'))
    model.add('FUEL_CANOPY', 'White', sh.box((x0, y + h, z0), (x1, y + h + .05, z1), tile=(2., 2.), faces='b'))
    # the shop, east of the canopy, and the small building south east of it
    for name, (bx, bz, bw, bd) in (('FUEL_SHOP', (193.5, -12.8, 11., 11.5)), ('FUEL_STORE', (183.5, -17.5, 8.5, 8.5))):
        building(model, name, bx, bz, bw, bd, 0., ground, 3.2, 'gable', mat_wall='GreyWall', mat_roof='Roof', ridge=1.,
                 windows=[('n', a, 1., 1.6, 1.3) for a in np.arange(2., bw - 1., 3.)], doors=[('w', bd / 2, 1.2, 2.2)], over=.4)
        yb = ground.low(rect(bx, bz, bw, bd, 0))
        pil = []
        for x, z in rect(bx, bz, bw + .1, bd + .1, 0):
            pil += sh.beam((x, yb - .3, z), (x, yb + 3.25, z), .3, tile=(.5, 1.))
        model.add(name + '_PILLARS', 'Pillar', pil)
        model.add('1WALL_' + name, 'GreyWall', sh.walls(rect(bx, bz, bw, bd, 0), yb - .3, yb + 3., tile=(2., 2.)))
    # the sign by the road, north of the canopy
    sx, sz = 196., -3.
    g = float(ground(sx, sz)[0])
    model.add('FUEL_SIGN_POST', 'Steel', sh.beam((sx, g - .3, sz), (sx, g + 3.2, sz), .3))
    panel = sh.box((sx - 1., g + 3.2, sz - .2), (sx + 1., g + 7.2, sz + .2), tile=(1., 1.), faces='ewtb')
    model.add('FUEL_SIGN_POST', 'Steel', panel)
    for zz, nrm in ((sz + .21, 1), (sz - .21, -1)):
        q = sh.quad((sx + nrm * 1., g + 3.2, zz), (sx - nrm * 1., g + 3.2, zz), (sx - nrm * 1., g + 7.2, zz), (sx + nrm * 1., g + 7.2, zz), tile=(2., 4.))
        q = [sh._part(V, N, np.c_[U[:, 0], 1 - U[:, 1]], F) for V, N, U, F in q]
        model.add('FUEL_SIGN', 'Pylon', q)


def banner(model, name, a, b, y0, y1, ground=None, lift=.06):
    """A banner from a to b (x, z), its bottom y0 and top y1 (over the ground at each end when
    ground is given), facing to its left going from a to b; text upright seen from that side."""
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]]) * lift
    ga, gb = (float(ground(*a)[0]), float(ground(*b)[0])) if ground else (0., 0.)
    p0, p1 = a + n, b + n
    q = sh.quad((p0[0], ga + y0, p0[1]), (p1[0], gb + y0, p1[1]), (p1[0], gb + y1, p1[1]), (p0[0], ga + y1, p0[1]),
                tile=(np.linalg.norm(b - a), y1 - y0))
    model.add('BANNER_' + name, 'Banner', [sh._part(V, N, np.c_[U[:, 0], 1 - U[:, 1]], F) for V, N, U, F in q])
    # its back, plain
    k = n / max(np.linalg.norm(n), 1e-9) * .01      # a centimetre behind it, not on it
    q0, q1 = p0 - k, p1 - k
    back = sh.quad((q0[0], ga + y0, q0[1]), (q0[0], ga + y1, q0[1]), (q1[0], gb + y1, q1[1]), (q1[0], gb + y0, q1[1]), tile=(1., 1.))
    model.add('BANNER_' + name + '_BACK', 'Fascia', back)


def flagpoles(model, ground, spots, h=10.):
    """Tall thin flagpoles in a row, as by the vp_1500's start."""
    out = []
    for x, z in spots:
        y = float(ground(x, z)[0])
        out += sh.cylinder((x, y - .3, z), .05, h + .3, seg=8, tile=(.3, 1.), r_top=.03)
        out += sh.sphere((x, y + h + .05, z), .07, seg=8, rings=4)
    model.add('FLAGPOLES', 'Steel', out)


def court(model, ground, cx, cz, w, d, yaw):
    """The sports court: its turf, a tall fence round it on posts."""
    corners = rect(cx, cz, w, d, yaw)
    y = ground.low(corners) + .05
    turf = sh.poly([(x, y, z) for x, z in corners], np.array([0, 1., 0]), tile=(1., 1.), axes=(np.array([1., 0, 0]), np.array([0, 0, 1.])))
    # its uv over the whole court, 0..1
    V, N, U, F = turf[0]
    q = (V[:, [0, 2]] - corners[0]) @ np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    U = np.c_[q[:, 0] / w, q[:, 1] / d]
    model.add('COURT_TURF', 'Turf', [sh._part(V, N, U, F)])
    ring = np.r_[corners, corners[:1]]
    flat = lambda a, b: np.full(np.size(a), y)
    posts = []
    L = LineString(ring)
    for v in np.arange(0, L.length, 3.):
        x, z = L.interpolate(v).coords[0]
        posts += sh.cylinder((x, y - .2, z), .04, 4.2, seg=6)
    model.add('COURT_POSTS', 'Steel', posts)
    model.add('COURT_MESH', 'Wire', sh.panels(ring, flat, 0., 4., tile=(1., 1.)))


def shed(model, name, ground, cx, cz, w, d, yaw, h=3.2):
    building(model, name, cx, cz, w, d, yaw, ground, h, 'gable', mat_wall='White', mat_roof='Roof', ridge=1.2, over=.4,
             windows=[('w', a, 1.2, 1.2, 1.) for a in np.arange(3., d - 2., 6.)], doors=[('s', w / 2, 2.4, 2.4)])


def pond(model, ground, lines):
    P = lines[446879561]
    y = ground.low(P[::3]) + .04
    model.add('POND', 'Water', sh.poly([(x, y, z) for x, z in P[:-1]], np.array([0, 1., 0]), tile=(6., 6.),
                                       axes=(np.array([1., 0, 0]), np.array([0, 0, 1.]))))


# ---- the layouts' model lists --------------------------------------------------------------------
def register():
    """velopark_outside.kn5 in every layout's models_*.ini, after the others."""
    added = 0
    for f in sorted(glob.glob(os.path.join(LAYOUTS, 'models_*.ini'))):
        text = open(f, newline='').read()
        if 'velopark_outside.kn5' in text:
            continue
        nl = '\r\n' if '\r\n' in text else '\n'
        n = text.count('[MODEL_')
        text = text.rstrip() + nl + nl + '[MODEL_%d]%sFILE=velopark_outside.kn5%sPOSITION=0,0,0%sROTATION=0,0,0%s' % (n, nl, nl, nl, nl)
        open(f, 'w', newline='').write(text)
        added += 1
    return added


def remove_old_fence():
    """The south west's old fence (WALL_100, chain link on Fence03) out of velopark.glb."""
    k = gltf_source.load_kn5(bp.TRACK)
    before = len(k['root']['children'])
    for parent, _ in __import__('kn5tree').walk(k['root']):
        parent['children'] = [c for c in parent['children'] if c['name'] != 'WALL_100']
    gone = sum(1 for n, m in bp.meshes(k) if n['name'] == 'WALL_100') == 0
    if gone and len(k['root']['children']) != before:
        bp.save(k, bp.TRACK)
        return True
    return False


def main():
    ground = Ground()
    lines = osm_lines()
    model = Model()
    # the dirt roads
    roads = {'NORTH': clip(lines[94747576]),
             'NORTH_W': clip(lines[94747823]),
             'WEST': clip(lines[412134877]), 'WEST_S': clip(lines[629631265]),
             'WEST_TRACK': clip(lines[412134879]),
             'EAST': clip(lines[1177404931])}
    for name, P in roads.items():
        if len(P) > 1:
            model.add('DIRT_' + name, 'Dirt', sh.strip(P, DIRT, ground, lift=.04, tile=(DIRT, 6.)))
    # the property's fence, between the town's dirt roads and the property's trees: along the
    # north road (from the west road to the east one) and down the west road to the driveway
    N = lines[94747576]
    nl = LineString(offset(N[(N[:, 0] > -300) & (N[:, 0] < 230)], -FENCE_IN))   # the road runs west: its right is south
    wl = LineString(offset(lines[412134877], -FENCE_IN))                          # runs south: its right is east
    el = LineString(offset(lines[1177404931], FENCE_IN))                          # runs south: its left is west
    # the fences that part the track from the parking: the vp_1000's round its east (from the north
    # fence) and the south west's, on the old one's line to the west fence
    T = track_fence_line()
    if T[0, 1] < T[-1, 1]:
        T = T[::-1]
    # its east side as the ground photo shows it, then close behind TYRES_52 to TYRES_54
    on = offset(np.array(TYRE_WALL), -FENCE_OFF)
    on = on[int(np.argmin(np.linalg.norm(on - (-122.5, 24.5), axis=1))):]
    cut = int(np.nonzero(T[:, 1] < TRACE_FROM_Z)[0].min())
    T = np.r_[T[:cut], EAST_TRACE, EAST_PATH, on]
    # its corners rounded a little, its ends where they are
    for _ in range(2):
        q = [T[0]]
        for p0, p1 in zip(T[:-1], T[1:]):
            q += [.75 * p0 + .25 * p1, .25 * p0 + .75 * p1]
        T = np.array(q + [T[-1]])
    top = nl.interpolate(nl.project(Point(T[0])))
    T = np.r_[[top.coords[0]], T]
    fences = {'NORTH': between(nl, wl, el), 'WEST': down_from(wl, nl, -60.),
              'SOUTH': np.array([(124.43, -9.33), (198.5, 8.6)]), 'SOUTH_W': np.array([(207.5, 10.8), (234., 17.2)]),
              'TRACK': T, 'TRACK_PARKING': np.array(PARKING_BRANCH, float)}
    posts = 0
    for name, P in fences.items():
        posts += fence(model, name, P, ground, metal=name.startswith('SOUTH'))
    # the pit exit's gate, slid open along the fence west of the opening
    sliding_gate(model, ground, (198.5, 8.6), (207.5, 10.8))
    # the globe lamps by the south east's wood: on the track's fence where it runs south of z 60
    T = fences['TRACK']
    lamp_run = T[T[:, 1] < 62]
    model.add('LAMPS_TRACK', 'Globe', [])
    if len(lamp_run) > 1:
        fence_l = LineString(lamp_run)
        out = []
        for v in np.arange(LAMP_APART / 2, fence_l.length, LAMP_APART):
            x, z = fence_l.interpolate(v).coords[0]
            y = float(ground(x, z)[0])
            out += sh.cylinder((x, y - .1, z), .045, 2.55, seg=8)
            out += sh.sphere((x, y + 2.68, z), .17)
        model.add('LAMPS_TRACK', 'Globe', out)
    # the buildings
    blue_shed(model, ground)
    building(model, 'FARM_HOUSE', -207.8, 313.5, 9.5, 10.6, 0., ground, 3.0, 'gable', mat_wall='Plaster', mat_roof='Roof', ridge=1.6,
             windows=[('w', a, 1., 1.2, 1.1) for a in (2.5, 8.)] + [('e', a, 1., 1.2, 1.1) for a in (3., 7.5)],
             doors=[('s', 5., 1., 2.1)])
    building(model, 'FARM_LEANTO', -200.4, 313.5, 5.2, 10.6, 0., ground, 2.6, 'flat', mat_wall='Plaster', mat_roof='Roof', over=.2,
             doors=[('w', 5.3, 2.6, 2.3)])
    bins(model, ground)
    building(model, 'FARM_BARN', -242.5, 310.8, 14., 18.5, 0., ground, 4.2, 'gable', mat_wall='White', mat_roof='Roof', ridge=2.2,
             doors=[('s', 7., 4., 3.6), ('w', 9., 3., 3.)], windows=[('e', a, 2., 2., 1.) for a in (4., 9., 14.)])
    building(model, 'FARM_LOW', -243.5, 289.2, 17., 10.5, 0., ground, 2.8, 'gable', mat_wall='White', mat_roof='Roof', ridge=1.,
             windows=[('s', a, 1., 1.5, 1.) for a in (3., 7., 11., 15.)], doors=[('w', 4.5, 1., 2.1)])
    tent(model, ground, -257.5, 286.5, 9.5)
    pond(model, ground, lines)
    tower(model, ground, -168.3, 159.8)
    # the Velogrill: an L of two wings with dark tiled hip roofs, turned as it stands
    yaw = math.radians(-33.)
    building(model, 'VELOGRILL_A', -207.5, 53., 22., 10., yaw, ground, 3.3, 'hip', mat_wall='Plaster', mat_roof='TilesDark', ridge=2.4,
             windows=[('s', a, 1., 1.6, 1.3) for a in (3., 7., 11., 15., 19.)] + [('n', a, 1., 1.6, 1.3) for a in (4., 10., 16.)],
             doors=[('s', 9., 1.8, 2.3)])
    building(model, 'VELOGRILL_B', -214.5, 41., 9., 12., yaw, ground, 3.3, 'hip', mat_wall='Plaster', mat_roof='TilesDark', ridge=2.,
             windows=[('w', a, 1., 1.4, 1.2) for a in (3., 8.)])
    # the two houses in the wood, and the court
    building(model, 'HOUSE_1', -119., -4., 14., 9., math.radians(-20.), ground, 2.9, 'hip', mat_wall='Plaster', mat_roof='Tiles', ridge=2.,
             windows=[('s', a, 1., 1.4, 1.2) for a in (3., 7., 11.)] + [('n', a, 1., 1.2, 1.1) for a in (4., 10.)], doors=[('e', 4.5, 1., 2.1)])
    building(model, 'HOUSE_2', -169., -31., 13., 9., math.radians(-35.), ground, 2.9, 'gable', mat_wall='Plaster', mat_roof='Tiles', ridge=2.,
             windows=[('s', a, 1., 1.4, 1.2) for a in (3., 7., 10.)], doors=[('w', 4.5, 1., 2.1)])
    court(model, ground, -151.5, -37.5, 24., 14., math.radians(-35.))
    marshal(model, ground, -87.4, 96.4, side=1., tag='_1000')
    marshal(model, ground, 3.2, 116.3, side=-1., tag='_1500')
    flagpoles(model, ground, [(-24., z) for z in (109., 113.5, 118., 122.5, 127.)])
    control_tower(model, ground, -47., 80.5, front='w')
    fuel_station(model, ground)
    # the banners: along the stand's front wall by R26, and on the south fence west of R15
    banner(model, 'STAND', (-19.01, -21.1), (-9.06, -38.03), .2, 1.45, lift=.5)
    banner(model, 'FENCE', (152., -2.6), (168., 1.27), .35, 2.35, ground=ground, lift=.18)
    # the sheds along the east road, beyond it
    east = LineString(lines[1177404931])
    for i, (cz, d) in enumerate(((135., 30.), (37., 24.), (-12., 14.))):
        # just beyond the road, clear of it
        road_x = min(x for x, z in east.intersection(LineString([(-400, cz), (0, cz)])).coords) if not             east.intersection(LineString([(-400, cz), (0, cz)])).is_empty else -258.
        shed(model, 'EAST_SHED_%d' % i, ground, road_x - DIRT / 2 - 1.5 - 4.5, cz, 9., d, 0.)
    # textures, the model, the layouts
    tex = {}
    made = textures()
    made['OUT_banner.dds'] = (banner_texture(), 'DXT1')
    made['OUT_pylon.dds'] = (pylon_texture(), 'DXT1')
    for name, (img, fmt) in made.items():
        data = kn5io.dds(img, fmt)
        open(os.path.join(TEX, name), 'wb').write(data)
        tex[name] = data
    tris, meshes = write(model, tex)
    print('%d fence posts; %d meshes, %d triangles, in %s' % (posts, meshes, tris, os.path.relpath(OUT, ROOT)))
    print('%d layouts given the model' % register())
    if remove_old_fence():
        print('the old fence out of velopark.glb')


if __name__ == '__main__':
    main()
