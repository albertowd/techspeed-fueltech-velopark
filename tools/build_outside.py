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
  the vp_1000's and the vp_1500's starts, with their chairs and cones; the vp_1500's flagpoles,
  where their shadows start on the ground photo;
- the south fence on thin steel posts, open at the pit exit (R27), its sliding gate slid aside;
- the Velogrill and the two houses in the south east's wood, and the sports court between them;
- the sheds along the east road, beyond it;
- by the vp_1500's start and south, the race control tower and the fuel station in the R27 loop,
  as the paddock's buildings are (tools/restyle_paddock.py takes the old ones out), and the
  Techspeed banners on the stand by R26 and on the south fence west of R15;
- the buildings in the middle of the track (MIDDLE) and the race control tower as the screenshots
  show them, in place of the paddock's (tools/restyle_paddock.py takes those out): grey block
  walls, a band of steel framed windows under the eaves, and one rounded roof of dark metal
  arched over each from west to east; the tower's upper floor with two windows a side, their
  top lights hinged at the top and opened out; plastered inside, with ceiling lamps, whose night
  lights it writes into the Custom Shaders Patch config (LIGHT_SERIES_3); and the big white tent
  by the pit boxes, north of the solar shed, its fabric rising like a ball from the square of
  its posts.

Everything stands on the ground as it is (after tools/level_ground.py). Its textures are made
here, into source/textures (OUT_*.dds), but the old roofs' sheets, which are BuildingRoof.dds. Run
again, it makes the model again, the same.
- the pit building between the two tracks' pits (pit_building), and the bar's terrace under the
  middle paddock building's roof by the vp_1500's pit entry (bar).
"""
import glob, io, json, math, os, re, sys

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import build_paths as bp
import build_trees
import elevation as ev
import gltf_source
import kn5tree
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
# the buildings in the middle of the track, made again as the screenshots show them: each its
# corners (as its walls to hit in velopark_paddock.glb) and its eaves' height
MIDDLE = {'PADDOCK_BUILDING_N': ([(-58.58, 54.83), (-40.63, 53.68), (-39.44, 73.32), (-57.03, 74.48)], 5.0),
          'PADDOCK_BUILDING_M': ([(-65.86, 32.09), (-36.7, 29.76), (-34.8, 53.09), (-63.46, 55.42)], 4.8),
          'PADDOCK_BUILDING_S': ([(-62.54, 17.37), (-45.08, 15.61), (-43.88, 30.09), (-61.59, 31.25)], 4.2),
          'GARAGE': ([(-47.26, 142.03), (-39.81, 141.45), (-37.18, 168.83), (-44.64, 169.66)], 4.1)}   # (its east wall on the pit building's)
# their roofs, each one rounded sheet of metal arched from west to east: ROOF_UP over the eaves
# over the walls, rising ROOF_RISE of the width between them in the middle, ROOF_OVER beyond
# them, ROOF_THICK thick, the arc in ROOF_STEPS
ROOF_UP, ROOF_RISE, ROOF_OVER, ROOF_THICK, ROOF_STEPS = .3, .12, .6, .14, 25
BAND, BAND_PANE, BAND_END = (1.5, .4), 1.6, .3   # the band of windows: from and to under the eaves,
                                                 # its panes' width, and the wall left at each end
FRAME = .06              # the windows' steel frames, their side
WALL = .15               # the walls' thickness, where they have windows
# the ceiling lamps: their side and depth, about how far apart, and their night lights' colour
# (brightness last) and reach
LAMP, LAMP_EVERY, LAMP_LIGHT, LAMP_RANGE = (.6, .08), 7., '1, 0.84, 0.62, 0.15', 8.
# the race control tower's upper floor: the windows' sill, the hinge of their top lights over the
# floor, the top lights' height, the wall left at each corner and between the two windows, and
# how far each top light is open (degrees; two a side, round from the south)
TOWER_SILL, TOWER_HINGE, TOWER_TOP = 5.0, 7.5, 1.
TOWER_MARGIN, TOWER_MULLION = .3, .2
TOWER_OPEN = [8., 22., 0., 30., 15., 35., 12., 25.]
# the big tent by the pit boxes, north of the solar shed, from the ground photo: its middle, width
# and depth, its depth's way (as the solar shed's posts run), its eaves, and how high its fabric
# rises over them
PIT_TENT = ((-41., 130.4), (10.4, 8.6), (.087, .996))   # (made again from the shed's floor by tent_on_floor())
SHED_TENT_IN = .3


def tent_on_floor():
    """The tent on the shed's floor: its middle (x, z), its size across and along, its direction along
    (x, z), SHED_TENT_IN in from the concrete's edges all round."""
    import drape_track as dt
    (xa, za), (xb, zb) = dt.LANE_EDGE
    edge = lambda z: xa + (xb - xa) * (z - za) / (zb - za)
    z_n = dt.SHED_NORTH + dt.SHED_EDGE + 1.3
    ez = np.array([xb - xa, zb - za], float)
    ez /= np.linalg.norm(ez)
    zm = (SLAB_SOUTH + z_n) / 2
    c = ((edge(zm) + SLAB_WEST) / 2, zm)
    w = (SLAB_WEST - edge(zm)) * ez[1] - 2 * SHED_TENT_IN
    return c, (w, (z_n - SLAB_SOUTH) - 2 * SHED_TENT_IN), tuple(ez)
PIT_TENT_EAVE, PIT_TENT_DOME = 3.2, 2.6
# the pit building between the two tracks' pits (screenshots 15-17 and 25-27 in
# tools/reference/site): a workshop for the karts in its middle, its walls cream over a grey
# skirt, a band of dark windows high in them, two big doorways on each pit lane; one roof of
# shallow curved metal sheets over it and over both pit lanes, on salmon red columns where the
# old ones stood, steel trusses under it, strip lights, and its signs at both ends over each lane:
# Techspeed over the vp_1000's (east), "curta vencer" over the vp_1500's. Its middle and its way north
PIT_BUILDING = ((-45.55, 110.3), (.0945, .9955))
ROAD_CONES = ((-31.1, 61.6),)        # (x, z: where the vp_1500's pit entry and the side road part)
PIT_CORE = (9.4, 25.8)          # the workshop's width (across) and length (along)
PIT_CORE_WEST = 2.5             # metres more of it on its west side, out to the 1500's pit lane (screenshot 17)
PIT_ROOF = (31.2, 30.5)         # the roof's, over both lanes
PIT_RISE = .045                 # the roof's rise over its width
PIT_EAVE = 4.4                  # metres from the columns' feet to the eaves
PIT_COLUMNS = [(-30.05, 122.6), (-30.65, 116.05), (-31.2, 109.45), (-31.75, 102.85), (-32.35, 96.25),
               (-58.75, 124.55), (-59.35, 118.05), (-59.9, 111.6), (-60.45, 105.1), (-61.05, 98.6)]
PIT_DOORS = (-6.3, 6.3)         # the doorways' middles along the workshop, on both its sides
PIT_DOOR = (4.4, 3.4)           # their width and height
PIT_SKIRT, PIT_BAND = 1.2, (2.4, 3.4)   # the grey skirt's height, the windows' band
# (the bar's west side, its posts, stands back off the road along it, R26, behind its fence)
# the fence along the road by the vp_1500's pit entry and the hotlap's pit lane (screenshots 13 and
# 14): along the side of these roads away from the track, PIT_FENCE_OFF off them, from in front of
# the changing rooms (the north paddock building) south past the bar and the wooden stands to the
# dirt path from the kids track to the car park, from z to z
PIT_FENCE_ROADS = ('1ROAD_25', '1ROAD_26')
PIT_FENCE_Z = (74., -45.)
PIT_FENCE_OFF = .35
# the storage building as a garage: its open bays on the lane's side and the road's, metres from its
# north end, their height, its louvred doors and its fire hose box on the lane's side
GARAGE_BAYS_LANE = ((.8, 4.2), (4.7, 8.1), (8.6, 12.), (12.5, 15.9))
GARAGE_BAYS_ROAD = ((6., 9.4), (9.9, 13.3), (13.8, 17.2))
GARAGE_BAY_H = 2.9
GARAGE_DOORS = ((18.5, 19.9), (23., 24.4))
GARAGE_HOSE = 21.3
WALKWAY = 2.5                   # metres between the changing rooms' walls and the bar's, under their roofs
BAR_DEPTH = 4.5                 # metres the bar's terrace reaches in under the middle paddock building's roof
# the stand by R26 (screenshot 09): a low stand of timber, its front along the banner, three tiers
# rising back from it: its middle, its way along, length, depth, the tiers' heights over its foot
STAND = ((-19.14, -32.58), (.509, -.861), 19.6, 5.4, (1.4, 1.8, 2.2))
# the other wooden stand (screenshot 08): a small raised one on posts, east of the vp_1500's straight
# north of that one, past the tyre wall and the grass, before the fence and the wood: its front's
# middle, its way along (as the tyre wall runs there), length, depth, its tiers' heights
STAND_2 = ((-28.7, -2.3), (.371, -.928), 8., 3., (1., 1.4))
STUB_WALL, STUB_BOX = '1WALL_161', ((0., 3.5), (-42.5, -37.5))   # the old wall's piece past the pit entry's tyres
SLAB_SOUTH, SLAB_WEST = 125., -35.6   # z, x: where the shed's floor's concrete begins from the pits, and ends at the ramp to the pit exit
FACE_FROM = .2                  # metres the shed's floor stands over the 1000's pit lane where its concrete face starts
PIT_WALL = 4.2                  # the cream walls' height; white panels braced with steel over them to the roof
# the prices on the fuel station's sign, in reais: the averages in Rio Grande do Sul in ANP's
# survey of early November 2026, written as the stations' lit boards show them, a dot between
# the reais and the centavos. The Custom Shaders Patch shows them from its config's script,
# which tools/build_outside.py writes there from these
FUEL_PRICES = (('ÁLCOOL', '4.55'), ('GASOLINA', '6.11'))
# the vp_1500's flagpoles where their shadows start on the ground photo
FLAGPOLES = [(-21.25 - .26 * i, 112.3 - 2. * i) for i in range(5)]


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
    # window glass: a little grey blue, mostly clear
    gl = np.zeros((n, n, 4)) + np.array([150., 168., 182., 48.])
    gl[..., :3] *= (.95 + .1 * noise(n, 4, 21))[..., None]
    T['OUT_glass.dds'] = (Image.fromarray(np.clip(gl, 0, 255).astype(np.uint8), 'RGBA'), 'DXT5')
    # the windows' steel frames, painted dark grey
    T['OUT_frame.dds'] = (rgb((58 + 8 * noise(n, 6, 23))[..., None] * np.array([1., 1.02, 1.05])), 'DXT1')
    # a ceiling lamp's diffuser
    T['OUT_lamp.dds'] = (rgb(np.full((n, n, 3), 1.) * np.array([246., 242., 230.])), 'DXT1')
    # the dark metal roofs: ribs along the slope, eight to a tile
    rib = .5 + .5 * np.cos(x * 8 * 2 * np.pi)
    T['OUT_roof_dark.dds'] = (rgb((84 + 14 * rib ** 4 + 5 * noise(n, 6, 24))[..., None] * np.array([1., 1.02, 1.05])), 'DXT1')
    # the fuel station canopy's fascia: dark grey, a light line along it
    fa = np.zeros((n, n, 3)) + 70 + 10 * noise(n, 6, 20)[..., None]
    fa[(y > .7) & (y < .76)] = (200, 205, 210)
    T['OUT_fascia.dds'] = (rgb(fa), 'DXT1')
    # painted orange steel (tool cabinets), and black rubber (tyres)
    T['OUT_orange.dds'] = (rgb(np.array([214., 98., 34.]) * (.92 + .1 * noise(n, 6, 25))[..., None]), 'DXT1')
    T['OUT_rubber.dds'] = (rgb((34 + 10 * noise(n, 30, 26) + 8 * (np.sin(x * 2 * np.pi * 24) > .6))[..., None] * np.ones(3)), 'DXT1')
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


def pit_signs_texture():
    """The pit building's signs, 4 wide to 1 high each: Techspeed on the top half, "curta VENCER"
    on the bottom, white on charcoal with the orange and red swoosh."""
    from PIL import ImageDraw
    W, H = 1024, 512
    img = Image.new('RGB', (W, H), (34, 34, 38))
    d = ImageDraw.Draw(img)
    for y0, words in ((0, (('TECHSPEED', 118),)), (256, (('curta', 46), ('VENCER', 110)))):
        for x0, flip in ((24, 1), (W - 24, -1)):
            d.polygon([(x0, y0 + 220), (x0 + flip * 150, y0 + 60), (x0 + flip * 190, y0 + 66), (x0 + flip * 48, y0 + 226)], fill=(226, 92, 38))
            d.polygon([(x0 + flip * 20, y0 + 236), (x0 + flip * 190, y0 + 92), (x0 + flip * 204, y0 + 100), (x0 + flip * 56, y0 + 242)], fill=(170, 30, 30))
        if len(words) == 1:
            d.text((W // 2, y0 + 128), words[0][0], fill=(250, 250, 250), font=sign_font(words[0][1]), anchor='mm')
        else:
            d.text((W // 2 - 150, y0 + 62), words[0][0], fill=(230, 230, 232), font=sign_font(words[0][1]), anchor='mm')
            d.text((W // 2, y0 + 150), words[1][0], fill=(250, 250, 250), font=sign_font(words[1][1]), anchor='mm')
    return img


def timing_texture():
    """The timing screen, 16 wide to 9 high: a hot lap session's times, the karts in order, white on
    near black, the best of each in green."""
    from PIL import ImageDraw
    W, H = 1024, 576
    img = Image.new('RGB', (W, H), (10, 12, 18))
    d = ImageDraw.Draw(img)
    small, big, row_f = sign_font(16), sign_font(28), sign_font(20)
    d.text((24, 16), 'Session:', fill=(200, 200, 205), font=small)
    d.text((110, 16), 'HOT LAP', fill=(245, 245, 245), font=small)
    d.text((24, 38), 'Timing:', fill=(200, 200, 205), font=small)
    d.text((110, 32), '01:26:21', fill=(245, 245, 245), font=big)
    d.text((24, 66), 'Time to go:', fill=(200, 200, 205), font=small)
    d.text((130, 66), '00:00:00', fill=(245, 245, 245), font=small)
    d.line((20, 92, W - 20, 92), fill=(220, 220, 225), width=2)
    rng = np.random.RandomState(26)
    best = 75.15
    for i in range(13):
        y = 100 + i * 33
        t = best + i * .35 + rng.uniform(0., .3)
        d.text((24, y + 4), str(i + 1), fill=(245, 245, 245), font=row_f)
        d.text((70, y), '#%02d  KART %d' % (rng.randint(1, 99), i + 1), fill=(245, 245, 245), font=row_f)
        d.text((70, y + 18), 'Laps: %d' % rng.randint(2, 8), fill=(190, 190, 195), font=sign_font(12))
        d.text((420, y + 6), 'Best Lap: %d' % rng.randint(1, 6), fill=(190, 190, 195), font=sign_font(13))
        d.text((560, y + 6), 'Best Time: 01:%06.3f' % (t - 60.), fill=(120, 220, 120) if i in (0, 7, 10) else (230, 230, 232), font=sign_font(13))
        d.text((780, y + 6), 'Diff: %.3f' % (t - best), fill=(190, 190, 195), font=sign_font(13))
        d.line((20, y + 32, W - 20, y + 32), fill=(90, 92, 100), width=1)
    d.text((24, H - 22), 'Timing and results are not official', fill=(235, 200, 60), font=sign_font(12))
    return img


def sign_font(size):
    from PIL import ImageFont
    for f in ('arialbd.ttf', 'C:/Windows/Fonts/arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


PYLON = (512, 1024, 540)    # the sign's texture, its width and height, and where its board starts


def price_boxes():
    """Where each price shows on the sign's texture: (u0, u1, v0, v1), 0 to 1 over it."""
    W, H, top = PYLON
    rows = len(FUEL_PRICES)
    out = []
    for i in range(rows):
        y0 = top + 30 + i * (H - top - 60) // rows
        y1 = y0 + (H - top - 60) // rows - 24
        out.append(((W - 236) / W, (W - 28) / W, y0 / H, y1 / H))
    return out


def prices_texture():
    """The prices as the sign shows them without the Custom Shaders Patch (and in BeamNG): each
    in red digits on black, a square of the texture each, one under the other. With the patch the
    script in its config draws them there instead, lit (FUEL_PRICES_SCRIPT)."""
    from PIL import ImageDraw
    n = len(FUEL_PRICES)
    img = Image.new('RGB', (512, 512 * n), (8, 8, 10))
    d = ImageDraw.Draw(img)
    for i, (fuel, price) in enumerate(FUEL_PRICES):
        d.text((480, 512 * i + 270), price, fill=(255, 40, 30), font=sign_font(210), anchor='rm')
    return img


def pylon_texture():
    """The fuel station's sign, 1 wide to 2 high: the Velopark logo on white (tools/reference/
    velopark_logo.png when it is there, else the name under a V), over a navy board with each
    fuel's name and a black box for its price, which the prices' own mesh covers."""
    from PIL import ImageDraw
    W, H, top = PYLON
    img = Image.new('RGB', (W, H), (242, 243, 245))
    d = ImageDraw.Draw(img)
    logo = os.path.join(ROOT, 'tools', 'reference', 'velopark_logo.png')
    if os.path.exists(logo):
        L = Image.open(logo).convert('RGBA')
        k = min((W - 60) / L.width, (top - 60) / L.height)
        L = L.resize((int(L.width * k), int(L.height * k)), Image.LANCZOS)
        img.paste(L, ((W - L.width) // 2, (top - L.height) // 2), L)
    else:
        navy = (28, 36, 70)
        d.polygon([(96, 60), (256, 380), (416, 60), (346, 60), (256, 250), (166, 60)], fill=navy)
        d.text((W // 2, 460), 'VELOPARK', fill=navy, font=sign_font(86), anchor='mm')
    d.rectangle((0, top, W, H), fill=(26, 32, 62))
    for (fuel, price), (u0, u1, v0, v1) in zip(FUEL_PRICES, price_boxes()):
        y0, y1 = v0 * H, v1 * H
        d.text((26, (y0 + y1) / 2 - 16), fuel, fill=(236, 238, 242), font=sign_font(38), anchor='lm')
        d.text((26, (y0 + y1) / 2 + 26), 'R$ / litro', fill=(170, 178, 200), font=sign_font(24), anchor='lm')
        d.rectangle((u0 * W - 6, y0 - 6, u1 * W + 6, y1 + 6), fill=(8, 8, 10))
    return img




# ---- the model -------------------------------------------------------------------------------------
PROPS = ('ksAmbient', 'ksDiffuse', 'ksSpecular', 'ksSpecularEXP', 'ksEmissive', 'ksAlphaRef')


def material(name, tex, amb=.3, dif=.6, spec=.08, exp=12., at=False, emissive=0., blend=False):
    vals = (amb, dif, spec, exp, emissive, 1.)
    return dict(name=name, shader='ksPerPixelAT' if at else 'ksPerPixel', blend=int(blend), alpha_tested=int(at), depth_mode=int(blend),
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
    'Glass': ('OUT_glass.dds', dict(spec=.9, exp=90., blend=True)),
    'Frame': ('OUT_frame.dds', dict(spec=.3, exp=30.)),
    'Lamp': ('OUT_lamp.dds', dict(amb=.5, dif=.6)),
    'RoofDark': ('OUT_roof_dark.dds', dict(spec=.2, exp=25.)),
    'GreyWall': ('BuildingWallGrey.dds', {}),
    'Pillar': ('BuildingPillar.dds', {}),
    'Fascia': ('OUT_fascia.dds', dict(spec=.2, exp=20.)),
    'Banner': ('OUT_banner.dds', dict(spec=.1, amb=.35, dif=.65)),
    'Pylon': ('OUT_pylon.dds', dict(spec=.2, exp=20., amb=.35, dif=.65)),
    'FuelPrices': ('OUT_fuel_prices.dds', dict(spec=.3, exp=40., amb=.35, dif=.65)),
    'Orange': ('OUT_orange.dds', dict(spec=.3, exp=30.)),
    'Rubber': ('OUT_rubber.dds', dict(spec=.05, exp=8.)),
    'PitSigns': ('OUT_pit_signs.dds', dict(spec=.2, exp=20., amb=.35, dif=.65)),
    'Timing': ('OUT_timing.dds', dict(spec=.4, exp=60., amb=.2, dif=.3, emissive=.9)),
    'Solar': ('BuildingSolar.dds', dict(spec=.5, exp=60.)),
    'HotlapPage': ('OUT_hotlap_page.dds', dict(spec=.4, exp=60., amb=.2, dif=.3, emissive=.9)),
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
        if name.endswith(('_WALLS', '_GABLES', '_WALL')) and not name.startswith('1WALL_'):
            # (the buildings' walls seen from inside them too: through their doors and bays, under their roofs)
            parts = parts + [sh._part(V, -N, U, np.asarray(F)[:, ::-1]) for V, N, U, F in parts]
        flags = None
        if name.startswith('1WALL_'):
            flags = dict(renderable=0, cast_shadows=0)
        elif name.startswith(('FENCE_MESH', 'LAMPS')):
            flags = dict(lod_out=450.)
        elif MATERIALS[mat][1].get('blend'):
            flags = dict(transparent=1, cast_shadows=0)
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
    windows (side, at along it, sill, w, h) and doors (side, at, w, h) on its walls. The windows
    are steel framed glass in holes in the walls, with a plastered room behind them, its ceiling
    lamps lit at night; the doors are painted on."""
    corners = rect(cx, cz, w, d, yaw)
    y0 = ground.low(corners)
    # walls and plinth, built at the origin and turned and moved after
    P = np.array([(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)])
    holes, wins = {}, []
    for side, at, sill, ww, hh in windows:
        i = {'s': 0, 'w': 1, 'n': 2, 'e': 3}[side]
        hole = (at - ww / 2, at + ww / 2, sill, sill + hh)
        holes.setdefault(i, []).append(hole)
        wins.append((i, hole, dict(bars=(at,) if ww > 1. else (), transoms=(sill + hh * .62,) if hh > 1.05 else ())))
    if wins:
        parts, inner = holed_walls(P, 0, wall_h, holes)
    else:
        parts = sh.walls(P, 0, wall_h, tile=(2., 2.))
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
    for side, at, sill, ww, hh in [(s_, a, 0., ww, hh) for s_, a, ww, hh in doors]:
        a, b, nrm = sides[side]
        a, b = np.array(a, float), np.array(b, float)
        dirv = (b - a) / np.linalg.norm(b - a)
        p = a + dirv * at
        o = np.array(nrm) * .02
        p0 = p - dirv * ww / 2 + o
        p1 = p + dirv * ww / 2 + o
        # uv: the doors the right half of the texture
        u0, u1 = .5, 1.
        V = np.array([(p0[0], sill, p0[1]), (p1[0], sill, p1[1]), (p1[0], sill + hh, p1[1]), (p0[0], sill + hh, p0[1])])
        U = np.array([(u0, 0), (u1, 0), (u1, 1), (u0, 1)])
        Nn = np.tile([nrm[0], 0, nrm[1]], (4, 1))
        F = sh._facing(V, [(0, 1, 2), (0, 2, 3)], np.array([nrm[0], 0, nrm[1]]))
        ops.append(sh._part(V, Nn, U, F))
    model.add(name + '_WALLS', mat_wall, sh.place(parts + plinth, yaw, (cx, y0, cz)))
    model.add(name + '_ROOF', mat_roof, sh.place(top, yaw, (cx, y0, cz)))
    if ops:
        model.add(name + '_OPENINGS', 'Openings', sh.place(ops, yaw, (cx, y0, cz)))
    if wins:
        room(model, name, P, 0., wall_h, inner, wins, yaw, (cx, y0, cz))
    return y0


def room(model, name, P, floor, ceiling, inner, wins, yaw=0., at=(0, 0, 0)):
    """Behind the windows wins of the walls round P: the windows, the walls' inner faces inner,
    a ceiling and a floor, and its ceiling lamps in rows, about LAMP_EVERY apart; built round the
    origin, turned by yaw and moved to at."""
    P = np.asarray(P, float)
    glazing(model, name, P, wins, yaw, at)
    model.add(name + '_INSIDE', 'Plaster', sh.place(inner + sh.poly([(x, ceiling, z) for x, z in P], n=np.array([0., -1., 0.]), tile=(2., 2.)), yaw, at))
    model.add(name + '_FLOOR', 'Concrete', sh.place(sh.poly([(x, floor + .02, z) for x, z in P], n=np.array([0., 1., 0.]), tile=(2., 2.)), yaw, at))
    nu = max(1, int(round(np.linalg.norm(P[1] - P[0]) / LAMP_EVERY)))
    nv = max(1, int(round(np.linalg.norm(P[3] - P[0]) / LAMP_EVERY)))
    spots = []
    for i in range(nu):
        for j in range(nv):
            u, v = (i + .5) / nu, (j + .5) / nv
            x, z = (1 - u) * (1 - v) * P[0] + u * (1 - v) * P[1] + u * v * P[2] + (1 - u) * v * P[3]
            spots.append((x, ceiling, z))
    lamps(model, name, spots, yaw, at)


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
    # the floor slab, the cabin's walls with a band of windows all round, its roof reaching over
    model.add('TOWER_CABIN', 'White', sh.box((cx - s / 2, y + floor - .15, cz - s / 2), (cx + s / 2, y + floor, cz + s / 2), tile=(2., 2.), faces='nsewb'))
    P = np.array([(-s / 2, -s / 2), (s / 2, -s / 2), (s / 2, s / 2), (-s / 2, s / 2)])
    hole = (.12, s - .12, 1., 2.1)
    holes = {i: [hole] for i in range(4)}
    wins = [(i, hole, dict(bars=(s / 2,))) for i in range(4)]
    walls, inner = holed_walls(P, 0., cab, holes)
    model.add('TOWER_CABIN', 'White', sh.place(walls, 0., (cx, y + floor, cz)))
    room(model, 'TOWER_CABIN', P, 0., cab, inner, wins, 0., (cx, y + floor, cz))
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


def tube(foot, head, r0, r1, seg=16):
    """A round column from foot to head, along its own slant, r0 thick at its foot tapering to r1:
    one smooth piece, its rings square to its axis, its top capped."""
    ax = (head - foot) / np.linalg.norm(head - foot)
    e1 = np.cross(ax, [0., 0., 1.] if abs(ax[2]) < .9 else [1., 0., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    a = np.linspace(0, 2 * np.pi, seg + 1)
    ring = np.outer(np.cos(a), e1) + np.outer(np.sin(a), e2)
    L = np.linalg.norm(head - foot)
    slope = (r0 - r1) / L
    N = ring + ax * slope
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    V = np.r_[foot + ring * r0, head + ring * r1]
    U = np.r_[np.c_[a / (2 * np.pi), np.zeros(seg + 1)], np.c_[a / (2 * np.pi), np.full(seg + 1, L)]]
    k = seg + 1
    F = [(i, i + k, i + 1) for i in range(seg)] + [(i + 1, i + k, i + k + 1) for i in range(seg)]
    out = [sh._part(V, np.r_[N, N], U, F)]
    # facing out
    v0, v1, v2 = V[F[0][0]], V[F[0][1]], V[F[0][2]]
    if np.cross(v1 - v0, v2 - v0) @ N[0] < 0:
        out = [sh._part(V, np.r_[N, N], U, [(i, c, b) for i, b, c in F])]
    cap = np.r_[[head], head + ring[:-1] * r1]
    Fc = [(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
    if np.cross(cap[Fc[0][1]] - cap[0], cap[Fc[0][2]] - cap[0]) @ ax < 0:
        Fc = [(i, c, b) for i, b, c in Fc]
    out.append(sh._part(cap, np.tile(ax, (len(cap), 1)), cap[:, [0, 2]] / .5, Fc))
    return out


def under(parts, x, z):
    """The lowest of the faces of PARTS over (x, z): where a column under them must stop; nan
    where none is over it."""
    best = np.nan
    for V, _, _, F in parts:
        T = V[F]
        a, b, c = T[:, 0, [0, 2]], T[:, 1, [0, 2]], T[:, 2, [0, 2]]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-12
        l1 = np.where(ok, ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (z - c[:, 1])) / np.where(ok, d, 1.), -1.)
        l2 = np.where(ok, ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (z - c[:, 1])) / np.where(ok, d, 1.), -1.)
        l3 = 1 - l1 - l2
        hit = ok & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if hit.any():
            y = (l1 * T[:, 0, 1] + l2 * T[:, 1, 1] + l3 * T[:, 2, 1])[hit].min()
            best = y if np.isnan(best) else min(best, y)
    return best


def column_to(roof, foot, top, r0, r1, gap=.02):
    """A round column from FOOT leaning to TOP's x and z, up to GAP under the roof all round its
    head (or to TOP's height where the roof is not over it)."""
    ring = [under(roof, top[0] + r1 * math.cos(a), top[2] + r1 * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    y = np.nanmin(ring) - gap if not np.all(np.isnan(ring)) else top[1]
    return tube(np.asarray(foot, float), np.array([top[0], y, top[2]]), r0, r1)


def control_tower(model, ground, cx, cz, w=5.6, d=5.6, front='w'):
    """The race control tower by the vp_1500's start: grey block walls, salmon red pillars at its
    corners, a rounded roof over it. Its upper floor has two windows a side, each a pane and a top
    light a metre high over it, hinged at its top and opened out, each as far as it is; its ground
    floor two windows on its front. The windows are steel framed glass, the rooms behind them
    plastered, each with a lamp in its ceiling."""
    corners = rect(cx, cz, w, d, 0)
    y = ground.low(corners)
    holes, wins = {}, []
    for i in range(4):
        L = np.linalg.norm(corners[(i + 1) % 4] - corners[i])
        ww = (L - 2 * TOWER_MARGIN - TOWER_MULLION) / 2
        for j in range(2):
            s0 = TOWER_MARGIN + j * (ww + TOWER_MULLION)
            hole = (s0, s0 + ww, y + TOWER_SILL, y + TOWER_HINGE)
            holes.setdefault(i, []).append(hole)
            wins.append((i, hole, dict(transoms=(y + TOWER_HINGE - TOWER_TOP,), open_top=(TOWER_TOP, TOWER_OPEN[2 * i + j]))))
    # its front's ground floor one band of windows across it, steel framed, as the screenshots
    # (13 to 15 in tools/reference/site) show
    side = {'s': 0, 'w': 1, 'n': 2, 'e': 3}[front]
    L = np.linalg.norm(corners[(side + 1) % 4] - corners[side])
    hole = (.35, L - .35, y + .5, y + 3.)
    holes.setdefault(side, []).append(hole)
    k = 4
    wins.append((side, hole, dict(bars=tuple(.35 + (L - .7) * j / k for j in range(1, k)), transoms=(y + 1.9,))))
    outer, inner = holed_walls(corners, y - .3, y + 8.1, holes)
    # the front cream, the rest dark grey
    a_, along_, L_, n_ = side_of(np.asarray(corners, float), side)
    fn = np.array([n_[0], 0., n_[1]])
    model.add('TOWER_CONTROL_WALLS', 'Plaster', [p for p in outer if p[1][0] @ fn > .9])
    model.add('TOWER_CONTROL_WALLS', 'GreyWall', [p for p in outer if not p[1][0] @ fn > .9])
    glazing(model, 'TOWER_CONTROL', np.asarray(corners, float), wins)
    # the two rooms: the ground floor's up to the upper floor's slab, the upper one's to the roof
    rooms = [(y + .02, y + 3.6), (y + 4.1, y + 8.1)]
    inside = inner + [p for f, c in rooms for p in sh.poly([(x, c, z) for x, z in corners], n=np.array([0., -1., 0.]), tile=(2., 2.))]
    model.add('TOWER_CONTROL_INSIDE', 'Plaster', inside)
    model.add('TOWER_CONTROL_FLOOR', 'Concrete', [p for f, c in rooms for p in sh.poly([(x, f, z) for x, z in corners], n=np.array([0., 1., 0.]), tile=(2., 2.))])
    lamps(model, 'TOWER_CONTROL', [(cx, c, cz) for f, c in rooms])
    # the roof, as the middle's buildings' are, arched from west to east
    roof, gables = barrel_roof(corners, y + 8.1, over=.9)
    # round orange columns at its corners, straight up from the ground to under the roof's edge
    pillars = []
    for x, z in rect(cx, cz, w + .5, d + .5, 0):
        pillars += column_to(roof, (x, y - .3, z), (x, y + 8.35, z), .28, .24)
    model.add('TOWER_CONTROL_PILLARS', 'Orange', pillars)
    model.add('TOWER_CONTROL_ROOF', 'RoofDark', roof)
    model.add('TOWER_CONTROL_WALLS', 'GreyWall', gable_walls(gables, (cx, cz)))
    model.add('1WALL_TOWER_CONTROL', 'GreyWall', sh.walls(corners, y - .3, y + 3., tile=(2., 2.)))


# ---- windows and lamps --------------------------------------------------------------------------
LAMP_SPOTS = []          # where every building's ceiling lamps hang, for their night lights


def side_of(P, i):
    """The side of P from corner i: its start, its way along, its length and its way out."""
    P = np.asarray(P, float)
    a, b = P[i], P[(i + 1) % len(P)]
    L = np.linalg.norm(b - a)
    along = (b - a) / L
    out = np.array([along[1], -along[0]])
    if out @ ((a + b) / 2 - P.mean(0)) < 0:
        out = -out
    return a, along, L, out


def wall_quad(a, along, n, s0, s1, h0, h1, tile=2.):
    p, q = a + along * s0, a + along * s1
    V = np.array([(p[0], h0, p[1]), (q[0], h0, q[1]), (q[0], h1, q[1]), (p[0], h1, p[1])])
    U = np.array([(s0, h0), (s1, h0), (s1, h1), (s0, h1)]) / tile
    nn = np.array([n[0], 0., n[1]])
    return sh._part(V, np.tile(nn, (4, 1)), U, sh._facing(V, [(0, 1, 2), (0, 2, 3)], nn))


def holed_walls(P, y0, y1, holes, thick=WALL):
    """Walls round P from y0 to y1 with holes in them: holes[i] the (s0, s1, h0, h1) on the side
    from P[i], s along it. Their faces outside, and inside (thick in from them, with the holes'
    reveals between)."""
    outer, inner = [], []
    for i in range(len(P)):
        a, along, L, n = side_of(P, i)
        hs = holes.get(i, [])
        b = a - n * thick
        for s0, s1, h0, h1 in hs:
            at = lambda s, h, o: (a[0] + along[0] * s - n[0] * o, h, a[1] + along[1] * s - n[1] * o)
            for V, f in (([at(s0, h0, 0), at(s1, h0, 0), at(s1, h0, thick), at(s0, h0, thick)], (0., 1., 0.)),
                         ([at(s0, h1, 0), at(s1, h1, 0), at(s1, h1, thick), at(s0, h1, thick)], (0., -1., 0.)),
                         ([at(s0, h0, 0), at(s0, h1, 0), at(s0, h1, thick), at(s0, h0, thick)], (along[0], 0., along[1])),
                         ([at(s1, h0, 0), at(s1, h1, 0), at(s1, h1, thick), at(s1, h0, thick)], (-along[0], 0., -along[1]))):
                V, f = np.array(V, float), np.array(f)
                inner.append(sh._part(V, np.tile(f, (4, 1)), np.zeros((4, 2)), sh._facing(V, [(0, 1, 2), (0, 2, 3)], f)))
        xs = sorted({0., L} | {v for h in hs for v in h[:2]})
        ys = sorted({y0, y1} | {v for h in hs for v in h[2:]})
        for xa, xb in zip(xs, xs[1:]):
            for ya, yb in zip(ys, ys[1:]):
                xm, ym = (xa + xb) / 2, (ya + yb) / 2
                if any(s0 < xm < s1 and h0 < ym < h1 for s0, s1, h0, h1 in hs):
                    continue
                outer.append(wall_quad(a, along, n, xa, xb, ya, yb))
                inner.append(wall_quad(b, along, -n, xa, xb, ya, yb))
    return outer, inner


def pane(V):
    """A pane of glass on the four points V, both its faces."""
    V = np.asarray(V, float)
    n = np.cross(V[1] - V[0], V[3] - V[0])
    n /= np.linalg.norm(n)
    U = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
    F = sh._facing(V, [(0, 1, 2), (0, 2, 3)], n)
    return [sh._part(V, np.tile(n, (4, 1)), U, F), sh._part(V, np.tile(-n, (4, 1)), U, F[:, ::-1])]


def window(a, along, n, hole, bars=(), transoms=(), open_top=None):
    """A steel framed window in the hole (s0, s1, h0, h1) on the side from a, along it, facing n:
    its frame, bars (at s) and transoms (at h), and the glass between them. With open_top (its
    height, degrees), the top light over the top transom is hinged at the hole's top and opened
    out. Its frame and its glass."""
    s0, s1, h0, h1 = hole
    at = lambda s, h, o=0.: np.array([a[0] + along[0] * s + n[0] * o, h, a[1] + along[1] * s + n[1] * o])
    f = FRAME / 2
    frame, glass = [], []
    beam = lambda p, q: sh.beam(p, q, FRAME, tile=(.5, 1.))
    for p, q in ((at(s0, h0), at(s1, h0)), (at(s0, h1), at(s1, h1)), (at(s0, h0), at(s0, h1)), (at(s1, h0), at(s1, h1))):
        frame += beam(p, q)
    top = h1
    if open_top:
        th, ang = open_top
        top = h1 - th
        r = math.radians(ang)
        # the top light: its frame round it, turned out about the hinge along the hole's top
        tl = lambda s, k: at(s, h1 - k * th * math.cos(r), k * th * math.sin(r))
        c = [tl(s0 + f, 1), tl(s1 - f, 1), tl(s1 - f, 0), tl(s0 + f, 0)]
        for p, q in zip(c, c[1:] + c[:1]):
            frame += beam(p, q)
        g = [tl(s0 + 2 * f, 1 - f / th), tl(s1 - 2 * f, 1 - f / th), tl(s1 - 2 * f, f / th), tl(s0 + 2 * f, f / th)]
        glass += pane(g)
    for h in transoms:
        frame += beam(at(s0, h), at(s1, h))
    for s in bars:
        frame += beam(at(s, h0), at(s, top))
    # the fixed panes, between the frame, bars and transoms up to the top light
    ss = [s0] + sorted(bars) + [s1]
    hh = [h0] + sorted(h for h in transoms if h < top - 1e-6) + [top]
    for sa, sb in zip(ss, ss[1:]):
        for ha, hb in zip(hh, hh[1:]):
            glass += pane([at(sa + f, ha + f), at(sb - f, ha + f), at(sb - f, hb - f), at(sa + f, hb - f)])
    return frame, glass


def glazing(model, name, P, wins, yaw=0., at=(0, 0, 0)):
    """The windows wins, each (side of P, hole, window's options), into the model, turned by yaw
    and moved to at."""
    frame, glass = [], []
    for i, hole, kw in wins:
        a, along, L, n = side_of(P, i)
        fr, gl = window(a, along, n, hole, **kw)
        frame += fr
        glass += gl
    model.add(name + '_FRAMES', 'Frame', sh.place(frame, yaw, at))
    model.add(name + '_GLASS', 'Glass', sh.place(glass, yaw, at))


def lamps(model, name, spots, yaw=0., at=(0, 0, 0)):
    """Ceiling lamps, a flat white box under the ceiling at each spot (x, ceiling, z), which glows
    at night, and a night light under it in the Custom Shaders Patch config; turned by yaw and
    moved to at."""
    out = []
    for x, c, z in spots:
        out += sh.box((x - LAMP[0] / 2, c - LAMP[1], z - LAMP[0] / 2), (x + LAMP[0] / 2, c, z + LAMP[0] / 2), faces='nsewb')
    model.add(name + '_LAMPS', 'Lamp', sh.place(out, yaw, at))
    S = np.array([(x, c - LAMP[1] - .05, z) for x, c, z in spots], float)
    S = sh.place([sh._part(S, np.zeros_like(S), np.zeros((len(S), 2)), np.zeros((0, 3), int))], yaw, at)[0][0]
    LAMP_SPOTS.extend(map(tuple, S))


def write_price_display():
    """The fuel station's prices as a display of the Custom Shaders Patch, as the start gantry's
    board is: a script that draws them on the prices' material in red digits, lit like LEDs and
    glowing at night, written into the track's config from FUEL_PRICES (each price a square of
    the display, one under the other, as prices_texture() paints them)."""
    path = os.path.join(LAYOUTS, 'extension', 'ext_config.ini')
    text = open(path, encoding='utf-8', newline='').read()
    n = len(FUEL_PRICES)
    lua = '\n'.join([
        '-- the prices, in reais: edit them here, or in FUEL_PRICES of tools/build_outside.py',
        'local PRICES = { %s }' % ', '.join('"%s"' % p for _, p in FUEL_PRICES),
        'local RED = rgbm(1, .06, .04, 1)',
        'local LETTER = vec2(104, 250)',
        'function update(dt)',
        '\tfor i, p in ipairs(PRICES) do',
        '\t\tdisplay.text{ pos = vec2(482 - LETTER.x * #p, (i - 1) * 512 + 130), letter = LETTER, text = p, font = "digital_big", color = RED }',
        '\tend',
        'end'])
    block = '\n'.join([
        "; The fuel station's prices: lit digits as the start gantry's board has, drawn by this",
        '; script, written by tools/build_outside.py from its FUEL_PRICES. Without the patch the',
        '; sign shows them painted.',
        '[SCRIPTABLE_DISPLAY_...]', 'MATERIALS = OutFuelPrices', 'RESOLUTION = 512, %d' % (512 * n),
        'DISPLAY_POS = 0, 0', 'DISPLAY_SIZE = 512, %d' % (512 * n), "SCRIPT = '", lua + "'", 'BACKGROUND = 0, 0, 0, 1', '',
        '[CustomEmissive]', 'ACTIVE = 1', "DESCRIPTION = Fuel station's prices", 'MATERIALS = OutFuelPrices',
        'Resolution = 512, %d' % (512 * n),
        '@ = CustomEmissive_Rect, Channel = 0, Start = "0, 0", Size = "512, %d", CornerRadius = 0.0' % (512 * n), '',
        '[MATERIAL_ADJUSTMENT_...]', 'ACTIVE = 1', "DESCRIPTION = Fuel station's prices at night",
        'MATERIALS = OutFuelPrices', 'CONDITION = NIGHT_SMOOTH', 'KEY_0 = ksEmissive',
        'VALUE_0 = 255, 255, 255, 0.094', 'VALUE_0_OFF = 0, 0, 0',
        "; (the end of the fuel station's prices)"]) + '\n\n'
    old = re.search(r"; The fuel station's prices: lit digits.*?; \(the end of the fuel station's prices\)\n\n", text, re.S)
    if old:
        text = text[:old.start()] + block + text[old.end():]
    else:
        at = text.index('; ---- Start gantry board')
        text = text[:at] + block + text[at:]
    open(path, 'w', encoding='utf-8', newline='').write(text)


def write_lights():
    """The buildings' lamps' night lights, as their series in the track's Custom Shaders Patch
    config (which tools/export_beamng.py makes BeamNG's night lights too)."""
    path = os.path.join(LAYOUTS, 'extension', 'ext_config.ini')
    text = open(path, encoding='utf-8', newline='').read()
    block = '\n'.join([
        '; The buildings in the middle of the track are lit inside at night by their ceiling lamps,',
        '; written by tools/build_outside.py where it hangs them.',
        '[LIGHT_SERIES_3]', 'ACTIVE = 1', 'DESCRIPTION = Building lamps',
        'COLOR = %s' % LAMP_LIGHT, 'DIRECTION = 0, -1, 0', 'SPOT = 170', 'SPOT_SHARPNESS = 0.2',
        'RANGE = %g' % LAMP_RANGE, 'RANGE_GRADIENT_OFFSET = 0.3', 'FADE_AT = 300', 'FADE_SMOOTH = 40',
        'DIFFUSE_CONCENTRATION = 0.6', 'SINGLE_FREQUENCY = 0', 'SHADOWS = 0', 'CONDITION = NIGHT_SMOOTH'] +
        ['POSITION_%d = %.3f, %.3f, %.3f' % (i, x, y, z) for i, (x, y, z) in enumerate(LAMP_SPOTS)]) + '\n\n'
    old = re.search(r'; The buildings in the middle of the track are lit inside.*?\n\n', text, re.S)
    if old:
        text = text[:old.start()] + block + text[old.end():]
    else:
        at = text.index('[MATERIAL_ADJUSTMENT_')
        at = text.rfind('\n\n', 0, at) + 2
        text = text[:at] + block + text[at:]
    open(path, 'w', encoding='utf-8', newline='').write(text)
    return len(LAMP_SPOTS)


# ---- the buildings in the middle of the track ---------------------------------------------------
def barrel_roof(P, eave, rise=ROOF_RISE, over=ROOF_OVER, thick=ROOF_THICK, tile=2., eaves=None):
    """One rounded metal roof over the four corners P, arched from its west side to its east side
    (across x) and straight along its north to south: an arc of a circle from eave + ROOF_UP over
    the walls on each side, rising by rise of the width between them in its middle, carrying on
    over beyond them by over. Its parts, and the gable walls' outlines under it (each a list of
    points (x, y, z) from one side's eave over to the other's)."""
    P = np.asarray(P, float)
    # the sides along z are the eaves (or P[eaves] to the next, when given): A to B one, D to C the other across from it
    e = eaves if eaves is not None else max(range(4), key=lambda i: abs(P[(i + 1) % 4][1] - P[i][1]) - abs(P[(i + 1) % 4][0] - P[i][0]))
    A, B, C, D = P[e], P[(e + 1) % 4], P[(e + 2) % 4], P[(e + 3) % 4]
    ov = over / np.linalg.norm(B - A)
    y_eave = eave + ROOF_UP
    cols, gables = [], []
    for t in (-ov, 0., 1., 1 + ov):
        q0, q1 = A + (B - A) * t, D + (C - D) * t
        W = np.linalg.norm(q1 - q0)
        u = (q1 - q0) / W
        r = rise * W
        R = (W * W / 4 + r * r) / (2 * r)
        yc = y_eave + r - R
        rows = []
        for s in np.linspace(-over, W + over, ROOF_STEPS):
            dx = s - W / 2
            y = yc + math.sqrt(R * R - dx * dx)
            rows.append((np.array([q0[0] + u[0] * s, y, q0[1] + u[1] * s]),
                         np.array([u[0] * dx / R, (y - yc) / R, u[1] * dx / R])))
        cols.append(rows)
        if t in (0., 1.):
            # under the roof, over the walls: from one eave over to the other
            g = [(q0[0], eave, q0[1])]
            for s in np.linspace(0, W, ROOF_STEPS):
                dx = s - W / 2
                g.append((q0[0] + u[0] * s, yc + math.sqrt(R * R - dx * dx) - thick, q0[1] + u[1] * s))
            g.append((q1[0], eave, q1[1]))
            gables.append(np.array(g))
    cols = [cols[0], cols[3]]
    top = np.array([[p for p, _ in col] for col in cols])      # (2, n, 3)
    nrm = np.array([[m for _, m in col] for col in cols])
    bot = top - nrm * thick
    n = top.shape[1]
    along = np.linalg.norm(top[1, 0] - top[0, 0])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(top[0], axis=0), axis=1))]
    out = []
    quad = lambda V, Nn: sh._part(V, Nn, np.zeros((4, 2)), sh._facing(V, [(0, 1, 2), (0, 2, 3)], Nn.mean(0)))
    for i in range(n - 1):
        for S, sgn in ((top, 1.), (bot, -1.)):
            V = np.array([S[0, i], S[1, i], S[1, i + 1], S[0, i + 1]])
            Nn = np.array([nrm[0, i], nrm[1, i], nrm[1, i + 1], nrm[0, i + 1]]) * sgn
            U = np.array([(arc[i], 0), (arc[i], along), (arc[i + 1], along), (arc[i + 1], 0)]) / tile
            out.append(sh._part(V, Nn, U, sh._facing(V, [(0, 1, 2), (0, 2, 3)], Nn.mean(0))))
    # its edges: the arcs at its north and south ends, and along the eaves
    side = top[1, 0] - top[0, 0]
    side /= np.linalg.norm(side)
    for k, sgn in ((0, -1.), (1, 1.)):
        for i in range(n - 1):
            out.append(quad(np.array([top[k, i], top[k, i + 1], bot[k, i + 1], bot[k, i]]), np.tile(side * sgn, (4, 1))))
    for i in (0, n - 1):
        d = (top[0, i] - top[0, n // 2]) * [1, 0, 1]
        out.append(quad(np.array([top[0, i], top[1, i], bot[1, i], bot[0, i]]), np.tile(d / np.linalg.norm(d), (4, 1))))
    return out, gables


def gable_walls(gables, centre, tile=(2., 2.)):
    """The walls closing the gables under a barrel roof, both ways (seen from a terrace under the roof too)."""
    out = []
    for g in gables:
        mid = g.mean(0)
        n = np.array([mid[0] - centre[0], 0., mid[2] - centre[1]])
        a = g[-1] - g[0]
        a = np.array([a[0], 0., a[2]]) / np.linalg.norm(a)
        n = n - a * (n @ a)
        n /= np.linalg.norm(n)
        face = sh.poly(g, n=n, tile=tile, axes=(a, np.array([0., 1., 0.])))
        out += face
        out += [sh._part(V, -N, U, np.asarray(F)[:, ::-1]) for V, N, U, F in face]     # (seen from under the roof too)
    return out


def middle_building(model, name, P, eave, ground, roof_P=None):
    """One of the buildings in the middle of the track: grey block walls round the corners P up to
    eave, a band of steel framed windows along them under it, plastered inside with ceiling lamps
    in rows, and one rounded roof arched over it from west to east, its gables walled up to it;
    over roof_P, when given, which the walls stand in (a terrace under the roof outside them)."""
    P = np.asarray(P, float)
    R = P if roof_P is None else np.asarray(roof_P, float)
    y0 = ground.low(R)
    holes, wins = {}, []
    for i in range(4):
        a, along, L, n = side_of(P, i)
        s0, s1 = BAND_END, L - BAND_END
        k = max(1, int(round((s1 - s0) / BAND_PANE)))
        hole = (s0, s1, eave - BAND[0], eave - BAND[1])
        holes[i] = [hole]
        wins.append((i, hole, dict(bars=tuple(s0 + (s1 - s0) * j / k for j in range(1, k)))))
    outer, inner = holed_walls(P, y0 - .3, eave, holes)
    model.add(name + '_WALLS', 'GreyWall', outer)
    roof, gables = barrel_roof(R, eave)
    model.add(name + '_ROOF', 'RoofDark', roof)
    model.add(name + '_WALLS', 'GreyWall', gable_walls(gables, R.mean(0)))
    room(model, name, P, y0, eave, inner, wins)
    return y0


def garage(model, ground):
    """The storage building north of the shed, by the 1000's pit lane, a garage as the screenshots
    show it (01, 02, 18 to 22 in tools/reference/site): cream walls, salmon red pillars between wide
    open bays on both its long sides, on the lane's from its north end (GARAGE_BAYS_LANE, metres from
    it) and on the road's to the pit exit (GARAGE_BAYS_ROAD), louvred doors and a fire hose box on
    the lane's further south; its rooms plastered, lit, under one rounded roof as the middle's."""
    name = 'GARAGE'
    P, eave = MIDDLE[name]
    P = np.asarray(P, float)
    y0 = ground.low(P)
    eave = max(eave, y0 + 3.2)
    sides = [side_of(P, i) for i in range(4)]
    lane = min(range(4), key=lambda i: (P[i][0] + P[(i + 1) % 4][0]))        # (x grows west)
    road = max(range(4), key=lambda i: (P[i][0] + P[(i + 1) % 4][0]))
    holes, pillars, doors, box = {}, [], [], []
    for i, bays in ((lane, GARAGE_BAYS_LANE), (road, GARAGE_BAYS_ROAD)):
        a, along, L, out = sides[i]
        s_of = (lambda d: d) if along[1] < 0 else (lambda d: L - d)            # metres from the north end
        at = lambda s, h, o=0.: (a[0] + along[0] * s + out[0] * o, h, a[1] + along[1] * s + out[1] * o)
        for d0, d1 in bays:
            s0, s1 = sorted((s_of(d0), s_of(d1)))
            holes.setdefault(i, []).append((s0, s1, y0 - .3, y0 + GARAGE_BAY_H))
        edges = sorted({d for b in bays for d in b})
        for d in [bays[0][0] - .25] + [(bays[k][1] + bays[k + 1][0]) / 2 for k in range(len(bays) - 1)] + [bays[-1][1] + .25]:
            s = s_of(d)
            pillars += sh.beam(at(s, y0 - .3, .2), at(s, eave - .05, .2), .4)
        if i == lane:
            for d0, d1 in GARAGE_DOORS:
                s0, s1 = sorted((s_of(d0), s_of(d1)))
                doors += sh.quad(at(s0, y0, .03), at(s1, y0, .03), at(s1, y0 + 2.3, .03), at(s0, y0 + 2.3, .03), tile=(1.2, 2.3), both=True)
            s = s_of(GARAGE_HOSE)
            c = np.array(at(s, y0 + 1.6, .14))
            box += sh.box(c - (.3, .4, .3), c + (.3, .4, .3), tile=(.6, .8))
    outer, inner = holed_walls(P, y0 - .3, eave, holes)
    model.add(name + '_WALLS', 'Plaster', outer)
    roof, gables = barrel_roof(P, eave)
    model.add(name + '_ROOF', 'RoofDark', roof)
    model.add(name + '_WALLS', 'Plaster', gable_walls(gables, P.mean(0)))
    model.add(name + '_PILLARS', 'Pillar', pillars)
    model.add(name + '_DOORS', 'Openings', doors)
    model.add(name + '_HOSE_BOX', 'Orange', box)
    room(model, name, P, y0, eave, inner, [])
    model.parts.pop((name + '_FLOOR', 'Concrete'), None)          # (its floor the paddock's asphalt, no step in)
    model.add('1WALL_' + name, 'Plaster', outer)                  # (its walls, open where its bays are)


def bar(model, ground):
    """The bar: the middle paddock building's west end, by the vp_1500's pit entry, open as a
    terrace under its roof (screenshots 06, 07 and 13 in tools/reference/site): its walls stand
    BAR_DEPTH in from the roof's west side; along that side salmon red posts leaning out to the
    roof's edge (the fence along its front is middle_fence's); on its concrete floor white
    plastic tables, four chairs at each, and lamps under the roof."""
    name = 'PADDOCK_BUILDING_M'
    P, eave = MIDDLE[name]
    P = np.asarray(P, float)
    # the west side is the one of the most x: from A to B; the walls BAR_DEPTH in from it
    i = max(range(4), key=lambda j: P[j][0] + P[(j + 1) % 4][0])
    A, B = P[i], P[(i + 1) % 4]
    Ain, Bin = P[(i - 1) % 4], P[(i + 2) % 4]
    v_a, v_b = (Ain - A) / np.linalg.norm(Ain - A), (Bin - B) / np.linalg.norm(Bin - B)
    walls = P.copy()
    walls[i], walls[(i + 1) % 4] = A + v_a * BAR_DEPTH, B + v_b * BAR_DEPTH
    y0 = middle_building(model, name, walls, eave, ground, roof_P=P)
    L = np.linalg.norm(B - A)
    u = (B - A) / L
    v = (v_a + v_b) / 2
    at = lambda s, d, h: (A[0] + u[0] * s + v[0] * d, y0 + h, A[1] + u[1] * s + v[1] * d)
    floor = [at(0, 0, .03), at(L, 0, .03), at(L, BAR_DEPTH, .03), at(0, BAR_DEPTH, .03)]
    model.add('BAR_FLOOR', 'Concrete', sh.poly(np.array(floor, float), n=np.array([0., 1., 0.]), tile=(2., 2.)))
    # the posts, round, leaning out to the roof's edge and stopping under it (the roof stands at
    # EAVE, whatever the floor's height)
    roof, _ = barrel_roof(P, eave)
    k = max(2, int(round(L / 3.4)))
    ss = np.linspace(.3, L - .3, k + 1)
    posts = []
    for s in ss:
        top = at(s, -.05, 0.)
        posts += column_to(roof, at(s, .55, -.3), (top[0], eave, top[2]), .12, .1)
    gap = (L / 2 - .9, L / 2 + .9)
    model.add('BAR_POSTS', 'Pillar', posts)
    # the tables and chairs, in two rows along it
    furniture = []
    for d in (1.5, 3.2):
        for s in np.arange(1.6, L - 1.2, 2.5):
            if gap[0] - .6 < s < gap[1] + .6 and d < 2:
                continue
            x, y, z = at(s, d, 0.)
            furniture += sh.cylinder((x, y + .70, z), .4, .03, seg=14, bottom=True)
            furniture += sh.cylinder((x, y, z), .04, .70, seg=6, top=False)
            for du, dv in ((.65, 0.), (-.65, 0.), (0., .65), (0., -.65)):
                c = np.array([x + u[0] * du + v[0] * dv, z + u[1] * du + v[1] * dv])
                back = c + (u * np.sign(du) + v * np.sign(dv)) * .2
                furniture += sh.box((c[0] - .21, y + .42, c[1] - .21), (c[0] + .21, y + .46, c[1] + .21))
                for lx, lz in ((-.18, -.18), (.18, -.18), (.18, .18), (-.18, .18)):
                    furniture += sh.beam((c[0] + lx, y, c[1] + lz), (c[0] + lx, y + .42, c[1] + lz), .03)
                furniture += sh.beam((back[0], y + .46, back[1]), (back[0], y + .9, back[1]), .4 if du == 0 else .04, .04 if du == 0 else .4)
    model.add('BAR_FURNITURE', 'White', furniture)
    lamps(model, 'BAR', [at(s, BAR_DEPTH / 2, eave - .02) for s in np.arange(2., L - 1., 4.)])


# the open end of the building along the hotlap's pit lane (its east half, along it from its east end
# to OPEN_TO, metres along its front from its first column): its front open, its columns only; its
# east end walled up to the roof; along its back, over its low wall, mesh fence up to the roof
OPEN_FROM, OPEN_TO, OPEN_BACK = -5.6, 45.6, -11.35    # along its front from its first column; its back wall, out from its front


def hotlap_building(ground):
    """The building along the hotlap's pit lane as it stands: its front (a point on it, its direction west
    to east, its normal out of it), its ends along its front, its depth, its eave's height (its
    columns' tops) and the paddock's model."""
    import restyle_paddock as rp
    from grade_paths import pieces as parts
    pad = gltf_source.load_kn5(bp.PADDOCK)
    track = gltf_source.load_kn5(bp.TRACK)
    foot, p0, d, nrm, xa, xb = rp.garages(pad, track)
    Fp = np.array(foot.exterior.coords)
    a, o = (Fp - p0) @ d, (Fp - p0) @ nrm
    col = next(n for n, _ in bp.meshes(pad) if n['name'] == rp.HOTLAP_COLUMNS)
    V = col['verts'][:, :3].astype(float)
    lab = parts(col['faces'].astype(int), len(V))
    tops = [V[lab == t][:, 1].max() for t in np.unique(lab)
            if abs((V[lab == t][:, [0, 2]].mean(0) - p0) @ nrm) < 1.]       # (the front's)
    return dict(p0=p0, d=d, nrm=nrm, ends=(OPEN_FROM, float(a.max())), depth=float(-o.min()),
                eave=float(np.median(tops)) - ROOF_UP + ROOF_THICK, pad=pad)


def hotlap_roof(model, ground):
    """The building along the hotlap's pit lane roofed as the middle buildings are: one rounded roof of
    dark metal arched over it from its front to its back (north to south), on its columns, its
    gables walled up to it, the solar panels on its north half."""
    b = hotlap_building(ground)
    p0, d, nrm, (s0, s1), D, eave = b['p0'], b['d'], b['nrm'], b['ends'], b['depth'], b['eave']
    P = np.array([p0 + d * s0, p0 + d * s1, p0 + d * s1 - nrm * D, p0 + d * s0 - nrm * D])
    roof, gables = barrel_roof(P, eave, eaves=0)
    model.add('HOTLAP_BUILDING_ROOF', 'RoofDark', roof)
    model.add('HOTLAP_BUILDING_GABLES', 'GreyWall', gable_walls(gables, P.mean(0)))
    # the solar panels: rows over its north half, a hand over the sheets
    W, r = D, ROOF_RISE * D
    R = (W * W / 4 + r * r) / (2 * r)
    yc = eave + ROOF_UP + r - R
    solar = []
    ss = np.linspace(W / 2 + .4, W - .3, 7)
    for t0, t1 in ((.4, s1 - s0 - .4),):
        for w0, w1 in zip(ss[:-1], ss[1:]):
            pts = []
            for w in (w0, w1):
                dx = w - W / 2
                y = yc + math.sqrt(R * R - dx * dx) + .05
                pts.append([p0 + d * (s0 + t) - nrm * w for t in (t0, t1)] + [y])
            (a0, a1, ya), (b0, b1, yb) = pts
            solar += sh.quad((a0[0], ya, a0[1]), (a1[0], ya, a1[1]), (b1[0], yb, b1[1]), (b0[0], yb, b0[1]), tile=(4., 4.))
    model.add('HOTLAP_BUILDING_SOLAR', 'Solar', solar)
    columns_to_roof(b, yc, R, W)
    return b


def columns_to_roof(b, yc, R, W):
    """The building's columns' tops (each one's upper part) just under its roof's underside over them
    (arched yc, R across its depth W): the drape moved each column with the ground under it, the roof
    is one."""
    import restyle_paddock as rp
    from grade_paths import pieces as parts
    pad, p0, nrm = b['pad'], b['p0'], b['nrm']
    col = next(n for n, _ in bp.meshes(pad) if n['name'] == rp.HOTLAP_COLUMNS)
    V = col['verts'].astype(float)
    lab = parts(col['faces'].astype(int), len(V))
    for t in np.unique(lab):
        sel = np.where(lab == t)[0]
        y = V[sel, 1]
        up = sel[y > y.min() + .6 * (y.max() - y.min())]
        w = np.clip(-((V[up][:, [0, 2]] - p0) @ nrm), 0., W)
        V[up, 1] = yc + np.sqrt(np.maximum(R * R - (w - W / 2) ** 2, 0.)) - ROOF_THICK - .02
    col['verts'] = V.astype(col['verts'].dtype)
    bp.sphere(col)
    bp.save(pad, bp.PADDOCK)


def open_area(model, ground, b):
    """The building along the hotlap's pit lane's open end: the mesh fence over its back's low wall up to
    the roof, and its east end walled up to its eave (its gable over it), with its wall to hit. The
    low wall's top from the paddock's model as it lies."""
    pad, p0, d, nrm, eave = b['pad'], b['p0'], b['d'], b['nrm'], b['eave']
    under = eave + ROOF_UP - ROOF_THICK                                 # (the roof's underside at its eaves)
    W = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(pad) if n['name'] in ('BLD_WALLS', 'BLD_ROOFS') and len(n['faces'])])
    P = W.reshape(-1, 3)
    a, o = (P[:, [0, 2]] - p0) @ d, (P[:, [0, 2]] - p0) @ nrm

    def low_top(s_):
        near = (np.abs(a - s_) < .8) & (np.abs(o - OPEN_BACK) < .4)
        g = float(ground(*(p0 + d * s_ + nrm * OPEN_BACK))[0])
        y = P[near, 1]
        y = y[y < g + 2.]
        return y.max() if len(y) else g + 1.
    ss = np.linspace(OPEN_FROM + .3, OPEN_TO - .3, int((OPEN_TO - OPEN_FROM) / 1.5) + 1)
    lows = np.array([low_top(s_) for s_ in ss])
    # (one straight line, never over the wall's top: its foot in the wall where the wall is higher)
    k_ = np.polyfit(ss, lows, 1)
    lows = np.polyval(k_, ss) + min(0., float((lows - np.polyval(k_, ss)).min())) - .01
    # (in the middle of the wall's thickness)
    g0 = float(ground(*(p0 + d * ss.mean() + nrm * OPEN_BACK))[0])
    wall = (a > OPEN_FROM) & (a < OPEN_TO) & (np.abs(o - OPEN_BACK) < .4) & (P[:, 1] < g0 + 2.)
    mid = (o[wall].min() + o[wall].max()) / 2 if wall.any() else OPEN_BACK + .12
    fence_, rail = [], []
    for s0_, s1_, l0, l1 in zip(ss[:-1], ss[1:], lows[:-1], lows[1:]):
        q0, q1 = p0 + d * s0_ + nrm * mid, p0 + d * s1_ + nrm * mid
        fence_ += sh.quad((q0[0], l0, q0[1]), (q1[0], l1, q1[1]), (q1[0], under, q1[1]), (q0[0], under, q0[1]), tile=(1., 1.), both=True)
        rail += sh.beam((q0[0], l0 + .02, q0[1]), (q1[0], l1 + .02, q1[1]), .05)
    model.add('OPEN_AREA_FENCE', 'Wire', fence_)
    model.add('OPEN_AREA_FENCE_RAIL', 'Steel', rail)
    # the east end walled up to its eave, from the back to the front
    wall, hit = [], []
    D = b['depth']
    oo = np.linspace(-D, 0., 9)
    for o0, o1 in zip(oo[:-1], oo[1:]):
        q0, q1 = p0 + d * (OPEN_FROM + .1) + nrm * o0, p0 + d * (OPEN_FROM + .1) + nrm * o1
        g0, g1 = float(ground(*q0)[0]) - .2, float(ground(*q1)[0]) - .2
        wall += sh.quad((q0[0], g0, q0[1]), (q1[0], g1, q1[1]), (q1[0], eave, q1[1]), (q0[0], eave, q0[1]), tile=(2., 2.), both=True)
        hit += sh.quad((q0[0], g0, q0[1]), (q1[0], g1, q1[1]), (q1[0], eave, q1[1]), (q0[0], eave, q0[1]), both=True)
    model.add('OPEN_AREA_EAST_WALL', 'GreyWall', wall)
    model.add('1WALL_OPEN_AREA_EAST', 'GreyWall', hit)


# the timing screens in the open end of the building along the hotlap's pit lane, in its north east
# corner, facing the crowd (south west): a long table (TIMING_AT, x, z, its middle) and on it two TVs,
# the times on the left as they look at it, the HotLap Velopark site's times on the right, the
# player's best lap among them
TIMING_AT, TIMING_YAW = (19.95, -31.8), math.radians(45.)
TIMING_TABLE, TIMING_TV = (2.9, .8, .75), (1.24, .72, .04)   # its top's width, depth, height; each TV's width, height, depth
TIMING_SCREENS = ((.72, 'Timing'), (-.72, 'HotlapPage'))   # each TV's middle across the table (+ on the left seen from in front), its screen
# the HotLap Velopark site's times (hotlap.albertowd.dev's results/results.json, a copy), each driver's
# best with each car, in order; on the TV HOTLAP_SHOW of them at a time under the site's header
HOTLAP_RESULTS = os.path.join(ROOT, 'tools', 'reference', 'site', 'hotlap_results.json')
HOTLAP_SHOW, HOTLAP_ROW, HOTLAP_TOP = 14, 30, 150      # rows shown, their height and where they begin on the 1024 x 576 screen
HOTLAP_COLS = ((24, '#'), (80, 'Marca'), (230, 'Modelo'), (470, 'Ano'), (540, 'Piloto(a)'), (900, 'Tempo'), (930, 'Edicao'))


def hotlap_rows():
    """The site's times: each driver's best with each car, fastest first, as (place in that list, brand,
    car, year, driver, time in ms, edition)."""
    import unicodedata
    plain = lambda t: unicodedata.normalize('NFKD', str(t or '')).encode('ascii', 'ignore').decode().replace("'", '').replace('"', '')
    rows = []
    for r in json.load(open(HOTLAP_RESULTS, encoding='utf-8')):
        m = re.match(r'(\d+):(\d+(?:\.\d+)?)$', str(r.get('time') or ''))
        if not m:
            continue
        rows.append((int(m.group(1)) * 60000 + int(round(float(m.group(2)) * 1000)), r))
    rows.sort(key=lambda t: t[0])
    out, seen = [], set()
    for ms, r in rows:
        key = (r.get('name'), r.get('brand'), r.get('car'))
        if key in seen:
            continue
        seen.add(key)
        out.append((len(out) + 1, plain(r.get('brand')), plain(r.get('car')), str(r.get('year') or ''), plain(r.get('name')) or '-', ms,
                    int(r.get('edition') or 0)))
    return out


def lap(ms):
    return '%d:%06.3f' % (ms // 60000, (ms % 60000) / 1000.)


def hotlap_page_texture():
    """The HotLap Velopark site's classification, 16 wide to 9 high, its fastest HOTLAP_SHOW as the site
    shows them (what the TV shows without the Custom Shaders Patch, and in BeamNG)."""
    from PIL import ImageDraw
    W, H = 1024, 576
    img = Image.new('RGB', (W, H), (18, 18, 18))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 64), fill=(33, 33, 33))
    d.polygon([(16, 22), (30, 40), (44, 14), (34, 42), (24, 44)], fill=(230, 110, 30))
    d.text((60, 32), 'HotLap', fill=(240, 240, 240), font=sign_font(26), anchor='lm')
    d.text((W // 2, 105), 'Classificação', fill=(255, 128, 0), font=sign_font(34), anchor='mm')
    f, fb = sign_font(17), sign_font(17)
    y0 = HOTLAP_TOP - HOTLAP_ROW
    d.rectangle((16, y0, W - 16, HOTLAP_TOP), fill=(30, 30, 31))
    for x, t in HOTLAP_COLS:
        d.text((x, y0 + HOTLAP_ROW // 2), 'Edição' if t == 'Edicao' else t, fill=(235, 235, 235), font=fb, anchor='rm' if t == 'Tempo' else 'lm')
    for i, (rank, brand, car, year, name, ms, ed) in enumerate(hotlap_rows()[:HOTLAP_SHOW]):
        y = HOTLAP_TOP + i * HOTLAP_ROW
        col = (255, 128, 0) if i < 3 else (40, 200, 180)
        d.rectangle((16, y, W - 16, y + HOTLAP_ROW - 1), fill=(30, 30, 31))
        d.line((16, y + HOTLAP_ROW - 1, W - 16, y + HOTLAP_ROW - 1), fill=(55, 55, 58))
        cy = y + HOTLAP_ROW // 2
        for (x, _), t in zip(HOTLAP_COLS, (str(rank), brand, car, year, '@ ' + name, lap(ms), '%d.' % ed)):
            d.text((x, cy), t.replace('.', 'ª') if x == 930 else t, fill=(240, 240, 240) if x == 930 else col, font=f, anchor='rm' if x == 900 else 'lm')
    return img


def write_timing_display():
    """The session's times on the TV beside the site's, as a display of the Custom Shaders Patch: a
    script that draws every car of the session in order of its best lap (its laps, its last and its
    best lap, how far behind the fastest), under the session's clock. Written into the track's
    config; without the patch the TV shows timing_texture() painted."""
    path = os.path.join(LAYOUTS, 'extension', 'ext_config.ini')
    text = open(path, encoding='utf-8', newline='').read()
    lua = '\n'.join([
        '-- the session: every car in order of its best lap, from tools/build_outside.py',
        'local WHITE, GREY, GREEN, AMBER = rgbm(.95, .95, .95, 1), rgbm(.7, .7, .74, 1), rgbm(.45, .86, .45, 1), rgbm(.92, .78, .24, 1)',
        'local L, S = vec2(11, 20), vec2(8, 14)',
        'local function get(o, k) local ok, v = pcall(function() return o[k] end) if ok then return v end return nil end',
        'local function fmt(ms) if not ms or ms <= 0 then return "--:--.---" end return string.format("%02d:%06.3f", math.floor(ms / 60000), (ms % 60000) / 1000) end',
        'local function clock(s) s = math.max(0, math.floor(s or 0)) return string.format("%02d:%02d:%02d", math.floor(s / 3600), math.floor(s / 60) % 60, s % 60) end',
        'local function put(x, y, t, c, l) display.text{ pos = vec2(x, y), letter = l or S, text = t, font = "aria", color = c } end',
        'function update(dt)',
        '\tlocal ok, sim = pcall(ac.getSim)',
        '\tif not ok or not sim then return end',
        '\tlocal n = get(sim, "carsCount") or 1',
        '\tlocal rows = {}',
        '\tfor i = 0, n - 1 do',
        '\t\tlocal okc, car = pcall(ac.getCar, i)',
        '\t\tif okc and car then',
        '\t\t\tlocal okn, name = pcall(ac.getDriverName, i)',
        '\t\t\trows[#rows + 1] = { name = okn and name or ("CAR " .. (i + 1)), best = get(car, "bestLapTimeMs") or 0,',
        '\t\t\t\tlast = get(car, "previousLapTimeMs") or 0, laps = get(car, "lapCount") or 0 }',
        '\t\tend',
        '\tend',
        '\ttable.sort(rows, function(a, b) if (a.best > 0) ~= (b.best > 0) then return a.best > 0 end return a.best < b.best end)',
        '\tlocal left = get(sim, "sessionTimeLeft")',
        '\tput(24, 16, "Session:", GREY) put(110, 16, "HOT LAP", WHITE)',
        '\tput(24, 40, "Timing:", GREY) put(110, 34, clock((get(sim, "timeHours") or 0) * 3600 + (get(sim, "timeMinutes") or 0) * 60 + (get(sim, "timeSeconds") or 0)), WHITE, vec2(15, 28))',
        '\tput(24, 68, "Time to go:", GREY) put(130, 68, clock(left and left / 1000 or 0), WHITE)',
        '\tdisplay.rect{ pos = vec2(20, 92), size = vec2(984, 2), color = rgbm(.86, .86, .88, 1) }',
        '\tlocal fastest = rows[1] and rows[1].best or 0',
        '\tfor k = 1, math.min(#rows, 13) do',
        '\t\tlocal r, y = rows[k], 100 + (k - 1) * 33',
        '\t\tput(24, y + 4, tostring(k), WHITE, L)',
        '\t\tput(70, y, string.upper(r.name), WHITE, L)',
        '\t\tput(70, y + 20, "Laps: " .. r.laps, GREY, vec2(6, 11))',
        '\t\tput(400, y + 6, "Last Time: " .. fmt(r.last), GREY)',
        '\t\tput(600, y + 6, "Best Time: " .. fmt(r.best), k == 1 and r.best > 0 and GREEN or WHITE)',
        '\t\tif k > 1 and r.best > 0 and fastest > 0 then put(820, y + 6, string.format("Diff: %.3f", (r.best - fastest) / 1000), GREY) end',
        '\t\tdisplay.rect{ pos = vec2(20, y + 32), size = vec2(984, 1), color = rgbm(.35, .36, .4, 1) }',
        '\tend',
        '\tput(24, 554, "Timing and results are not official", AMBER, vec2(7, 12))',
        'end'])
    block = '\n'.join([
        "; The session's times on the TV in the open end of the building by the hotlap's pit lane, drawn",
        "; by this script, written by tools/build_outside.py. Without the patch the TV shows them painted.",
        '[SCRIPTABLE_DISPLAY_...]', 'MATERIALS = OutTiming', 'RESOLUTION = 1024, 576',
        'DISPLAY_POS = 0, 0', 'DISPLAY_SIZE = 1024, 576', "SCRIPT = '", lua + "'", 'BACKGROUND = 0.04, 0.05, 0.07, 1',
        "; (the end of the session's times)"]) + '\n\n'
    old = re.search(r"; The session's times on the TV.*?; \(the end of the session's times\)\n\n", text, re.S)
    if old:
        text = text[:old.start()] + block + text[old.end():]
    else:
        at = text.index('; ---- Start gantry board')
        text = text[:at] + block + text[at:]
    open(path, 'w', encoding='utf-8', newline='').write(text)


def write_hotlap_display():
    """The HotLap Velopark site's times on the TV as a display of the Custom Shaders Patch: a script
    that draws them as the site shows them and, once the player has a best lap, puts it in among
    them, its place in the middle, the times either side round it; till then the whole list rolls
    by. Written into the track's config from hotlap_rows()."""
    path = os.path.join(LAYOUTS, 'extension', 'ext_config.ini')
    text = open(path, encoding='utf-8', newline='').read()
    rows = hotlap_rows()
    data = ',\n'.join('{%d,"%s","%s","%s","%s",%d,%d}' % (r, b[:12], c[:22], y, n[:30], ms, e) for r, b, c, y, n, ms, e in rows)
    cols = ', '.join(str(x) for x, _ in HOTLAP_COLS)
    lua = '\n'.join([
        '-- the HotLap Velopark times (rank, brand, car, year, driver, ms, edition), from tools/build_outside.py',
        'local R = {', data, '}',
        'local X = { %s }' % cols,
        'local SHOW, ROW, TOP = %d, %d, %d' % (HOTLAP_SHOW, HOTLAP_ROW, HOTLAP_TOP),
        'local ORANGE, TEAL, WHITE = rgbm(1, .5, 0, 1), rgbm(.16, .78, .7, 1), rgbm(.94, .94, .94, 1)',
        'local L = vec2(10, 18)',
        'local scroll = 0',
        'local function fmt(ms) return string.format("%d:%06.3f", math.floor(ms / 60000), (ms % 60000) / 1000) end',
        'local function put(x, y, t, c, right) if right then x = x - #t * L.x end display.text{ pos = vec2(x, y), letter = L, text = t, font = "aria", color = c } end',
        'function update(dt)',
        '\tdisplay.rect{ pos = vec2(0, 0), size = vec2(1024, 64), color = rgbm(.13, .13, .13, 1) }',
        '\tdisplay.text{ pos = vec2(60, 18), letter = vec2(15, 28), text = "HotLap", font = "aria", color = WHITE }',
        '\tdisplay.text{ pos = vec2(512 - 13 * 11, 84), letter = vec2(22, 38), text = "Classificacao", font = "aria", color = ORANGE }',
        '\tdisplay.rect{ pos = vec2(16, TOP - ROW), size = vec2(992, ROW), color = rgbm(.12, .12, .12, 1) }',
        '\tfor i, h in ipairs({ "#", "Marca", "Modelo", "Ano", "Piloto(a)", "Tempo", "Edicao" }) do put(X[i], TOP - ROW + 6, h, WHITE, i == 6) end',
        '\tlocal best = 0',
        '\tlocal ok, car = pcall(ac.getCar, 0)',
        '\tif ok and car and car.bestLapTimeMs and car.bestLapTimeMs > 0 then best = car.bestLapTimeMs end',
        '\tlocal you = nil',
        '\tif best > 0 then you = #R + 1 for i = 1, #R do if R[i][6] > best then you = i break end end end',
        '\tlocal n = #R + (you and 1 or 0)',
        '\tif you then',
        '\t\tlocal target = math.max(0, math.min(n - SHOW, you - 1 - math.floor(SHOW / 2)))',
        '\t\tscroll = scroll + (target - scroll) * math.min(1, dt * 2)',
        '\telse',
        '\t\tscroll = (scroll + dt * .7) % n',
        '\tend',
        '\tlocal first = math.floor(scroll)',
        '\tfor j = 0, SHOW - 1 do',
        '\t\tlocal p = (first + j) % n + 1',
        '\t\tlocal y = TOP + j * ROW',
        '\t\tif you and p == you then',
        '\t\t\tdisplay.rect{ pos = vec2(16, y), size = vec2(992, ROW - 1), color = rgbm(1, .5, 0, .35) }',
        '\t\t\tput(X[1], y + 6, tostring(you), WHITE) put(X[5], y + 6, "@ VOCE", WHITE) put(X[6], y + 6, fmt(best), WHITE, true)',
        '\t\telse',
        '\t\t\tlocal r = R[(you and p > you) and p - 1 or p]',
        '\t\t\tlocal c = r[1] <= 3 and ORANGE or TEAL',
        '\t\t\tdisplay.rect{ pos = vec2(16, y), size = vec2(992, ROW - 1), color = rgbm(.12, .12, .12, 1) }',
        '\t\t\tput(X[1], y + 6, tostring(r[1]), c) put(X[2], y + 6, r[2], c) put(X[3], y + 6, r[3], c) put(X[4], y + 6, r[4], c)',
        '\t\t\tput(X[5], y + 6, "@ " .. r[5], c) put(X[6], y + 6, fmt(r[6]), c, true) put(X[7], y + 6, r[7] .. ".", WHITE)',
        '\t\tend',
        '\tend',
        'end'])
    block = '\n'.join([
        "; The HotLap Velopark site's times on the TV in the open end of the building by the hotlap's pit",
        "; lane, drawn by this script, the player's best lap among them: written by tools/build_outside.py",
        "; from the site's results. Without the patch the TV shows the fastest painted.",
        '[SCRIPTABLE_DISPLAY_...]', 'MATERIALS = OutHotlapPage', 'RESOLUTION = 1024, 576',
        'DISPLAY_POS = 0, 0', 'DISPLAY_SIZE = 1024, 576', "SCRIPT = '", lua + "'", 'BACKGROUND = 0.07, 0.07, 0.07, 1',
        "; (the end of the HotLap Velopark site's times)"]) + '\n\n'
    old = re.search(r"; The HotLap Velopark site's times on the TV.*?; \(the end of the HotLap Velopark site's times\)\n\n", text, re.S)
    if old:
        text = text[:old.start()] + block + text[old.end():]
    else:
        at = text.index('; ---- Start gantry board')
        text = text[:at] + block + text[at:]
    open(path, 'w', encoding='utf-8', newline='').write(text)
    return len(rows)


def apron_step(model):
    """The step down from the pit apron by the hotlap's pit lane to the car park: a face along its edge
    from the car park up to the apron, but where the ramp joins them (tools/drape_track.py's
    PARK_RAMP)."""
    import restyle_paddock as rp
    import drape_track as dt
    edge = np.array(list(rp.APRON_SOUTH) + list(rp.APRON_EAST), float)
    track = gltf_source.load_kn5(bp.TRACK)
    pad = gltf_source.load_kn5(bp.PADDOCK)
    apron = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track) if n['name'] == rp.HOTLAP_BOX_ROAD]))
    park = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(pad) if n['name'].startswith(dt.PARK_MESH) and len(n['faces'])]))
    face = []
    for (ax, az), (bx, bz) in zip(edge[:-1], edge[1:]):
        L = np.hypot(bx - ax, bz - az)
        t = np.linspace(0., 1., max(2, int(L / .5) + 1))
        P = np.c_[ax + (bx - ax) * t, az + (bz - az) * t]
        u = np.array([bx - ax, bz - az]) / L
        out = np.array([u[1], -u[0]])                                    # (to the car park's side)
        if out[1] > 0:
            out = -out
        top = apron.ys(P - out * .15)
        low = park.ys(P + out * .15) - .02
        # (where either is not found, as its neighbours: no gap in the face)
        idx = np.arange(len(P))
        for v in (top, low):
            ok = np.isfinite(v)
            if ok.any() and not ok.all():
                v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
        for i in range(len(P) - 1):
            x0, x1 = P[i, 0], P[i + 1, 0]
            if dt.PARK_RAMP[0] - .5 < (x0 + x1) / 2 < dt.PARK_RAMP[1] + .5:
                continue
            if not np.isfinite([top[i], top[i + 1], low[i], low[i + 1]]).all():
                continue
            face += sh.quad((P[i, 0], low[i], P[i, 1]), (P[i + 1, 0], low[i + 1], P[i + 1, 1]),
                            (P[i + 1, 0], top[i + 1], P[i + 1, 1]), (P[i, 0], top[i], P[i, 1]), tile=(1., 1.), both=True)
    model.add('APRON_STEP', 'Concrete', face)


def timing_screen(model, ground):
    """A long table and two TVs on it, at TIMING_AT, their screens to TIMING_YAW (0 is south, the
    local -z the front)."""
    x, z = TIMING_AT
    y = float(ground(x, z)[0]) + .02
    w, dp, h = TIMING_TABLE
    table = sh.box((-w / 2, h - .04, -dp / 2), (w / 2, h, dp / 2), tile=(1., 1.))
    for lx, lz in ((-w / 2 + .06, -dp / 2 + .06), (w / 2 - .06, -dp / 2 + .06), (w / 2 - .06, dp / 2 - .06), (-w / 2 + .06, dp / 2 - .06)):
        table += sh.box((lx - .03, 0., lz - .03), (lx + .03, h - .04, lz + .03))
    model.add('TIMING_TABLE', 'Wood', sh.place(table, TIMING_YAW, (x, y, z)))
    tw, th, td = TIMING_TV
    lift = h + .12                                                  # (its foot on the table, the screen over it)
    b = .02                                                         # (the bezel round the screen)
    nn = np.array([0., 0., -1.])
    for cx, mat in TIMING_SCREENS:
        tv = sh.box((cx - tw / 2, lift, -td / 2), (cx + tw / 2, lift + th, td / 2), faces='nsewtb')
        tv += sh.box((cx - .04, h, -.02), (cx + .04, lift + .1, .02))       # (its stand)
        tv += sh.box((cx - .3, h, -.12), (cx + .3, h + .015, .12))           # (its foot)
        model.add('TIMING_TV', 'Frame', sh.place(tv, TIMING_YAW, (x, y, z)))
        V = np.array([(cx + tw / 2 - b, lift + th - b, -td / 2 - .002), (cx - tw / 2 + b, lift + th - b, -td / 2 - .002),
                      (cx - tw / 2 + b, lift + b, -td / 2 - .002), (cx + tw / 2 - b, lift + b, -td / 2 - .002)])
        U = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])      # (seen from the front, -x on the right)
        screen = [sh._part(V, np.tile(nn, (4, 1)), U, sh._facing(V, [(0, 1, 2), (0, 2, 3)], nn))]
        model.add('TIMING_SCREEN_' + mat.upper(), mat, sh.place(screen, TIMING_YAW, (x, y, z)))


def middle_fence(model, ground):
    """One tall mesh fence on thin steel posts, as the south ones, along the road by the vp_1500's
    pit entry and on beside the hotlap's pit lane (screenshots 13 and 14 in tools/reference/site):
    in front of the changing rooms and the bar, the race control's side and the wooden stands, to
    the dirt path from the kids track to the car park. On the side of the roads away from the
    track, PIT_FENCE_OFF off their edge."""
    k = gltf_source.load_kn5(bp.TRACK)
    tris = [bp.triangles(n, m)[:, :, [0, 2]] for n, m in bp.meshes(k) if n['name'] in PIT_FENCE_ROADS and len(n['faces'])]
    road = unary_union([Polygon(t).buffer(.01) for t in np.concatenate(tris) if Polygon(t).area > 1e-6])
    P = []
    for z in np.arange(PIT_FENCE_Z[0], PIT_FENCE_Z[1] - .01, -1.):
        cut = road.intersection(LineString([(-60., z), (20., z)]))
        if cut.is_empty:
            continue
        P.append((min(x for g in getattr(cut, 'geoms', [cut]) for x, _ in g.coords) - PIT_FENCE_OFF, z))
    P = np.array(P)
    # in front of the changing rooms, where the road runs up to their wall, half a metre out from it
    N = np.asarray(MIDDLE['PADDOCK_BUILDING_N'][0], float)
    (ax_, az_), (bx_, bz_) = N[1], N[2]                    # its west side, south to north
    on = (P[:, 1] >= az_) & (P[:, 1] <= bz_)
    P[on, 0] = np.maximum(P[on, 0], ax_ + (bx_ - ax_) * (P[on, 1] - az_) / (bz_ - az_) + .5)
    # its kinks eased where one road meets the other
    for _ in range(3):
        P[1:-1] = (P[:-2] + 2 * P[1:-1] + P[2:]) / 4
    fence(model, 'PIT_ENTRY', [tuple(p) for p in P], ground, metal=True)


def pit_tent(model, ground):
    """The big white tent north of the solar shed, by the pit boxes: on steel posts, all white
    fabric, rising from the square the posts make like a ball, open under it."""
    c, (w, d), ez = tent_on_floor()
    c, ez = np.array(c, float), np.array(ez, float)
    ez /= np.linalg.norm(ez)
    ex = np.array([ez[1], -ez[0]])
    corners = np.array([c - ex * w / 2 - ez * d / 2, c + ex * w / 2 - ez * d / 2, c + ex * w / 2 + ez * d / 2, c - ex * w / 2 + ez * d / 2])
    y = ground.low(corners)
    eave = y + PIT_TENT_EAVE
    posts = []
    for i in range(4):
        for f in (0., .5):
            x, z = corners[i] + (corners[(i + 1) % 4] - corners[i]) * f
            posts += sh.beam((x, y - .2, z), (x, eave, z), .1, tile=(.5, 1.))
    model.add('PIT_TENT_POLES', 'Steel', posts)
    # the fabric over the square: highest in its middle, down to the eaves all round its edges,
    # standing up from them as a ball's side does
    K = 24
    g = np.cos(np.linspace(np.pi, 0, K + 1))           # -1 to 1, closer at the edges
    U, Vv = np.meshgrid(g, g)
    h = eave + PIT_TENT_DOME * np.sqrt(np.clip((1 - U ** 2) * (1 - Vv ** 2), 0, 1))
    xz = c + U[..., None] * ex * w / 2 + Vv[..., None] * ez * d / 2
    V = np.c_[xz[..., 0].ravel(), h.ravel(), xz[..., 1].ravel()]
    F = []
    for i in range(K):
        for j in range(K):
            a, b = i * (K + 1) + j, i * (K + 1) + j + 1
            F += [(a, b, b + K + 1), (a, b + K + 1, a + K + 1)]
    F = np.array(F)
    N = sh._vertex_normals(V, F)
    N[N[:, 1] < 0] *= -1
    tri_n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (tri_n * N[F].mean(1)).sum(1) < 0
    F[flip] = F[flip][:, ::-1]
    U = V[:, [0, 2]] / 3.
    model.add('PIT_TENT_TOP', 'Fabric', [sh._part(V, N, U, F), sh._part(V, -N, U, F[:, ::-1])])


def pit_building(model, ground):
    """The pit building, as PIT_BUILDING and the rest above say: built round its middle, its x
    across it (east), its z along it (north), and turned and moved to where it stands."""
    c = np.array(PIT_BUILDING[0], float)
    ez = np.array(PIT_BUILDING[1], float)
    ez /= np.linalg.norm(ez)
    ex = np.array([ez[1], -ez[0]])
    yaw = math.atan2(-ez[0], ez[1])
    at = (c[0], 0., c[1])
    local = lambda p: np.array([(np.asarray(p) - c) @ ex, (np.asarray(p) - c) @ ez])
    world = lambda P: np.asarray(P, float) @ np.array([ex, ez]) + c
    place = lambda parts: sh.place(parts, yaw, at)
    cw, cl = PIT_CORE
    rw, rl = PIT_ROOF
    xe, xw = -cw / 2, cw / 2 + PIT_CORE_WEST      # its sides across: the 1000's (east), the 1500's (west, out to that lane)
    side_x = lambda side: xw if side > 0 else xe
    core = np.array([(xe, -cl / 2), (xw, -cl / 2), (xw, cl / 2), (xe, cl / 2)])
    # its floor over the highest ground under it, the paddock's asphalt not showing through it; its
    # walls down to the lowest
    gx, gz = np.meshgrid(np.linspace(xe, xw, 10), np.linspace(-cl / 2, cl / 2, 27))
    inside = world(np.c_[gx.ravel(), gz.ravel()])
    floor = float(np.max(ground(inside[:, 0], inside[:, 1]))) + .03
    foot = float(ground.low(world(core))) - .3
    cols = [local(p) for p in PIT_COLUMNS]
    feet = [float(ground.low(world(np.array([q + (-.2, -.2), q + (.2, -.2), q + (.2, .2), q + (-.2, .2)])))) for q in cols]
    eave = max(max(feet), floor) + PIT_EAVE
    # the roof's underside over a point across it, as barrel_roof arches it
    rr = PIT_RISE * rw
    R = (rw * rw / 4 + rr * rr) / (2 * rr)
    yc = eave + ROOF_UP + rr - R
    under = lambda x: yc + math.sqrt(max(R * R - x * x, 0.)) - ROOF_THICK
    outer = np.array([(-rw / 2, -rl / 2), (rw / 2, -rl / 2), (rw / 2, rl / 2), (-rw / 2, rl / 2)])
    roof, _ = barrel_roof(outer, eave, rise=PIT_RISE)
    model.add('PIT_BUILDING_ROOF', 'RoofDark', place(roof))
    # the workshop: walls up to the roof, doorways and the band of windows in them
    top = min(under(xe), under(xw)) - .02
    holes, wins = {}, []
    for i in range(4):
        a, along, L, n = side_of(core, i)
        hs, doors = [], []
        if abs(along[1]) > .9:
            for zd in PIT_DOORS:
                s = (zd - a[1]) / along[1]
                doors.append((s - PIT_DOOR[0] / 2, s + PIT_DOOR[0] / 2))
                hs.append((s - PIT_DOOR[0] / 2, s + PIT_DOOR[0] / 2, foot, floor + PIT_DOOR[1]))   # (open to the ground: the asphalt runs in)
        # the windows, in runs between the doorways and the corners
        cuts = sorted([(0., .6)] + [(d0 - .5, d1 + .5) for d0, d1 in doors] + [(L - .6, L)])
        for (_, e0), (s1, _) in zip(cuts, cuts[1:]):
            if s1 - e0 > 1.5:
                k = max(1, int(round((s1 - e0) / BAND_PANE)))
                hole = (e0, s1, floor + PIT_BAND[0], floor + PIT_BAND[1])
                hs.append(hole)
                wins.append((i, hole, dict(bars=tuple(e0 + (s1 - e0) * j / k for j in range(1, k)))))
        holes[i] = hs
    band = floor + PIT_WALL                 # the cream walls' top; white braced panels over it
    outer_w, inner_w = holed_walls(core, foot, band, holes)
    model.add('PIT_BUILDING_WALLS', 'Plaster', place(outer_w))
    model.add('PIT_BUILDING_INSIDE', 'GreyWall', place(inner_w))
    glazing(model, 'PIT_BUILDING', core, wins, yaw, at)

    def runs(i):
        """The side's stretches but its doorways."""
        a, along, L, n = side_of(core, i)
        edges = [0.]
        for s0, s1, h0, h1 in sorted(holes[i]):
            if h0 <= floor + .01:
                edges += [s0, s1]
        edges.append(L)
        return a, along, n, [(s0, s1) for s0, s1 in zip(edges[::2], edges[1::2]) if s1 - s0 > .05]
    # the grey skirt over the walls' feet, but in the doorways
    skirt = []
    for i in range(4):
        a, along, n, rs = runs(i)
        for s0, s1 in rs:
            skirt.append(wall_quad(a + n * .012, along, n, s0, s1, foot, floor + PIT_SKIRT))
    model.add('PIT_BUILDING_SKIRT', 'GreyWall', place(skirt))
    # the doorways: their rolled up doors' boxes over them, inside and out
    rolls = []
    for zd in PIT_DOORS:
        for xs in (-1, 1):
            x = side_x(xs)
            rolls += sh.box((x - .25, floor + PIT_DOOR[1], zd - PIT_DOOR[0] / 2 - .1), (x + .25, floor + PIT_DOOR[1] + .45, zd + PIT_DOOR[0] / 2 + .1))
    model.add('PIT_BUILDING_DOORS', 'Steel', place(rolls))
    # over the cream walls up to the roof, all round: white panels, dark steel crossed on them in
    # bays, as the screenshots show it
    gables, braces = [], []
    for i in range(4):
        a, along, L, n = side_of(core, i)
        k = max(2, int(round(L / 2.6)))
        ss = np.linspace(0., L, 2 * k + 1)
        pt = lambda s, o=0.: (a[0] + along[0] * s + n[0] * o, a[1] + along[1] * s + n[1] * o)
        roof_at = lambda s: under(pt(s)[0]) - .01
        P3 = [(pt(0.)[0], band, pt(0.)[1])] + [(pt(s)[0], roof_at(s), pt(s)[1]) for s in ss] + [(pt(L)[0], band, pt(L)[1])]
        gables += sh.poly(np.array(P3, float), n=np.array([n[0], 0., n[1]]), tile=(2., 2.),
                          axes=(np.array([along[0], 0., along[1]]), np.array([0, 1., 0])))
        q = lambda s, y: (pt(s, .05)[0], y, pt(s, .05)[1])
        for s0, s1 in zip(ss[::2], ss[2::2]):
            braces += sh.beam(q(s0, band), q(s1, roof_at(s1) - .05), .1) + sh.beam(q(s1, band), q(s0, roof_at(s0) - .05), .1)
            braces += sh.beam(q(s1, band), q(s1, roof_at(s1) - .05), .1)
        braces += sh.beam(q(0., band), q(L, band), .12)
    model.add('PIT_BUILDING_GABLES', 'White', place(gables))
    # the columns, and the trusses from each to the workshop's wall, under the roof
    columns = []
    for q, foot in zip(cols, feet):
        columns += sh.cylinder((q[0], foot - .2, q[1]), .19, eave - foot + .2, seg=12, top=False)
    model.add('PIT_BUILDING_COLUMNS', 'Pillar', place(columns))

    def truss(x0, x1, z, low):
        """A lattice truss across, from x0 to x1 at z: its bottom chord at low, its top under the roof."""
        n = max(2, int(round(abs(x1 - x0) / 1.25)))
        xx = np.linspace(x0, x1, n + 1)
        bot = [(x, low, z) for x in xx]
        tp = [(x, under(x) - .08, z) for x in xx]
        out = []
        for i in range(n):
            out += sh.beam(bot[i], bot[i + 1], .1) + sh.beam(tp[i], tp[i + 1], .1)
            out += sh.beam(bot[i], tp[i + 1] if i % 2 == 0 else tp[i], .06) + sh.beam(bot[i + 1], tp[i + 1], .06)
        return out + sh.beam(bot[0], tp[0], .06)
    steel = braces
    for q in cols:
        steel += truss(q[0], side_x(np.sign(q[0])), q[1], eave)
    for z in np.linspace(-cl / 2 + 2., cl / 2 - 2., 6):
        steel += truss(xe, xw, z, top - 1.)
    # purlins along under the roof, and the eaves' beams over the columns
    for x in np.arange(-rw / 2 + 1., rw / 2 - .5, 2.6):
        steel += sh.beam((x, under(x) - .07, -rl / 2 + .3), (x, under(x) - .07, rl / 2 - .3), .06, .1)
    for side in (-1, 1):
        q = sorted([p for p in cols if np.sign(p[0]) == side], key=lambda p: p[1])
        steel += sh.beam((q[0][0], eave, -rl / 2 + .3), (q[-1][0], eave, rl / 2 - .3), .2, .3)
    model.add('PIT_BUILDING_STEEL', 'Frame', place(steel))
    # strip lights: over each lane between the trusses, and in rows in the workshop
    tubes, spots = [], []
    for x in [side_x(side) + side * d for side in (-1, 1) for d in (3., 7.)]:
        for z in np.arange(-rl / 2 + 3., rl / 2 - 2.5, 4.4):
            tubes += sh.box((x - .05, eave - .1, z - .6), (x + .05, eave - .03, z + .6), faces='nsewb')
            spots.append((x, eave - .12, z))
    for x in (-2.3, 2.3):
        for z in np.arange(-cl / 2 + 2.2, cl / 2 - 1.5, 3.):
            tubes += sh.box((x - .05, top - 1.1, z - .6), (x + .05, top - 1.03, z + .6), faces='nsewb')
            spots.append((x, top - 1.12, z))
    model.add('PIT_BUILDING_LAMPS', 'Lamp', place(tubes))
    S = np.array(spots, float)
    S = sh.place([sh._part(S, np.zeros_like(S), np.zeros((len(S), 2)), np.zeros((0, 3), int))], yaw, at)[0][0]
    LAMP_SPOTS.extend(map(tuple, S))
    # the signs over each lane at both ends: Techspeed over the vp_1000's (east: AC's x grows to
    # the west, so its x here is less), "curta vencer" over the vp_1500's, each read left to right
    # from outside
    signs = []
    for zs in (-1, 1):
        z = zs * (rl / 2 + .02)
        for side, v0 in ((-1, 0.), (1, .5)):
            xa, xb = side_x(side) + side * .3, side * (rw / 2 - .3)
            lo, hi = eave - .15, under(rw / 2 - .3) - .02
            V = np.array([(xa, lo, z), (xb, lo, z), (xb, hi, z), (xa, hi, z)])
            # seen from the south (zs -1), east (less x) is on the right; from the north, west
            right_is_b = (side < 0) == (zs < 0)
            u_a, u_b = (0., 1.) if right_is_b else (1., 0.)
            U = np.array([(u_a, v0 + .5), (u_b, v0 + .5), (u_b, v0), (u_a, v0)])
            nn = np.array([0., 0., float(zs)])
            signs.append(sh._part(V, np.tile(nn, (4, 1)), U, sh._facing(V, [(0, 1, 2), (0, 2, 3)], nn)))
    model.add('PIT_BUILDING_SIGNS', 'PitSigns', place(signs))
    # inside: the floor, benches along the walls, tool cabinets, racks of tyres and parts
    # (its floor the pits' asphalt, running in through its doorways with no step)
    benches, cabinets, tyres = [], [], []
    for side in (-1, 1):
        x = side_x(side) - side * .5
        for z0, z1 in ((-cl / 2 + 1.6, PIT_DOORS[0] - PIT_DOOR[0] / 2 - .5),
                       (PIT_DOORS[0] + PIT_DOOR[0] / 2 + .5, PIT_DOORS[1] - PIT_DOOR[0] / 2 - .5)):
            if z1 - z0 < 1.5:
                continue
            benches += sh.box((x - .35, floor + .86, z0), (x + .35, floor + .92, z1 - .8))
            for z in np.linspace(z0 + .1, z1 - .9, max(2, int((z1 - z0) / 1.5) + 1)):
                for dx in (-.3, .3):
                    benches += sh.beam((x + dx, floor, z), (x + dx, floor + .86, z), .05)
            cabinets += sh.box((x - .3, floor, z1 - .7), (x + .3, floor + 1.05, z1 - .05))
    # racks of tyres along the north wall, shelves of parts in the south corners
    for x in np.arange(xe + 1., xw - .8, 1.):
        for k in range(4):
            tyres += sh.cylinder((x, floor + .02 + k * .26, cl / 2 - .45), .27, .25, seg=10)
    for x in (xe + .9, xw - .9):
        z = -cl / 2 + .5
        for h in (0., .7, 1.4, 2.1):
            benches += sh.box((x - .8, floor + h, z - .3), (x + .8, floor + h + .04, z + .3))
        for dx in (-.78, .78):
            for dz in (-.28, .28):
                benches += sh.beam((x + dx, floor, z + dz), (x + dx, floor + 2.2, z + dz), .04)
    model.add('PIT_BUILDING_BENCHES', 'Steel', place(benches))
    model.add('PIT_BUILDING_CABINETS', 'Orange', place(cabinets))
    model.add('PIT_BUILDING_TYRES', 'Rubber', place(tyres))
    # its walls to hit: the workshop's, but in its doorways
    hit = []
    for i in range(4):
        a, along, n, rs = runs(i)
        for s0, s1 in rs:
            hit.append(wall_quad(a, along, n, s0, s1, foot, floor + 3.))
    model.add('1WALL_PIT_BUILDING', 'GreyWall', place(hit))
    return floor


def wooden_stand(model, ground, name, front, ax, L, D, tiers, toward):
    """A wooden stand as the real ones are (screenshots 08 and 09): timber, low, tiers of planks
    rising back from a front of boards, rails round the tiers, steps at one end, all on posts. Its
    front's middle FRONT, along AX, L long and D deep, its tiers TIERS over its foot, its front
    the side towards TOWARD."""
    front, ax = np.array(front, float), np.array(ax, float) / np.linalg.norm(ax)
    f = np.array([ax[1], -ax[0]])
    if f @ (np.array(toward, float) - front) < 0:
        f = -f
    xz = lambda s, d: (front[0] + ax[0] * (s - L / 2) - f[0] * d, front[1] + ax[1] * (s - L / 2) - f[1] * d)
    foot2 = np.array([xz(0, 0), xz(L, 0), xz(L, D), xz(0, D)])
    y0 = float(ground.low(foot2))
    at = lambda s, d, h: (xz(s, d)[0], y0 + h, xz(s, d)[1])
    foot = [at(0, 0, 0), at(L, 0, 0), at(L, D, 0), at(0, D, 0)]
    planks, posts, rails = [], [], []
    depth = D / len(tiers)
    for t, h in enumerate(tiers):
        d0, d1 = t * depth, (t + 1) * depth
        for s0 in np.arange(0., L - .01, 2.45):
            s1 = min(s0 + 2.45, L)
            planks += sh.quad(at(s0, d0, h), at(s1, d0, h), at(s1, d1, h), at(s0, d1, h), tile=(2., 2.))
        # its riser, boards from the tier below
        lo = 0. if t == 0 else tiers[t - 1]
        planks += sh.quad(at(0, d0, lo - .3 if t == 0 else lo), at(L, d0, lo - .3 if t == 0 else lo), at(L, d0, h), at(0, d0, h), tile=(2., 2.), both=True)
        for s in np.linspace(0., L, int(L / 2.45) + 2):
            for d in (d0 + .1, d1 - .1):
                posts += sh.beam(at(s, d, -.3), at(s, d, h), .12)
    # the back's boards, and rails round the top: along the front, the ends and the back
    planks += sh.quad(at(L, D, -.3), at(0, D, -.3), at(0, D, tiers[-1]), at(L, D, tiers[-1]), tile=(2., 2.), both=True)
    for s0, d0, s1, d1, h in ((0, .05, L, .05, tiers[0]), (0, D - .05, L, D - .05, tiers[-1]),
                              (.05, 0, .05, D, None), (L - .05, 0, L - .05, D, None)):
        for k_ in range(int(np.hypot(s1 - s0, d1 - d0) / 2.45) + 2):
            u = k_ / max(1, int(np.hypot(s1 - s0, d1 - d0) / 2.45) + 1)
            s, d = s0 + (s1 - s0) * u, d0 + (d1 - d0) * u
            hh = h if h is not None else tiers[min(int(d / depth), len(tiers) - 1)]
            rails += sh.beam(at(s, d, hh), at(s, d, hh + 1.), .08)
        for hr in (.5, 1.):
            hh0 = h if h is not None else tiers[0]
            hh1 = h if h is not None else tiers[-1]
            rails += sh.beam(at(s0, d0, hh0 + hr), at(s1, d1, hh1 + hr), .06, .12)
    # the steps up at its end: six treads beside it, up to its first tier
    n_st = 6
    for j in range(n_st):
        h = tiers[0] * (j + 1) / n_st
        s0, s1 = -1.7 + j * .28, -1.7 + (j + 1) * .28
        planks += sh.quad(at(s0, .3, h), at(s1, .3, h), at(s1, 1.3, h), at(s0, 1.3, h), tile=(1., 1.), both=True)
        planks += sh.quad(at(s0, .3, h - tiers[0] / n_st), at(s0, 1.3, h - tiers[0] / n_st), at(s0, 1.3, h), at(s0, .3, h), tile=(1., 1.), both=True)
    model.add(name + '_PLANKS', 'Wood', planks)
    model.add(name + '_POSTS', 'Wood', posts + rails)
    model.add('1WALL_' + name, 'Wood', sh.walls(np.array([(p[0], p[2]) for p in foot]), y0 - .3, y0 + tiers[-1] + 1., tile=(2., 2.)))


def shed_floor(model, ground):
    """The shed's floor, the kart platform under the tent's arches beside the 1000's pit lane, at the
    pits' height (screenshots 01, 02, 23 and 24 in tools/reference/site), where tools/drape_track.py
    raises the ground: light concrete over it all, from the lane's west edge to the asphalt of the
    1500's pit exit (SLAB_WEST, x), from the pits to just past where the raised ground ends north;
    and a steep concrete face down from it to the lane, up the lane's ramp from where the floor is
    FACE_FROM over it, and along its north side down to the grass towards the storage building. The
    faces drawn, and walls to hit."""
    import drape_track as dt
    (xa, za), (xb, zb) = dt.LANE_EDGE
    edge = lambda z: xa + (xb - xa) * (np.asarray(z, float) - za) / (zb - za)       # the lane's west edge
    z_n = dt.SHED_NORTH + dt.SHED_EDGE + 1.3                                         # the north face
    in_x = dt.FACE_IN + 1.                                                            # where the floor is level
    floor = lambda x, z: ground(np.maximum(x, edge(z) + in_x), np.minimum(z, dt.SHED_NORTH - .5)) + .04
    # the floor, on a grid a metre square, its east side along the lane's edge
    zs = np.arange(SLAB_SOUTH, z_n + .01, 1.)
    zs[-1] = z_n
    top = []
    for z0, z1 in zip(zs[:-1], zs[1:]):
        x0s, x1s = np.linspace(edge(z0) + .05, SLAB_WEST, 12), np.linspace(edge(z1) + .05, SLAB_WEST, 12)
        for k in range(11):
            P = [(x0s[k], z0), (x1s[k], z1), (x1s[k + 1], z1), (x0s[k + 1], z0)]
            top += sh.quad(*[(x, float(np.ravel(floor(x, z))[0]), z) for x, z in P], tile=(2., 2.), uv0=(P[0][1] / 2., P[0][0] / 2.))   # (mapped as one)
    model.add('1ROAD_SHED_FLOOR', 'Concrete', top)
    face = []
    # (what lies round it as it is: the lane's asphalt and the ground)
    T = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(gltf_source.load_kn5(bp.TRACK))
                        if n['name'].startswith(('1ROAD', '1GRASS')) and len(n['faces'])])
    beside = bp.Surface(T)

    def foot(x, z):
        y = beside.ys(np.c_[np.atleast_1d(x), np.atleast_1d(z)])
        g = ground(x, z)
        return np.fmin(np.where(np.isnan(y), g, y), g) - .12
    # down to the lane, all along it, closed
    zz = np.linspace(SLAB_SOUTH, z_n, 80)
    xe = edge(zz) + .05
    lane = foot(edge(zz) - .3, zz)
    fl = floor(xe, zz)
    on = np.arange(len(zz))
    zz, xe, lane, fl = zz[on], xe[on], lane[on], fl[on]
    for i in range(len(zz) - 1):
        face += sh.quad((xe[i], lane[i], zz[i]), (xe[i + 1], lane[i + 1], zz[i + 1]),
                        (xe[i + 1], fl[i + 1], zz[i + 1]), (xe[i], fl[i], zz[i]), tile=(2., 2.), both=True)
    # down to the grass north of it, all across, closed
    xs = np.linspace(edge(z_n) + .05, SLAB_WEST, 60)
    grass = foot(xs, np.full(len(xs), z_n + .3))
    fn = floor(xs, np.full(len(xs), z_n))
    for i in range(len(xs) - 1):
        face += sh.quad((xs[i], grass[i], z_n), (xs[i + 1], grass[i + 1], z_n),
                        (xs[i + 1], fn[i + 1], z_n), (xs[i], fn[i], z_n), tile=(2., 2.), both=True)
    model.add('SHED_FLOOR_FACE', 'Concrete', face)
    model.add('1WALL_SHED_FLOOR_FACE', 'Concrete', face)
    print("the shed's floor: %.2f m over the 1000's pit lane at the north end of its face" % (fl[-1] - lane[-1] if len(on) else 0.))


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
            pil += sh.cylinder((x, yb - .3, z), .16, 3.45, seg=16, tile=(1., 1.))     # (round, up to under the roof's edge)
        model.add(name + '_PILLARS', 'Pillar', pil)
        model.add('1WALL_' + name, 'GreyWall', sh.walls(rect(bx, bz, bw, bd, 0), yb - .3, yb + 3., tile=(2., 2.)))
    # the sign by the road, north of the canopy, its faces to the west and the east, to the
    # traffic both ways along the road; the image the right way round seen from each side
    sx, sz = 196., -3.
    g = float(ground(sx, sz)[0])
    model.add('FUEL_SIGN_POST', 'Steel', sh.beam((sx, g - .3, sz), (sx, g + 3.2, sz), .3))
    panel = sh.box((sx - .2, g + 3.2, sz - 1.), (sx + .2, g + 7.2, sz + 1.), tile=(1., 1.), faces='nstb')
    model.add('FUEL_SIGN_POST', 'Steel', panel)
    U = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], float)
    for xx, nrm in ((sx + .21, 1.), (sx - .21, -1.)):
        # seen from out along nrm, its right is towards -z for the west face, +z for the east
        za, zb = (sz + 1., sz - 1.) if nrm > 0 else (sz - 1., sz + 1.)
        V = np.array([(xx, g + 3.2, za), (xx, g + 3.2, zb), (xx, g + 7.2, zb), (xx, g + 7.2, za)])
        n = np.array([nrm, 0., 0.])
        model.add('FUEL_SIGN', 'Pylon', [sh._part(V, np.tile(n, (4, 1)), U, sh._facing(V, [(0, 1, 2), (0, 2, 3)], n))])
        # the prices: a display of their own over the board's black boxes, each a square of it
        at = lambda u, v: (xx + nrm * .005, g + 7.2 - 4. * v, za + (zb - za) * u)
        k = len(FUEL_PRICES)
        for i, (u0, u1, v0, v1) in enumerate(price_boxes()):
            P = np.array([at(u0, v1), at(u1, v1), at(u1, v0), at(u0, v0)])
            UV = np.array([(0, (i + 1) / k), (1, (i + 1) / k), (1, i / k), (0, i / k)])
            model.add('FUEL_PRICES', 'FuelPrices', [sh._part(P, np.tile(n, (4, 1)), UV, sh._facing(P, [(0, 1, 2), (0, 2, 3)], n))])


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


def remove_wall_stub():
    """The piece of the old invisible wall behind the 1500's south straight's tyres (1WALL_161)
    left past the end of their wall at the pit entry, across the paving, out of velopark.glb."""
    k = gltf_source.load_kn5(bp.TRACK)
    taken = 0
    for n, m in bp.meshes(k):
        if n['name'] != STUB_WALL or not len(n['faces']):
            continue
        T = bp.triangles(n, m)
        (x0, x1), (z0, z1) = STUB_BOX
        inside = ((T[:, :, 0] > x0) & (T[:, :, 0] < x1) & (T[:, :, 2] > z0) & (T[:, :, 2] < z1)).all(1)
        if inside.any():
            n['faces'] = n['faces'][~inside]
            bp.sphere(n)
            taken += int(inside.sum())
    if taken:
        bp.save(k, bp.TRACK)
    return taken


def gantry_boxes():
    """Each start gantry's board closed as a box: its front's and its back's panels stand apart, and
    nothing went over their top, under their bottom or round their ends (seen from behind and a
    side, the inside of the front panel, which is not drawn). Its top, bottom and ends in the
    gantry's steel, as GANTRY_BOARD_SIDES. How many gantries."""
    import glob
    done = 0
    for f in sorted(glob.glob(os.path.join(bp.MODELS, 'velopark_gantry*.glb'))):
        k = gltf_source.load_kn5(f)
        if any(n['name'] == 'GANTRY_BOARD_SIDES' for n, _ in bp.meshes(k)):
            continue
        board = next(((n, m) for n, m in bp.meshes(k) if n['name'] == 'GANTRY_BOARD' and len(n['faces'])), None)
        steel = next((n for n, _ in bp.meshes(k) if n['name'] == 'GANTRY_STEEL'), None)
        if board is None or steel is None:
            continue
        T = bp.triangles(*board)
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        front = fn[np.argmax(np.abs(fn).sum(1))]
        nrm = front[[0, 2]] / np.linalg.norm(front[[0, 2]])
        u = np.array([-nrm[1], nrm[0]])
        sides = []
        for sgn in (1., -1.):
            P = T[(fn[:, [0, 2]] @ nrm) * sgn > 0].reshape(-1, 3)
            a_ = P[:, [0, 2]] @ u
            o = float(np.median(P[:, [0, 2]] @ nrm))
            ends = []
            for e in (a_.min(), a_.max()):
                q = P[np.abs(a_ - e) < .05]
                ends.append((e, q[:, 1].min(), q[:, 1].max()))
            sides.append((o, ends))
        (o0, e0), (o1, e1) = sides
        pt = lambda o, e, y: (u[0] * e + nrm[0] * o, y, u[1] * e + nrm[1] * o)
        parts = []
        for i in (0, 1):
            # the ends, each from the front's panel to the back's
            (ea, ya0, ya1), (eb, yb0, yb1) = e0[i], e1[i]
            parts += sh.quad(pt(o0, ea, ya0), pt(o1, eb, yb0), pt(o1, eb, yb1), pt(o0, ea, ya1), both=True)
        for top in (False, True):
            j = 2 if top else 1
            parts += sh.quad(pt(o0, e0[0][0], e0[0][j]), pt(o0, e0[1][0], e0[1][j]),
                             pt(o1, e1[1][0], e1[1][j]), pt(o1, e1[0][0], e1[0][j]), both=True)
        Pp, Np, Up, Fp = sh.merge(parts)
        Tg = gltf_source.tangents(Pp, Np, Up, Fp)
        V = np.zeros((len(Pp), steel['verts'].shape[1]))
        cols = np.c_[Pp, Np, Up, Tg]
        V[:, :min(V.shape[1], cols.shape[1])] = cols[:, :V.shape[1]]
        node = dict(steel, name='GANTRY_BOARD_SIDES', children=[], verts=V.astype(steel['verts'].dtype),
                    faces=np.asarray(Fp).astype('<u2'))                 # (at the model's root, where it stands)
        bp.sphere(node)
        k['root']['children'].append(node)
        bp.save(k, f)
        done += 1
    return done


def main():
    ground = Ground()
    lines = osm_lines()
    model = Model()
    # (--paddock: the paddock's and the track's own only (the buildings in its middle, the pit building, the
    # race control tower, the bar, the stands, the fuel station, the hotlap building's roof, the screens and
    # their lights, the judges' tower, the marshals' tents, the flagpoles); not the property round it)
    paddock_only = '--paddock' in sys.argv
    posts = 0
    if not paddock_only:
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
    if not paddock_only:
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
    # (the flag marshals' posts by the starts: objects of the road editor's now (objects.json: a marshal shelter, its
    # chair and its cones), laid by tools/place_objects.py; marshal() is what tools/templates.py takes the shelter from)
    flagpoles(model, ground, FLAGPOLES)
    control_tower(model, ground, -47., 80.5, front='w')
    for name, (P, eave) in MIDDLE.items():
        if name == 'PADDOCK_BUILDING_N':
            # the changing rooms: their walls WALKWAY short of the bar's, under the roof, a way
            # through between them
            P = np.asarray(P, float)
            walls = P.copy()
            for i, j in ((0, 3), (1, 2)):
                walls[i] = P[i] + (P[j] - P[i]) / np.linalg.norm(P[j] - P[i]) * WALKWAY
            middle_building(model, name, walls, eave, ground, roof_P=P)
        elif name == 'GARAGE':
            garage(model, ground)
        elif name != 'PADDOCK_BUILDING_M':
            middle_building(model, name, P, eave, ground)
    bar(model, ground)
    middle_fence(model, ground)
    pit_tent(model, ground)
    pit_building(model, ground)
    shed_floor(model, ground)
    # the stand by R26 with the banner, its front along the line the old concrete stand's was
    c, ax, L, D, tiers = STAND
    ax = np.array(ax, float) / np.linalg.norm(ax)
    f = np.array([ax[1], -ax[0]])
    if f @ (np.array([-19.01, -21.1]) - np.array(c)) < 0:
        f = -f
    wooden_stand(model, ground, 'STAND', np.array(c) + f * 5.85, ax, L, D, tiers, (-19.01, -21.1))
    # the small raised one north of it, its front to the track
    c, ax, L, D, tiers = STAND_2
    wooden_stand(model, ground, 'STAND_2', c, ax, L, D, tiers, (-18., 1.))
    print('%d building lamps lit at night' % write_lights())
    write_price_display()
    print("%d of the HotLap Velopark site's times on the TV" % write_hotlap_display())
    write_timing_display()
    fuel_station(model, ground)
    open_area(model, ground, hotlap_roof(model, ground))
    # a cone where the vp_1500's pit entry and the side road part (as the photos show it)
    cones = []
    for x, z in ROAD_CONES:
        g = float(ground(x, z)[0])
        cones += sh.box((x - .2, g, z - .2), (x + .2, g + .03, z + .2), tile=(.4, .4))
        cones += sh.cylinder((x, g + .03, z), .14, .67, seg=12, tile=(1., .7), r_top=.025)
    model.add('ROAD_CONES', 'Cone', cones)
    print('%d start gantries\' boards closed as boxes' % gantry_boxes())
    apron_step(model)
    timing_screen(model, ground)
    # the banners: along the stand's front wall by R26, and on the south fence west of R15
    banner(model, 'STAND', (-19.01, -21.1), (-9.06, -38.03), .2, 1.45, lift=.5)
    banner(model, 'FENCE', (152., -2.6), (168., 1.27), .35, 2.35, ground=ground, lift=.18)
    if not paddock_only:
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
    made['OUT_pit_signs.dds'] = (pit_signs_texture(), 'DXT1')
    made['OUT_fuel_prices.dds'] = (prices_texture(), 'DXT1')
    made['OUT_timing.dds'] = (timing_texture().resize((1024, 512), Image.LANCZOS), 'DXT1')   # (16 to 9 on the screen)
    made['OUT_hotlap_page.dds'] = (hotlap_page_texture().resize((1024, 512), Image.LANCZOS), 'DXT1')
    for name, (img, fmt) in made.items():
        data = kn5io.dds(img, fmt)
        open(os.path.join(TEX, name), 'wb').write(data)
        tex[name] = data
    tris, meshes = write(model, tex)
    print('%d fence posts; %d meshes, %d triangles, in %s' % (posts, meshes, tris, os.path.relpath(OUT, ROOT)))
    print('%d layouts given the model' % register())
    print("%d faces of the old wall past the 1500 pit entry's tyres out" % remove_wall_stub())
    if remove_old_fence():
        print('the old fence out of velopark.glb')


if __name__ == '__main__':
    main()
