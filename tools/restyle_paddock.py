"""The paddock's buildings as they are: grey block walls, salmon red pillars, darker roofs, in
source/models/velopark_paddock.glb.

    python tools/restyle_paddock.py

The paddock's buildings were all white, walls and posts alike. The real ones (the screenshots
of the vp_1500's start and the south) have walls of grey painted block and pillars and posts
painted salmon red. This takes every pillar and post out of the walls' mesh (a piece of it
narrower than PILLAR and taller than TALL) into a mesh of its own, BLD_PILLARS, of a material of
its own; and paints the walls grey (BuildingWallGrey.dds, made here). The roofs' sheets are made
darker (BuildingRoofDark.dds, ROOF_SHADE of BuildingRoof.dds).

It takes out too the buildings tools/build_outside.py makes again (REMADE: the race control
tower, the fuel station, the buildings in the middle of the track and the garage annex, now the
big tent by the pit boxes), each piece of the buildings' meshes whose middle stands in one, and
the walls to hit of those that have none now. Run again, it finds the paddock restyled, and does
only what is left to do.
"""
import os, sys

import numpy as np
from PIL import Image

import build_paths as bp
import gltf_source
import kn5tree
from plant_trees import pieces

TEX = os.path.join(bp.ROOT, 'source', 'textures')
PILLAR, TALL = .8, 2.         # metres: a piece narrower than this and taller than that is a pillar
ROOF_SHADE = .68
# the buildings made again by tools/build_outside.py: (x0, x1, z0, z1), and their walls to hit
# that go with them (the others' stay, as the new buildings stand where they did)
REMADE = {'tower': ((-52., -42., 75., 86.), ('1WALL_BLD_C3',)),
          'fuel station': ((178., 214., -23., -5.), ('1WALL_BLD_W1', '1WALL_BLD_W2')),
          'paddock building': ((-65.9, -33.5, 15.6, 74.5), ()),
          'garage': ((-47.4, -37.2, 141.4, 169.7), ('1WALL_BLD_C1',)),     # (its walls to hit: build_outside.py's, open at its bays)
          'pit tent': ((-46.5, -41.5, 123.9, 133.7), ('1WALL_BLD_C2',)),
          # the pit building, its roof over both pits; its columns' walls to hit stay
          'pit building': ((-62., -29., 95., 126.), ('1WALL_BLD_G1', '1WALL_BLD_G2'))}
BUILDING_MESHES = ('BLD_WALLS', 'BLD_ROOFS', 'BLD_DARK', 'BLD_DOORS', 'BLD_SOLAR')


def grey_wall(n=512, seed=3):
    """Painted concrete block: courses of blocks, a darker plinth, a little grime."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    g = np.asarray(Image.fromarray((rng.random((32, 32)) * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float) / 255
    course = (y * 10) % 1
    off = (np.floor(y * 10) % 2) * .5
    joint = ((course < .04) | (((x * 5 + off) % 1) < .02)).astype(float)
    base = 120 + 14 * (g - .5) - 28 * joint
    rgb = base[..., None] * np.array([.96, .98, 1.02])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def pillar_paint(n=256, seed=4):
    rng = np.random.default_rng(seed)
    g = np.asarray(Image.fromarray((rng.random((16, 16)) * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float) / 255
    rgb = (np.array([196., 104., 92.]) * (.9 + .15 * g[..., None]))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def inside(c, box, pad=1.):
    x0, x1, z0, z1 = box
    return (c[:, 0] > x0 - pad) & (c[:, 0] < x1 + pad) & (c[:, 1] > z0 - pad) & (c[:, 1] < z1 + pad)


def add_texture(k, name, img, fmt='DXT1'):
    import materials
    data = materials.save_texture(name, img, fmt)                          # (kept as PNG, AC's DDS made by its build)
    k['textures'] = [t for t in k['textures'] if t['name'] != name] + [dict(name=name, active=1, data=data)]


def take_remade(k, nodes):
    """The pieces of the buildings tools/build_outside.py makes again out, and their walls to hit."""
    taken = 0
    for name in BUILDING_MESHES:
        if name not in nodes:
            continue
        n, m = nodes[name]
        if not len(n['faces']):
            continue
        V = n['verts'][:, :3].astype(float) + m[3, :3]
        F = n['faces'].astype(int)
        lab = pieces(F, len(V))
        flab = lab[F[:, 0]]
        ids = np.unique(flab)
        c = np.array([V[lab == t][:, [0, 2]].mean(0) for t in ids])
        out = np.zeros(len(ids), bool)
        for box, _ in REMADE.values():
            out |= inside(c, box)
        if out.any():
            bp.split(n, np.isin(flab, ids[out]))
            taken += out.sum()
    gone_walls = {w for _, walls in REMADE.values() for w in walls}
    walls = 0
    for parent, _ in kn5tree.walk(k['root']):
        before = len(parent['children'])
        parent['children'] = [ch for ch in parent['children'] if ch['name'] not in gone_walls]
        walls += before - len(parent['children'])
    return taken, walls


def round_pillars(n):
    """The pillars of mesh n round, as the steel columns are: each one an upright tube as wide as it was
    across and as tall, its foot and its top where they were."""
    import shapes as sh
    from scipy.spatial import cKDTree
    V = n['verts'][:, :3].astype(float)
    F = n['faces'].astype(int)
    lab = pieces(F, len(V))
    ids = np.unique(lab[np.unique(F)])
    # (a pillar's sides are pieces of their own: those standing together one pillar)
    lo = np.array([V[lab == t].min(0) for t in ids])
    hi = np.array([V[lab == t].max(0) for t in ids])
    c2 = ((lo + hi) / 2)[:, [0, 2]]
    group = np.arange(len(ids))
    for a, b in cKDTree(c2).query_pairs(.35):
        ga, gb = group[a], group[b]
        group[group == max(ga, gb)] = min(ga, gb)
    parts = []
    for g in np.unique(group):
        m = group == g
        l, h = lo[m].min(0), hi[m].max(0)
        c = (l + h) / 2
        r = float(max(h[0] - l[0], h[2] - l[2])) / 2
        parts += sh.cylinder((c[0], l[1], c[2]), r, h[1] - l[1], seg=16, tile=(1., 1.), top=False)
    if not parts:
        return
    P, N, U, Fc = sh.merge(parts)
    T = gltf_source.tangents(P, N, U, Fc)
    out = np.zeros((len(P), n['verts'].shape[1]))
    cols = np.c_[P, N, U, T]
    out[:, :min(out.shape[1], cols.shape[1])] = cols[:, :out.shape[1]]
    n['verts'] = out.astype(n['verts'].dtype)
    n['faces'] = np.asarray(Fc).astype('<u2')


def paint(k, nodes):
    """The pillars out of the walls, into their own mesh, and the walls grey."""
    n, m = nodes['BLD_WALLS']
    V = n['verts'][:, :3].astype(float) + m[3, :3]
    F = n['faces'].astype(int)
    lab = pieces(F, len(V))
    flab = lab[F[:, 0]]
    ids = np.unique(flab)
    pillar = []
    for t in ids:
        p = V[lab == t]
        span = np.ptp(p[:, [0, 2]], axis=0)
        if span.max() < PILLAR and np.ptp(p[:, 1]) > TALL:
            pillar.append(t)
    sel = np.isin(flab, pillar)
    verts, faces = bp.split(n, sel)
    add_texture(k, 'BuildingWallGrey.dds', grey_wall())
    add_texture(k, 'BuildingPillar.dds', pillar_paint())
    wall_mat = next(i for i, mt in enumerate(k['materials']) if mt['name'] == 'BuildingWall')
    k['materials'][wall_mat]['textures'] = [dict(sampler='txDiffuse', slot=0, name='BuildingWallGrey.dds')]
    base = k['materials'][wall_mat]
    k['materials'].append(dict(base, name='BuildingPillar', props=[dict(p) for p in base['props']],
                               textures=[dict(sampler='txDiffuse', slot=0, name='BuildingPillar.dds')]))
    pnode = dict(n, name='BLD_PILLARS', children=[], material=len(k['materials']) - 1, verts=verts, faces=faces.astype('<u2'))
    bp.sphere(pnode)
    parent = next(p for p, _ in kn5tree.walk(k['root']) if any(ch is n for ch in p.get('children', [])))
    parent['children'].append(pnode)
    return len(pillar)


def darken_roofs(k):
    """The roofs' material onto a darker copy of its sheets; False when it is already."""
    mt = next(mt for mt in k['materials'] if mt['name'] == 'BuildingRoof')
    if mt['textures'][0]['name'] == 'BuildingRoofDark.dds':
        return False
    data = next(t['data'] for t in k['textures'] if t['name'] == mt['textures'][0]['name'])
    import io
    img = Image.open(io.BytesIO(data)).convert('RGB')
    add_texture(k, 'BuildingRoofDark.dds', Image.fromarray((np.asarray(img, float) * ROOF_SHADE).astype(np.uint8)))
    mt['textures'] = [dict(mt['textures'][0], name='BuildingRoofDark.dds')]
    return True


# the old track's stand by R26 (the GrandStand material), concrete steps 5 m high, which
# tools/build_outside.py makes again as the real one is: a low stand of timber with the banner on it
OLD_STAND = 'OBJ67'
# the 1000's pit lane north of the pits (screenshots 18 to 24 in tools/reference/site): its edges'
# x at z 126, and how much more a metre north (x grows west)
LANE_WEST, LANE_EAST, LANE_SLOPE = -49.76, -53.43, .1194
lane_west = lambda z: LANE_WEST + LANE_SLOPE * (np.asarray(z, float) - 126.)
lane_east = lambda z: LANE_EAST + LANE_SLOPE * (np.asarray(z, float) - 126.)
# grass, not asphalt: east of the lane north of the pits (from LANE_WIDER past its edge, where its
# own asphalt carries it on), between the shed's floor and the storage building beside it, and
# north of z 172 east of x -47 (x, z round each)
LANE_WIDER = .1
# the lane widened on its east side, LANE_EXTRA, from z to z (two and a half to three karts long
# across): its own mesh carried on past its edge, mapped as it is
LANE_EXTRA, WIDE_FROM, WIDE_TO = 1.3, 128., 200.
# its mouth onto the track: the corners between them rounded MOUTH_ROUND, as asphalt, within x, z
MOUTH_ROUND, MOUTH_BOX = 7., ((-70., -34.), (176., 206.))
MOUTH_FILL = 6.          # (its outline, the track's by it too, rounded out over this after all: no corner, notch or sliver)
# the 1500's pit exit's junction north of it, as drawn over the ground photo: the path from the north
# (path A) cut at JUNCTION_CUT (z) and laid on straight (its middle PATH_ON, PATH_WIDTH wide) into the
# top of the pit exit's curve, on its pits' side; the road from the storage building's west side
# (STORE_ROAD, STORE_WIDTH wide) across it, curving into the pit exit's curve on the track's side,
# the grass between them and the curve a triangle; a straight ramp (RAMP, x, z) from the shed's
# floor's concrete down to the pit exit; their corners rounded JUNCTION_ROUND; all as one asphalt,
# the pit exit's, laid on the roads' seams. All the paddock's asphalt else round them grass
# (JUNCTION_GRASS, x, z, but the shed's floor, the storage building and the pits)
PATH_ON, PATH_WIDTH = ((-26.8, 166.5), (-27.6, 149.), (-30.3, 133.6)), 3.6    # (path A's top: as wide, carried straight on from its middle)
STORE_ROAD, STORE_WIDTH, STORE_MESH = ((-38.4, 157.), (-36.7, 156.2), (-31.5, 152.4), (-27., 148.5), (-23.7, 143.2),
                                      (-20.9, 138.2), (-18.1, 134.6)), 3.6, '1ROAD_STORAGE'
JUNCTION_CUT, JUNCTION_ROUND = 166.5, 3.
RAMP = ((-35.65, -30.), (124.6, 134.6))
# the pit exit carried on across its curve's inside to under its tyre wall there (the stacks within
# EXIT_TYRES_NEAR of it, EXIT_TYRE_R round), the gaps between closed over EXIT_CLOSE: its own asphalt
EXIT_TYRES_NEAR, EXIT_TYRE_R, EXIT_CLOSE, EXIT_AREA = 3., .4, 1.6, ((-36., -6.), (108., 138.))
# where it meets the track past its tyre wall's end (EXIT_MERGE_AT, x, z), curving into the track's
# edge as a road merges, EXIT_MERGE round, not ending square across it; its tip, where it is thinner
# than EXIT_MERGE_MIN, left off (it ends on the track's edge, not trailing along beside it)
EXIT_MERGE, EXIT_MERGE_MIN, EXIT_MERGE_AT = 5., .6, ((-16., -8.), (104., 122.))
JUNCTION_GRASS = ((-49.5, -8.), (124.6, 172.))


GRASS_BACK = [[(float(lane_east(127.)) - LANE_WIDER, 127.), (float(lane_east(172.)) - LANE_WIDER, 172.), (-90., 172.), (-90., 127.)],
              [(float(lane_west(138.4)) + .3, 138.4), (-36.5, 138.4), (-36.5, 141.2), (float(lane_west(141.2)) + .3, 141.2)],
              [(-47., 172.), (-47., 200.), (-90., 200.), (-90., 172.)]]
GRASS_MESH = '1GRASS_PADDOCK_GRASS'
# the line the pit building's and the storage building's east walls stand on, beside the lane, the
# shed's floor's face on it too (x at z 126 and 142, as tools/drape_track.py's LANE_EDGE)
WALL_LINE = ((-48.78, 126.), (-47.26, 142.))
wall_line = lambda z: WALL_LINE[0][0] + (WALL_LINE[1][0] - WALL_LINE[0][0]) * (np.asarray(z, float) - WALL_LINE[0][1]) / (WALL_LINE[1][1] - WALL_LINE[0][1])
# the tyre wall along the lane, as the screenshots show it (19 to 24 in tools/reference/site and the
# one by the storage building): against the storage building's wall along its south end, and on to
# the shed's floor; none on the lane further, nor in front of the garage's
# openings. Its rows: metres in from the walls' line, from z to z; a stack every PIT_TYRE_STEP
PIT_TYRES = ('TYRES_100', 'TYRES_100F', '1WALL_TYRE_100')
PIT_TYRE_ROWS = ((.35, 138.6, 152.5),)
PIT_TYRE_STEP = .62
# the vp_1000's pit boxes stand in one row PIT_IN in from the walls' line, from the pit building to
# the storage building, clear of the tyres, with room to pass on the lane; each with its painted
# box (velopark_lines.glb's GRID_PITS, within PIT_PAINT of it)
PIT_LAYOUT, PIT_FROM, PIT_IN, PIT_PAINT = 'vp_1000', 90., 2.1, 2.5
NORTH_RUNOFF_Z = 185.
# where the ground is cut out under all the asphalt, old and new (x, z): the pits, the 1500's pit
# exit's junction
UNDER_PITS, UNDER_JUNCTION = ((-64., -27.), (92., 128.)), ((-50., -8.), (124., 172.))
UNDER_HOTLAP = ((-6., 140.), (-64., -28.))   # (the hotlap's pit lane, where it meets the road south, and the car park)
UNDER_LANE = ((-62., -44.), (126., 205.))     # (the 1000's pit lane, widened, beside the pits)
UNDER_PAVERS = ((114., 150.), (20., 42.))     # (the pavers along the main straight)
# the pit apron in front of the building along the hotlap's pit lane, as drawn over the ground photo:
# from the building's front (west of it from APRON_WEST, x) down to its south edge (APRON_SOUTH, x, z,
# west to east), round to its east corner (APRON_EAST, x, z); the hotlap's road curving into it on
# its outside over APRON_ROUND (m), APRON_ROUND_AT (x, z) the corner; one with the road along the
# building (HOTLAP_BOX_ROAD). The car park south of it a step under it (tools/drape_track.py), but
# where a ramp joins them (tools/drape_track.py's PARK_RAMP). The paddock's asphalt west of it, from
# the hotlap's road south, grass (APRON_GRASS, x, z)
APRON_WEST, APRON_SOUTH, APRON_EAST = 18., ((23.7, -67.4), (129.1, -43.6)), ()
APRON_NE = (123.3, -18.1)          # (its north east corner, on the building's front carried on)
# the car park south of it (x, z round it, as drawn): flat asphalt, PARK_MESH, a step under the apron
# where it meets it (tools/drape_track.py), but for the grass round its trees (tools/car_parks.py)
PARK, PARK_MESH = ((172.2, -33.6), (178.8, -71.5), (40., -97.5), (33.3, -59.7)), '1ROAD_PADDOCK_PARK'
APRON_ROUND, APRON_ROUND_AT = 0., ((6., 23.), (-64., -45.))
# the grass island between the hotlap's road and the road south where they come onto the apron, as
# drawn (x, z round it): out of the apron, grass
APRON_ISLAND = ((10., -47.1), (14., -47.6), (18., -48.2), (20.5, -48.8), (22.3, -49.6), (23., -51.), (23., -55.), (22.4, -56.3),
                (21., -56.7), (18., -56.3), (14., -54.4), (10., -52.2))
APRON_GRASS, APRON_MESH = ((-6., 23.5), (-68., -26.)), '1GRASS_PADDOCK_HOTLAP'
# the pit lane along the building, out from its front (its lines', boxes' and racing line's)
HOTLAP_EDGE = 7.57
# the hotlap's pit lane, from its fork off the track (where it runs on the track's mesh outside the
# track's edge line, z HOTLAP_LANE_Z) on its own roads (HOTLAP_ROADS: the road there, and the one
# along the pit garages' doors, its boxes on it) to the fence it joins the track through
# (HOTLAP_FENCE: x, z past which it has joined it)
HOTLAP_LANE_Z, HOTLAP_ROADS, HOTLAP_MESH = (30., 66.), ('1ROAD_13', '1ROAD_14'), 'LINE_VP_1500_HOTLAP_PIT'
# the building along the hotlap's pit lane: its front, the line of its columns (HOTLAP_COLUMNS, the
# paddock's mesh, within HOTLAP_COLUMNS_AREA, x, z), pulled back BUILDING_BACK to where the photo shows
# it (the shadow in front of it no part of it), its back where it is. In front of it the pit boxes'
# row, half the road (see hotlap_row), a yellow line along its far side; the hotlap's and the 2500's pit boxes in
# its middle; their pit lanes' racing lines, from their points HOTLAP_AI (where they come off the
# track onto the pit lane's road, to where they pass the fence), along the middle of the pit lane:
# its road, the lane beyond the boxes' row, and the road from the building's east end up to the
# fence, their corners rounded over HOTLAP_ROUND
HOTLAP_COLUMNS, HOTLAP_COLUMNS_AREA, BUILDING_BACK = 'BLD_PILLARS', ((14., 125.), (-50., -5.)), 3.4
HOTLAP_LAYOUTS, HOTLAP_BOX_ROAD, HOTLAP_BOX_ROW, HOTLAP_AI, HOTLAP_ROUND = ('vp_1500_hotlap', 'vp_2500'), '1ROAD_14', None, (25, 178), 10.
# in front of the building the pit lane's lines straight, the building's front's, the boxes' row's and
# the road's outer edge's, from a white line across it at the building's west end (HOTLAP_START, x
# on its front) to its east end's line
HOTLAP_START, HOTLAP_START_MESH = 20., 'LINE_VP_1500_HOTLAP_START'
# where its road meets the road along the building, the road widened on its outer side as it goes
# from after its tyre wall to the road along the building (from x HOTLAP_FLARE[0] on its edge to
# HOTLAP_FLARE[1] on the other's), curving out from one into the other; the road along the building
# grown over it
HOTLAP_FLARE = (8., 20.)    # (its end on the building's start, HOTLAP_START, its lines going straight on)
# and its inner side the same: from its road's edge at x HOTLAP_INNER curving onto the building's
# front at its start (HOTLAP_START), the asphalt grown over the grass between
HOTLAP_INNER = 10.
# ... onto its front at its east end (its corner), the roads north of that curve (between it and the
# grass east of the building) out: the lane's edge running on into the building's front

# the paddock's asphalt left over by the track's edge past T2 and by the hotlap's pit lane: grass (x, z boxes)
T2_STRAYS = (((-5.5, -2.5), (-11.5, -6.)), ((-12., -7.5), (-13., -9.5)))
# past the hairpin (HAIRPIN_OUT) the paddock's asphalt north of the building there out (north of the
# line of its north wall, BAR_LINE, two points x, z on it, from x HAIR_FROM to HAIR_TO); the track's
# road beyond the kerb (HAIR_ROAD) and its patches (HAIR_PATCHES: each the road it is of) one, and
# its outside smooth and round (closed and opened over HAIR_ROUND metres) in HAIR_BOX (x, z)
BAR_LINE, HAIR_FROM, HAIR_TO = ((23.4, -28.7), (14.9, -30.6)), 13., 45.
HAIR_ROAD, HAIR_PATCHES = '1ROAD_7', (('1ROAD_PATCH_HAIRPIN_OUT_1ROAD_3', '1ROAD_3'), ('1ROAD_PATCH_HAIRPIN_OUT_1ROAD_7', '1ROAD_7'))
HAIR_BOX, HAIR_ROUND = ((13.5, 40.), (-31., -14.)), 2.


def take_old_stand():
    """The old stand out of source/models/velopark.glb. Whether it was there."""
    k = gltf_source.load_kn5(bp.TRACK)
    root = k['root']
    keep = [c for c in root['children'] if c['name'] != OLD_STAND]
    if len(keep) == len(root['children']):
        return False
    root['children'] = keep
    bp.save(k, bp.TRACK)
    return True


def west_grass():
    """Round the pit exit's junction, where the paddock's asphalt was laid but grass is (the junction's
    roads and the ramp cut out of it after), x, z: JUNCTION_GRASS less the shed's floor, the storage
    building and the 1000's lane beside them."""
    import build_outside as bo
    import drape_track as dt
    from shapely.geometry import Polygon, box
    (x0, x1), (z0, z1) = JUNCTION_GRASS
    (xa, za), (xb, zb) = dt.LANE_EDGE
    line = lambda z: xa + (xb - xa) * (z - za) / (zb - za)
    east = Polygon([(line(z0), z0), (line(z1), z1), (-120., z1), (-120., z0)])       # (the lane's side of the walls' line)
    slab = box(bo.SLAB_WEST - 20., z0 - 1., bo.SLAB_WEST + .02, dt.SHED_NORTH + dt.SHED_EDGE + 1.3 + .02)
    return box(x0, z0, x1, z1).difference(east).difference(slab).difference(Polygon(bo.MIDDLE['GARAGE'][0]).buffer(.3))


def background(k):
    """The ground's material (Background, the track's) in model k, added if it has none: its index."""
    import copy
    mi = next((i for i, mt in enumerate(k['materials']) if mt['name'] == 'Background'), None)
    if mi is not None:
        return mi
    track = gltf_source.load_kn5(bp.TRACK)
    bg = next(mt for mt in track['materials'] if mt['name'] == 'Background')
    k['materials'].append(copy.deepcopy(bg))
    have = {t['name'] for t in k['textures']}
    for t in bg['textures']:
        if t['name'] not in have:
            k['textures'].append(next(x for x in track['textures'] if x['name'] == t['name']))
    return len(k['materials']) - 1


def grass_over(k, shape, name='1GRASS_PADDOCK_WEST', grassy=True):
    """The paddock's asphalt under shape grass, cut exactly along its edge (no faces half one, half
    the other): the asphalt there out, grass laid over just that, at the asphalt's height, the ground's
    photo on it. How many faces of grass."""
    if any(n['name'] == name for n, _ in bp.meshes(k)):
        return 0
    from shapely.geometry import box
    x0, z0, x1, z1 = shape.bounds
    area = ((x0 - 1., x1 + 1.), (z0 - 1., z1 + 1.))
    pad = [n for n, m in bp.meshes(k) if n['name'].startswith(bp.ASPHALT) and len(n['faces']) and np.allclose(m, np.eye(4))]
    if not pad:
        return 0
    region = roads_shape(k, area, lambda n: n.startswith(bp.ASPHALT)).intersection(shape).buffer(0)
    top = bp.Surface(np.concatenate([bp.triangles(n, np.eye(4)) for n in pad]))
    for n in pad:
        clip_in(n, shape)
    tris = fill(region, most=2.)
    if not tris:
        return 0
    P = np.concatenate(tris)
    y = top.ys(P)
    y = np.where(np.isnan(y), np.nanmedian(y), y)
    base = pad[0]
    V = np.zeros((len(P), base['verts'].shape[1]))
    V[:, 0], V[:, 1], V[:, 2] = P[:, 0], y, P[:, 1]
    V[:, 4] = 1.
    V[:, 6], V[:, 7] = .5 - P[:, 0] / 1060., .5 - P[:, 1] / 1060.
    if V.shape[1] > 8:
        V[:, 8] = 1.
    F = np.arange(len(P)).reshape(-1, 3)
    a, b, c = V[F[:, 0], :3], V[F[:, 1], :3], V[F[:, 2], :3]
    up = np.cross(b - a, c - a)[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)[np.abs(up) > 1e-8]
    mi = background(k)
    node = dict(base, name=name, children=[], material=mi, verts=V.astype('<f4'), faces=F.astype('<u2'))
    bp.sphere(node)
    k['root']['children'].append(node)
    import re
    csp = os.path.join(bp.ROOT, 'source', 'track', 'extension', 'ext_config.ini')
    text = open(csp, newline='').read()
    m_ = re.search(r'(?m)^GRASS_MESHES *= *(.*)$', text)
    if grassy and m_ and name not in m_.group(1):
        open(csp, 'w', newline='').write(text[:m_.end(1)] + ', ' + name + text[m_.end(1):])
    return len(F)


def grass_back(k, more=()):
    """The paddock's asphalt in GRASS_BACK as grass, the ground's photo on it as on the near
    ground's (u, v from x, z), in a mesh of its own. How many faces."""
    if any(n['name'] == GRASS_MESH for n, _ in bp.meshes(k)):
        return 0
    import copy, re
    import shapely
    from shapely.geometry import Polygon
    zones = [Polygon(z) for z in GRASS_BACK] + list(more)
    taken = []
    for n, m in list(bp.meshes(k)):
        if not n['name'].startswith(bp.ASPHALT) or not len(n['faces']):
            continue
        c = n['verts'][:, :3].astype(float)[n['faces'].astype(int)].mean(1)
        out = np.zeros(len(c), bool)
        for z in zones:
            out |= shapely.contains_xy(z, c[:, 0], c[:, 2])
        if out.any():
            taken.append(bp.split(n, out))
    if not taken:
        return 0
    starts = np.cumsum([0] + [len(v) for v, _ in taken])
    V = np.concatenate([v for v, _ in taken]).copy()
    F = np.concatenate([f + starts[i] for i, (_, f) in enumerate(taken)])
    V[:, 6] = .5 - V[:, 0] / 1060.
    V[:, 7] = .5 - V[:, 2] / 1060.
    track = gltf_source.load_kn5(bp.TRACK)
    bg = next(mt for mt in track['materials'] if mt['name'] == 'Background')
    mi = next((i for i, mt in enumerate(k['materials']) if mt['name'] == 'Background'), None)
    if mi is None:
        k['materials'].append(copy.deepcopy(bg))
        mi = len(k['materials']) - 1
        have = {t['name'] for t in k['textures']}
        for t in bg['textures']:
            if t['name'] not in have:
                k['textures'].append(next(x for x in track['textures'] if x['name'] == t['name']))
    node = dict(kind=2, name=GRASS_MESH, active=1, children=[], material=mi, cast_shadows=1, visible=1,
                transparent=0, layer=0, lod_in=0., lod_out=0., renderable=1, verts=V.astype('<f4'), faces=F.astype('<u2'))
    bp.sphere(node)
    k['root']['children'].append(node)
    # and grown with the Custom Shaders Patch's grass
    csp = os.path.join(bp.ROOT, 'source', 'track', 'extension', 'ext_config.ini')
    text = open(csp, newline='').read()
    m_ = re.search(r'(?m)^GRASS_MESHES *= *(.*)$', text)
    if m_ and GRASS_MESH not in m_.group(1):
        text = text[:m_.end(1)] + ', ' + GRASS_MESH + text[m_.end(1):]
        open(csp, 'w', newline='').write(text)
    return len(F)


def road_uv(T, U):
    """The map from x, z to u, v of a road's faces T (n, 3, 3) with their u, v U (n, 3, 2), fitted."""
    P = T.reshape(-1, 3)[:, [0, 2]]
    A = np.c_[P, np.ones(len(P))]
    return np.linalg.lstsq(A, U.reshape(-1, 2), rcond=None)[0]


def fill(shape, most=1.):
    """The triangles filling a shape exactly, each (3, 2) x, z, none over MOST square metres: points
    added inside it as needed, none on its edges (where it meets other faces, it meets their points)."""
    import triangle
    out = []
    for poly in getattr(shape, 'geoms', [shape]):
        if poly.is_empty or poly.area < 1e-6:
            continue
        pts, segs, holes = [], [], []
        for ring in [poly.exterior] + list(poly.interiors):
            q = np.array(ring.coords)[:-1]
            base = len(pts)
            pts += [tuple(p) for p in q]
            segs += [(base + i, base + (i + 1) % len(q)) for i in range(len(q))]
        for ring in poly.interiors:
            from shapely.geometry import Polygon
            holes.append(tuple(np.array(Polygon(ring).representative_point().coords)[0]))
        data = dict(vertices=np.array(pts), segments=np.array(segs))
        if holes:
            data['holes'] = np.array(holes)
        t = triangle.triangulate(data, 'pqYa%g' % most)
        if 'triangles' in t:
            out += [t['vertices'][f] for f in t['triangles']]
    return out

def lane_shape(track, every=False):
    """The 1000's pit lane from WIDE_FROM - 5 to WIDE_TO + 5 (or all of it), x, z, and its faces there."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    lane = next(n for n, _ in bp.meshes(track) if n['name'] == '1ROAD_24')
    V = lane['verts'].astype(float)
    F = lane['faces'].astype(int)
    T = V[F]
    c = T.mean(1)
    near = ((c[:, 2] > WIDE_FROM - 5.) & (c[:, 2] < WIDE_TO + 5.)) | every
    shape = unary_union([Polygon(t[:, [0, 2]]).buffer(1e-5, join_style=2) for t in T[near] if Polygon(t[:, [0, 2]]).area > 1e-10])
    return lane, shape, F[near]


def roads_shape(k, box_, keep=lambda name: True):
    """The union of k's roads and kerbs (1ROAD, 1KERB) whose faces lie in box_ ((x0, x1), (z0, z1)), x, z."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    (x0, x1), (z0, z1) = box_
    parts = []
    for n, m in bp.meshes(k):
        if not n['name'].startswith(('1ROAD', '1KERB')) or not len(n['faces']) or not keep(n['name']):
            continue
        T = bp.triangles(n, m)
        c = T.mean(1)
        for t in T[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 2] > z0) & (c[:, 2] < z1)]:
            q = Polygon(t[:, [0, 2]])
            if q.area > 1e-10:      # (the thinnest too: a seam's slivers left out notch it)
                parts.append(q.buffer(1e-5, join_style=2))
    return unary_union(parts) if parts else Polygon()


def wide_shape(track):
    """The strip the lane is widened by: LANE_EXTRA past its east edge, from WIDE_FROM to WIDE_TO."""
    import struct
    from shapely.geometry import Polygon, box
    lane, shape, _ = lane_shape(track)
    b = open(os.path.join(bp.ROOT, 'source', 'track', 'vp_1000', 'ai', 'pit_lane.ai'), 'rb').read()
    n_ = struct.unpack('<i', b[4:8])[0]
    ai = np.frombuffer(b[16:16 + n_ * 20], '<f4').reshape(n_, 5)[:, [0, 2]].astype(float)
    ai_run = ai.copy()                                    # (in the order driven)
    ai = ai[(ai[:, 1] > WIDE_FROM - 10.) & (ai[:, 1] < WIDE_TO + 10.) & (ai[:, 0] > -70.)]
    ai = ai[np.argsort(ai[:, 1])]
    east = Polygon(list(map(tuple, ai)) + [(-120., ai[-1, 1]), (-120., ai[0, 1])])
    strip = shape.buffer(LANE_EXTRA, join_style=2).difference(shape).intersection(east).intersection(box(-120., WIDE_FROM, 0., WIDE_TO))
    # its mouth: the corners between it and the track filled, rounded (closed over MOUTH_ROUND)
    (mx0, mx1), (mz0, mz1) = MOUTH_BOX
    others = [gltf_source.load_kn5(os.path.join(bp.MODELS, f)) for f in ('velopark_kerbs.glb', 'velopark_runoffs.glb')
              if os.path.exists(os.path.join(bp.MODELS, f))]
    from shapely.ops import unary_union
    track_shape = unary_union([roads_shape(track, MOUTH_BOX, lambda n: n != '1ROAD_24')]
                              + [roads_shape(k_, MOUTH_BOX) for k_ in others])
    lane_all = lane_shape(track, every=True)[1]
    both = lane_all.union(track_shape)
    mouth = both.buffer(MOUTH_ROUND).buffer(-MOUTH_ROUND).difference(both).intersection(box(mx0, mz0, mx1, mz1))
    # (only the corners at the lane's mouth, not the track's own bends there; and only the one the
    # karts come from along the track: past the mouth, the lane's other side meets the track at the
    # tight angle its curve makes, the pit lane leaving the track on its racing line)
    touch = lane_all.buffer(.05)
    along = ai_run[(ai_run[:, 1] > mz0 + 12.) & (ai_run[:, 0] > mx0 - 20.) & (ai_run[:, 0] < mx1)]
    way = along[-1] - along[0] if len(along) > 1 else np.array([1., 0.])
    way /= np.linalg.norm(way) or 1.
    at = np.array(lane_all.intersection(box(mx0, mz0 + 8., mx1, mz1)).centroid.coords[0])
    mouth = unary_union([g for g in getattr(mouth, 'geoms', [mouth]) if g.area > .01 and g.intersects(touch)
                         and (np.array(g.centroid.coords[0]) - at) @ way < 0])
    return strip.union(mouth).difference(track_shape).buffer(0)


def widen_lane(track):
    """The 1000's pit lane, 1ROAD_24, carried on over wide_shape: faces filling it in its own mesh,
    at the height of its edge beside them, mapped as its nearest face is. How many faces."""
    from scipy.spatial import cKDTree
    lane, shape, Fn = lane_shape(track)
    V = lane['verts'].astype(float)
    on = (V[:, 2] > WIDE_FROM + 5.) & (V[:, 2] < 165.)
    if on.any() and (V[on, 0] < lane_east(V[on, 2]) - LANE_EXTRA / 2).any():
        return 0                                          # (widened already)
    strip = wide_shape(track)
    near = lane_shape(track)[2]
    ref = dict(lane, faces=near.astype('<u2'))
    # each new point as high as the lane's and the track's points nearest it (not a face's plane
    # carried out, which leans)
    from scipy.spatial import cKDTree
    pts = np.concatenate([n['verts'][:, :3].astype(float) for n, _ in bp.meshes(track)
                          if n['name'] in ('1ROAD_24', '1ROAD_17', '1ROAD_18') and len(n['verts'])])
    tree = cKDTree(pts[:, [0, 2]])

    def height(x, z):
        d, i = tree.query((x, z), k=6)
        w = 1. / np.maximum(d, .05) ** 2
        return float((pts[i, 1] * w).sum() / w.sum())
    return grow(lane, fill(strip), height=height, uv_from=ref)


def grow(node, tris, pool=(), height=None, uv_from=None, snap=.002):
    """Faces filling tris ((3, 2) x, z each) added to mesh node: a corner where node has a point
    already is that point (one mesh, no seam); where a mesh of pool has one, a copy of it (the same
    place, height and mapping); else a point of its own at height(x, z), mapped as the nearest face of
    uv_from (node by default) maps x, z, facing up. How many faces."""
    from scipy.spatial import cKDTree
    if not tris:
        return 0
    V = node['verts'].astype(float)
    P = np.concatenate(tris)
    own = cKDTree(V[:, [0, 2]]) if len(V) else None
    others = np.concatenate([n_['verts'].astype(float) for n_ in pool]) if pool else np.zeros((0, V.shape[1]))
    other = cKDTree(others[:, [0, 2]]) if len(others) else None
    ref = uv_from if uv_from is not None else node
    Rv, Rf = ref['verts'].astype(float), ref['faces'].astype(int)
    RT = Rv[Rf]
    rtree = cKDTree(RT[:, :, [0, 2]].mean(1))
    rows, index, made = [], [], {}
    for x, z in P:
        if own is not None:
            d, i = own.query((x, z))
            if d < snap:
                index.append(int(i))
                continue
        key = (round(x, 4), round(z, 4))
        if key in made:
            index.append(made[key])
            continue
        if other is not None:
            d, i = other.query((x, z))
            if d < snap:
                v = others[i].copy()
                made[key] = len(V) + len(rows)
                rows.append(v)
                index.append(made[key])
                continue
        t = RT[rtree.query((x, z))[1]]
        A = np.c_[t[:, 0], t[:, 2], np.ones(3)]
        try:
            M = np.linalg.solve(A, t)
        except np.linalg.LinAlgError:
            M = np.linalg.lstsq(A, t, rcond=None)[0]
        v = np.array([x, z, 1.]) @ M
        v[0], v[2] = x, z
        if height is not None:
            h = float(height(x, z))
            if np.isfinite(h):
                v[1] = h                                  # (else the height of the nearest face's plane)
        v[3:6] = (0., 1., 0.)
        made[key] = len(V) + len(rows)
        rows.append(v)
        index.append(made[key])
    F = np.array(index).reshape(-1, 3)
    allV = np.r_[V, np.array(rows)] if rows else V
    a, b, c = allV[F[:, 0], :3], allV[F[:, 1], :3], allV[F[:, 2], :3]
    up = np.cross(b - a, c - a)[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)[(np.abs(up) > 1e-7) & (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    if not np.isfinite(allV).all():
        sys.exit('%s: points with no place (NaN)' % node['name'])
    node['verts'] = allV.astype(node['verts'].dtype)
    # (past 65536 points, 32-bit indices: the model's loader cuts such a mesh in pieces a kn5 mesh holds)
    node['faces'] = np.r_[node['faces'].astype(int).reshape(-1, 3), F].astype('<u2' if len(allV) < 65536 else '<u4')
    bp.sphere(node)
    return len(F)


def clip_out(n, shape):
    """The faces of mesh n under shape out of it, and those across its edge cut along it: their
    parts outside it as (points, faces) of n's layout, interpolated over the faces they were of."""
    from shapely.geometry import Polygon
    V = n['verts'].astype(float)
    F = n['faces'].astype(int)
    T = V[F][:, :, [0, 2]]
    x0, z0, x1, z1 = shape.bounds
    lo, hi = T.min(1), T.max(1)
    cand = np.nonzero((hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= z0) & (lo[:, 1] <= z1))[0]
    gone = np.zeros(len(F), bool)
    rows = []
    for i in cand:
        tri = Polygon(T[i])
        if tri.area < 1e-9 or not tri.intersects(shape):
            continue
        gone[i] = True
        rest = tri.difference(shape)
        if rest.is_empty or rest.area < 1e-6:
            continue
        a, b, c = T[i]
        M = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        for t in fill(rest):
            w = np.linalg.solve(M, (t - a).T).T
            rows.append(np.c_[1 - w.sum(1), w] @ V[F[i]])
    if not gone.any():
        return None
    n['faces'] = F[~gone].astype(n['faces'].dtype)
    bp.sphere(n)
    if not rows:
        return np.zeros((0, V.shape[1])), np.zeros((0, 3), int)
    P = np.concatenate(rows)
    P[:, 3:6] /= np.maximum(np.linalg.norm(P[:, 3:6], axis=1, keepdims=True), 1e-9)
    Fc = np.arange(len(P)).reshape(-1, 3)
    a, b, c = P[Fc[:, 0], :3], P[Fc[:, 1], :3], P[Fc[:, 2], :3]
    Fc = np.where((np.cross(b - a, c - a) @ np.array([0., 1., 0.]) < 0)[:, None], Fc[:, ::-1], Fc)
    return P, Fc


def clip_in(n, shape):
    """Mesh n cut along shape's edge, the part under it out and the faces across it kept in n as their
    parts outside it (one mesh still, its edge exactly along shape's)."""
    out = clip_out(n, shape)
    if out is None or not len(out[1]):
        return
    V = n['verts']
    # (past 65536 points, 32-bit indices: the model's loader cuts such a mesh in pieces a kn5 mesh holds)
    n['faces'] = np.r_[n['faces'].astype(int).reshape(-1, 3), out[1] + len(V)].astype('<u2' if len(V) + len(out[0]) < 65536 else '<u4')
    n['verts'] = np.r_[V, out[0].astype(V.dtype)]
    bp.sphere(n)


def holes(k, prefix, shape, name):
    """Holes in the meshes of k named from prefix, under shape; the cut faces round them as a mesh
    of their own, name (as the first such mesh is). How many faces cut."""
    if any(n['name'] == name for n, _ in bp.meshes(k)):
        return 0
    made, first = [], None
    for n, m in list(bp.meshes(k)):
        if not n['name'].startswith(prefix) or not len(n['faces']) or not np.allclose(m, np.eye(4)):
            continue
        out = clip_out(n, shape)
        if out is None:
            continue
        first = first or n
        if len(out[1]):
            made.append(out)
    if first is None:
        return 0
    if made:
        starts = np.cumsum([0] + [len(p) for p, _ in made])
        P = np.concatenate([p for p, _ in made])
        F = np.concatenate([f + starts[i] for i, (_, f) in enumerate(made)])
        node = dict(first, name=name, children=[], verts=P.astype(first['verts'].dtype), faces=F.astype('<u2'))
        bp.sphere(node)
        k['root']['children'].append(node)
    return sum(len(f) for _, f in made)


def column_lines():
    """The pit building's columns, as two lines from the roof's south end to its north: the west's
    (the 1500's side) and the east's (the 1000's), each (x, z) points south to north."""
    import build_outside as bo
    c = np.array(bo.PIT_BUILDING[0], float)
    ez = np.array(bo.PIT_BUILDING[1], float)
    ez /= np.linalg.norm(ez)
    rl = bo.PIT_ROOF[1]
    cols = np.array(bo.PIT_COLUMNS, float)
    lines = []
    for side in (cols[cols[:, 0] > c[0]], cols[cols[:, 0] <= c[0]]):
        side = side[np.argsort(side[:, 1])]
        d = side[-1] - side[0]
        d /= np.linalg.norm(d)
        ends = [(c + ez * e)[1] for e in (-rl / 2, rl / 2)]
        a = side[0] + d * (ends[0] - side[0][1]) / d[1]
        b = side[-1] + d * (ends[1] - side[-1][1]) / d[1]
        lines.append(np.r_[[a], side, [b]])
    return lines


def pits_shape(track, pad):
    """The pits under the pit building's roof, out to its columns' lines on both sides (the tyre
    walls along them), where no asphalt is yet, x, z."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    west, east = column_lines()
    area_poly = Polygon(list(map(tuple, west)) + list(map(tuple, east[::-1])))
    x0, z0, x1, z1 = area_poly.bounds
    area = ((x0 - 2., x1 + 2.), (z0 - 2., z1 + 2.))
    have = unary_union([roads_shape(track, area), roads_shape(pad, area)])
    gap = area_poly.difference(have)
    parts = [g for g in getattr(gap, 'geoms', [gap]) if g.area > .05]
    return unary_union(parts).buffer(0) if parts else Polygon()


def pits_fill(track, pad):
    """The pits' asphalt out to the columns (pits_shape), in the paddock's asphalt, mapped as it is,
    at the height of the asphalt beside it on their seams and of the ground between. How many faces."""
    name = bp.ASPHALT + '_PITS'
    if any(n['name'] == name for n, _ in bp.meshes(pad)):
        return 0
    import build_outside as bo
    from shapely.geometry import Polygon
    shape = pits_shape(track, pad)
    tris = fill(shape)
    if not tris:
        return 0
    # the paddock's asphalt under the roof out of its mesh into the pits' own
    c = np.array(bo.PIT_BUILDING[0], float)
    ez = np.array(bo.PIT_BUILDING[1], float)
    ez /= np.linalg.norm(ez)
    ex = np.array([ez[1], -ez[0]])
    rw, rl = bo.PIT_ROOF
    roof = Polygon([tuple(c + ex * a + ez * b) for a, b in ((-rw / 2 - 2., -rl / 2 - 2.), (rw / 2 + 2., -rl / 2 - 2.),
                                                             (rw / 2 + 2., rl / 2 + 2.), (-rw / 2 - 2., rl / 2 + 2.))])
    import shapely
    base = next(n for n, _ in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    F0 = base['faces'].astype(int)
    cen = base['verts'][:, :3].astype(float)[F0].mean(1)
    under = shapely.contains_xy(roof, cen[:, 0], cen[:, 2])
    v, f = bp.split(base, under)
    node = dict(base, name=name, children=[], verts=v, faces=f.astype('<u2'))
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                        if n['name'].startswith(('1GRASS_0', '1GRASS_PATHS')) and len(n['faces'])]))
    made = grow(node, tris, height=lambda x, z: ground.ys(np.array([[x, z]]))[0] + .02)
    # all of it mapped as one, as the most of the paddock's asphalt is (some of it there was stretched)
    V = node['verts'].astype(float)
    F = node['faces'].astype(int)
    J, ok = [], []
    for f in F:
        A = np.c_[V[f[1], [0, 2]] - V[f[0], [0, 2]], V[f[2], [0, 2]] - V[f[0], [0, 2]]]
        if abs(np.linalg.det(A)) < 1e-4:
            continue
        j = np.c_[V[f[1], 6:8] - V[f[0], 6:8], V[f[2], 6:8] - V[f[0], 6:8]] @ np.linalg.inv(A)
        sv = np.linalg.svd(j, compute_uv=False)
        if sv[1] > .02:
            J.append(j)
            ok.append(f[0])
    Jm = np.median(np.array(J), 0)
    b0 = np.median(V[ok, 6:8] - V[ok][:, [0, 2]] @ Jm.T, 0)
    V[:, 6:8] = V[:, [0, 2]] @ Jm.T + b0
    node['verts'] = V.astype(node['verts'].dtype)
    pad['root']['children'].append(node)
    return made


def exit_shape(track):
    """Where the pit exit is carried on over its curve's inside, to under its tyre wall, x, z."""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    exit_ = roads_shape(track, EXIT_AREA, lambda n: n == '1ROAD_25')
    others = roads_shape(track, EXIT_AREA, lambda n: n != '1ROAD_25')
    k = gltf_source.load_kn5(bp.TYRES)
    (x0, x1), (z0, z1) = EXIT_AREA
    rings = []
    for n, m in bp.meshes(k):
        if not n['name'].startswith('TYRES_') or not len(n['faces']):
            continue
        V = n['verts'][:, :3].astype(float)
        if not ((V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 2] > z0) & (V[:, 2] < z1)).any():
            continue
        drawn = [x for x, _ in bp.meshes(k) if x['name'] in (n['name'], n['name'] + 'F') and len(x['faces'])]
        _, mid = tyre_stacks(drawn)
        for c in mid:
            if x0 < c[0] < x1 and z0 < c[1] < z1:
                q = Point(c[0], c[1])
                if 0. < exit_.distance(q) <= EXIT_TYRES_NEAR and not others.contains(q):
                    rings.append(q.buffer(EXIT_TYRE_R))
    if not rings:
        from shapely.geometry import Polygon
        return Polygon()
    tyres = unary_union(rings)
    closed = unary_union([exit_, tyres]).buffer(EXIT_CLOSE).buffer(-EXIT_CLOSE)
    # (out from the tyres as far as they stand from it, so it meets the exit's edge all along)
    wide = closed.difference(exit_).intersection(tyres.buffer(EXIT_TYRES_NEAR + EXIT_CLOSE)).difference(others).buffer(0)
    # its end curving into the track's edge, the tyres past it left on the grass
    from shapely.geometry import box
    (mx0, mx1), (mz0, mz1) = EXIT_MERGE_AT
    have = unary_union([exit_, wide, others])
    fillet = have.buffer(EXIT_MERGE).buffer(-EXIT_MERGE).difference(have).intersection(box(mx0, mz0, mx1, mz1))
    fillet = [g for g in getattr(fillet, 'geoms', [fillet])
              if g.area > .01 and g.distance(wide) < .05 and g.distance(others) < .05]
    if fillet:
        wide = unary_union([wide] + fillet).difference(tyres.difference(wide.buffer(.01))).buffer(0)
        h = EXIT_MERGE_MIN / 2
        thin = wide.difference(wide.buffer(-h, join_style=2).buffer(h * 1.02, join_style=2)).intersection(box(mx0, mz0, mx1, mz1))
        wide = wide.difference(thin).buffer(0)
        wide = unary_union([g for g in getattr(wide, 'geoms', [wide]) if g.area > .01]) if not wide.is_empty else wide
    return wide


def inner_curve(track, pad, k=40):
    """The hotlap's pit lane's inner (north) edge from its road's at x HOTLAP_INNER onto the building's
    front at its east end, smooth: (k, 2) points, x, z."""
    from shapely.geometry import LineString
    near = ((HOTLAP_FLARE[0] - 10., HOTLAP_FLARE[1] + 10.), (-70., -20.))
    own = roads_shape(track, near, lambda n: n == HOTLAP_ROADS[0])

    def north(x):
        cut = LineString([(x, -80.), (x, 0.)]).intersection(own)
        return max(q[1] for g in getattr(cut, 'geoms', [cut]) for q in g.coords)
    _, p0, d, nrm, xa, xb = garages(pad, track)
    xi = HOTLAP_INNER
    a = np.array([xi, north(xi)])
    ta = np.array([1., north(xi + .5) - north(xi - .5)])
    ta /= np.linalg.norm(ta)
    b = p0 + d * ((xa - p0[0]) / d[0])
    L = np.linalg.norm(b - a)
    u = np.linspace(0., 1., k)[:, None]
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * L * ta
            + (-2 * u ** 3 + 3 * u ** 2) * b + (u ** 3 - u ** 2) * L * d)


def beyond_inner(track, pad):
    """North of the lane's inner edge's curve (inner_curve), between its ends, x, z: where no road is."""
    from shapely.geometry import Polygon
    _, p0, d, nrm, xa, xb = garages(pad, track)
    c = inner_curve(track, pad)
    return Polygon(list(map(tuple, c)) + [tuple(c[-1] - nrm * 8.), tuple(c[0] - nrm * 8.)]).buffer(0)


def round_out(track, pad, area, road_name, r):
    """The roads' outline in area (x, z box) rounded out over r metres (its corners filled with
    curves, its notches and the grass slivers in it closed): road road_name grown over what that adds,
    as high as the roads round it. (Faces grown, what was added, x, z.)"""
    from shapely.geometry import box
    from shapely.ops import unary_union
    from scipy.spatial import cKDTree
    (x0, x1), (z0, z1) = area
    big = ((x0 - 2 * r, x1 + 2 * r), (z0 - 2 * r, z1 + 2 * r))
    kerbs = gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_kerbs.glb'))
    runoffs = gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_runoffs.glb'))
    U = unary_union([roads_shape(track, big), roads_shape(pad, big), roads_shape(runoffs, big), roads_shape(kerbs, big)]).buffer(0)
    add = U.buffer(r).buffer(-r).difference(U).intersection(box(x0, z0, x1, z1)).buffer(-.01).buffer(.01)
    if add.is_empty or add.area < .02:
        return 0, add
    road = next(n for n, _ in bp.meshes(track) if n['name'] == road_name)
    V = np.concatenate([n['verts'][:, :3].astype(float) for n, _ in bp.meshes(track) if n['name'].startswith('1ROAD') and len(n['faces'])])
    a0, b0, a1, b1 = add.buffer(3.).bounds
    V = V[(V[:, 0] > a0) & (V[:, 0] < a1) & (V[:, 2] > b0) & (V[:, 2] < b1)]
    tree = cKDTree(V[:, [0, 2]])

    def height(x, z):
        dd, ii = tree.query((x, z), k=min(8, len(V)))
        w = 1. / np.maximum(dd, .05) ** 2
        return float((V[ii, 1] * w).sum() / w.sum())
    pool = [n for n, _ in bp.meshes(track) if n['name'].startswith('1ROAD') and n is not road and len(n['faces'])]
    return grow(road, fill(add, most=1.), pool=pool, height=height), add


def hairpin_cuts():
    """What of the paddock's asphalt goes, x, z: by T2 and the hotlap's pit lane (T2_STRAYS), and north
    of the building past the hairpin (BAR_LINE)."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    (ax, az), (bx, bz) = BAR_LINE
    k_ = (bz - az) / (bx - ax)
    at = lambda x: az + k_ * (x - ax)
    bar = Polygon([(HAIR_FROM, at(HAIR_FROM)), (HAIR_TO, at(HAIR_TO)), (HAIR_TO, -14.), (HAIR_FROM, -14.)])
    return unary_union([bar] + [box(x0, z0, x1, z1) for (x0, x1), (z0, z1) in T2_STRAYS])


def hairpin_out(track, pad, runoffs):
    """Past the hairpin: the track's patches there grown into the roads they are of, and the outside of
    its roads smooth and round in HAIR_BOX (the notches filled, as the road beyond the kerb, and what
    sticks out out). (Faces grown, faces cut, and what was grown, x, z.)"""
    from shapely.geometry import box
    from shapely.ops import unary_union
    from scipy.spatial import cKDTree
    grown = 0
    for pname, rname in HAIR_PATCHES:
        pn = next((n for n, m in bp.meshes(runoffs) if n['name'] == pname and len(n['faces'])), None)
        road = next((n for n, m in bp.meshes(track) if n['name'] == rname), None)
        if pn is None or road is None:
            continue
        T = bp.triangles(pn, np.eye(4))
        S = bp.Surface(T)
        e1, e2 = T[:, 1, [0, 2]] - T[:, 0, [0, 2]], T[:, 2, [0, 2]] - T[:, 0, [0, 2]]
        tris = list(T[np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-6][:, :, [0, 2]])
        pool = [n for n, _ in bp.meshes(track) if n['name'].startswith('1ROAD') and n is not road and len(n['faces'])]
        grown += grow(road, tris, pool=pool, height=lambda x, z, S=S: S.y(x, z))
        pn['faces'] = np.zeros((0, 3), '<u2')
    (x0, x1), (z0, z1) = HAIR_BOX
    big = ((x0 - 10., x1 + 10.), (z0 - 10., z1 + 10.))
    kerbs = gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_kerbs.glb'))
    U = unary_union([roads_shape(track, big), roads_shape(pad, big), roads_shape(runoffs, big), roads_shape(kerbs, big)]).buffer(0)
    r = HAIR_ROUND
    S_ = U.buffer(r).buffer(-r).buffer(-r).buffer(r)
    inner = box(x0, z0, x1, z1)
    add = S_.difference(U).intersection(inner).buffer(0)
    cut = U.difference(S_).intersection(inner).buffer(0)
    for n, _ in list(bp.meshes(track)):
        if n['name'] == HAIR_ROAD and len(n['faces']):
            clip_in(n, cut)
    for n, _ in list(bp.meshes(pad)):
        if n['name'].startswith(bp.ASPHALT) and len(n['faces']):
            clip_in(n, cut)
    road = next(n for n, _ in bp.meshes(track) if n['name'] == HAIR_ROAD)
    if not add.is_empty and add.area > .02:
        V = np.concatenate([n['verts'][:, :3].astype(float) for n, _ in bp.meshes(track) if n['name'].startswith('1ROAD') and len(n['faces'])])
        x0_, z0_, x1_, z1_ = add.buffer(3.).bounds
        V = V[(V[:, 0] > x0_) & (V[:, 0] < x1_) & (V[:, 2] > z0_) & (V[:, 2] < z1_)]
        tree = cKDTree(V[:, [0, 2]])

        def height(x, z):
            dd, ii = tree.query((x, z), k=min(8, len(V)))
            w = 1. / np.maximum(dd, .05) ** 2
            return float((V[ii, 1] * w).sum() / w.sum())
        pool = [n for n, _ in bp.meshes(track) if n['name'].startswith('1ROAD') and n is not road and len(n['faces'])]
        grown += grow(road, fill(add, most=1.), pool=pool, height=height)
    return grown, cut.area, add


def hotlap_junction_shape(track, pad=None):
    """Where the hotlap's pit lane's road meets the road along the building, its outer side widened
    from the one into the other, a smooth curve (see HOTLAP_FLARE), x, z."""
    from shapely.geometry import LineString, Polygon, box
    near = ((HOTLAP_FLARE[0] - 10., HOTLAP_FLARE[1] + 10.), (-70., -20.))
    own, along = [roads_shape(track, near, lambda n, w=w: n == w) for w in HOTLAP_ROADS]
    have = roads_shape(track, near)

    def south(shape, x):
        cut = LineString([(x, -80.), (x, 0.)]).intersection(shape)
        return min(q[1] for g in getattr(cut, 'geoms', [cut]) for q in g.coords)
    xa, xb = HOTLAP_FLARE
    a = np.array([xa, south(own, xa)])
    ta = np.array([1., south(own, xa + .5) - south(own, xa - .5)])
    if pad is not None:
        # (onto the straight outer edge of the road along the building, where it begins, along it)
        _, p0_, d_, nrm_, _, _ = garages(pad, track)
        b = p0_ + d_ * ((HOTLAP_START - p0_[0]) / d_[0]) + nrm_ * hotlap_row(track, pad)[0]
        tb = d_.copy()
    else:
        both = own.union(along)
        b = np.array([xb, south(both, xb)])
        tb = np.array([1., south(both, xb + .5) - south(both, xb - .5)])
    xb = float(b[0])
    ta, tb = ta / np.linalg.norm(ta), tb / np.linalg.norm(tb)
    L = np.linalg.norm(b - a)
    u = np.linspace(0., 1., 40)[:, None]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    curve = h00 * a + h10 * L * ta + h01 * b + h11 * L * tb
    # (the curve and, behind it, into the roads)
    poly = Polygon(list(map(tuple, curve)) + [(xb, b[1] + 4.), (xa, a[1] + 4.)]).buffer(0)
    fill_ = poly.difference(have)
    edge = LineString(curve).buffer(.05)
    parts = [g for g in getattr(fill_, 'geoms', [fill_]) if g.area > .05 and g.intersects(edge)]
    # its inner side: from its road's north edge onto the building's front at its start
    if pad is not None:
        _, p0, d, nrm, xa, xb = garages(pad, track)

        def north(shape, x):
            cut = LineString([(x, -80.), (x, 0.)]).intersection(shape)
            return max(q[1] for g in getattr(cut, 'geoms', [cut]) for q in g.coords)
        inner = inner_curve(track, pad)
        a, b, xi = inner[0], inner[-1], HOTLAP_INNER
        poly = Polygon(list(map(tuple, inner)) + [tuple(b + nrm * 4.), (xi, a[1] - 4.)]).buffer(0)
        fill_ = poly.difference(have)
        edge = LineString(inner).buffer(.05)
        parts += [g for g in getattr(fill_, 'geoms', [fill_]) if g.area > .05 and g.intersects(edge)]
    from shapely.ops import unary_union
    return unary_union(parts).buffer(0) if parts else box(0, 0, 0, 0)


def widen_hotlap_junction(track, pad=None):
    """The road along the building, HOTLAP_BOX_ROAD, grown over hotlap_junction_shape: faces in its own
    mesh, its points shared on the seam, as high as the roads' points nearest them, mapped as its
    nearest face. How many faces."""
    from scipy.spatial import cKDTree
    shape = hotlap_junction_shape(track, pad)
    if shape.is_empty or shape.area < .05:
        return 0
    road = next(n for n, _ in bp.meshes(track) if n['name'] == HOTLAP_BOX_ROAD)
    pool = [n for n, _ in bp.meshes(track) if n['name'] in HOTLAP_ROADS and n is not road]
    V = np.concatenate([n_['verts'][:, :3].astype(float) for n_ in [road] + pool])
    tree = cKDTree(V[:, [0, 2]])

    def height(x, z):
        d, i = tree.query((x, z), k=6)
        w = 1. / np.maximum(d, .05) ** 2
        return float((V[i, 1] * w).sum() / w.sum())
    return grow(road, fill(shape), pool=pool, height=height)


def hotlap_straight(track, pad):
    """The road along the building (HOTLAP_BOX_ROAD) cut along the building's front, so the pit lane's
    side there is straight: its part behind it (under the roof at the building's west end) out of it
    into the paddock's asphalt, a floor under the roof. How many faces moved."""
    import copy
    from shapely.geometry import Polygon, box
    foot, p0, d, nrm, xa, xb = garages(pad, track)
    sa, sb = (HOTLAP_START - p0[0]) / d[0], (xb - p0[0]) / d[0]
    behind = Polygon([tuple(p0 + d * sa), tuple(p0 + d * sb), tuple(p0 + d * sb - nrm * 20.), tuple(p0 + d * sa - nrm * 20.)])
    road = next(n for n, _ in bp.meshes(track) if n['name'] == HOTLAP_BOX_ROAD)
    c = road['verts'][:, [0, 2]].astype(float)[road['faces'].astype(int)].mean(1)
    if not ((c - p0) @ nrm < -.02).any():
        return 0
    part = copy.deepcopy(road)
    clip_in(part, box(-1e4, -1e4, 1e4, 1e4).difference(behind))      # (what is behind the front)
    clip_in(road, behind)                                              # (and the road without it)
    if not len(part['faces']):
        return 0
    base = next(n for n, _ in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    used = np.unique(part['faces'].astype(int))
    remap = np.full(len(part['verts']), -1)
    remap[used] = np.arange(len(used))
    V = part['verts'][used].astype(float)
    k = base['verts'].shape[1]
    W = np.zeros((len(V), k))
    W[:, :min(k, V.shape[1])] = V[:, :min(k, V.shape[1])]
    node = dict(base, name=bp.ASPHALT + '_HOTLAP_WEST', children=[], verts=W.astype(base['verts'].dtype),
                faces=remap[part['faces'].astype(int)].astype('<u2'))
    bp.sphere(node)
    pad['root']['children'].append(node)
    return len(node['faces'])


def asphalt_gaps(track, pad):
    """The gaps left between the roads and the car park's asphalt by the hotlap's pit lane (in
    UNDER_HOTLAP, x, z): small spaces the asphalt closes round, the ground showing. x, z."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    road = unary_union([roads_shape(track, UNDER_HOTLAP), roads_shape(pad, UNDER_HOTLAP)])
    gaps = [Polygon(h) for g in getattr(road, 'geoms', [road]) for h in g.interiors if .02 < Polygon(h).area < 10.]
    return unary_union(gaps).buffer(0) if gaps else Polygon()


def fill_gaps(track, pad, gaps):
    """The paddock's asphalt grown over gaps, sharing the roads' points round them, as high as their
    points nearest. How many faces."""
    from scipy.spatial import cKDTree
    if gaps.is_empty:
        return 0
    base = next(n for n, _ in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    pool = [n for k_ in (track, pad) for n, m in bp.meshes(k_) if n['name'].startswith('1ROAD') and len(n['faces'])
            and n is not base and np.allclose(m, np.eye(4))]
    x0, z0, x1, z1 = gaps.buffer(3.).bounds
    V = np.concatenate([n['verts'][:, :3].astype(float) for n in pool + [base]])
    V = V[(V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 2] > z0) & (V[:, 2] < z1)]
    tree = cKDTree(V[:, [0, 2]])

    def height(x, z):
        d, i = tree.query((x, z), k=6)
        w = 1. / np.maximum(d, .05) ** 2
        return float((V[i, 1] * w).sum() / w.sum())
    return grow(base, fill(gaps), pool=pool, height=height)


def widen_exit(track):
    """The pit exit, 1ROAD_25, carried on over exit_shape: faces in its own mesh, its points shared on
    the seam, as high as its points nearest them, mapped as its nearest face. How many faces."""
    from scipy.spatial import cKDTree
    exit_ = next(n for n, _ in bp.meshes(track) if n['name'] == '1ROAD_25')
    shape = exit_shape(track)
    if shape.is_empty:
        return 0
    V = exit_['verts'].astype(float)
    tree = cKDTree(V[:, [0, 2]])

    def height(x, z):
        d, i = tree.query((x, z), k=6)
        w = 1. / np.maximum(d, .05) ** 2
        return float((V[i, 1] * w).sum() / w.sum())
    return grow(exit_, fill(shape), height=height)


def junction_shapes(track, pad):
    """The pit exit's junction north of it: (the asphalt to lay, x, z; path A's part to cut out)."""
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    area = ((-50., 2.), (118., 172.))
    pit = roads_shape(track, area, lambda n: n not in ('1ROAD_24',))      # (the pit exit, the track)
    from shapely.geometry import Polygon as _P
    tops = [bp.top_of(n) for n, _ in bp.meshes(pad) if n['name'].startswith('1ROAD_PATH_A') and len(n['faces'])]
    path = unary_union([_P(t[:, [0, 2]]).buffer(1e-5) for T in tops for t in T if _P(t[:, [0, 2]]).area > 1e-6])
    cut = box(-50., 118., 2., JUNCTION_CUT)
    kept = path.difference(cut)                                          # (its top, its sides left out)
    path_on = LineString(PATH_ON).buffer(PATH_WIDTH / 2, cap_style=2, join_style=1)
    road = LineString(STORE_ROAD).buffer(STORE_WIDTH / 2, cap_style=2, join_style=1)
    (rx0, rx1), (rz0, rz1) = RAMP
    ramp = box(rx0, rz0, rx1, rz1)
    own = path_on.intersection(cut).union(road).union(ramp)
    # (rounded where its own roads meet, not against path A: it carries straight on from it)
    lay = own.buffer(JUNCTION_ROUND).buffer(-JUNCTION_ROUND).intersection(cut)
    # (the corners rounded where the roads meet, not along path A's own sides)
    lay = lay.intersection(own.buffer(JUNCTION_ROUND * 2))
    lay = lay.difference(pit).difference(kept).buffer(0)
    # (one piece: not the ramp's end over on the pit exit's inside, nor slivers along path A)
    parts = getattr(lay, 'geoms', [lay])
    lay = max(parts, key=lambda g: g.area) if parts else lay
    return lay, cut.union(lay).buffer(0)                               # (path A out under it too)


def store_road_shape(track):
    """The junction's asphalt, x, z."""
    return junction_shapes(track, gltf_source.load_kn5(bp.PADDOCK))[0]


def store_road(track):
    """The pit exit's junction north of it laid as one asphalt, the pit exit's (1ROAD_25's material,
    mapped as it is there), at the height of the roads it meets on their seams (path A's as it ran
    before, the pit exit's, the track's) and of the ground between. How many faces."""
    if any(n['name'] == STORE_MESH for n, _ in bp.meshes(track)):
        return 0
    pad = gltf_source.load_kn5(bp.PADDOCK)
    lay, _ = junction_shapes(track, pad)
    tris = fill(lay)
    if not tris:
        return 0
    pit = next(n for n, _ in bp.meshes(track) if n['name'] == '1ROAD_25')
    Fp = pit['faces'].astype(int)
    c = pit['verts'][:, :3].astype(float)[Fp].mean(1)
    ref = dict(pit, faces=Fp[(c[:, 2] > 122.) & (c[:, 2] < 140.) & (c[:, 0] > -30.) & (c[:, 0] < -6.)].astype('<u2'))
    seams = [n for k_ in (track, pad) for n, _ in bp.meshes(k_)
             if len(n['faces']) and (n['name'] in ('1ROAD_25', '1ROAD_3', '1ROAD_5') or n['name'].startswith('1ROAD_PATH_A'))]
    road_top = bp.Surface(np.concatenate([bp.triangles(n, np.eye(4)) for n in seams]))
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                        if n['name'].startswith(('1GRASS_0', '1GRASS_PATHS')) and len(n['faces'])]
                                       + [bp.triangles(n, m) for n, m in bp.meshes(pad)
                                          if n['name'].startswith(bp.ASPHALT) and len(n['faces'])]))

    def height(x, z):
        y = road_top.ys(np.array([[x, z]]))[0]
        return y if not np.isnan(y) else ground.ys(np.array([[x, z]]))[0] + .02
    node = dict(pit, name=STORE_MESH, children=[], verts=np.zeros((0, pit['verts'].shape[1]), pit['verts'].dtype),
                faces=np.zeros((0, 3), '<u2'))
    made = grow(node, fill(lay), pool=seams, height=height, uv_from=ref)
    track['root']['children'].append(node)
    return made


def building_back(k, track):
    """The building along the hotlap's pit lane, its front (columns, doors, walls, roof's edge) pulled
    back BUILDING_BACK to where the photo shows it, its back where it is, every piece of it between
    squeezed so; its walls to hit with it. How many points moved."""
    from shapely.geometry import Point
    from grade_paths import pieces as parts
    foot, p0, d, nrm, xa, xb = garages(k, track)
    names = [n for n, m in bp.meshes(k) if (n['name'].startswith('BLD_') or n['name'].startswith('1WALL_BLD'))
             and len(n['faces']) and np.allclose(m, np.eye(4))]
    # (its front and back across it: from its columns, out of it; its back the furthest in)
    sel = []
    for n in names:
        V = n['verts'][:, [0, 2]].astype(float)
        lab = parts(n['faces'].astype(int), len(V))
        for i in np.unique(lab[np.unique(n['faces'].astype(int))]):
            m_ = lab == i
            o = (V[m_] - p0) @ nrm
            a_ = (V[m_] - p0) @ d
            if o.min() > -16. and o.max() < .5 and a_.min() > (xa - p0[0]) / d[0] - 2. and a_.max() < (xb - p0[0]) / d[0] + 2.:
                sel.append((n, m_))
    if not sel:
        return 0
    back = min(((n['verts'][m_][:, [0, 2]].astype(float) - p0) @ nrm).min() for n, m_ in sel)
    front = max(((n['verts'][m_][:, [0, 2]].astype(float) - p0) @ nrm).max() for n, m_ in sel)
    if front < -BUILDING_BACK + .5:
        return 0                                                        # (pulled back already)
    scale = (front - BUILDING_BACK - back) / (front - back)
    moved = 0
    for n, m_ in sel:
        V = n['verts'].astype(float)
        P = V[m_][:, [0, 2]]
        o = (P - p0) @ nrm
        P = P + np.outer((back + (o - back) * scale) - o, nrm)
        V[np.nonzero(m_)[0], 0], V[np.nonzero(m_)[0], 2] = P[:, 0], P[:, 1]
        n['verts'] = V.astype(n['verts'].dtype)
        moved += int(m_.sum())
    for n in {id(n): n for n, _ in sel}.values():
        bp.sphere(n)
    return moved


def building_roof_out(k, track):
    """The flat roof of the building along the hotlap's pit lane out (its sheets, its solar panels, its
    ceiling and the fascias round it, all over HIGH_UP over the columns' feet): tools/build_outside.py
    roofs it as the middle buildings are. How many faces out."""
    from grade_paths import pieces as parts
    foot, p0, d, nrm, xa, xb = garages(k, track)
    sb = (xb - p0[0]) / d[0]
    col = next(n for n, _ in bp.meshes(k) if n['name'] == HOTLAP_COLUMNS)
    feet = float(np.median(col['verts'][:, 1]))                       # (about half way up them)
    out = 0
    for n, m in list(bp.meshes(k)):
        if n['name'] not in ('BLD_WALLS', 'BLD_ROOFS', 'BLD_SOLAR') or not len(n['faces']) or not np.allclose(m, np.eye(4)):
            continue
        V = n['verts'][:, :3].astype(float)
        F = n['faces'].astype(int)
        lab = parts(F, len(V))
        gone = np.zeros(len(V), bool)
        for t in np.unique(lab[np.unique(F)]):
            sel = lab == t
            P = V[sel]
            a, o = (P[:, [0, 2]] - p0) @ d, (P[:, [0, 2]] - p0) @ nrm
            if a.min() > -8. and a.max() < sb + 3. and o.min() > -13. and o.max() < 1. and P[:, 1].min() > feet + 1.:
                gone |= sel
        if gone.any():
            keep = ~gone[F].any(1)
            out += int((~keep).sum())
            n['faces'] = F[keep].astype(n['faces'].dtype)
            bp.sphere(n)
    # the walls left inside, and its columns, up to the new roof's underside (no more, no less): arched from its front to its back over
    # its depth, from its columns' tops at its eaves (as tools/build_outside.py roofs it)
    import build_outside as bo
    Fp = np.array(foot.exterior.coords)
    D = float(-((Fp - p0) @ nrm).min())
    Vc = col['verts'][:, :3].astype(float)
    lab = parts(col['faces'].astype(int), len(Vc))
    tops = [Vc[lab == t][:, 1].max() for t in np.unique(lab) if abs((Vc[lab == t][:, [0, 2]].mean(0) - p0) @ nrm) < 1.]
    eave = float(np.median(tops))
    r = bo.ROOF_RISE * D
    R = (D * D / 4 + r * r) / (2 * r)
    under = lambda w: eave + r - R + np.sqrt(np.maximum(R * R - (np.clip(w, 0., D) - D / 2) ** 2, 0.))
    for n, m in list(bp.meshes(k)):
        if n['name'] not in ('BLD_WALLS', HOTLAP_COLUMNS) or not len(n['faces']):
            continue
        V = n['verts'].astype(float)
        a, o = (V[:, [0, 2]] - p0) @ d, (V[:, [0, 2]] - p0) @ nrm
        up = (a > -8.) & (a < sb + 3.) & (o > -13.) & (o < .5) & (V[:, 1] > feet + 1.)
        if up.any():
            V[up, 1] = under(-o[up])
            n['verts'] = V.astype(n['verts'].dtype)
            bp.sphere(n)
    return out


def garages(pad, track):
    """The building along the hotlap's pit lane: its footprint (x, z: from its front, the line of its
    columns along the road in front of it, back to its walls' furthest, between its walls' ends) and
    that front, as a point on it, its direction (west to east), its normal out of it, and its ends'
    x along it. (From its columns and its walls as they stand, whatever of its roof is left.)"""
    from shapely.geometry import Polygon, Point
    from grade_paths import pieces as parts
    (x0, x1), (z0, z1) = HOTLAP_COLUMNS_AREA
    n = next(n for n, _ in bp.meshes(pad) if n['name'] == HOTLAP_COLUMNS)
    V = n['verts'][:, [0, 2]].astype(float)
    lab = parts(n['faces'].astype(int), len(V))
    C = np.array([V[lab == i].mean(0) for i in np.unique(lab[np.unique(n['faces'].astype(int))])])
    C = C[(C[:, 0] > x0) & (C[:, 0] < x1) & (C[:, 1] > z0) & (C[:, 1] < z1)]
    every = C.copy()
    # (the columns along the road: the row of them nearest its outer edge)
    road = roads_shape(track, HOTLAP_COLUMNS_AREA, lambda n_: n_ == HOTLAP_BOX_ROAD)
    edge = road.exterior if road.geom_type == 'Polygon' else max(road.geoms, key=lambda g: g.area).exterior
    dist = np.array([edge.distance(Point(*q)) for q in C])
    inroad = np.array([road.buffer(.6).contains(Point(*q)) for q in C])
    C = C[inroad & (dist < np.median(dist[inroad]) + 1.5) & (dist > np.median(dist[inroad]) - 1.5)]
    o = np.polyfit(C[:, 0], C[:, 1], 1)
    for _ in range(3):                                                  # (the odd post off the row left out)
        keep = np.abs(C[:, 1] - np.polyval(o, C[:, 0])) < .5
        o = np.polyfit(C[keep, 0], C[keep, 1], 1)
    C = C[np.abs(C[:, 1] - np.polyval(o, C[:, 0])) < .5]
    d = np.array([1., o[0]]) / np.hypot(1., o[0])
    p0 = np.array([C[:, 0].min(), np.polyval(o, C[:, 0].min())])
    nrm = np.array([-d[1], d[0]])
    if np.median((every - p0) @ nrm) > 0:                               # (out of it: its other columns behind)
        nrm = -nrm
    # its walls' ends along it, and their furthest back
    W = np.concatenate([n_['verts'][:, [0, 2]].astype(float) for n_, _ in bp.meshes(pad)
                        if n_['name'] in ('BLD_WALLS', HOTLAP_COLUMNS) and len(n_['verts'])])
    a, b = (W - p0) @ d, (W - p0) @ nrm
    near = (a > -8.) & (a < 100.) & (b > -17.) & (b < 1.)
    a0, a1, back = float(a[near].min()), float(a[near].max()), float(b[near].min())
    corners = [p0 + d * a0, p0 + d * a1, p0 + d * a1 + nrm * back, p0 + d * a0 + nrm * back]
    foot = Polygon([tuple(q) for q in corners])
    return foot, p0, d, nrm, float((p0 + d * a0)[0]), float((p0 + d * a1)[0])


def hotlap_row(track, pad):
    """The pit lane in front of the building along the hotlap's pit lane, out from its front: (its outer
    edge, HOTLAP_EDGE, the pit boxes' row's far side), the row half of it, so its three lines (along
    the front, the row's far side, the outer edge) stand evenly apart."""
    return HOTLAP_EDGE, HOTLAP_EDGE / 2


def apron_shape(track, pad):
    """The pit apron by the hotlap's pit lane (see APRON_SOUTH), x, z: from the building's front down to
    its south edge, the hotlap's road curving into it on its outside."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    _, p0, d, nrm, xa, xb = garages(pad, track)
    front = lambda x: p0 + d * ((x - p0[0]) / d[0])
    P = [tuple(front(APRON_WEST)), tuple(front(APRON_NE[0])), APRON_SOUTH[-1], APRON_SOUTH[0]]
    apron = Polygon(P).buffer(0)
    road = roads_shape(track, ((-20., 40.), (-80., -20.)), lambda n: n in HOTLAP_ROADS[:1])
    if APRON_ROUND > 0:
        (x0, x1), (z0, z1) = APRON_ROUND_AT
        r = APRON_ROUND
        fillet = unary_union([apron, road]).buffer(r).buffer(-r).intersection(box(x0, z0, x1, z1)).difference(road)
        apron = unary_union([apron, fillet])
    return apron.difference(Polygon(APRON_ISLAND).buffer(.6).buffer(-.6)).buffer(0)     # (the island's corners rounded)


def flat_map(node):
    """A mesh's uv made again as one flat map from above, as most of its faces were mapped (the faces
    grown onto it were mapped from their nearest, stretched the further they reach)."""
    V = node['verts'].astype(float)
    F = node['faces'].astype(int)
    J = []
    for f in F:
        A = np.c_[V[f[1], [0, 2]] - V[f[0], [0, 2]], V[f[2], [0, 2]] - V[f[0], [0, 2]]]
        if abs(np.linalg.det(A)) < 1e-4:
            continue
        j = np.c_[V[f[1], 6:8] - V[f[0], 6:8], V[f[2], 6:8] - V[f[0], 6:8]] @ np.linalg.inv(A)
        if np.linalg.svd(j, compute_uv=False)[1] > .02:
            J.append(j)
    if not J:
        return
    Jm = np.median(np.array(J), 0)
    s_ = np.sqrt(abs(np.linalg.det(Jm)))                                # (its scale, square: no stretch)
    V[:, 6:8] = V[:, [0, 2]] * s_
    V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
    node['verts'] = V.astype(node['verts'].dtype)


def park_shape(track, pad):
    """The car park as drawn (PARK), less the apron and the grass round its trees, x, z."""
    from shapely.geometry import Polygon
    import car_parks as cp
    return Polygon(PARK).buffer(0).difference(apron_shape(track, pad)).difference(cp.park_islands()).buffer(0)


def car_park(track, pad):
    """The car park laid again: flat asphalt over park_shape (its points every metre round its edges,
    on the ground as it is, mapped as the paddock's asphalt from above), a mesh of the paddock's.
    How many faces."""
    shape = park_shape(track, pad)
    if shape.is_empty or any(n['name'] == PARK_MESH for n, _ in bp.meshes(pad)):
        return 0
    base = next(n for n, _ in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                        if n['name'].startswith(('1GRASS_0', '1GRASS_PATHS')) and len(n['faces'])]))
    P = np.concatenate(fill(shape.segmentize(1.), most=4.))
    y = ground.ys(P)
    y = np.where(np.isnan(y), np.nanmedian(y), y) + .02
    V = np.zeros((len(P), base['verts'].shape[1]))
    V[:, 0], V[:, 1], V[:, 2] = P[:, 0], y, P[:, 1]
    V[:, 4] = 1.
    node = dict(base, name=PARK_MESH, children=[], verts=V.astype(base['verts'].dtype),
                faces=np.arange(len(P)).reshape(-1, 3).astype('<u2'))
    F = node['faces'].astype(int)
    a_, b_, c_ = V[F[:, 0], :3], V[F[:, 1], :3], V[F[:, 2], :3]
    up = np.cross(b_ - a_, c_ - a_)[:, 1]
    node['faces'] = np.where((up < 0)[:, None], F[:, ::-1], F).astype('<u2')
    # (mapped as the paddock's asphalt is, from above)
    ref = dict(base)
    flat_map(ref)
    sc = np.median(np.linalg.norm(ref['verts'][:, 6:8].astype(float), axis=1) / np.maximum(np.linalg.norm(ref['verts'][:, [0, 2]].astype(float), axis=1), 1e-6))
    V[:, 6:8] = V[:, [0, 2]] * sc
    V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], node['faces'].astype(int))[:, :3]
    node['verts'] = V.astype(base['verts'].dtype)
    bp.sphere(node)
    pad['root']['children'].append(node)
    return len(node['faces'])


def apron(track, pad):
    """The road along the building grown over the pit apron (apron_shape, where no road is yet): faces in
    its own mesh, its points shared on the seams, as high as the roads' points nearest them. How many
    faces."""
    from scipy.spatial import cKDTree
    shape = apron_shape(track, pad).difference(roads_shape(track, ((-20., 140.), (-80., 0.)))).buffer(0)
    if shape.is_empty or shape.area < .05:
        return 0
    road = next(n for n, _ in bp.meshes(track) if n['name'] == HOTLAP_BOX_ROAD)
    pool = [n for n, _ in bp.meshes(track) if n['name'].startswith('1ROAD') and n is not road and len(n['faces'])]
    base = next(n for n, _ in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    x0, z0, x1, z1 = shape.buffer(3.).bounds
    V = np.concatenate([n['verts'][:, :3].astype(float) for n in pool + [road, base]])
    V = V[(V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 2] > z0) & (V[:, 2] < z1)]
    tree = cKDTree(V[:, [0, 2]])

    def height(x, z):
        dd, ii = tree.query((x, z), k=8)
        w = 1. / np.maximum(dd, .05) ** 2
        return float((V[ii, 1] * w).sum() / w.sum())
    return grow(road, fill(shape, most=2.), pool=pool, height=height)


PIT_LINE_TYRES = (('TYRES_99', 'TYRES_99F', '1WALL_TYRE_99'), ('TYRES_101', 'TYRES_101F', '1WALL_TYRE_101'))


def tyre_stacks(drawn):
    """The stacks of the tyre meshes drawn, as one: each point's stack, and each stack's middle (x, z)."""
    from scipy.spatial import cKDTree
    from grade_paths import pieces as parts
    W = np.concatenate([n['verts'][:, :3].astype(float) for n in drawn])
    F, off = [], 0
    for n in drawn:
        F.append(n['faces'].astype(int) + off)
        off += len(n['verts'])
    lab = parts(np.concatenate(F), len(W))
    ids, inv = np.unique(lab, return_inverse=True)
    lo = np.full((len(ids), 3), np.inf)
    hi = np.full((len(ids), 3), -np.inf)
    np.minimum.at(lo, inv, W)
    np.maximum.at(hi, inv, W)
    c, size = ((lo + hi) / 2)[:, [0, 2]], (hi - lo)[:, [0, 2]]
    tree = cKDTree(c)
    Q = tree.query_pairs(.3, output_type='ndarray').reshape(-1, 2)
    d = np.abs(c[Q[:, 0]] - c[Q[:, 1]])
    within = np.zeros(len(Q), bool)
    for b in (0, 1):
        within |= (size[Q[:, b]].max(1) < 1.) & (d <= .3 * size[Q[:, b]]).all(1)
    st = np.arange(len(ids))

    def root(a):
        while st[a] != a:
            a = st[a]
        return a
    for a, b in np.r_[tree.query_pairs(.1, output_type='ndarray').reshape(-1, 2), Q[within]]:
        ra, rb = root(a), root(b)
        st[max(ra, rb)] = min(ra, rb)
    st = np.array([root(i) for i in range(len(st))])
    sid, sinv = np.unique(st, return_inverse=True)
    mid = np.array([c[sinv == s_].mean(0) for s_ in range(len(sid))])
    return sinv[inv], mid


def pits_tyre_walls():
    """The tyre walls along the pits under the roof moved onto the columns' lines (the 1500's west,
    the 1000's east), stack by stack across, the columns standing in them (screenshot 17). How many
    stacks moved."""
    k = gltf_source.load_kn5(bp.TYRES)
    moved = 0
    for names, line in zip(PIT_LINE_TYRES, column_lines()):
        nodes = [(n, m) for n, m in bp.meshes(k) if n['name'] in names and len(n['faces'])]
        if not nodes or any(not np.allclose(m, np.eye(4)) for _, m in nodes):
            continue
        at = lambda z: np.interp(z, line[:, 1], line[:, 0])
        drawn = [n for n, _ in nodes if not n['name'].startswith('1WALL')]
        if not drawn:
            continue
        point_stack, mid = tyre_stacks(drawn)
        shift = at(mid[:, 1]) - mid[:, 0]
        if np.abs(shift).max() < .05:
            continue
        off = 0
        for n in drawn:
            ps = point_stack[off:off + len(n['verts'])]
            off += len(n['verts'])
            n['verts'][:, 0] = (n['verts'][:, 0].astype(float) + shift[ps]).astype(n['verts'].dtype)
            bp.sphere(n)
        order = np.argsort(mid[:, 1])
        for n, _ in nodes:
            if n['name'].startswith('1WALL'):
                V = n['verts'].astype(float)
                n['verts'][:, 0] = (V[:, 0] + np.interp(V[:, 2], mid[order, 1], shift[order])).astype(n['verts'].dtype)
                bp.sphere(n)
        moved += len(mid)
    if moved:
        bp.save(k, bp.TYRES)
    return moved


def pits_tyre_rows():
    """The tyre walls along the pits under the pit building's roof laid again, each one row straight
    along its columns' line, a stack round each column (the column standing in its tyres) and the
    rest evenly between, no more than PIT_TYRE_STEP apart, sitting on the asphalt. How many stacks."""
    import build_outside as bo
    k = gltf_source.load_kn5(bp.TYRES)
    pad = gltf_source.load_kn5(bp.PADDOCK)
    track = gltf_source.load_kn5(bp.TRACK)
    top = bp.Surface(np.concatenate([bp.triangles(n, m) for k_ in (track, pad) for n, m in bp.meshes(k_)
                                     if n['name'].startswith('1ROAD') and len(n['faces'])]))
    made = 0
    for names, line in zip(PIT_LINE_TYRES, column_lines()):
        drawn = [n for n, m in bp.meshes(k) if n['name'] in names and not n['name'].startswith('1WALL') and len(n['faces'])]
        if not drawn:
            continue
        stack, mid = tyre_stacks(drawn)
        cols = line[1:-1]
        if all(np.min(np.hypot(*(mid - c).T)) < .03 for c in cols):
            continue                                                    # (laid so already)
        d = (line[-1] - line[0]) / np.linalg.norm(line[-1] - line[0])
        along = (mid - line[0]) @ d
        ca = (cols - line[0]) @ d
        ends = [min(along.min(), ca.min()), max(along.max(), ca.max())]
        anchors = np.unique(np.r_[ends[0], ca, ends[1]])
        at = [anchors[0]]
        for s0, s1 in zip(anchors[:-1], anchors[1:]):
            n_ = max(1, int(np.ceil((s1 - s0) / PIT_TYRE_STEP - 1e-6)))
            at += list(np.linspace(s0, s1, n_ + 1)[1:])
        at = np.array(at)
        P = line[0] + np.outer(at, d)
        # the stack nearest the line's middle as the one laid
        t = int(np.argmin(np.abs(along - along.mean())))
        tc = mid[t]
        base = top.ys(np.array([tc]))[0]
        off = 0
        for n in drawn:
            st = stack[off:off + len(n['verts'])]
            off += len(n['verts'])
            Vn = n['verts'].astype(float)
            Fn = n['faces'].astype(int)
            mine = st == t
            fsel = mine[Fn].all(1)
            used = np.unique(Fn[fsel])
            remap = np.full(len(Vn), -1)
            remap[used] = np.arange(len(used))
            Vt, Ft = Vn[used], remap[Fn[fsel]]
            ys = top.ys(P)
            rows, faces = [], []
            for i, (q, y) in enumerate(zip(P, ys)):
                v = Vt.copy()
                v[:, 0] += q[0] - tc[0]
                v[:, 2] += q[1] - tc[1]
                if np.isfinite(y) and np.isfinite(base):
                    v[:, 1] += y - base
                rows.append(v)
                faces.append(Ft + i * len(Vt))
            Vall = np.concatenate(rows)
            n['verts'] = Vall.astype(n['verts'].dtype)
            n['faces'] = np.concatenate(faces).astype('<u2' if len(Vall) < 65536 else '<u4')
            bp.sphere(n)
        made += len(P)
    if made:
        bp.save(k, bp.TYRES)
    return made


def pit_tyres():
    """The tyre wall along the 1000's pit lane as the real one: its stacks moved to PIT_TYRE_ROWS,
    one to each place along them, the rest out; its wall to hit made again along them. Whether
    it was changed."""
    from scipy.spatial import cKDTree
    from grade_paths import pieces as parts
    k = gltf_source.load_kn5(bp.TYRES)
    nodes = [(n, m) for n, m in bp.meshes(k) if n['name'] in PIT_TYRES and len(n['faces'])]
    if not nodes or any(not np.allclose(m, np.eye(4)) for _, m in nodes):
        return False
    drawn = [n for n, _ in nodes if not n['name'].startswith('1WALL')]
    W = np.concatenate([n['verts'][:, :3].astype(float) for n in drawn])
    if np.mean(W[:, 0] - lane_west(W[:, 2])) > -1.:
        return False                                      # (done already)
    F, off = [], 0
    for n in drawn:
        F.append(n['faces'].astype(int) + off)
        off += len(n['verts'])
    F = np.concatenate(F)
    lab = parts(F, len(W))
    ids, inv = np.unique(lab, return_inverse=True)
    lo = np.full((len(ids), 3), np.inf)
    hi = np.full((len(ids), 3), -np.inf)
    np.minimum.at(lo, inv, W)
    np.maximum.at(hi, inv, W)
    c, size = ((lo + hi) / 2)[:, [0, 2]], (hi - lo)[:, [0, 2]]
    tree = cKDTree(c)
    Q = tree.query_pairs(.3, output_type='ndarray').reshape(-1, 2)
    d = np.abs(c[Q[:, 0]] - c[Q[:, 1]])
    within = np.zeros(len(Q), bool)
    for b in (0, 1):
        within |= (size[Q[:, b]].max(1) < 1.) & (d <= .3 * size[Q[:, b]]).all(1)
    st = np.arange(len(ids))

    def root(a):
        while st[a] != a:
            a = st[a]
        return a
    for a, b in np.r_[tree.query_pairs(.1, output_type='ndarray').reshape(-1, 2), Q[within]]:
        ra, rb = root(a), root(b)
        st[max(ra, rb)] = min(ra, rb)
    st = np.array([root(i) for i in range(len(st))])
    sid, sinv = np.unique(st, return_inverse=True)
    mid = np.array([c[sinv == s_].mean(0) for s_ in range(len(sid))])
    # the places along the rows, each stack (south first) to one
    spots = [(float(wall_line(z)) - inn, z) for inn, z0, z1 in PIT_TYRE_ROWS for z in np.arange(z0, z1 + .01, PIT_TYRE_STEP)]
    order = np.argsort(mid[:, 1])
    keep = np.zeros(len(sid), bool)
    shift = np.zeros((len(sid), 2))
    for spot, s_ in zip(spots, order):
        keep[s_] = True
        shift[s_] = np.array(spot) - mid[s_]
    point_stack = sinv[inv]
    off = 0
    for n in drawn:
        ps = point_stack[off:off + len(n['verts'])]
        off += len(n['verts'])
        n['verts'][:, 0] = (n['verts'][:, 0].astype(float) + shift[ps, 0]).astype(n['verts'].dtype)
        n['verts'][:, 2] = (n['verts'][:, 2].astype(float) + shift[ps, 1]).astype(n['verts'].dtype)
        f = n['faces'].astype(int)
        bp.split(n, ~keep[ps[f[:, 0]]])
    # its wall to hit: along the rows' lane side, as high as the stacks
    y0 = float(lo[:, 1].min()) - .1
    y1 = y0 + float((hi[:, 1] - lo[:, 1]).max()) + .1
    quads = []
    for inn, z0, z1 in PIT_TYRE_ROWS:
        x0, x1 = float(wall_line(z0)) - inn - .32, float(wall_line(z1)) - inn - .32
        quads.append(((x0, z0 - .32), (x1, z1 + .32)))
    rows = []
    for (xa, za), (xb, zb) in quads:
        for p in ((xa, y0, za), (xb, y0, zb), (xb, y1, zb), (xa, y1, za)):
            v = np.zeros(drawn[0]['verts'].shape[1])
            v[:3] = p
            v[3] = -1.
            rows.append(v)
    Fw = []
    for q in range(len(quads)):
        b0 = 4 * q
        Fw += [(b0, b0 + 1, b0 + 2), (b0, b0 + 2, b0 + 3), (b0, b0 + 2, b0 + 1), (b0, b0 + 3, b0 + 2)]
    for n, _ in nodes:
        if n['name'].startswith('1WALL'):
            n['verts'] = np.array(rows).astype(n['verts'].dtype)
            n['faces'] = np.array(Fw).astype('<u2')
            bp.sphere(n)
    bp.save(k, bp.TYRES)
    print('  the tyre wall along the 1000\'s pit lane: %d stacks against the storage building and on to the shed, %d out'
          % (keep.sum(), (~keep).sum()))
    return True


def north_runoff_grass():
    """The paddock's old asphalt beside the 1000's north corner, by its pit entry (velopark_runoffs.glb's
    1ROAD_RUNOFF_PADDOCK north of NORTH_RUNOFF_Z), grass again, the ground's photo on it. How many faces."""
    path = os.path.join(bp.MODELS, 'velopark_runoffs.glb')
    if not os.path.exists(path):
        return 0
    k = gltf_source.load_kn5(path)
    name = '1GRASS_RUNOFF_PADDOCK_NORTH'
    if any(n['name'] == name for n, _ in bp.meshes(k)):
        return 0
    taken = []
    for n, m in list(bp.meshes(k)):
        if n['name'] != '1ROAD_RUNOFF_PADDOCK' or not len(n['faces']):
            continue
        c = n['verts'][:, :3].astype(float)[n['faces'].astype(int)].mean(1)
        out = c[:, 2] > NORTH_RUNOFF_Z
        if out.any():
            taken.append((n, bp.split(n, out)))
    if not taken:
        return 0
    starts = np.cumsum([0] + [len(v) for _, (v, _) in taken])
    V = np.concatenate([v for _, (v, _) in taken]).copy()
    F = np.concatenate([f + starts[i] for i, (_, (_, f)) in enumerate(taken)])
    V[:, 6] = .5 - V[:, 0] / 1060.
    V[:, 7] = .5 - V[:, 2] / 1060.
    mi = next(i for i, mt in enumerate(k['materials']) if mt['name'] == 'Background')
    node = dict(taken[0][0], name=name, children=[], material=mi, verts=V.astype('<f4'), faces=F.astype('<u2'))
    bp.sphere(node)
    k['root']['children'].append(node)
    bp.save(k, path)
    return len(F)


def main():
    if take_old_stand():
        print('the old concrete stand by R26 out of %s' % os.path.relpath(bp.TRACK, bp.ROOT))
    k = gltf_source.load_kn5(bp.PADDOCK)
    nodes = {n['name']: (n, m) for n, m in bp.meshes(k)}
    done = []
    taken, walls = take_remade(k, nodes)
    if taken or walls:
        done.append('%d pieces of the remade buildings out, %d of their walls to hit' % (taken, walls))
    if 'BLD_PILLARS' not in nodes:
        done.append('%d pillars and posts painted, the walls grey' % paint(k, nodes))
    pn = next((n for n, _ in bp.meshes(k) if n['name'] == 'BLD_PILLARS'), None)
    if pn is not None and len(pn['verts']) < 12 * len(np.unique(pieces(pn['faces'].astype(int), len(pn['verts'])))):
        round_pillars(pn)
        bp.sphere(pn)
        done.append('the pillars round')
    if darken_roofs(k):
        done.append('the roofs darker')
    track = gltf_source.load_kn5(bp.TRACK)
    pulled = building_back(k, track)
    for n_ in [n for n, m in bp.meshes(k) if n['name'] in ('BLD_WALLS', 'BLD_DARK') and len(n['faces'])]:
        if not n_.get('both'):
            # (its buildings' walls seen from inside them too)
            V_ = n_['verts']
            back = V_.copy()
            back[:, 3:6] = -back[:, 3:6]
            n_['faces'] = np.r_[n_['faces'].astype(int), n_['faces'].astype(int)[:, ::-1] + len(V_)].astype('<u2')
            n_['verts'] = np.r_[V_, back]
            n_['both'] = True
            bp.sphere(n_)
    roofed = building_roof_out(k, track)
    if roofed:
        done.append('the flat roof of the building by the hotlap\'s pit lane out, %d faces' % roofed)
    if pulled:
        done.append('the building by the hotlap\'s pit lane\'s front %.1f m back, as the photo shows it (%d points)' % (BUILDING_BACK, pulled))
    # (their shapes from the lane as it is, before it is widened)
    pits = pits_shape(track, k)
    import car_parks as cp
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    _area = ((-60., 200.), (-140., -20.))
    east = cp.east_shape().difference(unary_union([roads_shape(track, _area), roads_shape(k, _area)]).buffer(.02)).buffer(0)
    south_on = cp.south_on_shape().difference(unary_union([roads_shape(track, _area), roads_shape(k, _area, lambda n: n != bp.ASPHALT + '_2')]).buffer(.02)).buffer(0)
    south_on = south_on.difference(roads_shape(k, _area, lambda n: n == bp.ASPHALT + '_2')).buffer(0)
    parks = park_shape(track, k).union(cp.park_south_fill(k)).buffer(0)
    import shift_pit_entry as spe
    _pa = ((-60., -20.), (60., 125.))
    left = spe.vacated(track).difference(unary_union([roads_shape(track, _pa), roads_shape(k, _pa)]).buffer(.02)).buffer(0)
    lanes = cp.lanes_shape().difference(unary_union([roads_shape(track, _area), roads_shape(k, _area), east, south_on, parks]).buffer(.02)).buffer(0)
    new = wide_shape(track).union(store_road_shape(track)).union(exit_shape(track)).union(hotlap_junction_shape(track, k)).union(apron_shape(track, k)).union(parks)     # (what the paddock's asphalt gives way to)
    under = new.union(pits).union(east).union(south_on).union(lanes).union(left)
    gone = hairpin_cuts().union(beyond_inner(track, k))                # (no road there, the ground kept)
    # and under all the asphalt round them, the old with the new: the ground's holes under the old
    # did not quite meet its edges, which its ramps down to the grass hid, and the new carries on past
    from shapely.geometry import box
    others = [gltf_source.load_kn5(os.path.join(bp.MODELS, f)) for f in ('velopark_kerbs.glb', 'velopark_runoffs.glb')
              if os.path.exists(os.path.join(bp.MODELS, f))]
    for area in (UNDER_PITS, UNDER_JUNCTION, MOUTH_BOX, UNDER_HOTLAP, UNDER_LANE, UNDER_PAVERS):
        under = unary_union([under] + [roads_shape(m_, area) for m_ in [track] + others]
                            + [roads_shape(k, area, lambda n: not n.startswith('1ROAD_PATH_A')),
                               roads_shape(k, area, lambda n: n.startswith('1ROAD_PATH_A')).difference(box(-50., 118., 2., JUNCTION_CUT))])
    gaps = asphalt_gaps(track, k)
    under = under.union(gaps).difference(gone).buffer(-.02)
    wide, store = widen_lane(track), store_road(track)
    print('  the pit exit carried on to under its tyre wall, %d faces' % widen_exit(track))
    print("  the hotlap's pit lane's corners rounded where it meets the road along the building, %d faces" % widen_hotlap_junction(track, k))
    print("  the road along the building by the hotlap's pit lane cut along its front, %d faces of it a floor under its roof" % hotlap_straight(track, k))
    print("  the pit apron by the hotlap's pit lane, %d faces" % apron(track, k))
    # the lane's inner edge running on into the building's front: no road north of its curve
    for n_, _ in list(bp.meshes(track)):
        if n_['name'] in HOTLAP_ROADS and len(n_['faces']):
            clip_in(n_, beyond_inner(track, k))
    # the paddock's asphalt left over by T2, and north of the building past the hairpin, out; the
    # track's road there one with its patches, its outside smooth
    # (grass laid over where it was: the ground under it was cut away)
    grass_over(k, hairpin_cuts(), '1GRASS_PADDOCK_LEFT_OVER')
    _runoffs_path = os.path.join(bp.MODELS, 'velopark_runoffs.glb')
    _runoffs = gltf_source.load_kn5(_runoffs_path)
    _g, _c, _add = hairpin_out(track, k, _runoffs)
    _m, _madd = round_out(track, k, MOUTH_BOX, '1ROAD_24', MOUTH_FILL)
    _add = _add.union(_madd) if not _madd.is_empty else _add
    print("  the 1000's pit lane's mouth rounded out, %d faces" % _m)
    bp.save(_runoffs, _runoffs_path)
    print('  past the hairpin: %d faces grown into its road, %.1f m2 cut off its outside' % (_g, _c))
    flat_map(next(n for n, _ in bp.meshes(track) if n['name'] == HOTLAP_BOX_ROAD))
    # the ground and the paddock's asphalt under them out, as under every road
    under = under.union(_add.buffer(-.02)) if not _add.is_empty else under
    cut = holes(track, bp.GROUND, under, bp.GROUND + '_CUT') + holes(track, bp.FILLER, under, bp.FILLER + '_CUT')
    bp.save(track, bp.TRACK)
    if wide or store or cut:
        bp.save(track, bp.TRACK)
        print('  the 1000\'s pit lane %.1f m wider, %d faces; the road by the storage building to the pit exit, %d faces; '
              'the ground under them out, %d faces cut round them' % (LANE_EXTRA, wide, store, cut))
    holes(k, bp.ASPHALT, new.buffer(-.02), bp.ASPHALT + '_CUT')
    # the car parks and the road south carried on, their asphalt, the trees on them out
    print('  the car park south of the apron laid again, %d faces' % cp.lay_flat(track, k, parks, PARK_MESH))
    print('  the car park east of it, %d faces' % cp.lay_flat(track, k, east, cp.EAST_MESH))
    print('  the road south carried on to the dirt road, %d faces' % cp.lay_flat(track, k, south_on, cp.SOUTH_ON_MESH))
    print('  the road down between the car parks where it was not, %d faces' % cp.lay_flat(track, k, lanes, cp.LANES_MESH))
    print("  where the vp_1500's pit entry was before it moved west, %d faces" % cp.lay_flat(track, k, left, bp.ASPHALT + '_PIT_ENTRY'))
    print('  %d trees on them out' % cp.clear_trees(unary_union([parks, east, lanes, cp.south_on_shape().buffer(cp.SOUTH_ON_TREES)])))
    print('  the pits\' asphalt out to the columns under the pit building\'s roof, %d faces' % pits_fill(track, k))
    print("  %d faces of asphalt over the gaps between the roads by the hotlap's pit lane" % fill_gaps(track, k, gaps))
    # path A's old end, where the junction's asphalt and the grass take its place
    holes(k, '1ROAD_PATH_A', junction_shapes(track, k)[1], '1ROAD_PATH_A_CUT')
    print('  %d faces of the old asphalt by the 1000\'s north corner grass again' % north_runoff_grass())
    grass = grass_back(k)
    grass += grass_over(k, west_grass())
    from shapely.geometry import box as _box
    (gx0, gx1), (gz0, gz1) = APRON_GRASS
    _foot = garages(k, track)[0]
    _join = _box(20.5, gz0, gx1, min(q[1] for q in APRON_ISLAND))          # (the road south onto the apron below the island)
    grass += grass_over(k, _box(gx0, gz0, gx1, gz1).difference(apron_shape(track, k)).difference(_foot.buffer(.3)).difference(_join), APRON_MESH)
    grass += grass_over(k, cp.park_islands().intersection(Polygon(PARK)), '1GRASS_PARK_ISLANDS')
    grass += grass_over(k, cp.lanes_island(), '1GRASS_PADDOCK_LANES_ISLAND')
    if grass:
        done.append('%d faces of its asphalt by the 1000\'s pit lane grass again' % grass)
    pit_tyres()
    lined = pits_tyre_walls()
    if lined:
        print('  %d tyres of the pits\' tyre walls on the columns\' lines' % lined)
    rowed = pits_tyre_rows()
    if rowed:
        print('  the pits\' tyre walls laid again in one row each along the columns, %d stacks, a column in each of their tyres' % rowed)
    if not done:
        print('the paddock is restyled already')
        return
    # the textures no mesh uses any more out
    used = {t['name'] for mt in k['materials'] for t in mt['textures']}
    k['textures'] = [t for t in k['textures'] if t['name'] in used]
    doc, _ = gltf_source.read_glb(bp.PADDOCK)
    import urllib.parse
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(bp.PADDOCK, doc, binary)
    print('; '.join(done))


if __name__ == '__main__':
    main()
