"""Convert the built Assetto Corsa track into a BeamNG.drive level.

    python tools/export_beamng.py [layout] [track folder] [--no-pack]

Takes the models one layout loads (vp_1500 by default) from the Assetto Corsa build,
build/ac/content/tracks/techspeed_fueltech_velopark,
and writes build/beamng/levels/techspeed_fueltech_velopark with the meshes as Collada shapes, their
textures and materials, and the level's objects, then packs it as
build/dist/techspeed-fueltech-velopark-<version>-beamng.zip, beside the Assetto Corsa archive, to drop in BeamNG's mods
folder. release.ps1 runs it through build_tracks.py. Only the models that changed since the
last run are written again; --no-pack leaves out the zip, to run the level unpacked from
BeamNG's mods/unpacked folder.

The road, grass and kerb meshes AC drives on are solid, with the BeamNG ground model of their
surface (the grass one of the level's own, its soft layer 1 cm deep); the invisible walls are
solid and unseen, and so is a trunk for each tree; the cones AC knocks about are BeamNG's own
loose cones, every layout's, which the level's Lua spawns in free roam, and each race its own layout's;
all else is only seen, as in AC. A mesh AC
swaps for a simpler copy further away (the tyre stacks) is a shape of its own with both as its
detail levels, for BeamNG to swap about as far away. Karts
start on the grid's first slot of any layout, facing the way its lap runs; the layouts' racing
lines are BeamNG's road network (waypoints and map.json), for its map and AI, one road where
layouts share it. The lights of the Custom Shaders
Patch config are spot lights, lit at night only. The level stands LIFT metres up, above BeamNG's
ground. Each layout's race is a time trial mission (beamng_races.py); the gantry board and
lamps are run by the level's Lua (beamng_board.py). The other layouts' models
are not converted.
"""
import configparser, hashlib, io, json, math, os, re, shutil, sys, uuid, zipfile
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

import beamng_board
import beamng_grass
import beamng_races
import kn5io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVEL = 'techspeed_fueltech_velopark'
# AC's surface keys, from the start of a physics mesh's name, as BeamNG ground models
GROUND = {'ROAD': 'ASPHALT', 'GRASS': 'VELOPARK_GRASS', 'KERB': 'RUMBLE_STRIP', 'SAND': 'SAND', 'GRAVEL': 'GRAVEL',
          'WALL': 'ASPHALT', 'PIT': 'ASPHALT', 'TREE': 'WOOD'}
# the level's own ground models (levels/<level>/groundModels/*.json): BeamNG's GRASS is a soft
# layer 5 cm deep that wheels sink into, which hides half a kart's tyre; this is BeamNG's grass
# with a layer of 1 cm
GROUND_MODELS = {'VELOPARK_GRASS': {
    'staticFrictionCoefficient': 0.61, 'slidingFrictionCoefficient': 0.65, 'hydrodynamicFriction': 0.005,
    'stribeckVelocity': 4, 'strength': 1, 'roughnessCoefficient': 0.43,
    'fluidDensity': 8000, 'flowConsistencyIndex': 1500, 'flowBehaviorIndex': 0.7, 'dragAnisotropy': 0,
    'shearStrength': 4000, 'defaultDepth': 0.01, 'collisiontype': 'GRASS', 'skidMarks': False}}
PHYSICS = re.compile(r'^\d+([A-Z]+)')
# AC drives its kerbs as road; BeamNG rumbles over these materials (the tarmac beds of the
# kerbs, named 1ROAD_KERB_ too, stay road)
KERB_MATERIALS = {'KerbRamp', 'KerbConcrete', 'KerbFloor', 'KerbRound', 'Rumble1'}
# how far up the level stands: a level without terrain has BeamNG's ground at height 0, which
# vehicles stand on, and the track lies up to a few metres below that; its walls go down 43 m
LIFT = 50.
# BeamNG's lights fall off as the square of the distance, their brightness times their intensity:
# its own floodlit tracks (Hirochi Raceway, Driver Training) light theirs at an intensity of
# 12000 to 15000 and a brightness of 2.4 to 3. Without an intensity a light is next to nothing
# 16 m down from a pole. The brightness here is LIGHT_GAIN per unit of the Custom Shaders Patch
# intensity: the floods (1.69) at about 7, more than BeamNG's own as a pole here has one flood
# where theirs have two or more, the spill under each pole at about 1
LIGHT_INTENSITY = 15000.
LIGHT_GAIN = 4.
# pixels on screen per metre of a shape's radius at a metre away: a 1080 pixel high screen at
# BeamNG's 60 degree field of view, for where AC swaps a mesh for its far copy
SCREEN = 1080 / (2 * math.tan(math.radians(30)))
# raised when the shapes or textures are written differently, so a level built before is made again
FORMAT = 9


def to_beamng(p):
    """AC's y up, right handed, to BeamNG's z up: x stays, -z is north, y is up."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def floats(a, fmt='%.5g'):
    return ' '.join(fmt % x for x in np.ravel(a))


def file_name(texture):
    """A texture's file name with no spaces or other marks BeamNG's paths may not take."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', texture)


def compressed(data):
    """Whether a texture is a DDS BeamNG reads as it is: it takes only the compressed ones."""
    return data[:4] == b'DDS ' and data[84:88] in (b'DXT1', b'DXT3', b'DXT5')


def texture_file(name, data, normal=False):
    """The file a texture is written to: a DDS BeamNG cannot read becomes a PNG, and so does a
    DXT5 normal map, which BeamNG reads as a swizzled one (x in the alpha, as DXT5nm), tilting
    the normals of what it lies on as far as they go, so that it shows almost black by day."""
    f = file_name(name)
    dds = data is not None and data[:4] == b'DDS '
    return os.path.splitext(f)[0] + '.png' if dds and (not compressed(data) or normal and data[84:88] == b'DXT5') else f


def opacity_file(name):
    """The greyscale map of what of a texture shows, from its alpha: BeamNG reads opacity from
    a map's colour, so the texture itself would show its clear parts and hide the rest."""
    return os.path.splitext(file_name(name))[0] + '_opacity.png'


def payload(block):
    """A kn5 texture block's file bytes."""
    n = int.from_bytes(block[4:8], 'little')
    return block[8 + n + 4:]


def write_dae(path, meshes, collide=False, sizes=None, collision=None):
    """Meshes as (name, material, positions, normals, uvs, triangles), in BeamNG's axes.

    Laid out as BeamNG's own shapes are: base00 with the detail levels, then start01 with the
    meshes, each named with the detail it belongs to as the number its name ends in. All are
    drawn at one detail, 2, so a kn5 name ending in a number (1ROAD_3, TYRES_55) does not make a
    level of its own; or sizes gives each mesh's detail: the shape's size on screen, in pixels,
    from which it is drawn, BeamNG drawing the largest detail the shape is bigger than. With
    collide, each mesh is also a Colmesh at detail -1: what vehicles hit, the mesh's own
    triangles as they are, with its material for the ground model. collision gives meshes that
    are only that, drawn at no detail."""
    sizes = sizes or [2] * len(meshes)
    collision = collision or []
    collide = collide or bool(collision)
    geo, nodes, mats = [], [], {}
    for k, ((name, mat, P, N, U, F), size) in enumerate(list(zip(meshes, sizes)) + [(c, None) for c in collision]):
        mats[mat] = True
        g = 'g%d' % k
        geo.append(f'''<geometry id="{g}" name="{escape(name)}"><mesh>
<source id="{g}-p"><float_array id="{g}-pa" count="{P.size}">{floats(P)}</float_array><technique_common><accessor source="#{g}-pa" count="{len(P)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="{g}-n"><float_array id="{g}-na" count="{N.size}">{floats(N, '%.4g')}</float_array><technique_common><accessor source="#{g}-na" count="{len(N)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="{g}-t"><float_array id="{g}-ta" count="{U.size}">{floats(U, '%.6g')}</float_array><technique_common><accessor source="#{g}-ta" count="{len(U)}" stride="2"><param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
<vertices id="{g}-v"><input semantic="POSITION" source="#{g}-p"/><input semantic="NORMAL" source="#{g}-n"/></vertices>
<triangles material="{escape(mat)}" count="{len(F)}"><input semantic="VERTEX" source="#{g}-v" offset="0"/><input semantic="TEXCOORD" source="#{g}-t" offset="0" set="0"/><p>{' '.join(map(str, F.ravel()))}</p></triangles>
</mesh></geometry>''')
        bind = f'<bind_material><technique_common><instance_material symbol="{escape(mat)}" target="#{escape(mat)}"/></technique_common></bind_material>'
        if size is not None:
            nodes.append(f'<node id="n{k}" name="%s_%d_L%d"><instance_geometry url="#{g}">{bind}</instance_geometry></node>'
                         % (re.sub(r'[^A-Za-z0-9_]', '_', name), k, size))
        if size is None or collide and not collision:
            nodes.append(f'<node id="c{k}" name="Colmesh_{k}-1"><instance_geometry url="#{g}">{bind}</instance_geometry></node>')
    effects = ''.join(f'<effect id="{escape(m)}-fx"><profile_COMMON><technique sid="common"><lambert/></technique></profile_COMMON></effect>' for m in mats)
    materials = ''.join(f'<material id="{escape(m)}" name="{escape(m)}"><instance_effect url="#{escape(m)}-fx"/></material>' for m in mats)
    details = ''.join('<node id="detail%d" name="detail%d"/>' % (x, x) for x in sorted(set(sizes), reverse=True) if x is not None)
    details += '<node id="collision-1" name="collision-1"/>' if collide else ''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_effects>{effects}</library_effects>
<library_materials>{materials}</library_materials>
<library_geometries>{''.join(geo)}</library_geometries>
<library_visual_scenes><visual_scene id="scene"><node id="base00" name="base00">{details}<node id="start01" name="start01">{''.join(nodes)}</node></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
''')


def mesh_arrays(m):
    """A kn5 mesh's corners and triangles in BeamNG's axes, facing the way its normals do;
    None if nothing is left of it."""
    f = m['f'][(m['f'][:, 0] != m['f'][:, 1]) & (m['f'][:, 1] != m['f'][:, 2]) & (m['f'][:, 0] != m['f'][:, 2])]
    if not len(f):
        return None
    P = to_beamng(m['v'][:, :3])
    N = to_beamng(m['v'][:, 3:6])
    # DirectX's v runs down the texture, Collada's up
    U = np.c_[m['v'][:, 6], 1 - m['v'][:, 7]]
    t = P[f]
    geo = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (geo * N[f].sum(1)).sum(1) < 0
    f = f.copy()
    f[flip] = f[flip][:, ::-1]
    return P, N, U, f


def loose_cones(model):
    """The cones AC knocks about in a model: where each stands, [(x, y, foot)] in BeamNG's axes,
    and the middle of each, in AC's, to know the cone drawn there by. Each is an AC_POBJECT
    mesh, the unseen box AC moves it by, round the cone drawn."""
    feet, middles = [], []
    for m in model.meshes:
        if m['name'].startswith('AC_POBJECT') and len(m['v']):
            v = m['v'][:, :3]
            c = (v.min(0) + v.max(0)) / 2
            middles.append(c)
            if model.material_name(m) != BOARD_MATERIAL:
                feet.append(tuple(float(x) for x in to_beamng(np.array([c[0], v[:, 1].min(), c[2]]))))
    return feet, np.array(middles).reshape(-1, 3)


SAME = .3                           # metres apart two layouts' cones or boards are the same one
BOARD_MATERIAL = 'MarkerBoards'     # the distance boards', as their loose boxes have it


def loose_boards(model):
    """The distance boards AC knocks about in a model (tools/build_board_objects.py), in BeamNG's
    axes and the level's place: each as (the corners of its box, its mesh drawn (P, N, U, F))."""
    middle = lambda m: (m['v'][:, :3].min(0) + m['v'][:, :3].max(0)) / 2
    boxes = [m for m in model.meshes if m['name'].startswith('AC_POBJECT') and model.material_name(m) == BOARD_MATERIAL]
    out = []
    for m in model.meshes:
        if m['name'].startswith('AC_POBJECT') or not (m['visible'] and m['renderable']) or model.material_name(m) != BOARD_MATERIAL:
            continue
        box = next((b for b in boxes if np.linalg.norm(middle(b) - middle(m)) < .05), None)
        arrays = mesh_arrays(m)
        if box is None or arrays is None:
            continue
        P, N, U, F = arrays
        out.append((to_beamng(box['v'][:, :3]) + (0, 0, LIFT), (P + (0, 0, LIFT), N, U, F)))
    return out


def is_loose(m, middles):
    """Whether a mesh is a cone drawn round one of the loose cones' boxes."""
    if not len(middles) or not (m['visible'] and m['renderable']) or not len(m['v']):
        return False
    v = m['v'][:, :3]
    size = v.max(0) - v.min(0)
    return size.max() < 2 and np.linalg.norm(middles - (v.min(0) + v.max(0)) / 2, axis=1).min() < .05


def tree_feet(P, F):
    """Where each tree of a mesh stands, [(x, y, foot)] in BeamNG's axes: a tree is a run of
    crossed boards, apart from the others, standing round its trunk."""
    label = np.arange(len(P))
    while True:
        low = label[F].min(1)
        before = label.copy()
        for k in range(3):
            np.minimum.at(label, F[:, k], low)
        label = label[label]
        if (label == before).all():
            break
    out = []
    for t in np.unique(label[F[:, 0]]):
        p = P[label == t]
        c = (p.min(0) + p.max(0)) / 2
        out.append((float(c[0]), float(c[1]), float(p[:, 2].min())))
    return out


# the trunks vehicles hit, which AC's trees, boards only, do not have: as wide as the trunks drawn
# on the boards, from below the ground to above a kart. Trees whose boards stand further below the
# ground than this are buried there, out of sight, and get none
TRUNK_RADIUS = .5
TRUNK_HEIGHT = 4.
TRUNK_BURIED = 2.


def trunks(trees, ground_at):
    """The trunks of the trees standing on the ground, as one mesh (P, N, U, F), or None."""
    P, N, F = [], [], []
    sides = 8
    a = np.arange(sides) * 2 * np.pi / sides
    ring = np.c_[np.cos(a), np.sin(a)]
    for x, y, foot in trees:
        z = ground_at(x, y)
        if z is None or foot < z - TRUNK_BURIED:
            continue
        lo, hi = min(foot, z) - .3, max(foot, z) + TRUNK_HEIGHT
        k = len(P)
        for h in (lo, hi):
            P += [(x + TRUNK_RADIUS * c, y + TRUNK_RADIUS * s, h) for c, s in ring]
            N += [(c, s, 0.) for c, s in ring]
        P.append((x, y, hi))
        N.append((0., 0., 1.))
        for i in range(sides):
            j = (i + 1) % sides
            # counter-clockwise seen from outside
            F += [(k + i, k + j, k + sides + j), (k + i, k + sides + j, k + sides + i),
                  (k + sides + i, k + sides + j, k + 2 * sides)]
    if not F:
        return None
    P = np.array(P)
    return P, np.array(N), np.zeros((len(P), 2)), np.array(F)


# the plantation of tools/build_trees.py as BeamNG's forest: each tree made once, a shape with
# its trunk to hit, and the item data and placings that stand it where AC's copies of it stand,
# turned and sized as they are; AC's own copies of them are left out of the level's shapes
FOREST = os.path.join(ROOT, 'source', 'models', 'velopark_extra_trees.json')
FOREST_MODEL = 'velopark_extra.kn5'
FOREST_MESHES = ('EUCA_BARK', 'EUCA_LEAVES', 'EUCA_BROAD', 'EUCA_PALM', 'EUCA_BAMBOO', 'EUCA_CULM', '1WALL_EUCA_TRUNKS')
# swaying a little in the wind, less than BeamNG's own aspens
FOREST_WIND = dict(windScale=.25, branchAmp=.03, detailAmp=.05, detailFreq=4, trunkBendScale=.02, mass=40, rigidity=20,
                   tightnessCoefficient=4)


def facing(P, N, F):
    """Triangles turned to face the way their corners' normals do."""
    t = P[F]
    geo = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = (geo * N[F].sum(1)).sum(1) < 0
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    return F


def forest(level, data, textures):
    """The forest's shapes, textures, item data and placings written; its materials returned."""
    import build_trees
    shapes = os.path.join(level, 'art', 'shapes', 'forest')
    for d in (shapes, os.path.join(level, 'forest'), os.path.join(level, 'art', 'forest')):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    tex = '/levels/%s/art/shapes/forest/' % LEVEL
    pid = lambda k: str(uuid.uuid5(uuid.NAMESPACE_URL, LEVEL + '/forest/' + k))
    # a material for each part the trees have, its texture the one AC's copies of them have
    mats = {}
    for part, dds in build_trees.TEXTURES.items():
        name = 'velopark_forest_' + part
        png = os.path.splitext(dds)[0]
        img = Image.open(io.BytesIO(textures[dds]))
        img.convert('RGB').save(os.path.join(shapes, png + '.png'))
        stage = {'baseColorMap': tex + png + '.png', 'roughnessFactor': 1., 'metallicFactor': 0.}
        mats[name] = {'name': name, 'mapTo': name, 'class': 'Material', 'persistentId': pid(part), 'version': 1.5,
                      'annotation': 'NATURE', 'Stages': [stage, {}, {}, {}]}
        if part == 'bark':
            mats[name]['groundType'] = GROUND['TREE']
        else:
            img.convert('RGBA').getchannel('A').save(os.path.join(shapes, png + '_opacity.png'))
            stage['opacityMap'] = tex + png + '_opacity.png'
            mats[name].update(alphaTest=True, alphaRef=100)
    items = {}
    sides = 6
    a = np.arange(sides) * 2 * np.pi / sides
    for name, v in data['variants'].items():
        parts, r = build_trees.shape_of(v)
        meshes = []
        for part, mat in ((p, 'velopark_forest_' + p) for p in parts):
            n, Ps, Ns, Us, Fs = 0, [], [], [], []
            for V, N, U, F in parts[part]:
                Ps.append(V)
                Ns.append(N)
                Us.append(U)
                Fs.append(F + n)
                n += len(V)
            P, N, U = to_beamng(np.concatenate(Ps)), to_beamng(np.concatenate(Ns)), np.concatenate(Us)
            F = np.concatenate(Fs)
            # the leaves' cards are both ways round already, each way its own triangles
            meshes.append((part, mat, P, N, np.c_[U[:, 0], 1 - U[:, 1]], F if part != 'bark' else facing(P, N, F)))
        # the trunk to hit, as AC's wall round it
        ring = np.c_[np.cos(a), np.sin(a), np.zeros(sides)] * r * 1.1
        Pc = np.r_[ring + (0, 0, -.3), ring + (0, 0, build_trees.TRUNK)]
        Nc = np.r_[ring, ring] / np.linalg.norm(ring[0])
        Fc = np.array([(i, (i + 1) % sides, sides + (i + 1) % sides) for i in range(sides)] +
                      [(i, sides + (i + 1) % sides, sides + i) for i in range(sides)])
        write_dae(os.path.join(shapes, name + '.dae'), meshes, sizes=[2] * len(meshes),
                  collision=[('trunk', 'velopark_forest_bark', Pc, Nc, np.zeros((len(Pc), 2)), facing(Pc, Nc, Fc))])
        items[name] = dict({'name': name, 'internalName': name, 'class': 'TSForestItemData', 'persistentId': pid(name),
                            'radius': float(r), 'shapeFile': 'levels/%s/art/shapes/forest/%s.dae' % (LEVEL, name)}, **FOREST_WIND)
    json.dump(items, open(os.path.join(level, 'art', 'forest', 'managedItemData.json'), 'w'), indent=1)
    # where each stands: AC's turn about its up, y, made BeamNG's about its up, z; the
    # rotationMatrix lists the axes x, y, z in turn
    T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    lines = {}
    for name, x, y, z, yaw, scale in data['trees']:
        c, s_ = math.cos(yaw), math.sin(yaw)
        Rb = T @ np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]]) @ T.T
        pos = to_beamng(np.array([x, y, z])) + (0, 0, LIFT)
        lines.setdefault(name, []).append(json.dumps({'ctxid': 0, 'pos': [float(q) for q in pos],
                                                      'rotationMatrix': [float(q) for q in Rb.T.ravel()],
                                                      'scale': float(scale), 'type': name}))
    for name, ls in lines.items():
        open(os.path.join(level, 'forest', name + '.forest4.json'), 'w').write('\n'.join(ls) + '\n')
    print('%-28s %-5s %4d trees of %d kinds' % ('forest', 'items', len(data['trees']), len(items)))
    return mats


def racing_line(path):
    """An AC racing line in BeamNG's axes, (n, 3), and how far the road reaches to either side of
    each point, (n, 2), or None if the file does not say."""
    b = open(path, 'rb').read()
    n = int.from_bytes(b[4:8], 'little')
    line = to_beamng(np.frombuffer(b, dtype='<f4', count=n * 5, offset=16).reshape(-1, 5)[:, :3])
    p = 16 + n * 20
    extra = int.from_bytes(b[p:p + 4], 'little') if len(b) >= p + 4 else 0
    sides = None
    if extra == n and len(b) >= p + 4 + n * 72:
        sides = np.frombuffer(b, dtype='<f4', count=n * 18, offset=p + 4).reshape(-1, 18)[:, 5:7]
    return line, sides


# waypoints along each racing line, metres apart, and the road's half width they carry, at most
# (AC's side distances run on over the run-offs)
WAYPOINT_STEP = 5.
WAYPOINT_RADIUS = (2., 4.5)
# the roads off the laps the map draws too. The paths across the infield, in AC's (x, z) along
# their middle, with their half width, as tools/build_paths.py lays them (A and D there): from
# R20 down beside the start straight to the vp_1500 pits, and from R19 across to R20
SIDE_PATHS = {'path_r20_pits': (((-1., 226.1), (-8.4, 222.9), (-14.4, 217.8), (-18.1, 211.6), (-20.8, 204.5),
                                 (-23., 197.1), (-24.7, 189.5), (-25.9, 181.9), (-26.5, 174.), (-26.8, 166.1),
                                 (-26.1, 158.5), (-24.4, 152.1), (-21.9, 147.1), (-17.8, 140.3), (-16.5, 137.8),
                                 (-15.2, 135.2)), 1.8),
              'path_r19_r20': (((47., 236.6), (42.4, 235.9), (36.5, 235.), (32.5, 234.4), (29.6, 234.2), (26.5, 234.2),
                                (22., 233.5), (17., 232.5), (13.4, 231.9)), 1.4)}
# and the pit lanes, one for each that layouts share, by their AC pit lane lines, as wide as the
# asphalt round them: the lanes beside the centre buildings (R24 for vp_1000, R25 for vp_1500),
# and the pits along the long pit building (R14, for vp_2500 and the hotlap)
PIT_RADIUS = {'vp_1000': 5., 'vp_1500': 5., 'vp_1500_amateur': 5., 'vp_1500_short': 5., 'vp_1500_short_amateur': 5.,
              'vp_1500_hotlap': 9., 'vp_2500': 9.}


def side_roads(track, layouts, on_road):
    """The roads off the laps, {name: (line in BeamNG's axes, half width)}: the paths, and each
    pit lane once, named after the first layout that has it."""
    out = {}
    for name, (pts, r) in SIDE_PATHS.items():
        pts = np.asarray(pts, float)
        dense = [pts[0]]
        for p, q in zip(pts, pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(q - p) / 1.)))
            dense += [p + (q - p) * s for s in np.linspace(0, 1, n + 1)[1:]]
        line = []
        for x, z in dense:
            p = on_road(to_beamng(np.array([x, 1000., z])))
            if p[2] > 900:
                sys.exit('%s leaves the ground at AC x %.1f, z %.1f' % (name, x, z))
            # as the racing lines run, a little over the road
            line.append(p + (0, 0, .3))
        out[name] = (np.array(line), r)
    lanes = []
    for lay in layouts:
        path = os.path.join(track, lay, 'ai', 'pit_lane.ai')
        if lay not in PIT_RADIUS or not os.path.exists(path):
            continue
        line = racing_line(path)[0]
        # layouts sharing pits have lanes alike to a few centimetres, not to the byte
        if any(len(o) == len(line) and np.abs(o - line).max() < .5 for o in lanes):
            continue
        lanes.append(line)
        out['pits_' + lay] = (line, PIT_RADIUS[lay])
    return out


# where the layouts' racing lines run on the same road, one road of the network: a point of a
# line this near across a line before it, running its way, is that line's nearest waypoint (and
# it keeps to that line until JOIN_LEAVE across), so the layouts share their road and part at junctions, as the real
# track's roads do. BeamNG's AI routes its race from checkpoint to checkpoint along the network:
# with each layout a loop of its own, laid over the others, a checkpoint took the nearest of them,
# often another layout's, which no road joined to the next, and the racers stopped there
JOIN_NEAR, JOIN_LEAVE = 1.5, 2.5
JOIN_FACING = .8         # cosine of the angle the lines may run apart at, to be one road
SLACK = 3.               # metres off its lap a race's AI route may run, on the same asphalt


def road_network(roads, side=None):
    """Waypoints [(name, position, radius)], segments {name: [names]} and each layout's lap as the
    waypoints it runs through, {layout: [names]}, for the racing lines {layout: (line, sides)},
    longest first, laid on one network: where a line runs along one before it, it runs through
    that line's waypoints. Each stretch between junctions is one segment, given once: BeamNG's
    map, merging a run of waypoints into one stretch, fails where that stretch is there already.
    The roads off the laps {name: (line, half width)} are open segments of their own, apart."""
    from scipy.spatial import cKDTree
    waypoints, segments = [], {}
    for name, (line, r) in (side or {}).items():
        step = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.r_[0, np.cumsum(step)]
        picks = list(np.searchsorted(s, np.arange(0, s[-1], WAYPOINT_STEP)))
        if picks[-1] != len(line) - 1:
            picks.append(len(line) - 1)
        nodes = []
        for i in picks:
            node = 'wp_%s_%d' % (name, len(nodes))
            waypoints.append((node, line[i] - (0, 0, .3), float(r)))
            nodes.append(node)
        segments[name] = nodes
    pos, rad, facing = [], [], []          # the laps' waypoints
    edges = set()
    laps = {}
    for lay, (line, sides) in sorted(roads.items(), key=lambda kv: -len(kv[1][0])):
        step = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.r_[0, np.cumsum(step)]
        picks = np.searchsorted(s, np.arange(0, s[-1], WAYPOINT_STEP))
        n = len(line)
        known = len(pos)
        tree = cKDTree(np.array(pos)[:, :2]) if known else None
        seq, on = [], False
        for i in picks:
            # the racing line runs a little over the road
            at = line[i] - (0, 0, .3)
            d = line[(i + 2) % n] - line[(i - 2) % n]
            d = d[:2] / (np.linalg.norm(d[:2]) or 1)
            r = float(np.clip(sides[i].mean(), *WAYPOINT_RADIUS)) if sides is not None else WAYPOINT_RADIUS[1]
            k = None
            if tree is not None:
                best = None
                # how far across the other line, not from its waypoint: they are WAYPOINT_STEP apart
                # along it, and the point may fall anywhere between two
                for j in tree.query_ball_point(at[:2], WAYPOINT_STEP):
                    rel = at[:2] - pos[j][:2]
                    along, across = rel @ facing[j], abs(rel[0] * facing[j][1] - rel[1] * facing[j][0])
                    gap = np.hypot(along, across)
                    if (facing[j] @ d > JOIN_FACING and abs(along) < WAYPOINT_STEP * .6
                            and across < (JOIN_LEAVE if on else JOIN_NEAR) and (best is None or gap < best)):
                        k, best = j, gap
            on = k is not None
            if k is None:
                k = len(pos)
                pos.append(at)
                rad.append(r)
                facing.append(d)
            else:
                rad[k] = max(rad[k], r)
            if not seq or seq[-1] != k:
                seq.append(k)
        if len(seq) > 1 and seq[-1] == seq[0]:
            seq.pop()
        # between two waypoints of the lines before it that it reaches in turn, it runs along
        # their road between them, not straight across
        adj = {}
        for a, b in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        lap = []
        for a, b in zip(seq, seq[1:] + seq[:1]):
            lap.append(a)
            if a < known and b < known and b not in adj.get(a, ()):
                came, todo = {a: None}, [a]
                for _ in range(4):
                    nxt = []
                    for x in todo:
                        for y in adj.get(x, ()):
                            if y not in came:
                                came[y] = x
                                nxt.append(y)
                    todo = nxt
                    if b in came:
                        break
                if b in came:
                    via = []
                    x = came[b]
                    while x != a:
                        via.append(x)
                        x = came[x]
                    lap += via[::-1]
        for a, b in zip(lap, lap[1:] + lap[:1]):
            if a != b:
                edges.add((min(a, b), max(a, b)))
        laps[lay] = lap
    # waypoints of two lines that still lie on one road, nearer than JOIN_NEAR and running its
    # way, made one: else the network splits round them in small diamonds, the AI's route on one
    # side of each and the lap on the other
    P2 = np.array(pos)[:, :2]
    root = list(range(len(pos)))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for a, b in cKDTree(P2).query_pairs(JOIN_NEAR):
        if facing[a] @ facing[b] > JOIN_FACING:
            ra, rb = find(a), find(b)
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
    for x in range(len(pos)):
        r_ = find(x)
        if r_ != x:
            rad[r_] = max(rad[r_], rad[x])
    for lay, lap in laps.items():
        out = []
        for x in lap:
            x = find(x)
            if not out or out[-1] != x:
                out.append(x)
        if len(out) > 1 and out[-1] == out[0]:
            out.pop()
        laps[lay] = out
    edges = {(min(a, b), max(a, b)) for lap in laps.values() for a, b in zip(lap, lap[1:] + lap[:1]) if a != b}
    # the stretches between junctions, each once (two between the same junctions are two roads,
    # as where layouts take different corners)
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    junction = {k for k, v in adj.items() if len(v) != 2}
    seen, chains = set(), []
    for start in sorted(junction) + sorted(set(adj) - junction):
        for nxt in sorted(adj[start]):
            if (min(start, nxt), max(start, nxt)) in seen:
                continue
            chain, prev, cur = [start], start, nxt
            seen.add((min(start, nxt), max(start, nxt)))
            while True:
                chain.append(cur)
                if cur in junction or cur == start:
                    break
                prev, cur = cur, next(x for x in adj[cur] if x != prev)
                seen.add((min(prev, cur), max(prev, cur)))
            chains.append(chain)
    name = lambda k: 'wp_lap_%d' % k
    waypoints += [(name(k), pos[k], float(rad[k])) for k in sorted(adj)]
    for k, chain in enumerate(chains):
        segments['road_%d' % k] = [name(x) for x in chain]
    return waypoints, segments, {lay: [name(x) for x in lap] for lay, lap in laps.items()}


def route_races(races, laps, waypoints, segments, clear=4.):
    """Each race's checkpoints on its own layout's road, and as many more as it takes for BeamNG's
    AI to keep to that road. Its race takes the AI's route as BeamNG does: from each checkpoint
    the nearer end of the stretch of road nearest it, then the shortest way along the network to
    the next one's. Where layouts take different corners between the same junctions, that way
    took another layout's corner wherever it was the shorter and the race had no checkpoint on
    its own one; so a checkpoint is added halfway along its own road there, until none leaves
    its lap. Checkpoints stand on a stretch between two of its waypoints, clear of junctions and
    of the roads off the laps (clear metres)."""
    import heapq
    # where the level puts them, LIFT up, as the checkpoints are: BeamNG finds the nearest road
    # in three dimensions
    at = {n: np.asarray(p, float) + (0, 0, LIFT) for n, p, _ in waypoints}
    edges = [(a, b) for nodes in segments.values() for a, b in zip(nodes, nodes[1:])]
    adj = {}
    for a, b in edges:
        w = float(np.linalg.norm(at[a] - at[b]))
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    E0 = np.array([at[a] for a, _ in edges])
    ED = np.array([at[b] for _, b in edges]) - E0
    off = np.array([at[n] for name, nodes in segments.items() if not name.startswith('road_') for n in nodes])
    near_off = lambda q: len(off) and np.linalg.norm(off[:, :2] - q[:2], axis=1).min() < clear

    def nearest(q):
        t = np.clip(((q - E0) * ED).sum(1) / (ED * ED).sum(1), 0, 1)
        k = int(np.linalg.norm(E0 + ED * t[:, None] - q, axis=1).argmin())
        return edges[k][1] if t[k] > .5 else edges[k][0]

    def way(a, b):
        best, prev, todo = {a: 0.}, {}, [(0., a)]
        while todo:
            dist, x = heapq.heappop(todo)
            if x == b:
                break
            if dist > best[x]:
                continue
            for y, w in adj[x]:
                if dist + w < best.get(y, 1e18):
                    best[y], prev[y] = dist + w, x
                    heapq.heappush(todo, (dist + w, y))
        if b not in best:
            return None
        out = [b]
        while out[-1] != a:
            out.append(prev[out[-1]])
        return out[::-1]

    bad = 0
    for r in races:
        path, lap = r['path'], laps[r['layout']]
        n = len(lap)
        where = {x: i for i, x in enumerate(lap)}
        own = {frozenset(e) for e in zip(lap, lap[1:] + lap[:1])}
        L0 = np.array([at[x][:2] for x in lap])
        LD = np.roll(L0, -1, axis=0) - L0

        def strays(w):
            """Whether a route leaves the lap: a waypoint of it off the lap's own stretches more
            than SLACK from the lap (where two lines join, their waypoints a metre or two apart
            on the same asphalt are no other road)."""
            for x, y in zip(w, w[1:]):
                if frozenset((x, y)) in own:
                    continue
                for q in (at[x][:2], at[y][:2]):
                    t = np.clip(((q - L0) * LD).sum(1) / (LD * LD).sum(1), 0, 1)
                    if np.linalg.norm(L0 + LD * t[:, None] - q, axis=1).min() > SLACK:
                        return True
            return False
        # the lap's stretches a checkpoint may stand on: (i, i + 1) along it
        fit = [i for i in range(n) if len(adj[lap[i]]) == 2 and len(adj[lap[(i + 1) % n]]) == 2
               and not near_off(at[lap[i]]) and not near_off(at[lap[(i + 1) % n]])]
        F0 = np.array([at[lap[i]] for i in fit])
        FD = np.array([at[lap[(i + 1) % n]] for i in fit]) - F0

        def stand(q):
            """q moved onto the nearest stretch it may stand on."""
            t = np.clip(((q - F0)[:, :2] * FD[:, :2]).sum(1) / (FD[:, :2] ** 2).sum(1), .25, .75)
            foot = F0 + FD * t[:, None]
            k = int(np.linalg.norm((foot - q)[:, :2], axis=1).argmin())
            return np.array([foot[k][0], foot[k][1], q[2]]), FD[k] / np.linalg.norm(FD[k])

        for p in path['pathnodes']:
            q, _ = stand(np.array(p['pos']))
            p['pos'] = [float(v) for v in q]
        added = 0
        for _ in range(50):
            nodes = path['pathnodes']
            picks = [nearest(np.array(p['pos'])) for p in nodes]
            wrong = None
            for k, (a, b) in enumerate(zip(picks, picks[1:] + picks[:1])):
                w = way(a, b)
                if w is None or strays(w):
                    wrong = k
                    break
            if wrong is None:
                break
            a, b = picks[wrong], picks[(wrong + 1) % len(picks)]
            if a not in where or b not in where:
                break
            i, j = where[a], where[b]
            mid = lap[(i + ((j - i) % n) // 2) % n]
            q, normal = stand(at[mid] + (0, 0, nodes[wrong]['pos'][2] - at[mid][2]))
            ids = [x['oldId'] for x in path['pathnodes'] + path['startPositions'] + path['segments']]
            new = dict(nodes[wrong], oldId=max(ids) + 1, name='Checkpoint %d' % (max(ids) + 1), pos=[float(v) for v in q],
                       normal=[float(v) for v in normal], useAsSplit=False)
            nodes.insert(wrong + 1, new)
            added += 1
        # the segments, from each checkpoint to the next, again
        top = max(x['oldId'] for x in path['pathnodes'] + path['startPositions'])
        path['segments'] = [{'capsules': {}, 'from': p['oldId'], 'mode': 'waypoint', 'name': 'Segment %d' % (k + 1),
                             'oldId': top + k + 1, 'to': path['pathnodes'][(k + 1) % len(path['pathnodes'])]['oldId']}
                            for k, p in enumerate(path['pathnodes'])]
        if wrong is not None:
            bad += 1
            print('warning: %s: BeamNG\'s AI would leave its lap between checkpoints %d and %d' % (r['layout'], wrong, wrong + 1))
        elif added:
            print('%-28s %-5s %d checkpoints added to keep its AI on its lap' % (r['layout'], 'race', added))
    if not bad:
        print('%-28s %-5s every race\'s AI route on its own lap' % ('roads', 'ai'))


def lights(track, obj):
    """The track's lights in its Custom Shaders Patch config, as BeamNG spot lights: each light of
    a series where it stands, pointing its own way or the series', reaching as far as its range.
    They are BeamNG's night lights, as its own floodlit tracks' are: off, and switched on through
    the night by the time of day, as the series' NIGHT_SMOOTH condition does in AC."""
    path = os.path.join(track, 'extension', 'ext_config.ini')
    if not os.path.exists(path):
        return []
    text = open(path, encoding='utf-8', errors='replace').read()
    out = []
    for n, body in re.findall(r'\[LIGHT_SERIES_(\d+)\]\n(.*?)(?=\n\[|\Z)', text, re.S):
        key = lambda k, d=None: (re.search(r'^%s\s*=\s*(.*)$' % k, body, re.M) or [None, d])[1]
        if key('ACTIVE', '1').strip() != '1':
            continue
        vec = lambda v: np.array([float(x) for x in v.split(',')])
        colour = vec(key('COLOR', '1, 1, 1, 1'))
        spot = float(key('SPOT', '90'))
        outer = min(spot, 170.)
        inner = outer * (1 - float(key('SPOT_SHARPNESS', '0.5')) * .5)
        common = vec(key('DIRECTION', '0, -1, 0'))
        for k, pos in re.findall(r'^POSITION_(\d+)\s*=\s*(.*)$', body, re.M):
            own = key('DIRECTION_' + k)
            d = to_beamng(vec(own) if own else common)
            y = d / np.linalg.norm(d)
            # a light shines along its own y: x and z square to it, z as near up as it goes. A
            # level's rotationMatrix lists the object's axes in turn, x then y then z (BeamNG's
            # own spot lights all have the second pointing down)
            x = np.cross(y, [0, 0, 1.]) if abs(y[2]) < .99 else np.array([1., 0, 0])
            x /= np.linalg.norm(x)
            z = np.cross(x, y)
            out.append(obj('SpotLight', 'light_%s_%s' % (n, k), position=[float(v) for v in to_beamng(vec(pos)) + (0, 0, LIFT)],
                           rotationMatrix=[float(v) for v in np.r_[x, y, z]],
                           color=[float(c) for c in colour[:3]] + [1.], brightness=float(colour[3] * LIGHT_GAIN) if len(colour) > 3 else LIGHT_GAIN,
                           intensity=LIGHT_INTENSITY, attenuationRatio=[1, 0, 0],
                           range=float(key('RANGE', '50')), innerAngle=float(inner), outerAngle=float(outer),
                           castShadows=False, isEnabled=False, nightLight='1'))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    export(args[0] if args else 'vp_1500', args[1] if len(args) > 1 else os.path.join(ROOT, 'build', 'ac', 'content', 'tracks', LEVEL),
           pack='--no-pack' not in sys.argv)


def export(layout, track, out=None, pack=True):
    """The level in out (build/beamng), and its zip in dist (build/dist) unless pack is False. Returns the
    zip's path, or the level's."""
    out = out or os.path.join(ROOT, 'build', 'beamng')
    level = os.path.join(out, 'levels', LEVEL)
    shapes = os.path.join(level, 'art', 'shapes')
    os.makedirs(os.path.join(shapes, 'textures'), exist_ok=True)
    # what each model was when its shapes and textures were last written
    state_path = os.path.join(out, 'models.json')
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if state.get('layout') != layout or state.get('format') != FORMAT:
        state = {'layout': layout, 'format': FORMAT}
    written = set()
    ui = json.load(open(os.path.join(track, 'ui', layout, 'ui_track.json'), encoding='utf-8'))
    version = ui.get('version', '0')

    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(track, 'models_%s.ini' % layout))
    files = []
    for s in cfg.sections():
        pos, rot = cfg[s].get('POSITION', '0,0,0'), cfg[s].get('ROTATION', '0,0,0')
        if any(float(x) for x in (pos + ',' + rot).split(',')):
            print('warning: %s is placed at %s turned %s, which is not carried over' % (cfg[s]['FILE'], pos, rot))
        files.append(cfg[s]['FILE'])

    forest_data = json.load(open(FOREST)) if os.path.exists(FOREST) and FOREST_MODEL in files else None
    materials, statics, dummies, ground = {}, [], {}, []
    trees = []             # where each tree stands, (x, y, foot), in BeamNG's axes
    cones = []             # where each loose cone stands, the same way
    boards = []            # the distance boards knocked about, as loose_boards gives them
    lamps, gantries = {}, []
    road_map = []          # the roads and kerbs from above, for the starts' maps
    models, hashes = {}, []
    for fn in files:
        raw = open(os.path.join(track, fn), 'rb').read()
        model = kn5io.Model(raw)
        stem = os.path.splitext(fn)[0]
        h = hashlib.sha1(raw).hexdigest()
        fresh = state.get(stem) != h
        models[fn] = model
        hashes.append(h)
        dummies.update(model.dummies)
        # the textures, in a folder of their own: two models may hold different ones of a name
        tex_dir = os.path.join(shapes, 'textures', stem)
        written.add(tex_dir)
        normals = {m['tex'].get('txNormal') for m in model.materials} - {None}
        if fresh:
            if os.path.exists(tex_dir):
                shutil.rmtree(tex_dir)
            os.makedirs(tex_dir)
            see_through = {m['tex'].get('txDiffuse') for m in model.materials if m['alpha_tested'] or m['blend'] == 1}
            for name, block in model.textures.items():
                data = payload(block)
                out_file = os.path.join(tex_dir, texture_file(name, data, name in normals))
                if out_file.endswith('.png') and not name.lower().endswith('.png'):
                    img = Image.open(io.BytesIO(data))
                    (img.convert('RGB') if name in normals else img).save(out_file)
                else:
                    open(out_file, 'wb').write(data)
                if name in see_through:
                    img = Image.open(io.BytesIO(data))
                    alpha = img.getchannel('A') if 'A' in img.getbands() else Image.new('L', img.size, 255)
                    alpha.save(os.path.join(tex_dir, opacity_file(name)))
        data_of = {name: payload(block) for name, block in model.textures.items()}
        tex_path = lambda name: '/levels/%s/art/shapes/textures/%s/%s' % (LEVEL, stem, texture_file(name, data_of.get(name), name in normals))

        def material(mi, surface=None, hidden=False):
            m = model.materials[mi]
            name = re.sub(r'[^A-Za-z0-9_]', '_', '%s_%s' % (stem, m['name'])) + ('_' + surface if surface else '') + ('_hidden' if hidden else '')
            if name not in materials:
                stage = {}
                if hidden:
                    stage['baseColorFactor'] = [1, 1, 1, 0]
                    stage['opacityFactor'] = 0
                else:
                    # rough and not metal, as AC's materials: BeamNG's default shines like glass
                    stage.update(roughnessFactor=1., metallicFactor=0.)
                    if 'txDiffuse' in m['tex']:
                        stage['baseColorMap'] = tex_path(m['tex']['txDiffuse'])
                        # BeamNG takes what shows through from its own map, not the colour's alpha
                        if (m['alpha_tested'] or m['blend'] == 1) and m['tex']['txDiffuse'] in model.textures:
                            stage['opacityMap'] = '/levels/%s/art/shapes/textures/%s/%s' % (LEVEL, stem, opacity_file(m['tex']['txDiffuse']))
                    if 'txNormal' in m['tex']:
                        stage['normalMap'] = tex_path(m['tex']['txNormal'])
                mat = {'name': name, 'mapTo': name, 'class': 'Material', 'persistentId': str(uuid.uuid4()),
                       'version': 1.5, 'Stages': [stage, {}, {}, {}]}
                if m['alpha_tested']:
                    mat.update(alphaTest=True, alphaRef=100)
                if m['blend'] == 1 or hidden:
                    mat['translucent'] = True
                if surface:
                    # a mesh's material names its ground in groundType (groundmodelName is the
                    # terrain's), or BeamNG drives it as its default, asphalt
                    mat['groundType'] = GROUND.get(surface, 'ASPHALT')
                materials[name] = mat
            return name

        parts = {'solid': [], 'walls': [], 'seen': []}
        seen = []
        feet, drawn = loose_cones(model)
        cones += feet
        boards += loose_boards(model)
        if not beamng_races.BOARD_TEXTURE:
            tex = next((n for n in model.textures if n.lower().startswith('markerboards')), None)
            if tex:
                beamng_races.BOARD_TEXTURE = payload(model.textures[tex])
        for m in model.meshes:
            arrays = mesh_arrays(m)
            if forest_data and fn == FOREST_MODEL and m['name'].startswith(FOREST_MESHES):
                # BeamNG's forest stands these trees
                continue
            if arrays is None or m['name'].startswith('AC_POBJECT') or is_loose(m, drawn):
                # the loose cones are BeamNG's own, which the level's Lua spawns
                continue
            key = PHYSICS.match(m['name'])
            surface = key.group(1) if key else None
            if surface == 'ROAD' and model.materials[m['mat']]['name'] in KERB_MATERIALS:
                surface = 'KERB'
            if surface and not (m['visible'] and m['renderable']):
                parts['walls'].append((m['name'], material(m['mat'], surface, hidden=True)) + arrays)
            elif surface:
                parts['solid'].append((m['name'], material(m['mat'], surface)) + arrays)
                ground.append(arrays[0][arrays[3]])
                if surface in ('ROAD', 'KERB'):
                    road_map.append(arrays[0][arrays[3]][:, :, :2])
            elif m['visible'] and m['renderable'] and (beamng_board.LAMPS.match(m['name']) or beamng_board.FLAGS.match(m['name'])):
                # the gantry's lamps: shapes of their own, which the level's Lua lights
                P = arrays[0]
                centre = (P.min(0) + P.max(0)) / 2
                shape = '%s_%s.dae' % (stem, m['name'])
                written.add(os.path.join(shapes, shape))
                if fresh or not os.path.exists(os.path.join(shapes, shape)):
                    write_dae(os.path.join(shapes, shape), [(m['name'], 'velopark_lamp', P - centre) + arrays[1:]], sizes=[1])
                statics.append(dict(name=shape[:-4], shapeName='/levels/%s/art/shapes/%s' % (LEVEL, shape),
                                    collisionType='None', position=centre,
                                    extra={'useInstanceRenderData': True, 'instanceColor': [0, 0, 0, 1]}))
                lamps.setdefault(stem, []).append(m['name'])
            elif m['visible'] and m['renderable']:
                seen.append((m, (m['name'], material(m['mat'])) + arrays))
                if model.materials[m['mat']].get('shader') == 'ksTree':
                    trees += tree_feet(arrays[0], arrays[3])
        # AC's far-only stand-ins (the tyre stacks' lidded copies), each with the near mesh it
        # takes over from: a shape of their own, as BeamNG picks a shape's detail by how big
        # the whole shape is on screen
        swaps = []
        for far, fm in seen:
            if far['lod_in'] <= 0:
                continue
            fc = (fm[2].min(0) + fm[2].max(0)) / 2
            near = next(((m, x) for m, x in seen if m['lod_in'] <= 0 and m['lod_out'] == far['lod_in']
                         and m['mat'] == far['mat'] and not any(x is s[1] for s in swaps)
                         and np.linalg.norm((x[2].min(0) + x[2].max(0)) / 2 - fc) < .5), None)
            if near:
                swaps.append((far['lod_in'], near[1], fm))
        swapped = {id(x) for _, n, f in swaps for x in (n, f)}
        # the far-only ones with no near one are left out: they would be drawn over what is near
        parts['seen'] = [x for m, x in seen if m['lod_in'] <= 0 and id(x) not in swapped]
        for part, meshes in parts.items():
            if not meshes:
                continue
            shape = '%s_%s.dae' % (stem, part)
            written.add(os.path.join(shapes, shape))
            if fresh or not os.path.exists(os.path.join(shapes, shape)):
                write_dae(os.path.join(shapes, shape), meshes, collide=part != 'seen')
            statics.append(dict(name='%s_%s' % (stem, part), shapeName='/levels/%s/art/shapes/%s' % (LEVEL, shape),
                                collisionType='None' if part == 'seen' else 'Collision Mesh'))
            print('%-28s %-5s %4d meshes, %7d triangles%s' % (fn, part, len(meshes), sum(len(x[5]) for x in meshes),
                                                               '' if fresh else ', as before'))
        for dist, near, far in swaps:
            P = np.r_[near[2], far[2]]
            centre = (P.min(0) + P.max(0)) / 2
            radius = float(np.linalg.norm(P - centre, axis=1).max())
            # BeamNG measures the distance to the shape's origin, put at the meshes' middle; the
            # near one is drawn till even the closest end of a long row is as far as AC swaps it
            meshes = [x[:2] + (x[2] - centre,) + x[3:] for x in (near, far)]
            shape = '%s_lod_%s.dae' % (stem, re.sub(r'[^A-Za-z0-9_]', '_', near[0]))
            written.add(os.path.join(shapes, shape))
            if fresh or not os.path.exists(os.path.join(shapes, shape)):
                write_dae(os.path.join(shapes, shape), meshes, sizes=[max(2, round(radius * SCREEN / (dist + radius))), 1])
            statics.append(dict(name=shape[:-4], shapeName='/levels/%s/art/shapes/%s' % (LEVEL, shape),
                                collisionType='None', position=centre))
        if swaps:
            print('%-28s %-5s %4d shapes, near %7d, far %7d triangles%s' % (
                fn, 'lod', len(swaps), sum(len(n[5]) for _, n, _ in swaps), sum(len(f[5]) for _, _, f in swaps),
                '' if fresh else ', as before'))
        if any(x['name'] == beamng_board.BOARD for x in model.meshes):
            names = sorted(lamps.get(stem, []))
            gantries.append(beamng_board.gantry(
                stem, model, to_beamng, LIFT,
                lamps=['%s_%s' % (stem, n) for n in sorted((n for n in names if beamng_board.LAMPS.match(n)),
                                                          key=lambda n: int(beamng_board.LAMPS.match(n).group(2)))],
                flags=['%s_%s' % (stem, n) for n in names if beamng_board.FLAGS.match(n)]))
        state[stem] = h
    # grass where AC grows its own, made again when any model changed
    grass_key = hashlib.sha1((' '.join(hashes) + ' all layouts %r' % [beamng_grass.BAND, beamng_grass.STEP, beamng_grass.TILE,
                                                         beamng_grass.HEIGHT, beamng_grass.WIDTH, beamng_grass.GREEN]).encode()).hexdigest()
    grass_tex = os.path.join(shapes, 'textures', 'grass')
    written.add(grass_tex)
    old_tiles = sorted(f for f in os.listdir(shapes) if f.startswith('grass_') and f.endswith('.dae'))
    if state.get('grass') != grass_key or not old_tiles:
        for f in old_tiles:
            os.remove(os.path.join(shapes, f))
        os.makedirs(grass_tex, exist_ok=True)
        beamng_grass.texture(os.path.join(grass_tex, 'blades.png'))
        # along every layout's racing line, as the level starts them all, not the one it is made from
        lines = []
        for lay in sorted(os.listdir(track)):
            ai = os.path.join(track, lay, 'ai', 'fast_lane.ai')
            if os.path.exists(ai):
                b = open(ai, 'rb').read()
                lines.append(np.frombuffer(b, dtype='<f4', count=int.from_bytes(b[4:8], 'little') * 5, offset=16).reshape(-1, 5)[:, [0, 2]])
        line = np.unique(np.round(np.concatenate(lines), 1), axis=0)
        at = beamng_grass.places(track, models, line)
        for (i, j), (P, N, U, F) in beamng_grass.clumps(at, to_beamng).items():
            write_dae(os.path.join(shapes, 'grass_%d_%d.dae' % (i, j)), [('grass', 'velopark_grass', P, N, U, F)])
        state['grass'] = grass_key
        print('grass: %d clumps' % len(at))
    for f in sorted(os.listdir(shapes)):
        if f.startswith('grass_') and f.endswith('.dae'):
            written.add(os.path.join(shapes, f))
            statics.append(dict(name=f[:-4], shapeName='/levels/%s/art/shapes/%s' % (LEVEL, f), collisionType='None'))
    tex = '/levels/%s/art/shapes/textures/grass/' % LEVEL
    materials['velopark_grass'] = {'name': 'velopark_grass', 'mapTo': 'velopark_grass', 'class': 'Material',
                                   'persistentId': str(uuid.uuid5(uuid.NAMESPACE_URL, LEVEL + '/grass')), 'version': 1.5,
                                   'Stages': [{'roughnessFactor': 1., 'metallicFactor': 0., 'baseColorMap': tex + 'blades.png',
                                               'opacityMap': tex + 'blades_opacity.png'}, {}, {}, {}],
                                   'alphaTest': True, 'alphaRef': 90, 'doubleSided': True, 'castShadows': False}
    # the trees' trunks, unseen, made below once the ground is known
    trunk_shape = os.path.join(shapes, 'velopark_trunks.dae')
    written.add(trunk_shape)
    materials['velopark_trunk_hidden'] = {'name': 'velopark_trunk_hidden', 'mapTo': 'velopark_trunk_hidden', 'class': 'Material',
                                          'persistentId': str(uuid.uuid5(uuid.NAMESPACE_URL, LEVEL + '/trunk')), 'version': 1.5,
                                          'Stages': [{'baseColorFactor': [1, 1, 1, 0], 'opacityFactor': 0}, {}, {}, {}],
                                          'translucent': True, 'groundType': GROUND['TREE']}
    if forest_data:
        materials.update(forest(level, forest_data, {n: payload(b) for n, b in models[FOREST_MODEL].textures.items()}))
    else:
        for d in (os.path.join(shapes, 'forest'), os.path.join(level, 'forest'), os.path.join(level, 'art', 'forest')):
            if os.path.exists(d):
                shutil.rmtree(d)
    # the gantry board's characters and the lamps' glow
    beamng_board.write_shapes(shapes, write_dae)
    materials.update(beamng_board.materials(LEVEL))
    # the shapes and textures of models the layout no longer loads
    for f in os.listdir(shapes):
        if f.endswith('.dae') and os.path.join(shapes, f) not in written:
            os.remove(os.path.join(shapes, f))
    for d in os.listdir(os.path.join(shapes, 'textures')):
        if os.path.join(shapes, 'textures', d) not in written:
            shutil.rmtree(os.path.join(shapes, 'textures', d))
    state = {k: v for k, v in state.items() if k in ('layout', 'format', 'grass') or k + '.kn5' in files}
    json.dump(materials, open(os.path.join(shapes, 'main.materials.json'), 'w'), indent=1)

    # a start for each layout: its grid's first slot (the hotlap's own start for a hotlap),
    # facing the way its lap runs; AC drops the kart from a few metres up onto the road,
    # BeamNG starts it where it is placed: just over the road
    t = np.concatenate(ground)
    a, b, c = t[:, 0, :2], t[:, 1, :2], t[:, 2, :2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    # not those standing on edge, which have no inside seen from above
    flat = np.abs(d) > 1e-6
    d = np.where(flat, d, 1.)

    def on_road(at):
        l1 = ((b[:, 1] - c[:, 1]) * (at[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (at[1] - c[:, 1])) / d
        l2 = ((c[:, 1] - a[:, 1]) * (at[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (at[1] - c[:, 1])) / d
        l3 = 1 - l1 - l2
        hit = flat & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        z = l1 * t[:, 0, 2] + l2 * t[:, 1, 2] + l3 * t[:, 2, 2]
        below = z[hit & (z < at[2] + 1)]
        return np.r_[at[:2], below.max()] if len(below) else at

    def ground_at(x, y):
        p = on_road(np.array([x, y, 1000.]))
        return float(p[2]) if p[2] < 900 else None
    trunk = trunks(trees, ground_at)
    if trunk:
        write_dae(trunk_shape, [('trunks', 'velopark_trunk_hidden') + trunk], collide=True)
        statics.append(dict(name='velopark_trunks', shapeName='/levels/%s/art/shapes/velopark_trunks.dae' % LEVEL,
                            collisionType='Collision Mesh'))
        print('%-28s %-5s %4d trunks' % ('trees', 'walls', len(trunk[3]) // 24))
    elif os.path.exists(trunk_shape):
        os.remove(trunk_shape)
    def on_ground(foot):
        """A loose cone's foot on the ground under it, in the level's place."""
        p = on_road(np.array(foot) + (0, 0, .5))
        return tuple(float(x) for x in p + (0, 0, LIFT))
    # the level's own ground models
    gm_dir = os.path.join(level, 'groundModels')
    os.makedirs(gm_dir, exist_ok=True)
    json.dump(GROUND_MODELS, open(os.path.join(gm_dir, 'velopark.json'), 'w'), indent=1)

    layouts = sorted(x for x in os.listdir(track) if os.path.exists(os.path.join(track, x, 'ai', 'fast_lane.ai'))
                     and os.path.exists(os.path.join(track, 'models_%s.ini' % x)))
    spawns, roads, races = [], {}, []
    every_cone, every_board = [], []    # every layout's, for free roam
    for lay in layouts:
        lc = configparser.ConfigParser()
        lc.read(os.path.join(track, 'models_%s.ini' % lay))
        found, lay_cones, lay_boards = {}, [], []
        for sec in lc.sections():
            fn = lc[sec]['FILE']
            m = models.get(fn) or kn5io.Model(os.path.join(track, fn))
            found.update(m.dummies)
            lay_cones += loose_cones(m)[0]
            lay_boards += loose_boards(m)
        every_cone += lay_cones
        every_board += lay_boards
        line, sides = racing_line(os.path.join(track, lay, 'ai', 'fast_lane.ai'))
        roads[lay] = (line, sides)

        def placed(name):
            """Where a dummy of the layout is, on the road, and which way its lap runs there."""
            w = found.get(name)
            if w is None:
                return None
            at = on_road(to_beamng(w[3, :3]))
            fwd = to_beamng(w[2, :3])
            k = np.linalg.norm(line[:, :2] - at[:2], axis=1).argmin()
            if (line[(k + 3) % len(line)] - line[k])[:2] @ fwd[:2] < 0:
                fwd = -fwd
            fwd = np.r_[fwd[:2] / np.linalg.norm(fwd[:2]), 0]
            return at, fwd
        hotlap = placed('AC_HOTLAP_START_0')
        grid = placed('AC_START_0')
        # the whole grid, its slots in order, for the AI races' racers
        slots = []
        while placed('AC_START_%d' % len(slots)) is not None:
            slots.append(placed('AC_START_%d' % len(slots)))
        start = hotlap if 'hotlap' in lay and hotlap else grid
        if start is None:
            continue
        at, fwd = start
        lui = json.load(open(os.path.join(track, 'ui', lay, 'ui_track.json'), encoding='utf-8'))
        title = re.sub(r'^Techspeed FuelTech Velopark\s*', '', lui.get('name', lay)) or lay
        spawns.append(('spawn_' + lay, title, at, fwd[:2]))
        gates = []
        while 'AC_TIME_%d_L' % len(gates) in found and 'AC_TIME_%d_R' % len(gates) in found:
            gates.append(tuple(to_beamng(found['AC_TIME_%d_%s' % (len(gates), x)][3, :3]) for x in 'LR'))
        # not the hotlap: it is its layout's race from a rolling start, which every race offers
        if gates and 'hotlap' not in lay:
            # the car's front a metre on from where AC puts the kart
            path, length = beamng_races.race_path(line, sides, gates, (at + fwd, fwd), hotlap or start, on_road, LIFT)
            races.append(dict(layout=lay, title='Techspeed FuelTech Velopark ' + title, description=lui.get('description', ''),
                              author=lui.get('author', ''), path=path, length=length,
                              trigger=(at + (0, 0, LIFT), fwd), preview=os.path.join(level, 'spawn_%s_preview.jpg' % lay),
                              cones=[on_ground(c) for c in lay_cones], boards=lay_boards,
                              grid=[(at_ + (0, 0, LIFT), f_) for at_, f_ in slots]))
    if 'spawn_' + layout not in [x[0] for x in spawns]:
        raise SystemExit('layout %s has no start' % layout)
    # the loose cones and distance boards of every layout, which the level's Lua spawns in free
    # roam, each once where layouts share it (within SAME metres); a race spawns its own
    # layout's instead
    free_cones, free_boards = [], []
    for cone in every_cone:
        if all(np.hypot(cone[0] - q[0], cone[1] - q[1]) > SAME for q in free_cones):
            free_cones.append(cone)
    for board in every_board:
        if all(np.linalg.norm(board[0].mean(0) - q[0].mean(0)) > SAME for q in free_boards):
            free_boards.append(board)
    beamng_races.cone_prefab(os.path.join(level, 'cones.prefab.json'), [on_ground(c) for c in free_cones], LEVEL + '/cones',
                             free_boards)
    print('%-28s %-5s %4d loose cones and %d boards of every layout' % ('cones', 'lua', len(free_cones), len(free_boards)))
    # each start's picture: a map of where it is on the track, its layout's racing line on it
    road_map = np.concatenate(road_map)
    frame = (road_map.reshape(-1, 2).min(0), road_map.reshape(-1, 2).max(0))
    for name, title, at, fwd in spawns:
        beamng_races.map_preview(os.path.join(level, name + '_preview.jpg'), road_map, roads[name[len('spawn_'):]][0][:, :2],
                                 at, fwd, title, frame)
    # the layouts' racing lines as BeamNG's road network, which its map draws and its AI drives,
    # and the paths and pit lanes off them, which it draws as well; each race's checkpoints on its
    # own layout's road of it
    waypoints, segments, laps = road_network(roads, side_roads(track, layouts, on_road))
    route_races(races, laps, waypoints, segments)
    beamng_races.write(out, LEVEL, races)
    # the gantries' boards, each showing the races that start under it, and the Lua that runs them
    beamng_board.write_config(level, gantries, {r['layout']: next(p['pos'] for p in r['path']['pathnodes']
                                                                   if p['oldId'] == r['path']['startNode']) for r in races})
    for top in ('lua', 'scripts'):
        if os.path.exists(os.path.join(out, top)):
            shutil.rmtree(os.path.join(out, top))
        shutil.copytree(os.path.join(ROOT, 'tools', 'beamng', top), os.path.join(out, top))
    # the cones: each layout's one vehicle, a part of the cones prop of the level's own
    beamng_races.write_cone_sets(out)

    def obj(cls, name, **kw):
        return dict({'class': cls, 'name': name, 'persistentId': str(uuid.uuid4()), '__parent': 'MissionGroup'}, **kw)
    items = [obj('LevelInfo', 'theLevelInfo', gravity=-9.81, visibleDistance=1500, canvasClearColor=[0, 0, 0, 255],
                 globalEnviromentMap='BNG_Sky_02_cubemap'),
             obj('TimeOfDay', 'tod', time=0.1, play=False, dayLength=1200, azimuthOverride=0),
             obj('ScatterSky', 'sunsky', position=[0, 0, 0]),
             ]
    # BeamNG's vehicles face their -y; the rotationMatrix lists the axes x, y, z in turn
    for name, title, at, fwd in spawns:
        items.append(obj('SpawnSphere', name, dataBlock='SpawnSphereMarker', position=[float(x) for x in at + (0, 0, LIFT + .5)],
                         rotationMatrix=[float(x) for x in (-fwd[1], fwd[0], 0, -fwd[0], -fwd[1], 0, 0, 0, 1)], radius=1))
    items += [obj('BeamNGWaypoint', name, position=[float(x) for x in at + (0, 0, LIFT)], scale=[float(r)] * 3)
              for name, at, r in waypoints]
    items += lights(track, obj)
    if forest_data:
        mid = to_beamng(np.mean([t[1:4] for t in forest_data['trees']], axis=0)) + (0, 0, LIFT)
        items += [obj('Forest', 'theForest', lodReflectScalar=.15),
                  obj('ForestWindEmitter', 'velopark_wind', position=[float(q) for q in mid], strength=.15)]
    items += [obj('TSStatic', st['name'], shapeName=st['shapeName'], collisionType=st['collisionType'],
                  position=[float(x) for x in np.asarray(st.get('position', (0, 0, 0))) + (0, 0, LIFT)],
                  decalType='Collision Mesh' if st['collisionType'] != 'None' else 'None', **st.get('extra', {}))
              for st in statics]
    main = os.path.join(level, 'main')
    os.makedirs(os.path.join(main, 'MissionGroup'), exist_ok=True)
    json.dump({'class': 'SimGroup', 'name': 'MissionGroup', 'persistentId': str(uuid.uuid4()), 'enabled': '1'},
              open(os.path.join(main, 'items.level.json'), 'w'))
    open(os.path.join(main, 'MissionGroup', 'items.level.json'), 'w').write(''.join(json.dumps(i) + '\n' for i in items))

    shutil.copy(os.path.join(track, 'ui', layout, 'preview.png'), os.path.join(level, '%s_preview.png' % LEVEL))
    for f in os.listdir(level):
        if f.startswith('spawn_') and f.endswith('_preview.jpg') and f[:-len('_preview.jpg')] not in [x[0] for x in spawns]:
            os.remove(os.path.join(level, f))
    json.dump({'title': 'Techspeed FuelTech Velopark', 'description': ui.get('description', ''), 'authors': ui.get('author', ''),
               'previews': ['%s_preview.png' % LEVEL], 'size': [1024, 1024], 'country': ui.get('country', ''),
               # a kart track has no road traffic, and BeamNG's freeroam fills any level with it
               # unless told so: the road network is for the map and the AI of the cars one picks
               'supportsTraffic': False,
               'spawnPoints': [{'objectname': name, 'translationId': title, 'preview': name + '_preview.jpg'}
                               for name, title, _, _ in spawns],
               'defaultSpawnPointName': 'spawn_' + layout},
              open(os.path.join(level, 'info.json'), 'w'), indent=1)
    json.dump({'segments': {lay: {'nodes': nodes, 'drivability': 1, 'autoJunction': False} for lay, nodes in segments.items()}},
              open(os.path.join(level, 'map.json'), 'w'), indent=1)

    json.dump(state, open(state_path, 'w'), indent=1)
    if not pack:
        print('%d materials, %d starts, %d waypoints; level in %s' % (len(materials), len(spawns), len(waypoints), level))
        return level
    # packed beside the final name, so a failed pack leaves the old zip intact
    dist = os.path.join(os.path.dirname(out), 'dist')
    os.makedirs(dist, exist_ok=True)
    zip_path = os.path.join(dist, 'techspeed-fueltech-velopark-%s-beamng.zip' % version)
    with zipfile.ZipFile(zip_path + '.new', 'w', zipfile.ZIP_DEFLATED) as z:
        for top in (level, os.path.join(out, 'gameplay'), os.path.join(out, 'lua'), os.path.join(out, 'scripts'), os.path.join(out, 'vehicles')):
            for base, _, fs in os.walk(top):
                for f in fs:
                    full = os.path.join(base, f)
                    z.write(full, os.path.relpath(full, out))
    os.replace(zip_path + '.new', zip_path)
    print('%d materials, %d starts, %d waypoints; packed %s (%.1f MB)' % (len(materials), len(spawns), len(waypoints), zip_path, os.path.getsize(zip_path) / 2 ** 20))
    return zip_path


if __name__ == '__main__':
    main()
