"""Races for the BeamNG.drive level: a time trial mission for each layout, which BeamNG's
freeroam shows as a marker at the layout's start. Driving into it offers the race (with a list
to pick from where layouts share a start); starting it puts the car on the grid, counts down and
times the laps.

Each race is a closed path of checkpoints along the layout's racing line, as BeamNG's own race
tracks have: the start and finish where AC times the lap (its AC_TIME_0 gate), a split at each
of AC's other timing gates, and a checkpoint every CHECKPOINT_STEP metres between them, so a lap
cannot be cut short. A checkpoint spans the road there, wherever on it the car runs. Each has a
place to put a car back on the road ("recover"). The standing start is the grid's first slot (the
hotlap's own start for a hotlap); the rolling start, which the race's settings offer, is AC's
hotlap start, where a flying lap begins.
"""
import io, json, os, re, shutil, time, uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHECKPOINT_STEP = 40.    # metres along the racing line between checkpoints
HALF_ROAD = 6.           # metres either side of the racing line a checkpoint reaches, at most
# (AC's side distances run on over the run-offs)
LAPS = 3
# the lap times for one to five stars, from the lap's length: metres per second on average
STAR_SPEEDS = (11, 13, 15, 17, 19)


# the cones AC knocks about, as cones of BeamNG's cones prop, in orange and white as AC's are.
# Neither of its cones is AC's: the large one, 0.92 m high and 4.2 kg, all but flips a kart that
# runs into it; so each is the cone made AC's size, CONE_WIDTH wide and CONE_HEIGHT high, and
# CONE_MASS as a plastic cone of that size, sliding on the asphalt. A vehicle of its own each, as
# BeamNG's time trials place theirs, a layout's 75 cones took a minute to spawn as its race
# started and stalled the frames as they woke; so all the cones of a layout are one vehicle, a
# part of the cones prop of the level's own (written by write_cone_sets): each cone still its
# own nodes and beams, apart from the others, knocked about on its own
CONE_WIDTH, CONE_HEIGHT, CONE_MASS = .4, .7, 1.2
CONE_FRICTION = .7
CONE = {'dataBlock': 'default_vehicle', 'jbeam': 'cones',
        'color': [0.853, 0.388, 0.046, 2], 'colorPalette0': [0.95, 0.95, 0.95, 1], 'colorPalette1': [0.853, 0.388, 0.046, 2],
        'metallicPaintData': [[0, 0.344, 0, 0], [1, 0.526, 0, 0], [0, 0.344, 0, 0], None],
        'paints': '{{["baseColor"]={0.853,0.388,0.046,2},["roughness"]=0.344,["clearcoat"]=0,["metallic"]=0,["clearcoatRoughness"]=0},'
                  '{["baseColor"]={0.95,0.95,0.95,1},["roughness"]=0.526,["clearcoat"]=0,["metallic"]=1,["clearcoatRoughness"]=0},'
                  '{["baseColor"]={0.853,0.388,0.046,2},["roughness"]=0.344,["clearcoat"]=0,["metallic"]=0,["clearcoatRoughness"]=0}}',
        # a cone spawned is not the one the player drives
        'autoEnterVehicle': 'false'}
CONE_DROP = .03          # metres over the ground a cone is put, to settle on it


CONE_SETS = {}          # the sets the prefabs place, for write_cone_sets: name: ([cone foot], [board])
BOARD_TEXTURE = None     # the distance boards' texture, as the AC track has it
BOARD_MASS = 2.5         # kilograms, a distance board, a plastic panel on its foot


def cone_prefab(path, feet, tag, boards=()):
    """A prefab of the loose cones standing on feet, [(x, y, z)] in the level's place, and the
    distance boards (as export_beamng.loose_boards gives them), as one vehicle, its first cone's
    foot (or board's) where it is placed; written to path as BeamNG writes them: an object a line,
    in the group the last line makes. tag names the set."""
    name = re.sub(r'[^a-z0-9]+', '_', tag.lower()).strip('_')
    origin = np.array(feet[0], float) if len(feet) else boards[0][0].mean(0) * [1, 1, 0] + [0, 0, boards[0][0][:, 2].min()]
    CONE_SETS[name] = ([np.array(f, float) - origin for f in feet],
                       [(box - origin, (P - origin, N, U, F)) for box, (P, N, U, F) in boards])
    lines = [dict({'internalName': 'cones', 'class': 'BeamNGVehicle', 'persistentId': str(uuid.uuid5(uuid.NAMESPACE_URL, tag)),
                   '__parent': 'SimGroup_', 'position': [float(x) for x in origin + (0, 0, CONE_DROP)],
                   'rotationMatrix': [1, 0, 0, 0, 1, 0, 0, 0, 1]}, **CONE, partConfig='vehicles/cones/%s.pc' % name)]
    lines.append({'name': 'SimGroup_', 'class': 'SimGroup', 'groupPosition': '0 0 0'})
    open(path, 'w').write(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in lines))


def write_cone_sets(out):
    """The parts of the cones prop the prefabs' vehicles are, and a config for each, in
    out/vehicles/cones, made again each time: in each, a cone at each foot of its set, as BeamNG's
    own cone is built (four nodes round its foot, one at its top, beams between them, its mesh
    on them), each its own group, so each mesh bends with its own cone only."""
    if os.path.exists(os.path.join(out, 'vehicles')):
        shutil.rmtree(os.path.join(out, 'vehicles'))
    folder = os.path.join(out, 'vehicles', 'cones')
    os.makedirs(folder)
    w = CONE_WIDTH / 2
    corner = CONE_MASS * .85 / 4
    # BeamNG's small cone, 0.3 m wide and 0.47 m high, made this one's size
    scale = {'x': CONE_WIDTH / .3, 'y': CONE_WIDTH / .3, 'z': CONE_HEIGHT / .47}
    parts = {}
    meshes = []
    for name, (feet, boards) in sorted(CONE_SETS.items()):
        flex = [['mesh', '[group]:']]
        nodes = [['id', 'posX', 'posY', 'posZ'], {'frictionCoef': CONE_FRICTION}, {'nodeMaterial': '|NM_PLASTIC'},
                 {'collision': True}, {'selfCollision': False}]
        beams, limits, tris = [], [], []
        for k, f in enumerate(feet):
            g = 'cone%d' % k
            n = ['k%d_%s' % (k, s) for s in ('1r', '1l', '2r', '2l', '3')]
            p = [f + d for d in ((-w, w, 0), (w, w, 0), (-w, -w, 0), (w, -w, 0), (0, 0, CONE_HEIGHT))]
            flex += [{'pos': {'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])}}, ['cone_01a', [g], {'scale': scale}]]
            nodes += [{'group': g}, {'nodeWeight': corner}] + [[n[i]] + [round(float(c), 4) for c in p[i]] for i in range(4)]
            nodes += [{'nodeWeight': CONE_MASS * .15}, [n[4]] + [round(float(c), 4) for c in p[4]]]
            beams += [[n[a], n[b]] for a, b in ((0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))]
            limits += [[n[a], n[4]] for a in range(4)]
            tris += [{'group': g}] + [[n[a], n[b], n[c]] for a, b, c in ((4, 2, 3), (4, 0, 2), (4, 1, 0), (4, 3, 1), (1, 3, 2), (1, 2, 0))]
        # the boards: a node at each corner of its box, held square by beams between them all, its
        # mesh on them
        if boards:
            # the cones' offset (pos) carries on to what follows it: none for the boards, in place
            flex.append({'pos': {'x': 0, 'y': 0, 'z': 0}})
        for k, (box, (P, N, U, F)) in enumerate(boards):
            g = 'board%d' % k
            n = ['b%d_%d' % (k, j) for j in range(8)]
            mesh = 'tsvp_%s_board%d' % (name, k)
            meshes.append((mesh, P, N, U, F))
            flex.append(['tsvp_%s_board%d' % (name, k), [g]])
            nodes += [{'group': g}, {'nodeWeight': BOARD_MASS / 8}] + [[n[j]] + [round(float(c), 4) for c in box[j]] for j in range(8)]
            beams += [[n[a], n[b]] for a in range(8) for b in range(a + 1, 8)]
            # its box's faces, as the AC box's (tools/build_board_objects.py)
            tris += [{'group': g}] + [[n[a], n[b], n[c]] for a, b, c in
                                      ((0, 1, 2), (2, 3, 0), (4, 5, 6), (6, 7, 4), (0, 3, 5), (5, 4, 0), (7, 6, 2), (2, 1, 7),
                                       (1, 0, 4), (4, 7, 1), (3, 2, 6), (6, 5, 3))]
        nodes.append({'group': ''})
        tris.append({'group': ''})
        parts['cone_' + name] = {
            'information': {'authors': 'Techspeed FuelTech Velopark', 'name': 'Techspeed FuelTech Velopark cones (%s)' % name},
            'slotType': 'cone',
            'flexbodies': flex,
            'refNodes': [['ref:', 'back:', 'left:', 'up:'], ['k0_2r', 'k0_1r', 'k0_2l', 'k0_3'] if feet else ['b0_3', 'b0_0', 'b0_2', 'b0_1']],
            'nodes': nodes,
            'beams': [['id1:', 'id2:'],
                      {'beamPrecompression': 1, 'beamType': '|NORMAL', 'beamLongBound': 1, 'beamShortBound': 1},
                      {'beamSpring': 15000, 'beamDamp': 60}, {'beamDeform': 10000, 'beamStrength': 'FLT_MAX'},
                      {'deformLimitExpansion': 1.1}] + beams +
                     [{'beamType': '|BOUNDED', 'beamLongBound': 0, 'beamShortBound': 1}, {'beamSpring': 0, 'beamDamp': 0},
                      {'beamLimitSpring': 15000, 'beamLimitDamp': 60}, {'beamDeform': 'FLT_MAX', 'beamStrength': 'FLT_MAX'}] + limits +
                     [{'beamPrecompression': 1, 'beamType': '|NORMAL', 'beamLongBound': 1, 'beamShortBound': 1}],
            'triangles': [['id1:', 'id2:', 'id3:'], {'dragCoef': 50}] + tris,
        }
        json.dump({'format': 2, 'mainPartName': 'cones', 'model': 'cones', 'vars': {},
                   'parts': {'cone': 'cone_' + name, 'paint_design': 'cone_skin_single'}},
                  open(os.path.join(folder, name + '.pc'), 'w'), indent=1)
    json.dump(parts, open(os.path.join(folder, 'techspeed_fueltech_velopark_cones.jbeam'), 'w'), indent=1)
    if meshes:
        boards_dae(os.path.join(folder, 'techspeed_fueltech_velopark_boards.dae'), meshes)
        Image.open(io.BytesIO(BOARD_TEXTURE)).convert('RGB').save(os.path.join(folder, 'techspeed_fueltech_velopark_boards.png'))
        json.dump({'tsvp_marker_boards': {'name': 'tsvp_marker_boards', 'mapTo': 'tsvp_marker_boards', 'class': 'Material',
                                          'persistentId': str(uuid.uuid5(uuid.NAMESPACE_URL, 'techspeed_fueltech_velopark/boards')), 'version': 1.5,
                                          'Stages': [{'baseColorMap': '/vehicles/cones/techspeed_fueltech_velopark_boards.png',
                                                      'roughnessFactor': .8, 'metallicFactor': 0.}, {}, {}, {}]}},
                  open(os.path.join(folder, 'techspeed_fueltech_velopark_boards.materials.json'), 'w'), indent=1)
    CONE_SETS.clear()
    return folder


def boards_dae(path, meshes):
    """The boards' meshes for the flexbodies, each a node of its own name, in the vehicle's axes."""
    geo, nodes = [], []
    f = lambda a, fmt='%.5g': ' '.join(fmt % x for x in np.ravel(a))
    for k, (name, P, N, U, F) in enumerate(meshes):
        g = 'g%d' % k
        geo.append(f'''<geometry id="{g}" name="{name}"><mesh>
<source id="{g}-p"><float_array id="{g}-pa" count="{P.size}">{f(P)}</float_array><technique_common><accessor source="#{g}-pa" count="{len(P)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="{g}-n"><float_array id="{g}-na" count="{N.size}">{f(N, '%.4g')}</float_array><technique_common><accessor source="#{g}-na" count="{len(N)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="{g}-t"><float_array id="{g}-ta" count="{U.size}">{f(U, '%.6g')}</float_array><technique_common><accessor source="#{g}-ta" count="{len(U)}" stride="2"><param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
<vertices id="{g}-v"><input semantic="POSITION" source="#{g}-p"/><input semantic="NORMAL" source="#{g}-n"/></vertices>
<triangles material="tsvp_marker_boards" count="{len(F)}"><input semantic="VERTEX" source="#{g}-v" offset="0"/><input semantic="TEXCOORD" source="#{g}-t" offset="0" set="0"/><p>{' '.join(map(str, np.ravel(F)))}</p></triangles>
</mesh></geometry>''')
        nodes.append(f'<node id="{name}" name="{name}"><instance_geometry url="#{g}"><bind_material><technique_common>'
                     f'<instance_material symbol="tsvp_marker_boards" target="#tsvp_marker_boards"/></technique_common></bind_material></instance_geometry></node>')
    open(path, 'w', encoding='utf-8').write(f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_effects><effect id="tsvp_marker_boards-fx"><profile_COMMON><technique sid="common"><lambert/></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="tsvp_marker_boards" name="tsvp_marker_boards"><instance_effect url="#tsvp_marker_boards-fx"/></material></library_materials>
<library_geometries>{''.join(geo)}</library_geometries>
<library_visual_scenes><visual_scene id="scene">{''.join(nodes)}</visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>''')


MAP_SIZE = (800, 600)    # as BeamNG's own spawn previews
MAP_BACK, MAP_ROAD, MAP_LINE = (24, 30, 26), (92, 96, 102), (255, 118, 34)


def _font(size):
    for name in ('arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def map_preview(path, roads, line, at, fwd, title, frame):
    """A start's picture: the track's roads from above, north up, with the layout's racing line
    and an arrow at the start the way the lap runs, and its name. roads: (n, 3, 2) triangles,
    line: (n, 2), at, fwd: the start and its heading, frame: (min, max) corners, all in
    BeamNG's x, y. Drawn at twice the size and scaled down, for smooth edges."""
    k = 2
    w, h = MAP_SIZE[0] * k, MAP_SIZE[1] * k
    lo, hi = np.asarray(frame[0], float), np.asarray(frame[1], float)
    margin = 36 * k
    s = min((w - 2 * margin) / (hi[0] - lo[0]), (h - 2 * margin - 40 * k) / (hi[1] - lo[1]))
    mid = (lo + hi) / 2
    to_px = lambda p: np.c_[w / 2 + (np.asarray(p)[..., 0] - mid[0]) * s, h / 2 + 20 * k - (np.asarray(p)[..., 1] - mid[1]) * s]
    img = Image.new('RGB', (w, h), MAP_BACK)
    d = ImageDraw.Draw(img)
    for t in roads:
        d.polygon([tuple(q) for q in to_px(t)], fill=MAP_ROAD)
    pts = to_px(np.r_[line, line[:1]])
    d.line([tuple(q) for q in pts], fill=MAP_LINE, width=5 * k, joint='curve')
    # the start: a white arrow over a dark ring, pointing the way the lap runs
    c = to_px(at[:2])[0]
    f = np.array([fwd[0], -fwd[1]], float)
    f /= np.linalg.norm(f)
    side = np.array([-f[1], f[0]])
    r = 15 * k
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(20, 20, 20), outline='white', width=3 * k)
    tip, back = c + f * r * .75, c - f * r * .55
    d.polygon([tuple(tip), tuple(back + side * r * .55), tuple(c - f * r * .2), tuple(back - side * r * .55)], fill='white')
    d.text((margin, 20 * k), title, font=_font(34 * k), fill='white')
    img.resize(MAP_SIZE, Image.LANCZOS).save(path, quality=92)


def quat(fwd, up=(0, 0, 1.)):
    """The rotation BeamNG's race start positions take, as x, y, z, w: its y facing fwd, as
    BeamNG's quatFromDir(fwd, up) makes it. BeamNG's quaternions are the conjugates of the usual
    ones (its setFromDir lays the axes in the matrix's rows, and quat * vec3 turns the other way),
    so the usual quaternion of the turn, with x, y and z negated: given as it is, a start faced
    fwd mirrored about the north, the karts turned off their grid slots."""
    y = np.asarray(fwd, float)
    y = y / np.linalg.norm(y)
    x = np.cross(y, up)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    m = np.c_[x, y, z]
    w = np.sqrt(max(0., 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    if w > 1e-6:
        q = ((m[1, 2] - m[2, 1]) / (4 * w), (m[2, 0] - m[0, 2]) / (4 * w), (m[0, 1] - m[1, 0]) / (4 * w), w)
    else:
        # turned half round: about the axis the matrix leaves where it is
        k = int(np.argmax(np.diag(m)))
        axis = np.zeros(3)
        axis[k] = 1
        q = (*axis, 0.)
    return [float(v) for v in q]


def _floats(p):
    return [float(v) for v in p]


def race_path(line, sides, gates, standing, rolling, on_road, lift):
    """BeamNG's race path (race.race.json) for a racing line (n, 3) and its sides (n, 2) or None,
    in BeamNG's axes. gates: [(left, right)] AC's timing gates, the first the start and finish.
    standing, rolling: (position, forward) of the starts. Returns the path and its lap length."""
    n = len(line)
    step = np.linalg.norm(np.roll(line, -1, 0) - line, axis=1)
    s = np.r_[0, np.cumsum(step)]
    length = float(s[-1])

    def along(i):
        d = line[(i + 2) % n] - line[(i - 2) % n]
        d[2] = 0
        return d / np.linalg.norm(d)

    def across(i):
        """The road's middle and half width at a point of the line."""
        f = along(i)
        left = np.array([-f[1], f[0], 0])
        sl, sr = (np.minimum(sides[i], HALF_ROAD) if sides is not None else (HALF_ROAD / 2,) * 2)
        mid = line[i] + left * (sl - sr) / 2
        return on_road(mid), (sl + sr) / 2

    # the gates where the racing line crosses them; the start's first, and the line from it on
    picks = []
    for g, (l, r) in enumerate(gates):
        mid = (l + r) / 2
        i = int(np.linalg.norm(line[:, :2] - mid[:2], axis=1).argmin())
        picks.append((i, g, float(np.linalg.norm(l[:2] - r[:2]) / 2)))
    first = picks[0][0]
    pos = lambda i: (s[i] - s[first]) % length
    nodes = sorted(picks, key=lambda x: pos(x[0]))
    # checkpoints between them, at most CHECKPOINT_STEP apart
    full = []
    for k, (i, g, half) in enumerate(nodes):
        full.append((i, g, half))
        a = pos(i)
        b = pos(nodes[k + 1][0]) if k + 1 < len(nodes) else length
        extra = int(np.ceil((b - a) / CHECKPOINT_STEP)) - 1
        for e in range(1, extra + 1):
            at = (s[first] + a + (b - a) * e / (extra + 1)) % length
            full.append((int(np.searchsorted(s, at)) % n, None, None))

    ids = iter(range(1, 10000))
    pathnodes, starts, segments = [], [], []

    def start_position(name, group, at, fwd):
        sid = next(ids)
        starts.append({'group': group, 'name': name, 'oldId': sid, 'pos': _floats(on_road(at) + (0, 0, lift)),
                       'rot': quat(fwd)})
        return sid

    for k, (i, g, half) in enumerate(full):
        mid, width = across(i)
        if g is not None:
            # a gate of AC's: as wide as AC draws it
            radius = max(width, min(half, 2 * HALF_ROAD)) + 1
        else:
            radius = width + 1
        nid = next(ids)
        name = 'Start' if k == 0 else ('Split %d' % g if g is not None else 'Checkpoint %d' % k)
        rec = start_position('%s Recovery Forward' % name, 'recovery', line[(i - 4) % n], along((i - 4) % n))
        pathnodes.append({'customFields': {'names': {}, 'tags': {}, 'types': {}, 'values': {}}, 'mode': 'manual',
                          'name': name, 'navRadiusScale': 1, 'normal': _floats(along(i)), 'oldId': nid,
                          'pos': _floats(mid + (0, 0, lift)), 'radius': float(radius), 'recovery': rec,
                          'reverseRecovery': -1, 'sidePadding': [1, 1], 'useAsSplit': g is not None, 'visible': True})
    for k, p in enumerate(pathnodes):
        segments.append({'capsules': {}, 'from': p['oldId'], 'mode': 'waypoint', 'name': 'Segment %d' % (k + 1),
                         'oldId': next(ids), 'to': pathnodes[(k + 1) % len(pathnodes)]['oldId']})
    standing_id = start_position('Start Position fwd', 'start', *standing)
    rolling_id = start_position('Start Position fwd roll', 'start', *rolling)
    # where the car stops after the finish: a little on from the line
    after = int(np.searchsorted(s, (s[first] + 30) % length)) % n
    start_position('finalPosFwd', 'start', across(after)[0], along(after))
    path = {'authors': '', 'classification': {'allowRollingStart': True, 'branching': False, 'closed': True,
                                              'reversible': False},
            'date': 0, 'defaultLaps': LAPS, 'defaultStartPosition': standing_id, 'description': '', 'difficulty': 24,
            'endNode': -1, 'forwardPrefabs': {}, 'hideMission': False, 'name': 'Path', 'pacenotes': {},
            'pathnodes': pathnodes, 'prefabs': {}, 'reversePrefabs': {}, 'reverseStartPosition': -1,
            'rollingReverseStartPosition': -1, 'rollingStartPosition': rolling_id, 'segments': segments,
            'simplifyAiPath': False, 'startNode': pathnodes[0]['oldId'], 'startPositions': starts}
    return path, length


# when the races were made, which BeamNG's challenge list shows as their date (0 shows 1969)
DATE = int(time.time())

# the AI races: the racers, karts as the player's own, on the layout's grid
AI_RACERS = 9            # racers but the player, at first
AI_MAX = 35              # the most the race's settings offer, as the grid allows


def ai_path(path, grid):
    """The race's path with the grid's slots as the AI race's racers' start positions: Racer Start 1
    on, as BeamNG's AI race places its racers, the first slot the front of the grid."""
    path = json.loads(json.dumps(path))
    sid = max(p['oldId'] for p in path['startPositions'] + path['pathnodes']) + 1
    for k, (at, fwd) in enumerate(grid):
        path['startPositions'].append({'group': 'start', 'name': 'Racer Start %d' % (k + 1), 'oldId': sid + k,
                                       'pos': _floats(at), 'rot': _floats(quat(np.r_[fwd[:2], 0]))})
    return path


def ai_mission(level, folder, title, description, author, trigger, racers):
    """An AI race's info.json: the layout raced against karts as the player's, on its grid, the player
    at its back. trigger: (position, forward) of its marker; racers: how many the grid holds."""
    most = max(1, min(AI_MAX, racers - 1))
    data = {'allowFlip': True, 'allowRecover': True, 'camPathActive': False, 'damageFailActive': False,
            'damageFailText': 'missions.aiRace.general.damageFail', 'damageLimit': 10000, 'flipLimit': -1,
            'goalCollisionsLimit': 1, 'goalCollisionsTarget': 5, 'goalLaps': LAPS, 'goalRecoveries': 0, 'goalTime': 0,
            'lapCount': LAPS, 'mapPreviewMode': 'navgraph', 'maxRacerCount': most, 'minRacerCount': 1,
            'playerStartIndex': 0, 'prefabActive': True, 'raceFile': '/gameplay/missions/%s/race.race.json' % folder,
            'racerAggression': 1, 'racerAwareness': True, 'racerCount': min(AI_RACERS, most), 'racerRandomnessScale': 1,
            'racerRough': False, 'racerRubberbandMode': False, 'racerSkill': 1, 'recoverLimit': -1,
            # the player on the grid's next slot after the racers (playerStartIndex 0); without it the
            # player goes to the race's own start, the grid's first slot, on top of the first racer
            'shuffleGroup': False, 'smartStartPlace': True, 'stoppedLimit': 0,
            'startText': '%d laps against karts like yours, from the back of the grid. Finish in the top three for '
                         'a star, first for all three.' % LAPS,
            'useGroundmarkers': False,
            # the racers drive what the player drives: karts
            'vehicleGroupActive': False, 'vehicleGroupByClass': 'same', 'vehicleGroupByClassActive': True,
            'vehiclePerformanceOffset': 0,
            'vehicleFilters': {'allowAuxiliaryVehicles': False, 'allowLoadedTrailers': False,
                               'baseFilter': {'blackList': [], 'whiteList': []}, 'manualAdditions': {},
                               'manualRemovals': {}, 'maxVehicles': 0, 'popAttribute': 'Population',
                               'probabilitySettings': {}}}
    at, fwd = trigger
    return {
        'name': title + ' race', 'description': description, 'author': author, 'date': DATE, 'devMission': False,
        'missionType': 'aiRace', 'missionTypeData': data,
        'additionalAttributes': {'difficulty': 'medium'}, 'customAdditionalAttributes': {},
        'careerSetup': {'showInCareer': False, 'showInFreeroam': True, 'skill': '(none)', 'starOutroTexts': {},
                        'starRewards': {}, 'defaultStarKeys': ['finishBronze', 'finishSilver', 'finishGold'],
                        'starsActive': {'finishBronze': True, 'finishSilver': True, 'finishGold': True}},
        'grouping': {'id': '', 'label': ''}, 'isAvailableAsScenario': True,
        'layers': [{'dir': '/gameplay/missions/%s/' % folder, 'fixed': True, 'isMissionFolderDir': True, 'isMissionTypeDir': False},
                   {'dir': '/gameplay/missionTypes/aiRace/', 'fixed': True, 'isMissionFolderDir': False, 'isMissionTypeDir': True}],
        'retryBehaviour': 'infiniteRetries',
        'setupModules': {'traffic': {'enabled': False},
                         'vehicles': {'enabled': True, 'includePlayerVehicle': True, 'playerVehicleDiffs': {},
                                      'prioritizePlayerVehicle': True, 'vehicles': {}}},
        'startCondition': {'type': 'automatic'}, 'visibleCondition': {'type': 'automatic'},
        'startTrigger': {'type': 'coordinates', 'level': level, 'pos': _floats(at), 'radius': 4,
                         'rot': _floats(quat(np.r_[fwd[:2], 0]))},
    }


def mission(level, folder, title, description, author, trigger, length):
    """A time trial's info.json. trigger: (position, forward) of its marker."""
    times = [round(length / v) for v in STAR_SPEEDS]
    data = {'allowFlip': True, 'allowRecover': True, 'allowRollingStart': True, 'canChangeLapAmount': True,
            'closed': True, 'defaultLaps': LAPS, 'reversible': False, 'mapPreviewMode': 'navgraph',
            'flipLimit': -1, 'flipPenalty': 5, 'recoverLimit': -1, 'recoverPenalty': 5, 'justFinishPenalty': 30,
            'startScreenText': 'A time trial of %d laps, from a standing or a rolling start. The stars are for '
                               'your best lap.' % LAPS}
    data.update({'lapTime%d' % (k + 1): t for k, t in enumerate(times)})
    # the older medals, by the whole race's time
    for medal, t in (('bronze', times[0]), ('silver', times[2]), ('gold', times[4])):
        data.update({medal + 'Time': t * LAPS, medal + 'TimeTotal': t * LAPS, medal + 'TimePenalty': 0})
    return {
        'author': author, 'date': DATE, 'devMission': False, 'name': title, 'description': description,
        'missionType': 'timeTrial', 'missionTypeData': data,
        'additionalAttributes': {'difficulty': 'medium'}, 'customAdditionalAttributes': {},
        'careerSetup': {'showInFreeroam': True, 'skill': '(none)', 'starOutroTexts': {}, 'starRewards': {},
                        'defaultStarKeys': ['lapTime%d' % k for k in range(1, 6)],
                        'starsActive': dict({'justFinish': False}, **{'lapTime%d' % k: True for k in range(1, 6)})},
        'grouping': {'id': '', 'label': ''},
        'layers': [{'dir': '/gameplay/missions/%s/' % folder, 'fixed': True, 'isMissionFolderDir': True,
                    'isMissionTypeDir': False},
                   {'dir': '/gameplay/missionTypes/timeTrial/', 'fixed': True, 'isMissionFolderDir': False,
                    'isMissionTypeDir': True}],
        'retryBehaviour': 'infiniteRetries',
        'setupModules': {'environment': {'enabled': False}, 'traffic': {'enabled': False},
                         'vehicles': {'enabled': True, 'includePlayerVehicle': True, 'playerVehicleDiffs': {},
                                      'prioritizePlayerVehicle': False, 'vehicles': {}}},
        'startCondition': {'type': 'automatic'},
        'startTrigger': {'level': level, 'type': 'coordinates', 'pos': _floats(trigger[0]), 'radius': 3.,
                         'rot': quat(trigger[1])},
        'visibleCondition': {'type': 'always'},
    }


def write(out, level, races):
    """The missions in out/gameplay/missions/<level>/timeTrial, made again each time, and the same
    races as the level's time trials in out/levels/<level>/quickrace, which BeamNG lists in its
    Time Trials. races: a list of dicts with layout, title, description, author, path, length,
    trigger, preview (an image file: the start's map) and cones (where the layout's loose cones
    stand)."""
    root = os.path.join(out, 'gameplay', 'missions', level, 'timeTrial')
    for old in (root, os.path.join(out, 'gameplay', 'missions', level, 'aiRace')):
        if os.path.exists(old):
            shutil.rmtree(old)
    quick = os.path.join(out, 'levels', level, 'quickrace')
    if os.path.exists(quick):
        shutil.rmtree(quick)
    os.makedirs(quick)
    for k, r in enumerate(races):
        folder = '%s/timeTrial/%03d-%s' % (level, k + 1, r['layout'])
        d = os.path.join(out, 'gameplay', 'missions', *folder.split('/'))
        os.makedirs(d, exist_ok=True)
        json.dump(mission(level, folder, r['title'], r['description'], r['author'], r['trigger'], r['length']),
                  open(os.path.join(d, 'info.json'), 'w'), indent=1)
        json.dump(r['path'], open(os.path.join(d, 'race.race.json'), 'w'), indent=1)
        # the layout's cones, which the time trial spawns as it starts and stands up again on a retry
        if r.get('cones') or r.get('boards'):
            cone_prefab(os.path.join(d, 'mainPrefab.prefab.json'), r.get('cones', []), '%s/%s/cones' % (level, r['layout']),
                        r.get('boards', []))
        img = Image.open(r['preview']).convert('RGB')
        img.save(os.path.join(d, 'preview.jpg'), quality=92)
        # the list shows it small: wide enough to tell the layouts apart
        img.resize((320, round(320 * img.height / img.width)), Image.LANCZOS).save(os.path.join(d, 'thumbnail.jpg'), quality=90)
        # the time trial: the race's path with its name, its picture and its cones; not made a
        # mission of its own, as the race above is that already
        race = dict(r['path'], name=r['title'], description=r['description'], authors=r['author'], ignoreAsMission=True,
                    prefabs=[])
        if r.get('cones') or r.get('boards'):
            # named as the track, which BeamNG loads with it
            shutil.copy(os.path.join(d, 'mainPrefab.prefab.json'), os.path.join(quick, r['layout'] + '.prefab.json'))
        json.dump(race, open(os.path.join(quick, r['layout'] + '.race.json'), 'w'), indent=1)
        img.save(os.path.join(quick, r['layout'] + '.jpg'), quality=92)
        # the AI race: against karts, from the layout's grid
        if len(r.get('grid', [])) > 1:
            folder = '%s/aiRace/%03d-%s' % (level, k + 1, r['layout'])
            a = os.path.join(out, 'gameplay', 'missions', *folder.split('/'))
            os.makedirs(a, exist_ok=True)
            json.dump(ai_mission(level, folder, r['title'], r['description'], r['author'], r['trigger'], len(r['grid'])),
                      open(os.path.join(a, 'info.json'), 'w'), indent=1)
            json.dump(ai_path(r['path'], r['grid']), open(os.path.join(a, 'race.race.json'), 'w'), indent=1)
            if os.path.exists(os.path.join(d, 'mainPrefab.prefab.json')):
                # the cones, as the mission type loads its deco prefab
                shutil.copy(os.path.join(d, 'mainPrefab.prefab.json'), os.path.join(a, 'deco.prefab.json'))
            for f in ('preview.jpg', 'thumbnail.jpg'):
                shutil.copy(os.path.join(d, f), os.path.join(a, f))
    return root
