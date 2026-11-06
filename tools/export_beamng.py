"""Convert the built Assetto Corsa track into a BeamNG.drive level.

    python tools/export_beamng.py [layout] [track folder] [--no-pack]

Takes the models one layout loads (vp_1500 by default) from content/tracks/techspeed_velopark,
and writes build/beamng/levels/techspeed_velopark with the meshes as Collada shapes, their
textures and materials, and the level's objects, then packs it as
build/velopark-beamng-<version>.zip, beside the Assetto Corsa archive, to drop in BeamNG's mods
folder. release.ps1 runs it through build_tracks.py. Only the models that changed since the
last run are written again; --no-pack leaves out the zip, to run the level unpacked from
BeamNG's mods/unpacked folder.

The road, grass and kerb meshes AC drives on are solid, with the BeamNG ground model of their
surface; the invisible walls are solid and unseen; all else is only seen, as in AC. A mesh AC
swaps for a simpler copy further away (the tyre stacks) is a shape of its own with both as its
detail levels, for BeamNG to swap about as far away. Karts
start on the grid's first slot of any layout, facing the way its lap runs; the layouts' racing
lines are BeamNG's road network (waypoints and map.json), for its map and AI. The lights of the Custom Shaders
Patch config are spot lights, lit by day too. The level stands LIFT metres up, above BeamNG's
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
LEVEL = 'techspeed_velopark'
# AC's surface keys, from the start of a physics mesh's name, as BeamNG ground models
GROUND = {'ROAD': 'ASPHALT', 'GRASS': 'GRASS', 'KERB': 'RUMBLE_STRIP', 'SAND': 'SAND', 'GRAVEL': 'GRAVEL',
          'WALL': 'ASPHALT', 'PIT': 'ASPHALT'}
PHYSICS = re.compile(r'^\d+([A-Z]+)')
# how far up the level stands: a level without terrain has BeamNG's ground at height 0, which
# vehicles stand on, and the track lies up to a few metres below that; its walls go down 43 m
LIFT = 50.
# the brightness of a light per unit of its Custom Shaders Patch intensity, and how much further
# it reaches than its range there: BeamNG's lights are far weaker for the same numbers (its own
# spot lights run at 100 over 20 m), and at 3 and 1.5 left the track dark between poles
LIGHT_GAIN = 12.
LIGHT_REACH = 2.
# pixels on screen per metre of a shape's radius at a metre away: a 1080 pixel high screen at
# BeamNG's 60 degree field of view, for where AC swaps a mesh for its far copy
SCREEN = 1080 / (2 * math.tan(math.radians(30)))
# raised when the shapes or textures are written differently, so a level built before is made again
FORMAT = 5


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


def texture_file(name, data):
    """The file a texture is written to: a DDS BeamNG cannot read becomes a PNG."""
    f = file_name(name)
    return os.path.splitext(f)[0] + '.png' if data is not None and data[:4] == b'DDS ' and not compressed(data) else f


def opacity_file(name):
    """The greyscale map of what of a texture shows, from its alpha: BeamNG reads opacity from
    a map's colour, so the texture itself would show its clear parts and hide the rest."""
    return os.path.splitext(file_name(name))[0] + '_opacity.png'


def payload(block):
    """A kn5 texture block's file bytes."""
    n = int.from_bytes(block[4:8], 'little')
    return block[8 + n + 4:]


def write_dae(path, meshes, collide=False, sizes=None):
    """Meshes as (name, material, positions, normals, uvs, triangles), in BeamNG's axes.

    Laid out as BeamNG's own shapes are: base00 with the detail levels, then start01 with the
    meshes, each named with the detail it belongs to as the number its name ends in. All are
    drawn at one detail, 2, so a kn5 name ending in a number (1ROAD_3, TYRES_55) does not make a
    level of its own; or sizes gives each mesh's detail: the shape's size on screen, in pixels,
    from which it is drawn, BeamNG drawing the largest detail the shape is bigger than. With
    collide, each mesh is also a Colmesh at detail -1: what vehicles hit, the mesh's own
    triangles as they are, with its material for the ground model."""
    sizes = sizes or [2] * len(meshes)
    geo, nodes, mats = [], [], {}
    for k, ((name, mat, P, N, U, F), size) in enumerate(zip(meshes, sizes)):
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
        nodes.append(f'<node id="n{k}" name="%s_%d_L%d"><instance_geometry url="#{g}">{bind}</instance_geometry></node>'
                     % (re.sub(r'[^A-Za-z0-9_]', '_', name), k, size))
        if collide:
            nodes.append(f'<node id="c{k}" name="Colmesh_{k}-1"><instance_geometry url="#{g}">{bind}</instance_geometry></node>')
    effects = ''.join(f'<effect id="{escape(m)}-fx"><profile_COMMON><technique sid="common"><lambert/></technique></profile_COMMON></effect>' for m in mats)
    materials = ''.join(f'<material id="{escape(m)}" name="{escape(m)}"><instance_effect url="#{escape(m)}-fx"/></material>' for m in mats)
    details = ''.join('<node id="detail%d" name="detail%d"/>' % (x, x) for x in sorted(set(sizes), reverse=True))
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


def road_network(roads):
    """Waypoints [(name, position, radius)] and segments {layout: [names]} for the racing lines
    {layout: (line, sides)}, each a closed loop of its own. Layouts do not share waypoints where
    they share road: BeamNG's map, merging a run of waypoints into one stretch, fails where that
    stretch is already there, as it is round the triangles shared waypoints make."""
    waypoints, segments = [], {}
    for lay, (line, sides) in roads.items():
        step = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.r_[0, np.cumsum(step)]
        picks = np.searchsorted(s, np.arange(0, s[-1], WAYPOINT_STEP))
        nodes = []
        for i in picks:
            # the racing line runs a little over the road
            at = line[i] - (0, 0, .3)
            r = float(np.clip(sides[i].mean(), *WAYPOINT_RADIUS)) if sides is not None else WAYPOINT_RADIUS[1]
            name = 'wp_%s_%d' % (lay, len(nodes))
            waypoints.append((name, at, r))
            nodes.append(name)
        segments[lay] = nodes + nodes[:1]
    return waypoints, segments


def lights(track, obj):
    """The track's lights in its Custom Shaders Patch config, as BeamNG spot lights: each light of
    a series where it stands, pointing its own way or the series'. BeamNG has no lights that
    come on at night only: they shine by day too, under the sun."""
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
            # a light shines along its own y: x and z square to it, z as near up as it goes
            x = np.cross(y, [0, 0, 1.]) if abs(y[2]) < .99 else np.array([1., 0, 0])
            x /= np.linalg.norm(x)
            z = np.cross(x, y)
            out.append(obj('SpotLight', 'light_%s_%s' % (n, k), position=[float(v) for v in to_beamng(vec(pos)) + (0, 0, LIFT)],
                           rotationMatrix=[float(v) for v in np.c_[x, y, z].ravel()],
                           color=[float(c) for c in colour[:3]] + [1.], brightness=float(colour[3] * LIGHT_GAIN) if len(colour) > 3 else LIGHT_GAIN,
                           range=float(key('RANGE', '50')) * LIGHT_REACH, innerAngle=float(inner), outerAngle=float(outer),
                           castShadows=False, isEnabled=True))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    export(args[0] if args else 'vp_1500', args[1] if len(args) > 1 else os.path.join(ROOT, 'content', 'tracks', LEVEL),
           pack='--no-pack' not in sys.argv)


def export(layout, track, out=None, pack=True):
    """The level in out (build/beamng), and its zip beside it unless pack is False. Returns the
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

    materials, statics, dummies, ground = {}, [], {}, []
    lamps, gantries = {}, []
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
        if fresh:
            if os.path.exists(tex_dir):
                shutil.rmtree(tex_dir)
            os.makedirs(tex_dir)
            see_through = {m['tex'].get('txDiffuse') for m in model.materials if m['alpha_tested'] or m['blend'] == 1}
            for name, block in model.textures.items():
                data = payload(block)
                out_file = os.path.join(tex_dir, texture_file(name, data))
                if out_file.endswith('.png') and not name.lower().endswith('.png'):
                    Image.open(io.BytesIO(data)).save(out_file)
                else:
                    open(out_file, 'wb').write(data)
                if name in see_through:
                    img = Image.open(io.BytesIO(data))
                    alpha = img.getchannel('A') if 'A' in img.getbands() else Image.new('L', img.size, 255)
                    alpha.save(os.path.join(tex_dir, opacity_file(name)))
        data_of = {name: payload(block) for name, block in model.textures.items()}
        tex_path = lambda name: '/levels/%s/art/shapes/textures/%s/%s' % (LEVEL, stem, texture_file(name, data_of.get(name)))

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
                    mat['groundModelName'] = GROUND.get(surface, 'ASPHALT')
                materials[name] = mat
            return name

        parts = {'solid': [], 'walls': [], 'seen': []}
        seen = []
        for m in model.meshes:
            arrays = mesh_arrays(m)
            if arrays is None:
                continue
            key = PHYSICS.match(m['name'])
            surface = key.group(1) if key else None
            if surface and not (m['visible'] and m['renderable']):
                parts['walls'].append((m['name'], material(m['mat'], surface, hidden=True)) + arrays)
            elif surface:
                parts['solid'].append((m['name'], material(m['mat'], surface)) + arrays)
                ground.append(arrays[0][arrays[3]])
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
    grass_key = hashlib.sha1((' '.join(hashes) + ' %r' % [beamng_grass.BAND, beamng_grass.STEP, beamng_grass.TILE,
                                                         beamng_grass.HEIGHT, beamng_grass.WIDTH, beamng_grass.GREEN]).encode()).hexdigest()
    grass_tex = os.path.join(shapes, 'textures', 'grass')
    written.add(grass_tex)
    old_tiles = sorted(f for f in os.listdir(shapes) if f.startswith('grass_') and f.endswith('.dae'))
    if state.get('grass') != grass_key or not old_tiles:
        for f in old_tiles:
            os.remove(os.path.join(shapes, f))
        os.makedirs(grass_tex, exist_ok=True)
        beamng_grass.texture(os.path.join(grass_tex, 'blades.png'))
        ai = os.path.join(track, layout, 'ai', 'fast_lane.ai')
        b = open(ai, 'rb').read()
        line = np.frombuffer(b, dtype='<f4', count=int.from_bytes(b[4:8], 'little') * 5, offset=16).reshape(-1, 5)[:, [0, 2]]
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

    layouts = sorted(x for x in os.listdir(track) if os.path.exists(os.path.join(track, x, 'ai', 'fast_lane.ai'))
                     and os.path.exists(os.path.join(track, 'models_%s.ini' % x)))
    spawns, roads, races = [], {}, []
    for lay in layouts:
        lc = configparser.ConfigParser()
        lc.read(os.path.join(track, 'models_%s.ini' % lay))
        found = {}
        for sec in lc.sections():
            fn = lc[sec]['FILE']
            m = models.get(fn) or kn5io.Model(os.path.join(track, fn))
            found.update(m.dummies)
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
        start = hotlap if 'hotlap' in lay and hotlap else grid
        if start is None:
            continue
        at, fwd = start
        lui = json.load(open(os.path.join(track, 'ui', lay, 'ui_track.json'), encoding='utf-8'))
        title = re.sub(r'^Techspeed Velopark\s*', '', lui.get('name', lay)) or lay
        spawns.append(('spawn_' + lay, title, at, fwd[:2]))
        gates = []
        while 'AC_TIME_%d_L' % len(gates) in found and 'AC_TIME_%d_R' % len(gates) in found:
            gates.append(tuple(to_beamng(found['AC_TIME_%d_%s' % (len(gates), x)][3, :3]) for x in 'LR'))
        # not the hotlap: it is its layout's race from a rolling start, which every race offers
        if gates and 'hotlap' not in lay:
            # the car's front a metre on from where AC puts the kart
            path, length = beamng_races.race_path(line, sides, gates, (at + fwd, fwd), hotlap or start, on_road, LIFT)
            races.append(dict(layout=lay, title='Techspeed Velopark ' + title, description=lui.get('description', ''),
                              author=lui.get('author', ''), path=path, length=length,
                              trigger=(at + (0, 0, LIFT), fwd), preview=os.path.join(track, 'ui', lay, 'preview.png')))
    if 'spawn_' + layout not in [x[0] for x in spawns]:
        raise SystemExit('layout %s has no start' % layout)
    beamng_races.write(out, LEVEL, races)
    # the gantries' boards, each showing the races that start under it, and the Lua that runs them
    beamng_board.write_config(level, gantries, {r['layout']: next(p['pos'] for p in r['path']['pathnodes']
                                                                   if p['oldId'] == r['path']['startNode']) for r in races})
    for top in ('lua', 'scripts'):
        if os.path.exists(os.path.join(out, top)):
            shutil.rmtree(os.path.join(out, top))
        shutil.copytree(os.path.join(ROOT, 'tools', 'beamng', top), os.path.join(out, top))

    def obj(cls, name, **kw):
        return dict({'class': cls, 'name': name, 'persistentId': str(uuid.uuid4()), '__parent': 'MissionGroup'}, **kw)
    items = [obj('LevelInfo', 'theLevelInfo', gravity=-9.81, visibleDistance=1500, canvasClearColor=[0, 0, 0, 255],
                 globalEnviromentMap='BNG_Sky_02_cubemap'),
             obj('TimeOfDay', 'tod', time=0.1, play=False, dayLength=1200, azimuthOverride=0),
             obj('ScatterSky', 'sunsky', position=[0, 0, 0]),
             ]
    # BeamNG's vehicles face their -y
    for name, title, at, fwd in spawns:
        cs, sn = -fwd[1], fwd[0]
        items.append(obj('SpawnSphere', name, dataBlock='SpawnSphereMarker', position=[float(x) for x in at + (0, 0, LIFT + .5)],
                         rotationMatrix=[float(x) for x in (cs, -sn, 0, sn, cs, 0, 0, 0, 1)], radius=1))
    # the layouts' racing lines as BeamNG's road network, which its map draws and its AI drives
    waypoints, segments = road_network(roads)
    items += [obj('BeamNGWaypoint', name, position=[float(x) for x in at + (0, 0, LIFT)], scale=[float(r)] * 3)
              for name, at, r in waypoints]
    items += lights(track, obj)
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
    # each start shown with its layout's picture
    for name, _, _, _ in spawns:
        Image.open(os.path.join(track, 'ui', name[len('spawn_'):], 'preview.png')).convert('RGB').save(
            os.path.join(level, name + '_preview.jpg'), quality=92)
    for f in os.listdir(level):
        if f.startswith('spawn_') and f.endswith('_preview.jpg') and f[:-len('_preview.jpg')] not in [x[0] for x in spawns]:
            os.remove(os.path.join(level, f))
    json.dump({'title': 'Techspeed Velopark', 'description': ui.get('description', ''), 'authors': ui.get('author', ''),
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
    zip_path = os.path.join(os.path.dirname(out), 'velopark-beamng-%s.zip' % version)
    with zipfile.ZipFile(zip_path + '.new', 'w', zipfile.ZIP_DEFLATED) as z:
        for top in (level, os.path.join(out, 'gameplay'), os.path.join(out, 'lua'), os.path.join(out, 'scripts')):
            for base, _, fs in os.walk(top):
                for f in fs:
                    full = os.path.join(base, f)
                    z.write(full, os.path.relpath(full, out))
    os.replace(zip_path + '.new', zip_path)
    print('%d materials, %d starts, %d waypoints; packed %s (%.1f MB)' % (len(materials), len(spawns), len(waypoints), zip_path, os.path.getsize(zip_path) / 2 ** 20))
    return zip_path


if __name__ == '__main__':
    main()
