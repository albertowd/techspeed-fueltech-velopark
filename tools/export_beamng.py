"""Convert the built Assetto Corsa track into a BeamNG.drive level, to try the track there.

    python tools/export_beamng.py [layout] [track folder]

Takes the models one layout loads (vp_1500 by default) from content/tracks/techspeed_velopark,
and writes build/beamng/levels/techspeed_velopark with the meshes as Collada shapes, their
textures and materials, and the level's objects, then packs it as
build/velopark-beamng-<version>.zip, beside the Assetto Corsa archive, to drop in BeamNG's mods
folder. release.ps1 runs it after building that archive.

The road, grass and kerb meshes AC drives on are solid, with the BeamNG ground model of their
surface; the invisible walls are solid and unseen; all else is only seen, as in AC. Karts
start on the grid's first slot, facing the way the lap runs. Lights, the other layouts and
timing are not converted.
"""
import configparser, io, json, os, re, shutil, sys, uuid, zipfile
from xml.sax.saxutils import escape

import numpy as np

import kn5io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVEL = 'techspeed_velopark'
# AC's surface keys, from the start of a physics mesh's name, as BeamNG ground models
GROUND = {'ROAD': 'ASPHALT', 'GRASS': 'GRASS', 'KERB': 'RUMBLE_STRIP', 'SAND': 'SAND', 'GRAVEL': 'GRAVEL',
          'WALL': 'ASPHALT', 'PIT': 'ASPHALT'}
PHYSICS = re.compile(r'^\d+([A-Z]+)')


def to_beamng(p):
    """AC's y up, right handed, to BeamNG's z up: x stays, -z is north, y is up."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def floats(a, fmt='%.5g'):
    return ' '.join(fmt % x for x in np.ravel(a))


def payload(block):
    """A kn5 texture block's file bytes."""
    n = int.from_bytes(block[4:8], 'little')
    return block[8 + n + 4:]


def write_dae(path, meshes):
    """Meshes as (name, material, positions, normals, uvs, triangles), in BeamNG's axes."""
    geo, nodes, mats = [], [], {}
    for k, (name, mat, P, N, U, F) in enumerate(meshes):
        mats[mat] = True
        g = 'g%d' % k
        geo.append(f'''<geometry id="{g}" name="{escape(name)}"><mesh>
<source id="{g}-p"><float_array id="{g}-pa" count="{P.size}">{floats(P)}</float_array><technique_common><accessor source="#{g}-pa" count="{len(P)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="{g}-n"><float_array id="{g}-na" count="{N.size}">{floats(N, '%.4g')}</float_array><technique_common><accessor source="#{g}-na" count="{len(N)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="{g}-t"><float_array id="{g}-ta" count="{U.size}">{floats(U, '%.6g')}</float_array><technique_common><accessor source="#{g}-ta" count="{len(U)}" stride="2"><param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
<vertices id="{g}-v"><input semantic="POSITION" source="#{g}-p"/><input semantic="NORMAL" source="#{g}-n"/><input semantic="TEXCOORD" source="#{g}-t" set="0"/></vertices>
<triangles material="{escape(mat)}" count="{len(F)}"><input semantic="VERTEX" source="#{g}-v" offset="0"/><p>{' '.join(map(str, F.ravel()))}</p></triangles>
</mesh></geometry>''')
        nodes.append(f'''<node id="n{k}" name="{escape(name)}"><instance_geometry url="#{g}"><bind_material><technique_common><instance_material symbol="{escape(mat)}" target="#{escape(mat)}"/></technique_common></bind_material></instance_geometry></node>''')
    effects = ''.join(f'<effect id="{escape(m)}-fx"><profile_COMMON><technique sid="common"><lambert/></technique></profile_COMMON></effect>' for m in mats)
    materials = ''.join(f'<material id="{escape(m)}" name="{escape(m)}"><instance_effect url="#{escape(m)}-fx"/></material>' for m in mats)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_effects>{effects}</library_effects>
<library_materials>{materials}</library_materials>
<library_geometries>{''.join(geo)}</library_geometries>
<library_visual_scenes><visual_scene id="scene"><node id="base00" name="base00">{''.join(nodes)}</node></visual_scene></library_visual_scenes>
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


def main():
    layout = sys.argv[1] if len(sys.argv) > 1 else 'vp_1500'
    track = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'content', 'tracks', LEVEL)
    out = os.path.join(ROOT, 'build', 'beamng')
    level = os.path.join(out, 'levels', LEVEL)
    if os.path.exists(level):
        shutil.rmtree(level)
    shapes = os.path.join(level, 'art', 'shapes')
    os.makedirs(shapes)
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
    for fn in files:
        model = kn5io.Model(os.path.join(track, fn))
        stem = os.path.splitext(fn)[0]
        dummies.update(model.dummies)
        # the textures, in a folder of their own: two models may hold different ones of a name
        tex_dir = os.path.join(shapes, 'textures', stem)
        os.makedirs(tex_dir, exist_ok=True)
        for name, block in model.textures.items():
            open(os.path.join(tex_dir, name), 'wb').write(payload(block))
        tex_path = lambda name: '/levels/%s/art/shapes/textures/%s/%s' % (LEVEL, stem, name)

        def material(mi, surface=None, hidden=False):
            m = model.materials[mi]
            name = re.sub(r'[^A-Za-z0-9_]', '_', '%s_%s' % (stem, m['name'])) + ('_' + surface if surface else '') + ('_hidden' if hidden else '')
            if name not in materials:
                stage = {}
                if hidden:
                    stage['baseColorFactor'] = [1, 1, 1, 0]
                    stage['opacityFactor'] = 0
                else:
                    if 'txDiffuse' in m['tex']:
                        stage['baseColorMap'] = tex_path(m['tex']['txDiffuse'])
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
            elif m['visible'] and m['renderable']:
                parts['seen'].append((m['name'], material(m['mat'])) + arrays)
        for part, meshes in parts.items():
            if not meshes:
                continue
            shape = '%s_%s.dae' % (stem, part)
            write_dae(os.path.join(shapes, shape), meshes)
            statics.append(dict(name='%s_%s' % (stem, part), shapeName='/levels/%s/art/shapes/%s' % (LEVEL, shape),
                                collisionType='None' if part == 'seen' else 'Visible Mesh'))
            print('%-28s %-5s %4d meshes, %7d triangles' % (fn, part, len(meshes), sum(len(x[5]) for x in meshes)))
    json.dump(materials, open(os.path.join(shapes, 'main.materials.json'), 'w'), indent=1)

    # the grid's first slot, facing the way the lap runs; AC drops the kart from a few metres
    # up onto the road, BeamNG starts it where it is placed: just over the road
    w = dummies['AC_START_0']
    at = to_beamng(w[3, :3])
    t = np.concatenate(ground)
    a, b, c = t[:, 0, :2], t[:, 1, :2], t[:, 2, :2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    # not those standing on edge, which have no inside seen from above
    flat = np.abs(d) > 1e-6
    d = np.where(flat, d, 1.)
    l1 = ((b[:, 1] - c[:, 1]) * (at[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (at[1] - c[:, 1])) / d
    l2 = ((c[:, 1] - a[:, 1]) * (at[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (at[1] - c[:, 1])) / d
    l3 = 1 - l1 - l2
    hit = flat & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
    z = l1 * t[:, 0, 2] + l2 * t[:, 1, 2] + l3 * t[:, 2, 2]
    below = z[hit & (z < at[2] + 1)]
    if len(below):
        at = np.r_[at[:2], below.max()]
    fwd = to_beamng(w[2, :3])
    ai = os.path.join(track, layout, 'ai', 'fast_lane.ai')
    if os.path.exists(ai):
        b = open(ai, 'rb').read()
        n = int.from_bytes(b[4:8], 'little')
        line = to_beamng(np.frombuffer(b, dtype='<f4', count=n * 5, offset=16).reshape(-1, 5)[:, :3])
        k = np.linalg.norm(line[:, :2] - at[:2], axis=1).argmin()
        if (line[(k + 3) % n] - line[k])[:2] @ fwd[:2] < 0:
            fwd = -fwd
    fwd = fwd[:2] / np.linalg.norm(fwd[:2])
    # BeamNG's vehicles face their -y
    c, s = -fwd[1], fwd[0]

    def obj(cls, name, **kw):
        return dict({'class': cls, 'name': name, 'persistentId': str(uuid.uuid4()), '__parent': 'MissionGroup'}, **kw)
    items = [obj('LevelInfo', 'theLevelInfo', gravity=-9.81, visibleDistance=1500, canvasClearColor=[0, 0, 0, 255]),
             obj('TimeOfDay', 'tod', time=0.1, play=False, dayLength=1200, azimuthOverride=0),
             obj('ScatterSky', 'sunsky', position=[0, 0, 0]),
             obj('SpawnSphere', 'spawn_grid', dataBlock='SpawnSphereMarker', position=[float(x) for x in at + (0, 0, .5)],
                 rotationMatrix=[float(x) for x in (c, -s, 0, s, c, 0, 0, 0, 1)], radius=1)]
    items += [obj('TSStatic', st['name'], shapeName=st['shapeName'], position=[0, 0, 0], collisionType=st['collisionType'],
                  decalType='Collision Mesh' if st['collisionType'] != 'None' else 'None') for st in statics]
    main = os.path.join(level, 'main')
    os.makedirs(os.path.join(main, 'MissionGroup'))
    json.dump({'class': 'SimGroup', 'name': 'MissionGroup', 'persistentId': str(uuid.uuid4()), 'enabled': '1'},
              open(os.path.join(main, 'items.level.json'), 'w'))
    open(os.path.join(main, 'MissionGroup', 'items.level.json'), 'w').write(''.join(json.dumps(i) + '\n' for i in items))

    shutil.copy(os.path.join(track, 'ui', layout, 'preview.png'), os.path.join(level, '%s_preview.png' % LEVEL))
    json.dump({'title': 'Techspeed Velopark', 'description': ui.get('description', ''), 'authors': ui.get('author', ''),
               'previews': ['%s_preview.png' % LEVEL], 'size': [1024, 1024], 'country': ui.get('country', ''),
               'spawnPoints': [{'objectname': 'spawn_grid', 'translationId': 'Grid'}], 'defaultSpawnPointName': 'spawn_grid'},
              open(os.path.join(level, 'info.json'), 'w'), indent=1)

    # packed beside the final name, so a failed pack leaves the old zip intact
    pack = os.path.join(ROOT, 'build', 'velopark-beamng-%s.zip' % version)
    with zipfile.ZipFile(pack + '.new', 'w', zipfile.ZIP_DEFLATED) as z:
        for base, _, fs in os.walk(level):
            for f in fs:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, out))
    os.replace(pack + '.new', pack)
    print('%d materials, spawn at (%.1f, %.1f, %.1f); packed %s (%.1f MB)' % (len(materials), *at, pack, os.path.getsize(pack) / 2 ** 20))


if __name__ == '__main__':
    main()
