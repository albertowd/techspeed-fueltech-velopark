"""Assetto Corsa's Custom Shaders Patch config (extension/ext_config.ini), made by its build: AC's own settings as kept
in source/track/extension/ext_config.ini (grass, rain, lighting), its grass on the meshes the models say it grows
on (extras.track `grass`: GRASS_MESHES) and kept off where a material lies that covers it (materials.json
`covers_grass`: OCCLUDING_MATERIALS), then what the track's data has (TRACK_FORMAT.md, "Lights and screens"):
- each light series as a LIGHT_SERIES, its lights' positions and directions in it (track_lights.data());
- each material that glows only at night (materials.json `glow`) as a MATERIAL_ADJUSTMENT of its ksEmissive, and
  each one with `targets.ac.night` as one of the values it names, at night and by day;
- each screen as a SCRIPTABLE_DISPLAY, its script tools/ac/screens/<shows>.lua with the screen's data put in for its
  `-- @data` line, and a CustomEmissive when its material glows.
"""
import json
import os
import re
import struct

import materials
import track_lights

TOOLS = os.path.dirname(os.path.abspath(__file__))
SCREENS = os.path.join(TOOLS, 'ac', 'screens')
BASE = os.path.join(os.path.dirname(TOOLS), 'source', 'track', 'extension', 'ext_config.ini')
REL = os.path.join('extension', 'ext_config.ini')


def num(v):
    return '%g' % v if isinstance(v, float) else str(v)


def nums(v):
    return ', '.join(num(x) for x in v)


def lua(v, depth=0):
    """A JSON value as Lua: lists as tables (a list of lists one row a line), strings quoted."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, str):
        return '"%s"' % v.replace('\\', '\\\\').replace('"', '\\"').replace("'", '`')
    if isinstance(v, (int, float)):
        return num(v)
    if isinstance(v, dict):
        return '{ %s }' % ', '.join('%s = %s' % (k, lua(x, depth + 1)) for k, x in v.items())
    if v and all(isinstance(x, list) for x in v):
        return '{\n%s\n}' % ',\n'.join('{%s}' % ','.join(lua(y, depth + 1) for y in x) for x in v)
    return '{ %s }' % ', '.join(lua(x, depth + 1) for x in v)


def series(d):
    out = []
    for i, (name, s) in enumerate(d['series'].items()):
        rows = ['[LIGHT_SERIES_%d]' % i, 'ACTIVE = 1', 'DESCRIPTION = %s' % name,
                'COLOR = %s, %s' % (nums(s['colour']), num(s.get('strength', 1.)))]
        if s.get('facing'):
            rows.append('DIRECTION = %s' % nums(s['facing']))
        rows.append('SPOT = %s' % num(s.get('spot', 180)))
        if 'sharpness' in s:
            rows.append('SPOT_SHARPNESS = %s' % num(s['sharpness']))
        rows.append('RANGE = %s' % num(s.get('range', 50)))
        for k, v in s.get('targets', {}).get('ac', {}).items():
            rows.append('%s = %s' % (k, nums(v) if isinstance(v, list) else num(v)))
        if s.get('night_only'):
            rows.append('CONDITION = NIGHT_SMOOTH')
        for k, L in enumerate(x for x in d['lights'] if x['series'] == name):
            rows.append('POSITION_%d = %.3f, %.3f, %.3f' % (k, *L['at']))
            if L.get('facing'):
                rows.append('DIRECTION_%d = %.3f, %.3f, %.3f' % (k, *L['facing']))
        out.append('\n'.join(rows))
    return out


def adjustments(lib):
    out = []
    for name, m in lib.items():
        if m.get('glow') and m.get('night_only'):
            c = [int(round(x * 255)) for x in m['glow']]
            out.append('\n'.join(['[MATERIAL_ADJUSTMENT_...]', 'ACTIVE = 1', 'DESCRIPTION = %s glows at night' % name,
                                  'MATERIALS = %s' % name, 'CONDITION = NIGHT_SMOOTH', 'KEY_0 = ksEmissive',
                                  'VALUE_0 = %s, %s' % (nums(c), num(float(m.get('glow_strength', 1.)))), 'VALUE_0_OFF = 0, 0, 0']))
    groups = {}
    for name, m in lib.items():
        night = m.get('targets', {}).get('ac', {}).get('night')
        if night:
            groups.setdefault(json.dumps(night), []).append(name)
    for key, names in groups.items():
        rows = ['[MATERIAL_ADJUSTMENT_...]', 'ACTIVE = 1', 'DESCRIPTION = Night response: %s' % ', '.join(names),
                'MATERIALS = %s' % ', '.join(names), 'CONDITION = NIGHT_SMOOTH']
        for i, (k, (at_night, by_day)) in enumerate(json.loads(key).items()):
            rows += ['KEY_%d = %s' % (i, k), 'VALUE_%d = %s' % (i, num(at_night)), 'VALUE_%d_OFF = %s' % (i, num(by_day))]
        out.append('\n'.join(rows))
    return out


def script(sc):
    text = open(os.path.join(SCREENS, sc['shows'] + '.lua'), encoding='utf-8').read().rstrip('\n')
    data = '\n'.join('local %s = %s' % (k.upper(), lua(v)) for k, v in (sc.get('data') or {}).items())
    return text.replace('-- @data', data)


def screens(d, lib):
    out = []
    for sc in d['screens']:
        w, h = sc['pixels']
        x, y, aw, ah = sc.get('area') or (0, 0, w, h)
        rows = ['[SCRIPTABLE_DISPLAY_...]', 'MATERIALS = %s' % sc['material'], 'RESOLUTION = %d, %d' % (w, h),
                'DISPLAY_POS = %d, %d' % (x, y), 'DISPLAY_SIZE = %d, %d' % (aw, ah), "SCRIPT = '", script(sc) + "'",
                'BACKGROUND = %s, 1' % nums(sc.get('background', [0, 0, 0]))]
        out.append('\n'.join(rows))
        if lib.get(sc['material'], {}).get('glow'):
            out.append('\n'.join(['[CustomEmissive]', 'ACTIVE = 1', 'DESCRIPTION = %s glows as drawn' % sc['material'],
                                  'MATERIALS = %s' % sc['material'], 'Resolution = %d, %d' % (w, h),
                                  '@ = CustomEmissive_Rect, Channel = 0, Start = "%d, %d", Size = "%d, %d", CornerRadius = 0.0' % (x, y, aw, ah)]))
    return out


def grass_meshes(models):
    """The meshes grass grows on (extras.track `grass`) in the models (.glb) of the folder models, by AC's names."""
    hook = materials.Hook()
    out = []
    for f in sorted(os.listdir(models)):
        if not f.endswith('.glb'):
            continue
        with open(os.path.join(models, f), 'rb') as fh:
            head = fh.read(20)
            g = json.loads(fh.read(struct.unpack('<I', head[12:16])[0]))
        for node in g.get('nodes', []):
            track = (node.get('extras') or {}).get('track') or {}
            if track.get('grass'):
                out.append(hook.node_in(node['name'], track, {})[0])
    return out


def text(models=None):
    """The whole config: AC's own settings (its grass on the meshes the models in the folder models say it grows on,
    kept off the materials that cover it), then the lights, the materials' night and the screens."""
    base = open(BASE, encoding='utf-8').read().rstrip('\n')
    lib = materials.library()['materials']
    for key, more in (('GRASS_MESHES', grass_meshes(models) if models else []),
                      ('OCCLUDING_MATERIALS', [n for n, m in lib.items() if m.get('covers_grass')])):
        m = re.search(r'(?m)^%s *= *(.*)$' % key, base)
        if m and more:
            have = [x.strip() for x in m.group(1).split(',') if x.strip()]
            for x in more:                                  # (each once: a mesh cut in pieces has its name on each)
                if x not in have:
                    have.append(x)
            base = base[:m.start(1)] + ', '.join(have) + base[m.end(1):]
    d = track_lights.data()
    parts = [base, '; ---- Lights, night materials and screens, from source/lights.json and materials.json ----']
    parts += series(d) + adjustments(lib) + screens(d, lib)
    return '\n\n'.join(parts) + '\n'


def write(track_folder, models=None):
    """The config written into the built track (when it changed), grass also on what the models in the folder models
    say it grows on. Whether it changed."""
    p = os.path.join(track_folder, REL)
    new = text(models)
    if os.path.exists(p) and open(p, encoding='utf-8', newline='').read() == new:
        return False
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w', encoding='utf-8', newline='').write(new)
    return True
