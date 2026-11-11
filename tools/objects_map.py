"""Draw what stands round the track from above, each thing by name, to talk about it by name:
python tools/objects_map.py [out.png]. Reads the Assetto Corsa build in build/ac (python
tools/build_tracks.py ac first) and the forest's stands (source/models/velopark_extra_trees.json),
x to the left and z up, as it lies on the ground, as tools/track_map.py draws the roads.

The grass, the trees, the dirt roads, the pond and the roads are background colours (the trees
too many to draw each); the buildings, tyre walls, fences, gates, light poles, gantries, cones,
tents, flagpoles, banners and signs are drawn each, with its name: a tyre wall, a pole or a
fence by its mesh's name (TYRES_50 is velopark_tyres.kn5's, or velopark_kerbs.kn5's), a building
by what it is.
"""
import json, os, re, sys

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

import kn5io
from track_map import TRACK, font, triangles as flat_triangles


def triangles(m):
    """The mesh's triangles in x, z, the upright ones too (fences, walls)."""
    f = m['f']
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return m['v'][f][:, :, [0, 2]]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'objects_map.png')
FOREST = os.path.join(ROOT, 'source', 'models', 'velopark_extra_trees.json')
SCALE = 4.            # pixels a metre
X0, X1, Z0, Z1 = -290., 270., -140., 360.
COLOURS = dict(grass=(204, 220, 178), trees=(120, 160, 104), dirt=(206, 182, 140), water=(130, 170, 205),
               road=(158, 158, 158), paddock=(186, 180, 168), paving=(210, 204, 190),
               building=(196, 112, 96), tyres=(45, 45, 45), fence=(30, 60, 160), gate=(200, 0, 160),
               pole=(250, 200, 20), cone=(240, 110, 20), gantry=(120, 40, 160), object=(0, 140, 140), banner=(20, 30, 90))
# the paddock's buildings, by the walls to hit each has (1WALL_BLD_<key><n>)
PADDOCK_NAMES = {'C1': 'garage', 'C2': 'garage annex', 'C4': 'paddock building', 'C5': 'paddock building',
                 'C6': 'paddock building', 'G': 'solar shed rooms', 'H': 'long shed', 'P': 'long shed', 'D': 'long shed garages',
                 'M': 'L building', 'S1': 'north-south shed', 'S2': 'hut', 'B': 'small building', 'W': 'fuel station'}
# velopark_outside.kn5's meshes: their names' start and what they are
OUTSIDE = [('DIRT_', 'dirt'), ('POND', 'water'), ('FENCE_MESH_', 'fence'), ('GATE_', 'gate'), ('LAMPS_', 'lamps'),
           ('BANNER_', 'banner'), ('FLAGPOLES', 'object'), ('MARSHAL_', 'object'), ('FUEL_SIGN', 'object')]
SKIP = ('1WALL_', 'FENCE_POSTS_', 'BANNER_FENCE_BACK', 'BANNER_STAND_BACK')


def to_px(x, z):
    return (X1 - x) * SCALE, (Z1 - z) * SCALE


def footprint(tri):
    """Triangles from above as one shape: the flat ones as they lie, the upright ones (walls,
    panels) as their edges, a little wide."""
    polys = []
    for t in tri:
        p = Polygon(t)
        polys.append(p.buffer(.05) if p.area > .01 else LineString(np.r_[t, t[:1]]).buffer(.12))
    return unary_union(polys)


def fill(d, g, colour, outline=None, holes=False):
    """A shape filled; its holes painted grass again when holes (the roads' infields), else
    filled too (a building drawn as its walls)."""
    for p in getattr(g, 'geoms', [g]):
        if p.is_empty or p.geom_type != 'Polygon':
            continue
        d.polygon([to_px(*c) for c in p.exterior.coords], fill=colour, outline=outline)
        for h in p.interiors if holes else ():
            d.polygon([to_px(*c) for c in h.coords], fill=COLOURS['grass'])


def label(d, text, x, z, size=11, colour=(0, 0, 0), bold=False):
    px, py = to_px(x, z)
    f = font(size, bold)
    d.text((px, py), text, fill=colour, font=f, anchor='mm', stroke_width=2, stroke_fill=(255, 255, 255))


# the outside's things by name: the start of their meshes' names, what each is called (None: unnamed)
NAMES = [('TOWER_CONTROL', 'race control tower'), ('TOWER_', "judges' tower"), ('SHED_', 'blue shed'),
         ('FUEL_SHOP', 'fuel station shop'), ('FUEL_STORE', 'fuel station store'), ('FUEL_SIGN', 'fuel sign'),
         ('FUEL_', 'fuel station'), ('MARSHAL_POLES_1000', 'marshal tent 1000'), ('MARSHAL_POLES_1500', 'marshal tent 1500'),
         ('MARSHAL_', None), ('BANNER_STAND', 'banner'), ('BANNER_FENCE', 'banner'), ('COURT_', 'sports court'),
         ('VELOGRILL_A', 'Velogrill'), ('VELOGRILL_', None), ('FARM_LEANTO', None), ('FARM_HOUSE', 'farm house'),
         ('FARM_BARN', 'barn'), ('FARM_LOW', 'farm building'), ('BINS', 'round bins'), ('TENT_', 'white tent'),
         ('EAST_SHED', 'east shed'), ('HOUSE_', 'house'), ('FLAGPOLES', 'flagpoles'), ('GATE_', 'gate')]


def canon(name):
    """What an outside mesh is called on the map, and the group it is drawn and named with."""
    for start, called in NAMES:
        if name.startswith(start):
            # each house and shed its own, by its number
            num = re.match(r'^(HOUSE_\d+|EAST_SHED_\d+)', name)
            group = num.group(1) if num else start.rstrip('_')
            return group, called
    return pretty(name), pretty(name).lower().replace('_', ' ')


def pretty(name):
    return re.sub(r'_(WALLS|ROOF|OPENINGS|PILLARS|GLASS|STEEL|CABIN|WINDOWS|TOP|POLES|CONES|POSTS|ISLANDS|PUMPS|CANOPY|MESH|SLAB|TIMBER|BLUE_WALL|TURF|_\d+)$', '', name)


def main(out):
    load = lambda f: kn5io.Model(os.path.join(TRACK, f))
    img = Image.new('RGB', (int((X1 - X0) * SCALE), int((Z1 - Z0) * SCALE)), COLOURS['grass'])
    d = ImageDraw.Draw(img)
    models = {f: load(f) for f in sorted(os.listdir(TRACK)) if f.endswith('.kn5') and f.startswith('velopark')}

    # the trees: their canopies from the forest's stands
    forest = json.load(open(FOREST))
    radius = {'euca': 2.5, 'young': 1.5, 'broad': 4., 'palm': 2.5, 'bamboo': 2.2}
    crowns = [Polygon(np.c_[np.cos(a), np.sin(a)] * radius.get(forest['variants'][s[0]].get('kind', 'euca'), 3.) * s[5] + (s[1], s[3]))
              for s in forest['trees'] for a in [np.linspace(0, 2 * np.pi, 10, endpoint=False)]]
    fill(d, unary_union(crowns), COLOURS['trees'])

    # the outside's dirt roads and pond, then the roads, paddock and paving
    outside = models.get('velopark_outside.kn5')
    for m in outside.meshes if outside else []:
        kind = next((k for p, k in OUTSIDE if m['name'].startswith(p)), None)
        if kind in ('dirt', 'water'):
            fill(d, footprint(triangles(m)), COLOURS[kind])
    for f in ('velopark.kn5', 'velopark_kerbs.kn5', 'velopark_kids.kn5', 'velopark_paddock.kn5'):
        mdl = models[f]
        for m in mdl.meshes:
            n, mat = m['name'], mdl.material_name(m)
            if not m['visible'] or not n.startswith(('1ROAD', '1KERB')):
                continue
            t = flat_triangles(m)
            if len(t):
                c = COLOURS['paddock'] if 'PADDOCK' in n else COLOURS['paving'] if mat == 'Pavers' else COLOURS['road']
                fill(d, footprint(t), c, holes=True)

    labels = []
    # the paddock's buildings
    pad = models['velopark_paddock.kn5']
    walls = {m['name'][len('1WALL_BLD_'):]: m['v'][:, [0, 2]].mean(0) for m in pad.meshes if m['name'].startswith('1WALL_BLD_')}
    shapes = []
    for m in pad.meshes:
        if m['name'].startswith('BLD_') and len(m['f']):
            shapes.append(footprint(triangles(m)))
    bld = unary_union(shapes)
    fill(d, bld, COLOURS['building'], outline=(90, 40, 30))
    for p in getattr(bld, 'geoms', [bld]):
        if p.area < 4:
            continue
        c = p.representative_point()
        key = min(walls, key=lambda k: np.hypot(*(walls[k] - (c.x, c.y)))) if walls else None
        name = next((v for k, v in PADDOCK_NAMES.items() if key and key.startswith(k)), None) if key else None
        if name and p.area > 20:
            labels.append((name, c.x, c.y, 11, (60, 20, 10), True))
    # the stand, the gantries
    for m in models['velopark.kn5'].meshes:
        if models['velopark.kn5'].material_name(m) == 'GrandStand':
            g = footprint(triangles(m))
            fill(d, g, COLOURS['building'], outline=(90, 40, 30))
            c = g.representative_point()
            labels.append(('stand', c.x, c.y, 11, (60, 20, 10), True))
    for f in ('velopark_gantry_1000.kn5', 'velopark_gantry_1500.kn5'):
        g = unary_union([footprint(triangles(m)) for m in models[f].meshes if m['name'] == 'GANTRY_STEEL'])
        fill(d, g, COLOURS['gantry'])
        c = g.centroid
        labels.append(('gantry ' + f[len('velopark_gantry_'):-4], c.x, c.y - 3, 11, COLOURS['gantry'], True))

    # the outside's things
    groups = {}
    for m in outside.meshes if outside else []:
        n = m['name']
        if n.startswith(SKIP) or not len(m['f']):
            continue
        kind = next((k for p, k in OUTSIDE if n.startswith(p)), 'building')
        if kind in ('dirt', 'water'):
            continue
        group, called = canon(n)
        g = groups.setdefault((group, kind), [called, []])
        g[0] = g[0] or called
        g[1].append(m)
    for (name, kind), (called, ms) in sorted(groups.items(), key=lambda a: a[0][1] == 'building', reverse=True):
        t = np.concatenate([triangles(m) for m in ms])
        if not len(t):
            continue
        if kind == 'fence':
            for tr in t:
                d.line([to_px(*c) for c in tr[:2]], fill=COLOURS['fence'], width=3)
            mid = t[len(t) // 2].mean(0)
            labels.append((name.replace('FENCE_MESH_', 'fence '), mid[0], mid[1], 10, COLOURS['fence'], True))
        elif kind == 'gate':
            for tr in t:
                d.line([to_px(*c) for c in tr[:2]], fill=COLOURS['gate'], width=4)
            labels.append(('gate', *t.reshape(-1, 2).mean(0), 10, COLOURS['gate'], True))
        elif kind == 'lamps':
            P = t.reshape(-1, 2)
            for c in np.unique(np.round(P), axis=0)[::3]:
                px, py = to_px(*c)
                d.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 255, 255), outline=(80, 80, 80))
        else:
            g = footprint(t)
            fill(d, g, COLOURS['banner' if kind == 'banner' else kind if kind == 'object' else 'building'],
                 outline=(90, 40, 30) if kind == 'building' else None)
            c = g.representative_point()
            if called and not called.endswith('back'):
                labels.append((called, c.x, c.y, 10, (60, 20, 10) if kind == 'building' else COLOURS[kind], kind == 'building'))

    # the tyre walls
    for f in ('velopark_tyres.kn5', 'velopark_kerbs.kn5'):
        for m in models[f].meshes:
            n = m['name']
            if not n.startswith('TYRES_') or n.endswith('F') or not m['renderable'] or not len(m['f']):
                continue
            t = triangles(m)
            if not len(t):
                continue
            P = t.reshape(-1, 2)
            g = unary_union([Polygon(tr).buffer(.15) for tr in t[::6] if Polygon(tr).area > .001] or [LineString(P[:2]).buffer(.3)])
            fill(d, g, COLOURS['tyres'], holes=True)
            c = P.mean(0)
            labels.append((n, c[0], c[1], 9, (0, 0, 0), False))
    # the kids' entry
    for m in models['velopark_kids.kn5'].meshes:
        if m['name'] in ('ENTRY_FENCE', 'ENTRY_GATE'):
            for tr in triangles(m):
                d.line([to_px(*c) for c in tr[:2]], fill=COLOURS['gate' if 'GATE' in m['name'] else 'fence'], width=3)
    # the light poles
    for m in models['velopark_poles.kn5'].meshes:
        mt = re.match(r'POLE_(\d+)_MAST$', m['name'])
        if mt:
            x, z = m['v'][:, [0, 2]].mean(0)
            px, py = to_px(x, z)
            d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=COLOURS['pole'], outline=(0, 0, 0))
            labels.append(('POLE_' + mt.group(1), x, z + 3.5, 10, (110, 80, 0), True))
    # the loose cones
    for name, mtx in models['velopark.kn5'].dummies.items():
        if name.startswith('AC_POBJECT'):
            px, py = to_px(mtx[3, 0], mtx[3, 2])
            d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=COLOURS['cone'], outline=(90, 40, 0))

    for text, x, z, size, colour, bold in labels:
        if X0 < x < X1 and Z0 < z < Z1:
            label(d, text, x, z, size, colour, bold)
    # the title and the legend
    d.rectangle((0, 0, img.width, 64), fill=(255, 255, 255))
    d.text((12, 8), 'Techspeed FuelTech Velopark: what stands round the track', fill=(0, 0, 0), font=font(22, True))
    d.text((12, 38), 'Drawn from the Assetto Corsa build by tools/objects_map.py; x runs left, z up. Tyre walls and poles by '
                     'their mesh names; trees and grass as background colours.', fill=(60, 60, 60), font=font(12))
    keys = [('grass', 'grass'), ('trees', 'trees'), ('dirt', 'dirt road'), ('water', 'pond'), ('road', 'road'),
            ('paddock', 'paddock'), ('paving', 'paving'), ('building', 'building'), ('tyres', 'tyre wall'), ('fence', 'fence'),
            ('gate', 'gate'), ('pole', 'light pole'), ('cone', 'cone'), ('gantry', 'gantry'), ('object', 'tent, sign, flagpole'),
            ('banner', 'banner')]
    x = img.width - 12
    for k, text in reversed(keys):
        w = d.textlength(text, font=font(12)) + 26
        x -= w
        d.rectangle((x, 42, x + 14, 54), fill=COLOURS[k], outline=(0, 0, 0))
        d.text((x + 18, 41), text, fill=(0, 0, 0), font=font(12))
    # a scale of 50 m
    y = img.height - 20
    d.line((20, y, 20 + 50 * SCALE, y), fill=(0, 0, 0), width=3)
    d.text((20, y - 16), '50 m', fill=(0, 0, 0), font=font(12))
    img.save(out)
    print('drew %s (%d by %d)' % (os.path.relpath(out, ROOT), img.width, img.height))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
