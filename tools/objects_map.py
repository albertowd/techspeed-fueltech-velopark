"""Draw what stands round the track from above, each thing by name, to talk about it by name:
python tools/objects_map.py [out.png]. Reads the Assetto Corsa build in build/ac (the Top Race
Editor builds it) and the editor's objects and trees (tools/reference/track.json), x to the left
and z up, as it lies on the ground, as tools/track_map.py draws the roads.

The grass, the trees, the dirt roads, the pond and the roads are background colours (the trees
too many to draw each); the buildings, tyre walls, fences, gates, light poles, gantries, cones,
marshal posts, flagpoles, banners and signs are drawn each, named as the editor names them (its
objects' names: TYRES_50, POLE_3, BUILDING_RACE_CONTROL).
"""
import json, os, sys

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
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
TRACK_JSON = os.path.join(ROOT, 'tools', 'reference', 'track.json')
SCALE = 4.            # pixels a metre
X0, X1, Z0, Z1 = -290., 270., -250., 360.
COLOURS = dict(grass=(204, 220, 178), trees=(120, 160, 104), dirt=(206, 182, 140), water=(130, 170, 205),
               road=(158, 158, 158), paving=(210, 204, 190),
               building=(196, 112, 96), tyres=(45, 45, 45), fence=(30, 60, 160), gate=(200, 0, 160),
               pole=(250, 200, 20), cone=(240, 110, 20), gantry=(120, 40, 160), object=(0, 140, 140), banner=(20, 30, 90))
# the trees' crowns' radius, by their model's name's start
CROWNS = {'euca': 2.5, 'young': 1.5, 'broad': 4., 'palm': 2.5, 'bamboo': 2.2}
# the editor's objects' models drawn as small things (teal), and those not named on the map
SMALL = {'marshal_post', 'open_passage', 'timing_table', 'flagpoles', 'banner_fence', 'pit_entry_fence'}
UNNAMED = {'cone', 'board', 'chair', 'kart', 'globe_lamp'}


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
    d.text((px, py), text, fill=colour, font=font(size, bold), anchor='mm', stroke_width=2, stroke_fill=(255, 255, 255))


def curve_points(o):
    return np.array([n['p'] for n in o.get('curve', [])], dtype=float).reshape(-1, 2)


def middle_of(P):
    """A line's point halfway along it."""
    if len(P) < 2:
        return P[0]
    line = LineString(P)
    q = line.interpolate(line.length / 2)
    return np.array([q.x, q.y])


def main(out):
    load = lambda f: kn5io.Model(os.path.join(TRACK, f))
    img = Image.new('RGB', (int((X1 - X0) * SCALE), int((Z1 - Z0) * SCALE)), COLOURS['grass'])
    d = ImageDraw.Draw(img)
    track = json.load(open(TRACK_JSON, encoding='utf-8'))
    objects = track['objects']['objects']

    # the trees: their crowns where the editor plants them
    crowns = [Polygon(np.c_[np.cos(a), np.sin(a)] * next((r for k, r in CROWNS.items() if t['model'].startswith(k)), 3.) * t.get('scale', 1.)
                      + (t['x'], t['z']))
              for t in track.get('trees', []) for a in [np.linspace(0, 2 * np.pi, 10, endpoint=False)]]
    fill(d, unary_union(crowns), COLOURS['trees'])

    # the ground's dirt roads, pond, roads and paving
    ground = load('track.kn5')
    for m in ground.meshes:
        n, mat = m['name'], ground.material_name(m)
        if not m['visible'] or not n.startswith(('1ROAD', '1KERB')):
            continue
        t = flat_triangles(m)
        if not len(t):
            continue
        kind = 'dirt' if mat == 'OutDirt' else 'water' if mat == 'OutWater' else 'paving' if mat in ('Pavers', 'OutTurf') else 'road'
        fill(d, footprint(t), COLOURS[kind], holes=kind == 'road')

    # what stands: by its meshes' names (the editor's object's name, then its parts')
    stand = load('track_objects.kn5')
    meshes = [m for m in stand.meshes if len(m['f']) and not m['name'].startswith('1')]
    named = lambda o: [m for m in meshes if m['name'] == o['name'] or m['name'].startswith(o['name'] + '_')]
    labels = []
    for o in sorted(objects, key=lambda o: o['model'] in SMALL):
        model, name, P = o['model'], o['name'], curve_points(o)
        if not len(P):
            continue
        if model == 'tyre_stack':
            ms = named(o)
            t = np.concatenate([triangles(m) for m in ms]) if ms else np.zeros((0, 3, 2))
            g = unary_union([Polygon(tr).buffer(.15) for tr in t[::6] if Polygon(tr).area > .001]) if len(t) else (
                LineString(P).buffer(.4) if len(P) > 1 else Point(P[0]).buffer(.6))
            fill(d, g, COLOURS['tyres'], holes=True)
            labels.append((name, *middle_of(P), 9, (0, 0, 0), False))
        elif model == 'light_pole':
            px, py = to_px(*P[0])
            d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=COLOURS['pole'], outline=(0, 0, 0))
            labels.append((name, P[0][0], P[0][1] + 3.5, 10, (110, 80, 0), True))
        elif model in ('cone', 'board'):
            px, py = to_px(*P[0])
            d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=COLOURS['cone'], outline=(90, 40, 0))
        elif model.startswith('fence'):
            for a, b in zip(P[:-1], P[1:]):
                d.line([to_px(*a), to_px(*b)], fill=COLOURS['fence'], width=3)
            labels.append((name, *middle_of(P), 10, COLOURS['fence'], True))
        elif model.endswith('gate'):
            ms = named(o)
            for tr in (np.concatenate([triangles(m) for m in ms]) if ms else []):
                d.line([to_px(*c) for c in tr[:2]], fill=COLOURS['gate'], width=4)
            labels.append((name, *P[0], 10, COLOURS['gate'], True))
        elif model == 'globe_lamp':
            for c in P:
                px, py = to_px(*c)
                d.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 255, 255), outline=(80, 80, 80))
        elif model == 'gantry':
            continue
        elif model not in UNNAMED:
            ms = named(o)
            if not ms:
                continue
            g = footprint(np.concatenate([triangles(m) for m in ms]))
            kind = 'banner' if model == 'banner_fence' else 'object' if model in SMALL else 'building'
            fill(d, g, COLOURS[kind], outline=(90, 40, 30) if kind == 'building' else None)
            c = g.representative_point()
            labels.append((name, c.x, c.y, 10, (60, 20, 10) if kind == 'building' else COLOURS[kind], kind == 'building'))
    # the parked karts and the marshals' chairs, small
    for o in objects:
        if o['model'] in ('kart', 'chair') and o.get('curve'):
            px, py = to_px(*curve_points(o)[0])
            d.rectangle((px - 2, py - 2, px + 2, py + 2), fill=COLOURS['object'])
    # the gantries, from their own models
    for f in sorted(os.listdir(TRACK)):
        if f.startswith('track_gantry_') and f.endswith('.kn5') and not f.endswith('_still.kn5'):
            mdl = load(f)
            g = unary_union([footprint(triangles(m)) for m in mdl.meshes if m['visible'] and len(m['f']) and not m['name'].startswith('1')])
            fill(d, g, COLOURS['gantry'])
            c = g.centroid
            labels.append(('gantry ' + f[len('track_gantry_'):-4], c.x, c.y - 3, 11, COLOURS['gantry'], True))

    for text, x, z, size, colour, bold in labels:
        if X0 < x < X1 and Z0 < z < Z1:
            label(d, text, x, z, size, colour, bold)
    # the title and the legend
    d.rectangle((0, 0, img.width, 64), fill=(255, 255, 255))
    d.text((12, 8), 'Techspeed FuelTech Velopark: what stands round the track', fill=(0, 0, 0), font=font(22, True))
    d.text((12, 38), 'Drawn from the Assetto Corsa build by tools/objects_map.py, each thing by its name in the editor; x runs left, z up. '
                     'Trees and grass as background colours.', fill=(60, 60, 60), font=font(12))
    keys = [('grass', 'grass'), ('trees', 'trees'), ('dirt', 'dirt road'), ('water', 'pond'), ('road', 'road'),
            ('paving', 'paving'), ('building', 'building'), ('tyres', 'tyre wall'), ('fence', 'fence'),
            ('gate', 'gate'), ('pole', 'light pole'), ('cone', 'cone, board'), ('gantry', 'gantry'),
            ('object', 'marshal post, kart, flagpoles'), ('banner', 'banner')]
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
    print('drew %s (%d by %d)' % (out, img.width, img.height))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
