"""Draw the track from above with the name of every road and kerb on it, to talk about them by
name: python tools/track_map.py [out.png]. Reads the Assetto Corsa build in build/ac (python
the Top Race Editor builds it), x to the left and z up, as it lies on the ground."""
import os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from shapely.ops import unary_union

import kn5io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.path.join(ROOT, 'build', 'ac', 'content', 'tracks', 'techspeed_fueltech_velopark')
OUT = os.path.join(ROOT, 'docs', 'track_map.png')
SCALE = 9.            # pixels a metre
GRASS, ROAD, PADDOCK, PAVING = (196, 214, 170), (150, 150, 150), (178, 172, 160), (206, 200, 186)
KERBS = {'KerbConcrete': ((40, 90, 200), 'concrete kerb'), 'KerbRamp': ((214, 50, 50), 'ramp kerb'),
         'KerbFloor': ((230, 140, 20), 'floor kerb'),
         'KerbRound': ((235, 195, 0), 'round kerb'), 'Rumble1': ((200, 60, 200), 'old kerb')}


def font(size, bold=False):
    for name in (('arialbd.ttf', 'DejaVuSans-Bold.ttf') if bold else ('arial.ttf', 'DejaVuSans.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def triangles(m):
    """The mesh's triangles in x, z, without the ones blanked out of it."""
    f = m['f']
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    t = m['v'][f][:, :, [0, 2]]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return t[np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-8]


def shape(tri):
    return unary_union([Polygon(t) for t in tri]).buffer(.05).buffer(-.05)


def kerb_label(n):
    """A kerb's name on the map: the editor's 1ROAD_KERB_E_<road>_<n> as <road>.<n> (17.3: road 17's fourth kerb)."""
    rest = n[len('1ROAD_KERB_'):] if n.startswith('1ROAD_KERB_') else n.replace('1ROAD_', 'R')
    if rest.startswith('E_') and '_' in rest[2:]:
        road, i = rest[2:].rsplit('_', 1)
        return f'{road}.{i}'
    return rest


def largest(g):
    return max(getattr(g, 'geoms', [g]), key=lambda p: p.area)


def main(out):
    # (the editor's build: the roads, runoffs and the kids track's in track.kn5, the kerbs in track_kerbs.kn5)
    model = kn5io.Model(os.path.join(TRACK, 'track.kn5'))
    kerbs = kn5io.Model(os.path.join(TRACK, 'track_kerbs.kn5'))
    roads, kerb_shapes, paving = [], [], []
    for mdl in (model, kerbs):
        for m in mdl.meshes:
            n, mat = m['name'], mdl.material_name(m)
            if mat in KERBS:
                t = triangles(m)
                if len(t):
                    kerb_shapes.append((kerb_label(n), mat, t))
            elif mat == 'Pavers':
                t = triangles(m)
                if len(t):
                    paving.append((n if 'PATCH' not in n and 'PAVERS' not in n else None, t))
            elif n.startswith('1ROAD') and mat.startswith('Tarmac'):
                t = triangles(m)
                if len(t):
                    roads.append((n, t))
    pad = np.zeros((0, 3, 2))
    every = np.concatenate([t for _, t in roads] + [t for _, t in paving])
    lo, hi = every.reshape(-1, 2).min(0) - 10, every.reshape(-1, 2).max(0) + 10
    W, H = ((hi - lo) * SCALE).astype(int) + (0, 120)
    # x runs left and z up, so the track lies as it does on the ground
    P = lambda q: [(float((hi[0] - x) * SCALE), float((hi[1] - z) * SCALE) + 120) for x, z in q]
    im = Image.new('RGB', (int(W), int(H)), GRASS)
    d = ImageDraw.Draw(im)
    for t in pad:
        d.polygon(P(t), fill=PADDOCK)
    for _, tri in roads:
        for t in tri:
            d.polygon(P(t), fill=ROAD)
    for _, tri in paving:
        for t in tri:
            d.polygon(P(t), fill=PAVING)
    # the road meshes' outlines, so where one ends and the next starts shows
    for name, tri in roads:
        if name:
            for g in getattr(shape(tri), 'geoms', [shape(tri)]):
                d.line(P(np.array(g.exterior.coords)), fill=(110, 110, 110), width=1)
    for _, mat, tri in kerb_shapes:
        for t in tri:
            d.polygon(P(t), fill=KERBS[mat][0])
    name_font = font(20, True)
    for name, tri in roads + paving:
        if not name:
            continue
        g = largest(shape(tri))
        if g.area < 2:
            continue
        x, y = P([g.representative_point().coords[0]])[0]
        d.text((x, y), name.replace('1ROAD_RUNOFF_', 'RUNOFF ').replace('1ROAD_PATH_', 'PATH ').replace('1ROAD_', 'R').replace('1KERB_', 'R'), font=name_font, fill=(30, 30, 30), anchor='mm',
               stroke_width=2, stroke_fill=(235, 235, 235))
    for label, mat, tri in kerb_shapes:
        c = tri.reshape(-1, 2)
        p = c[np.linalg.norm(c - c.mean(0), axis=1).argmin()]
        x, y = P([p])[0]
        d.text((x, y), label, font=name_font, fill=KERBS[mat][0], anchor='mm', stroke_width=3, stroke_fill='white')
    # key
    d.rectangle((0, 0, W, 110), fill='white')
    d.text((20, 14), 'Techspeed FuelTech Velopark: roads and kerbs by name', font=font(30, True), fill=(20, 20, 20))
    d.text((20, 58), 'Roads are the 1ROAD_ meshes of track.kn5 (R17 = 1ROAD_17); kerbs are the 1ROAD_KERB_ meshes of '
                     'track_kerbs.kn5, by their road and number (17.3 = 1ROAD_KERB_E_17_3), as the editor builds them. x runs left, z up.',
           font=font(17), fill=(60, 60, 60))
    x = 20
    for col, what in [(ROAD, 'road'), (PAVING, 'paving')] + [KERBS[k] for k in KERBS if any(m == k for _, m, _ in kerb_shapes)]:
        d.rectangle((x, 86, x + 22, 104), fill=col, outline=(80, 80, 80))
        d.text((x + 30, 84), what, font=font(17), fill=(40, 40, 40))
        x += 60 + int(d.textlength(what, font=font(17)))
    # 10 m
    d.line((W - 140, 95, W - 140 + 10 * SCALE * 5, 95), fill=(20, 20, 20), width=4)
    d.text((W - 140, 70), '50 m', font=font(17), fill=(20, 20, 20))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im.save(out, optimize=True)
    print('%s: %d roads, %d paved, %d kerbs, %dx%d' % (out, sum(1 for n, _ in roads if n), len(paving), len(kerb_shapes), W, H))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
