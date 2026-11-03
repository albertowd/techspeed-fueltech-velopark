"""Side-by-side sheets from the renders, plus the kerb cross-sections."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'tools', 'render')
A = os.path.join(ROOT, 'assets')
BG, FG, DIM = (24, 26, 30), (236, 236, 232), (160, 164, 170)


def font(px, bold=False):
    return ImageFont.truetype('arialbd.ttf' if bold else 'arial.ttf', px)


def pair(left, right, l_label, r_label, out, h=900, title=None):
    a, b = [Image.open(p).convert('RGB') for p in (left, right)]
    a = a.resize((a.width * h // a.height, h), Image.LANCZOS)
    b = b.resize((b.width * h // b.height, h), Image.LANCZOS)
    pad, top = 24, 92
    im = Image.new('RGB', (a.width + b.width + pad * 3, h + top + pad), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 12), title, font=font(30, True), fill=FG)
    d.text((pad, top - 36), l_label, font=font(24), fill=DIM)
    d.text((a.width + pad * 2, top - 36), r_label, font=font(24), fill=DIM)
    im.paste(a, (pad, top))
    im.paste(b, (a.width + pad * 2, top))
    im.save(os.path.join(R, out))


def label(name, tags, out):
    im = Image.open(os.path.join(R, name)).convert('RGB')
    d = ImageDraw.Draw(im)
    f = font(30, True)
    for x, y, t in tags:
        box = d.textbbox((x, y), t, font=f)
        d.rectangle((box[0] - 10, box[1] - 7, box[2] + 10, box[3] + 7), fill=BG)
        d.text((x, y), t, font=f, fill=FG)
    im.save(os.path.join(R, out))


def sections(out):
    W, H = 1700, 900
    im = Image.new('RGB', (W, H), (246, 245, 241))
    d = ImageDraw.Draw(im)
    ink, grey, red = (34, 36, 40), (96, 98, 104), (176, 62, 66)
    d.text((50, 26), 'Cross-sections. Height is drawn 2x so the shape can be read.', font=font(26, True), fill=ink)
    sx, sz = 720, 720 * 2

    def dim_h(x1, x2, y, t):
        d.line([(x1, y), (x2, y)], fill=grey, width=2)
        for x in (x1, x2):
            d.line([(x, y - 8), (x, y + 8)], fill=grey, width=2)
        b = d.textbbox((0, 0), t, font=font(19))
        d.text(((x1 + x2 - b[2]) / 2, y + 10), t, font=font(19), fill=grey)

    # concrete kerb, track on the left, infield on the right
    x0, y0 = 240, 290
    ts = [0, .02, .18, .20, .95, 1.0, 1.03, 1.05, 1.08, 2.0]
    zs = [0, .012, .020, .024, .098, .108, .112, .104, .095, 0]
    pts = [(x0 + t * sx, y0 - z * sz) for t, z in zip(ts, zs)]
    d.text((50, 80), 'Concrete kerb (the section is taken along a rib)', font=font(24, True), fill=ink)
    d.polygon(pts + [(pts[-1][0], y0 + 86), (pts[0][0], y0 + 86)], fill=(214, 210, 200))
    d.rectangle((x0 - 190, y0, x0, y0 + 86), fill=(70, 70, 72))
    d.text((x0 - 180, y0 + 12), 'asphalt', font=font(20), fill=(230, 230, 230))
    d.line(pts, fill=ink, width=4)
    d.line(pts[3:8], fill=red, width=9)
    tg = np.array([1.12, 2.0])
    zg = np.interp(tg, ts, zs) - .055
    d.line([(x0 + t * sx, y0 - z * sz) for t, z in zip(tg, zg)], fill=grey, width=3)
    d.text((x0 + 1.55 * sx, y0 + 40), 'groove floor', font=font(19), fill=grey)
    y = y0 + 104
    dim_h(x0, x0 + .18 * sx, y, '0.18 m strip')
    dim_h(x0 + .2 * sx, x0 + 1.05 * sx, y, '0.85 m painted ramp')
    dim_h(x0 + 1.05 * sx, x0 + 2.0 * sx, y, '0.95 m ribs, 0.28 m pitch, 55 mm deep')
    d.text((x0 + 1.03 * sx - 66, y0 - .112 * sz - 36), '112 mm crest', font=font(21, True), fill=ink)

    # floor kerb, section along the track through the grooves
    x0, y0 = 240, 640
    hw = .068
    d.text((50, 480), 'Floor kerb (the section runs along the track, through the grooves of a red block)',
           font=font(24, True), fill=ink)
    c = [.3 * (k + .5) for k in range(4)]
    pts = []
    for u in np.linspace(0, 1.2, 601):
        z = 0.
        for cc in c:
            q = abs(u - cc)
            if q < hw:
                z = -.032 * (1 - q / hw)
        pts.append((x0 + u * sx, y0 - z * sz))
    d.polygon(pts + [(pts[-1][0], y0 + 130), (pts[0][0], y0 + 130)], fill=(214, 200, 196))
    d.rectangle((x0 - 190, y0, x0, y0 + 130), fill=(70, 70, 72))
    d.rectangle((x0 + 1.2 * sx, y0, x0 + 1.2 * sx + 190, y0 + 130), fill=(70, 70, 72))
    d.line(pts, fill=red, width=5)
    d.line([(x0 - 190, y0), (x0, y0)], fill=ink, width=4)
    d.line([(pts[-1][0], y0), (pts[-1][0] + 190, y0)], fill=ink, width=4)
    d.text((x0 - 180, y0 + 12), 'next block', font=font(20), fill=(230, 230, 230))
    dim_h(x0, x0 + 1.2 * sx, y0 + 152, '1.2 m block: 4 grooves on red, 5 on white. Top is level with the asphalt.')
    dim_h(x0 + .15 * sx - hw * sx, x0 + .15 * sx + hw * sx, y0 - 34, '136 mm')
    d.text((x0 + .45 * sx + 50, y0 - 40), '32 mm deep', font=font(21, True), fill=ink)
    d.text((x0 + 1.2 * sx + 215, y0 - 8), 'Block is 0.92 m across the track.\nGrooves are 0.74 m long.', font=font(21), fill=grey)
    im.save(os.path.join(R, out))


def stack(items, out, width=1570):
    pad, cap = 24, 46
    rows = []
    for p, t in items:
        a = Image.open(p).convert('RGB')
        rows.append((a.resize((width, a.height * width // a.width), Image.LANCZOS), t))
    im = Image.new('RGB', (width + pad * 2, sum(a.height + cap for a, _ in rows) + pad), BG)
    d = ImageDraw.Draw(im)
    y = 0
    for a, t in rows:
        d.text((pad, y + 10), t, font=font(24), fill=DIM)
        im.paste(a, (pad, y + cap))
        y += a.height + cap
    im.save(os.path.join(R, out))


def simple():
    r = lambda n: os.path.join(R, n + '.png')
    sections('6_cross_sections.png')
    pair(r('concrete_photo'), r('simple_concrete'), 'Detailed, 500 triangles per metre', 'Simplified',
         '7_concrete_simplified.png', title='Concrete kerb')
    pair(os.path.join(A, 'rumble-floor.jpeg'), r('simple_floor'), 'Photo', 'V grooves',
         '8_floor_simplified.png', title='Floor kerb')
    pair(r('pole_photo'), r('simple_pole'), 'First version, 3100 triangles', 'Rounder, within 1752 triangles',
         '9_pole_simplified.png', h=1400, title='Pole')
    stack([(r('simple_lod_near'), 'Near version at distance'), (r('simple_lod_far'), 'Far version at distance')],
          '10_kerbs_far_version.png')
    stack([(os.path.join(A, 'start-panel.jpeg'), 'Photo'), (r('gantry_current'), 'Current'), (r('gantry_lowered'), 'Lowered, legs on the panel ends'), (r('gantry_lowered_end'), 'Lowered, seen from the side')],
          '11_start_panel_lowered.png', width=1100)


if __name__ == '__main__':
    import sys
    if 'simple' in sys.argv:
        simple()
        sys.exit()
    pair(os.path.join(A, 'rumble-with-concrete.jpeg'), os.path.join(R, 'concrete_photo.png'), 'Photo', 'New model',
         '1_concrete_vs_photo.png', title='Concrete kerb')
    pair(os.path.join(A, 'rumble-floor.jpeg'), os.path.join(R, 'floor_photo.png'), 'Photo', 'New model',
         '2_floor_vs_photo.png', title='Floor kerb')
    pair(os.path.join(A, 'pole.jpeg'), os.path.join(R, 'pole_photo.png'), 'Photo', 'New model',
         '3_pole_vs_photo.png', h=1400, title='Pole')
    label('compare_pole.png', [(330, 1040, 'Current 8.4 m'), (800, 1040, 'New 17 m')], '4_pole_current_vs_new.png')
    label('compare_kerb.png', [(980, 215, 'Current kerb'), (30, 372, 'New concrete kerb'), (30, 800, 'New floor kerb')],
          '5_kerbs_current_vs_new.png')
    sections('6_cross_sections.png')
