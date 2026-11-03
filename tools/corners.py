"""Number the corners of a layout from its AI line and map the kerbs near them."""
import os, struct, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kn5_scan import read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.path.join(ROOT, 'base', 'velopark-0.5.0', 'content', 'tracks', 'techspeed_velopark')


def ai_line(layout):
    b = open(os.path.join(TRACK, layout, 'ai', 'fast_lane.ai'), 'rb').read()
    ver, n, lap, samples = struct.unpack_from('<4i', b, 0)
    d = np.frombuffer(b, dtype='<f4', count=n * 5, offset=16).reshape(-1, 5)
    return d[:, :3].copy()


def corners(p, min_turn=25):
    """Runs of the line that keep turning the same way. Returns start, end, signed angle."""
    seg = np.roll(p, -1, 0) - p
    L = np.linalg.norm(seg[:, [0, 2]], axis=1)
    h = np.unwrap(np.arctan2(seg[:, 2], seg[:, 0]))
    win = 12                                    # samples either side
    rate = (np.roll(h, -win) - np.roll(h, win))
    rate[:win] = rate[win]; rate[-win:] = rate[-win - 1]
    turning = np.abs(rate) > np.radians(8)
    out, i, n = [], 0, len(p)
    while i < n:
        if turning[i]:
            j = i
            while j + 1 < n and turning[j + 1] and np.sign(rate[j + 1]) == np.sign(rate[i]):
                j += 1
            ang = np.degrees(h[j] - h[i])
            if abs(ang) >= min_turn:
                out.append((i, j, ang))
            i = j + 1
        else:
            i += 1
    return out


def kerbs():
    b, ver, tex, mats, nodes = read(os.path.join(TRACK, 'velopark.kn5'))
    out = []
    for n in nodes:
        if mats[n['mat']]['name'] != 'Rumble1':
            continue
        d = np.frombuffer(b, dtype='<f4', count=n['vc'] * 11, offset=n['vp']).reshape(-1, 11)[:, :3].copy()
        out.append((n['name'], d))
    return out


if __name__ == '__main__':
    layout = sys.argv[1] if len(sys.argv) > 1 else 'vp_1500'
    p = ai_line(layout)
    cs = corners(p)
    ks = kerbs()
    print(len(p), 'samples')
    all_xz = p[:, [0, 2]]
    lo, hi = all_xz.min(0) - 25, all_xz.max(0) + 25
    S = 1500 / (hi - lo).max()
    W, H = ((hi - lo) * S).astype(int) + 40
    im = Image.new('RGB', (int(W), int(H)), (250, 249, 245))
    d = ImageDraw.Draw(im)
    # z runs up the map so north stays at the top of the screen when x runs right
    P = lambda q: tuple(((q - lo) * S + 20))
    f = ImageFont.truetype('arialbd.ttf', 24)
    f2 = ImageFont.truetype('arial.ttf', 15)
    for name, k in ks:
        pts = k[:, [0, 2]]
        for q in pts[::3]:
            x, y = P(q)
            d.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(214, 72, 72))
    d.line([P(q) for q in all_xz] + [P(all_xz[0])], fill=(40, 42, 48), width=3)
    x, y = P(all_xz[0])
    d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(30, 150, 70))
    d.text((x + 10, y - 10), 'start', font=f2, fill=(30, 110, 60))
    a = all_xz[30] - all_xz[0]
    d.line([P(all_xz[0]), P(all_xz[60])], fill=(30, 150, 70), width=6)
    for k, (i, j, ang) in enumerate(cs):
        m = (i + j) // 2
        q = all_xz[m]
        x, y = P(q)
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=(30, 90, 190))
        d.text((x - 8 if k < 9 else x - 14, y - 13), str(k + 1), font=f, fill='white')
        near = []
        for name, kp in ks:
            dist = np.linalg.norm(kp[:, [0, 2]][:, None] - all_xz[i:j + 1:4][None], axis=2).min()
            if dist < 7:
                side = kp[:, [0, 2]].mean(0)
                near.append('%s(%.1f)' % (name, dist))
        print('T%d samples %d-%d turn %+.0f at (%.0f, %.0f) y %.2f kerbs: %s' % (k + 1, i, j, ang, q[0], q[1], p[m, 1], ' '.join(near)))
    im.save(os.path.join(ROOT, 'tools', 'render', 'corners_%s.png' % layout))
