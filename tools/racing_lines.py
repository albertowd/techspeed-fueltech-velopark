"""The layouts' racing lines as data (TRACK_FORMAT.md, "Racing lines"): each layout's fast line and pit lane line, as
Béziers in the objects' curve form, the pit line forking from the fast line and merging back into it.

    python tools/racing_lines.py read [--write]     (today's AC AI lines, track/ac, read into objects.json once)

read: each layout's fast line (ai/fast_lane.ai) and pit lane line (ai/pit_lane.ai) fitted with as few points as keep
the curve within TOLERANCE of the line as AC drives it; the pit line from where it leaves the fast line to where it
joins it again (`leaves`, `joins`: metres along the fast line), its ends on the fast line, their handles along it.
With --write, into tools/reference/objects.json's layouts, through the road editor's own objects store (its format,
a backup of the file before).
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AC_MADE = os.path.join(ROOT, 'track', 'ac')
OBJECTS = os.path.join(ROOT, 'tools', 'reference', 'objects.json')
EDITOR = os.environ.get('TOP_RACE_EDITOR', os.path.join(os.path.dirname(ROOT), 'top-race-editor'))
TOLERANCE = .05       # metres the curve may lie off the line it is read from
FIRST_STEP = 25.      # metres between the points it starts from, before adding where it lies off
NEAR = .3             # metres: a pit line this near the fast line runs on it (before its fork, after its merge)


def read_ai(path):
    """An AC AI line's points, (n, 3), in the track's coordinates."""
    b = open(path, 'rb').read()
    n = struct.unpack_from('<4i', b, 0)[1]
    return np.frombuffer(b, '<f4', n * 5, 16).reshape(-1, 5)[:, :3].astype(float)


def bezier(p0, h0, h1, p1, t):
    """Points of a cubic Bézier at t (m,), from p0 (its out handle's end h0) to p1 (its in handle's end h1)."""
    t = t[:, None]
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * h0 + 3 * u * t * t * h1 + t ** 3 * p1


def handles(P, keys, closed, ends=None):
    """Each key point's handles (in, out), relative to it: along the line's way there (from the key before to the one
    after), a third of the way to its neighbour on either side. ends: the first and last key's way, if given."""
    K = P[keys]
    n = len(K)
    out = []
    for i in range(n):
        a = K[i - 1] if (closed or i > 0) else K[i]
        b = K[(i + 1) % n] if (closed or i < n - 1) else K[i]
        way = b - a
        if ends is not None and not closed and i in (0, n - 1):
            way = ends[0 if i == 0 else 1]
        way = way / max(np.linalg.norm(way), 1e-9)
        back = np.linalg.norm(K[i] - K[i - 1]) if (closed or i > 0) else 0.
        fwd = np.linalg.norm(K[(i + 1) % n] - K[i]) if (closed or i < n - 1) else 0.
        out.append((-way * back / 3, way * fwd / 3))
    return out


def curve_points(curve, closed, per=40):
    """A curve (objects' form) sampled, (m, 2)."""
    out = []
    n = len(curve)
    for i in range(n if closed else n - 1):
        a, b = curve[i], curve[(i + 1) % n]
        p0, p1 = np.array(a['p']), np.array(b['p'])
        out.append(bezier(p0, p0 + a['out'], p1 + b['in'], p1, np.linspace(0, 1, per, endpoint=False)))
    if not closed:
        out.append(np.array([curve[-1]['p']]))
    return np.concatenate(out)


def as_curve(P, keys, closed, ends=None):
    hs = handles(P, keys, closed, ends)
    return [dict(p=[round(float(P[k][0]), 4), round(float(P[k][1]), 4)],
                 **{'in': [round(float(h[0][0]), 4), round(float(h[0][1]), 4)],
                    'out': [round(float(h[1][0]), 4), round(float(h[1][1]), 4)]}) for k, h in zip(keys, hs)]


def fit(P, closed, ends=None, tol=TOLERANCE):
    """A line's points (n, 2) as a curve with as few points as keep it within tol of them: points every FIRST_STEP
    metres, then, where a stretch between two of them lies further off than tol, the line's point furthest off added
    between them, till none does. The curve, and how far off it lies at most."""
    from scipy.spatial import cKDTree
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keys = sorted(set(np.searchsorted(s, np.arange(0., s[-1], FIRST_STEP)).tolist()) | ({len(P) - 1} if not closed else set()))
    if not closed and keys[0] != 0:
        keys = [0] + keys
    while True:
        curve = as_curve(P, keys, closed, ends)
        n = len(keys)
        worst = []
        for i in range(n if closed else n - 1):
            a, b = keys[i], keys[(i + 1) % n]
            idx = np.arange(a, b + 1) if b > a else np.r_[np.arange(a, len(P)), np.arange(0, b + 1)]
            c0, c1 = curve[i], curve[(i + 1) % n]
            p0, p1 = np.array(c0['p']), np.array(c1['p'])
            seg = bezier(p0, p0 + c0['out'], p1 + c1['in'], p1, np.linspace(0, 1, 200))
            d, _ = cKDTree(seg).query(P[idx])
            j = int(np.argmax(d))
            worst.append((float(d[j]), int(idx[j]), i))
        bad = [(d, k) for d, k, _ in worst if d > tol and k not in keys]
        if not bad:
            return curve, max(d for d, _, _ in worst)
        keys = sorted(set(keys) | {k for _, k in bad})


def along(F, p):
    """Where p (x, z) is along a closed line F (n, 2): metres from its start to its nearest point, and that point's index."""
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(F, axis=0), axis=1))]
    i = int(np.argmin(np.linalg.norm(F - p, axis=1)))
    return float(s[i]), i


def read_layout(layout):
    """A layout's lines from its AI files: {'fast': ..., 'pit': ...} (pit left out if it has none), and how far each
    curve lies off the line it was read from at most."""
    folder = os.path.join(AC_MADE, layout, 'ai')
    F = read_ai(os.path.join(folder, 'fast_lane.ai'))[:, [0, 2]]
    if np.linalg.norm(F[0] - F[-1]) < 1.:
        F = F[:-1]                                              # (its last point its first again)
    fast, off_fast = fit(F, True)
    lines, off = dict(fast=dict(curve=fast, closed=True)), dict(fast=off_fast)
    pit_path = os.path.join(folder, 'pit_lane.ai')
    if os.path.exists(pit_path):
        from scipy.spatial import cKDTree
        Q = read_ai(pit_path)[:, [0, 2]]
        d, _ = cKDTree(F).query(Q)
        away = np.flatnonzero(d > NEAR)
        if len(away):
            a, b = max(away[0] - 1, 0), min(away[-1] + 1, len(Q) - 1)      # (its last point on the fast line, its first back on it)
            leaves, i = along(F, Q[a])
            joins, j = along(F, Q[b])
            Qp = Q[a:b + 1].copy()
            Qp[0], Qp[-1] = F[i], F[j]                                    # (its ends the fast line's there)
            way = lambda k: F[(k + 1) % len(F)] - F[k - 1]
            pit, off_pit = fit(Qp, False, ends=(way(i), way(j)))
            lines['pit'] = dict(curve=pit, closed=False, leaves=round(leaves, 3), joins=round(joins, 3))
            off['pit'] = off_pit
    return lines, off


def layouts():
    return sorted(d for d in os.listdir(AC_MADE) if os.path.exists(os.path.join(AC_MADE, d, 'ai', 'fast_lane.ai')))


def read(write=False):
    got = {}
    for lay in layouts():
        lines, off = read_layout(lay)
        got[lay] = lines
        print('%-22s fast %3d points (off %.3f m)%s' % (lay, len(lines['fast']['curve']), off['fast'],
              ', pit %d points (off %.3f m), leaves at %.1f m, joins at %.1f m' % (
                  len(lines['pit']['curve']), off['pit'], lines['pit']['leaves'], lines['pit']['joins']) if 'pit' in lines else ''))
    if write:
        sys.path.insert(0, os.path.join(EDITOR, 'engine'))
        from pathlib import Path
        from top_race_engine.objects import ObjectsStore
        store = ObjectsStore(Path(OBJECTS), Path(ROOT, 'build', 'cache', 'roads_backup'), seed=lambda: {})
        data, _ = store.load()
        for lay, lines in got.items():
            if lay in data['layouts']:
                data['layouts'][lay]['lines'] = lines
        store.save(data)
        print('written into %s' % os.path.relpath(OBJECTS, ROOT))
    return got


if __name__ == '__main__':
    if sys.argv[1:2] == ['read']:
        read('--write' in sys.argv)
    else:
        print(__doc__)
