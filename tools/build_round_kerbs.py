"""The yellow round kerbs of the middle of the track, in source/models/velopark_kerbs.glb.

    python tools/build_round_kerbs.py

The real track has a kerb of another kind round the S2 chicane and the link road R9: a little higher
than the ramp kerbs, rounded across, one long hump without blocks, painted yellow all over, the
road's edge line beside it. This lays it, WIDTH wide and HEIGHT high, its paint the ramp kerb's
made yellow:

- in place of the ramp kerbs S1_OUT and S2_EXIT, along the road's edge they lay on;
- in place of the middle of S2_LEFT (MIDDLE), the ramp kerb staying at its ends;
- along both sides of R9, but where it meets the other roads.

Its free ends grow out of the road's edge over TAPER, as the ramp kerbs' do, wider and higher
together; where it meets a ramp kerb, as wide as it, it takes the ramp kerb's shape over MORPH,
so the one carries on into the other. Its texture, source/textures/kerb_round.dds, is made here
from the ramp kerb's, mapped as it is. Run again, it finds the kerbs laid and changes nothing.
"""
import io, os, urllib.parse

import numpy as np
from PIL import Image
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

import build_paths as bp
import gltf_source
import kn5io

ROOT = bp.ROOT
KERBS = os.path.join(bp.MODELS, 'velopark_kerbs.glb')
TEXTURE = 'kerb_round.dds'
MATERIAL = 'KerbRound'
WIDTH = .6           # metres from the road's edge to the grass, as the ramp kerb
HEIGHT = .09         # at its middle
ACROSS = 16          # faces across it
STEP = .25           # metres between its rows along it
TAPER = 2.4          # metres over which a free end grows out of the road, two blocks as the ramp kerb
MORPH = 1.5          # and over which an end that meets a ramp kerb takes its shape
DROP = .1            # how far its back and square ends go into the ground
CORNER = 30.         # degrees a road's edge turns at where it ends, at a junction
TILE = 4.8           # metres along it the texture repeats over, as the ramp kerb's
DEPTH = 2.05 * .6 / 1.25    # and metres across it to a whole of the texture, as the ramp kerb's
RAMP = 'kerb_ramp.dds'
PAINT = 314          # rows of the ramp kerb's texture that are its paint, from its track side
YELLOW = (226., 186., 38.)
REPLACE = {'S1_OUT': (0., 1.), 'S2_EXIT': (0., 1.)}
MIDDLE = {'S2_LEFT': (.2, .8)}      # the part of the ramp kerb that becomes round, by length
SIDES = {'R9': '1ROAD_9'}           # roads with the kerb along both their free sides


def profile(t):
    """Height over the ground across it, t from its track side, 0, to its back, WIDTH."""
    return HEIGHT * np.sin(np.pi * np.clip(t / WIDTH, 0, 1))


def texture(ramp):
    """The ramp kerb's texture, its red and white blocks painted yellow: each pixel as much
    lighter or darker than the block it is in, the rubber on it staying dark and grey."""
    img = np.asarray(Image.open(io.BytesIO(ramp)).convert('RGB')).astype(float)
    paint = img[:PAINT]
    lum = paint @ [.299, .587, .114]
    red = np.clip((paint[..., 0] - paint[..., 1:].max(-1) - 10) / 60, 0, 1)
    usual = red * np.median(lum[red > .9]) + (1 - red) * np.median(lum[red < .1])
    rel = lum / usual
    # and every column as light as the next, so no block shows where one ended
    rel /= np.median(rel, axis=0)[None, :]
    yellow = np.array(YELLOW) * rel[..., None]
    # rubber: where it is darker than its block, it greys as it darkens
    dark = np.clip((.8 - rel) / .5, 0, 1)[..., None]
    img[:PAINT] = yellow * (1 - dark) + lum[..., None] * dark
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def resample(P, step):
    """Points along a line, step apart, with how far along it each is."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1))]
    s = np.linspace(0, d[-1], max(2, int(round(d[-1] / step)) + 1))
    return np.stack([np.interp(s, d, P[:, c]) for c in range(3)], 1), s


def outward(P, sign):
    """Square to the line, on the side sign says, flat."""
    t = np.gradient(P[:, [0, 2]], axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return sign * np.c_[t[:, 1], -t[:, 0]]


def ramp_edge(node, u0, u1):
    """The road's edge a ramp kerb lies along, from u0 to u1 of its texture along it, and
    which side of it the kerb lies on."""
    V = node['verts'].astype(float)
    on = np.abs(V[:, 7]) < 1e-4
    us = np.unique(np.round(V[on, 6], 4))
    us = us[(us >= u0 - 1e-6) & (us <= u1 + 1e-6)]
    E = np.array([V[on & (np.abs(V[:, 6] - u) < 1e-4), :3].mean(0) for u in us])
    # each row's furthest from the road
    away = []
    for u in us:
        row = V[np.abs(V[:, 6] - u) < 1e-4]
        away.append(row[np.argmax(row[:, 7]), :3])
    away = np.array(away) - E
    side = np.sign(np.sum(outward(E, 1) * away[:, [0, 2]]))
    return E, side


def section(node, u):
    """A ramp kerb's cross-section at u along it: metres from the road's edge, and height."""
    V = node['verts'].astype(float)
    row = V[(np.abs(V[:, 6] - u) < 1e-4) & (V[:, 7] >= -1e-4)]
    row = row[np.argsort(row[:, 7])]
    t = np.linalg.norm(row[:, [0, 2]] - row[0, [0, 2]], axis=1)
    return np.c_[np.maximum.accumulate(t), row[:, 1]]


def free_sides(node, others):
    """The edges of a road where no other road goes on from it, as lines, with the side of
    them the road is not on. Its outline is taken as the union of its faces, as a road's mesh
    may have faces on top of each other and corners twice over."""
    tris = bp.triangles(node, np.eye(4))
    me = bp.Surface(tris)
    shape = unary_union([Polygon(t[:, [0, 2]]).buffer(.01) for t in tris if Polygon(t[:, [0, 2]]).area > 1e-6]).buffer(-.01)
    if shape.geom_type == 'MultiPolygon':
        shape = max(shape.geoms, key=lambda g: g.area)
    ring = shape.exterior
    L = np.array([ring.interpolate(d).coords[0] for d in np.arange(0, ring.length, .25)])
    nxt = np.roll(L, -1, 0)
    d = nxt - L
    perp = np.c_[d[:, 1], -d[:, 0]] / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    mid = (L + nxt) / 2
    inside = contains_xy(shape, *(mid + .05 * perp).T)
    perp[inside] *= -1
    free = np.isnan(others.ys(mid + .3 * perp))
    # its height: the road's, just inside its edge
    y = me.ys(L - .05 * np.c_[np.roll(perp, 1, 0) + perp] / 2)
    ok = ~np.isnan(y)
    y = np.interp(np.arange(len(y)), np.nonzero(ok)[0], y[ok], period=len(y))
    L = np.c_[L[:, 0], y, L[:, 1]]
    # the runs of free edges, round the outline
    k = int(np.argmin(free)) if not free.all() else 0
    order = np.roll(np.arange(len(L)), -k)
    runs, cur = [], []
    for i in order:
        if free[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    lines = []
    for r in runs:
        pts = np.r_[L[r], L[(r[-1] + 1) % len(L)][None]]
        # only its longest stretch without a corner: a run can turn into a junction's mouth
        d = np.diff(pts[:, [0, 2]], axis=0)
        a = np.arctan2(d[:, 1], d[:, 0])
        turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi) > np.radians(CORNER)
        cuts = np.r_[0, np.nonzero(turn)[0] + 1, len(pts) - 1]
        size = [np.linalg.norm(np.diff(pts[i:j + 1, [0, 2]], axis=0), axis=1).sum() for i, j in zip(cuts, cuts[1:])]
        i = int(np.argmax(size))
        pts = pts[cuts[i]:cuts[i + 1] + 1]
        if size[i] < 3:
            continue
        side = np.sign((outward(pts, 1) * perp[r].mean(0)).sum(1).mean())
        lines.append((pts, side))
    return lines


def kerb(name, edge, side, ground, ends, template, mat):
    """The kerb along a road's edge: its top, a mesh cars drive on, and its sides, one they do not.
    ends says, for its start and its end, True if it goes down to the road there, or else the
    cross-section of the ramp kerb it meets there, as (metres from the road's edge, height)."""
    tapered = [e is True for e in ends]
    E, s = resample(edge, STEP)
    out = outward(E, side)
    t = np.linspace(0, WIDTH, ACROSS + 1)
    # its free ends grow out of the road's edge, wider and higher together
    f = np.ones(len(s))
    if tapered[0]:
        f = np.minimum(f, s / TAPER)
    if tapered[1]:
        f = np.minimum(f, (s[-1] - s) / TAPER)
    f = np.clip(f, 0, 1)
    xz = E[:, None, [0, 2]] + out[:, None, :] * (t[None, :] * f[:, None])[..., None]
    g = ground.ys(xz.reshape(-1, 2)).reshape(xz.shape[:2])
    back = np.where(np.isnan(g[:, -1]), E[:, 1], g[:, -1])
    line = E[:, 1, None] + (back - E[:, 1])[:, None] * t / WIDTH * f[:, None]
    base = np.fmax(line, g)
    base[:, 0] = E[:, 1]
    prof = np.tile(profile(t), (len(s), 1))
    for end, x in zip((0, -1), ends):
        if x is True:
            continue
        their = np.interp(t, x[:, 0], x[:, 1], right=np.nan) - base[end]
        their = np.clip(np.nan_to_num(their), 0, None)
        w = np.clip(1 - (s if end == 0 else s[-1] - s) / MORPH, 0, 1)
        w = w * w * (3 - 2 * w)
        prof = prof * (1 - w[:, None]) + their[None, :] * w[:, None]
    y = base + prof * f[:, None]
    top = np.dstack([xz[..., 0], y, xz[..., 1]])
    uv = np.dstack([np.repeat(s[:, None] / TILE, len(t), 1), np.repeat(t[None, :] / DEPTH, len(s), 0)])
    node = grid_node(template, '1ROAD_KERB_' + name, top, uv, mat, up=True)
    # its back into the ground, and the ends that do not go down to the road, square
    parts = []
    bottom = top[:, -1] - [0, DROP, 0]
    parts.append((np.stack([top[:, -1], bottom], 1), np.stack([uv[:, -1], uv[:, -1]], 1)))
    for end, tp in ((0, tapered[0]), (-1, tapered[1])):
        if not tp:
            low = top[end].copy()
            low[:, 1] = base[end] - DROP
            parts.append((np.stack([top[end], low], 1), np.stack([uv[end], uv[end] + [0, 0]], 1)))
    side_node = None
    for grid, guv in parts:
        n = grid_node(template, 'KERB_%s_SIDE' % name, grid, guv, mat, up=False, both=True)
        side_node = n if side_node is None else join(side_node, n)
    return node, side_node


def grid_node(template, name, grid, uv, mat, up, both=False):
    rows, cols = grid.shape[:2]
    P = grid.reshape(-1, 3)
    U = uv.reshape(-1, 2)
    idx = np.arange(len(P)).reshape(rows, cols)
    F = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            F += [(a, b, d), (a, d, c)]
    F = np.array(F, int)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    if up:
        down = fn[:, 1] < 0
        F[down] = F[down][:, ::-1]
    if both:
        # seen from either side: its own vertices for each, so each has its normals
        n = len(P)
        P, U = np.r_[P, P], np.r_[U, U]
        F = np.r_[F, F[:, ::-1] + n]
    N = np.zeros_like(P)
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    T = gltf_source.tangents(P, N, U, F)
    node = dict(template)
    node.update(name=name, children=[], material=mat, verts=np.c_[P, N, U, T].astype('<f4'), faces=F.astype('<u2'))
    bp.sphere(node)
    return node


def join(a, b):
    n = len(a['verts'])
    a['verts'] = np.r_[a['verts'], b['verts']]
    a['faces'] = np.r_[a['faces'].astype(int), b['faces'].astype(int) + n].astype('<u2')
    bp.sphere(a)
    return a


def cut(node, u0, u1):
    """The faces of a ramp kerb between u0 and u1 along it taken out."""
    V, F = node['verts'], node['faces'].astype(int)
    u = V[F, 6]
    gone = (u >= u0 - 1e-4).all(1) & (u <= u1 + 1e-4).all(1)
    bp.split(node, gone)


def save(k, path):
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)


def main():
    k = gltf_source.load_kn5(KERBS)
    root = k['root']['children']
    by = {n['name']: n for n in root}
    if any(k['materials'][n['material']]['name'] == MATERIAL for n in root if n['kind'] == 2):
        print('the round kerbs are laid already')
        return
    # its texture and material, as the ramp kerb's but for them
    ramp_tex = next(t['data'] for t in k['textures'] if t['name'] == RAMP)
    tex = kn5io.dds(texture(ramp_tex), 'DXT1')
    open(os.path.join(ROOT, 'source', 'textures', TEXTURE), 'wb').write(tex)
    k['textures'].append(dict(name=TEXTURE, active=1, data=tex))
    ramp = next(m for m in k['materials'] if m['name'] == 'KerbRamp')
    mat = dict(ramp, name=MATERIAL, props=[dict(p) for p in ramp['props']],
               textures=[dict(t, name=TEXTURE) if t['sampler'] == 'txDiffuse' else dict(t) for t in ramp['textures']])
    k['materials'].append(mat)
    mi = len(k['materials']) - 1

    # the ground under it: the track's, and the kerbs' beds and patches
    track = gltf_source.load_kn5(bp.TRACK)
    tris, roads = [], []
    for model in (track, k):
        for n, m in bp.meshes(model):
            nm = n['name']
            if nm[:1].isdigit() and 'WALL' not in nm and 'KERB' not in nm[:12] and len(n['faces']):
                t = bp.triangles(n, m)
                if len(t):
                    tris.append(t)
                    if model is track and nm.startswith('1ROAD_'):
                        roads.append((nm, t))
    ground = bp.Surface(np.concatenate(tris))
    template = by['1ROAD_KERB_S1_OUT']
    new = []

    for name, (a, b) in list(REPLACE.items()) + list(MIDDLE.items()):
        top = by['1ROAD_KERB_' + name]
        u = top['verts'][:, 6]
        rows = np.unique(np.round(u, 4))
        lo, hi = u.min(), u.max()
        # to whole rows of the ramp kerb, so its cut ends lie under the round kerb's square ones
        ua = rows[np.argmin(np.abs(rows - (lo + a * (hi - lo))))]
        ub = rows[np.argmin(np.abs(rows - (lo + b * (hi - lo))))]
        edge, side = ramp_edge(top, ua, ub)
        whole = name in REPLACE
        own = name if whole else name + '_MID'
        new += kerb(own, edge, side, ground, (True, True) if whole else (section(top, ua), section(top, ub)), template, mi)
        for n in (top, by['KERB_%s_SIDE' % name]):
            if whole:
                root.remove(n)
            else:
                cut(n, ua, ub)
        print('  %-10s %s, %.1f m' % (own, 'in place of the ramp kerb' if whole else 'in the middle of the ramp kerb', np.linalg.norm(np.diff(edge[:, [0, 2]], axis=0), axis=1).sum()))

    for name, road in SIDES.items():
        node = next(n for n, m in bp.meshes(track) if n['name'] == road)
        others = bp.Surface(np.concatenate([t for nm, t in roads if nm != road]))
        lines = free_sides(node, others)
        for pts, side in sorted(lines, key=lambda x: -x[0][:, 2].mean()):
            own = '%s_%s' % (name, 'NORTH' if pts[:, 2].mean() == max(p[:, 2].mean() for p, _ in lines) else 'SOUTH')
            new += kerb(own, pts, side, ground, (True, True), template, mi)
            print('  %-10s along %s, %.1f m' % (own, road, np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1).sum()))

    root.extend(new)
    save(k, KERBS)
    print('%d round kerbs laid in %s' % (len(new) // 2, os.path.relpath(KERBS, ROOT)))


if __name__ == '__main__':
    main()
