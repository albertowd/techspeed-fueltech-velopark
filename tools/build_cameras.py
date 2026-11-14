"""Replay cameras for each layout of the Assetto Corsa track: source/track/<layout>/data/cameras.ini.

    python tools/build_cameras.py [layout ...] [--render]

Reads the Assetto Corsa build (run tools/build_tracks.py ac first) for what stands round the
track, and each layout's AI line for its lap. Cameras stand where Assetto Corsa's replays can use
them: in the race control tower's upper room, on the start gantry over the grid, in the judges'
tower, and round the lap off the roads, where nothing of the track is in their way (trees,
buildings, tyre walls; glass and wire fences are seen through). Each one is tried against every
point of the lap it can see: how far, from in front or behind, from how high. The lap is then
cut between them so each corner and straight goes to the camera that sees it best, with few cuts:
the race control room takes the start straight, the gantry the grid, as they see them.

--render also draws, in build/cameras, a map of each layout's cameras and a picture from each
camera of a kart at the middle of its stretch, in plain colours, to check them by eye.

Needs bpy (Blender as a Python module) for its ray casting.
"""
import configparser, json, math, os, sys

import bpy  # noqa: F401  (loads mathutils)
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import kn5io
from corners import corners

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = 'techspeed_fueltech_velopark'
AC = os.path.join(ROOT, 'build', 'ac', 'content', 'tracks', TRACK)
SOURCE = os.path.join(ROOT, 'source', 'track')
OUT = os.path.join(ROOT, 'build', 'cameras')

STEP = 2.                # metres between the lap samples
MAX_DIST = 110.         # farthest a camera follows a kart, m
KART = (.45, 1.)        # heights over the road a kart is seen at: its body, the driver's head
FRAME = 9.              # width of the picture round the kart, m, which sets each camera's zoom
SWITCH = 4.             # cost of a cut, in the cost of one lap sample
MIN_RUN = 12            # fewest lap samples a camera keeps (24 m)
STATIONS = 12.          # metres between the places cameras are tried along the lap
SIDES = (6., 10., 15., 22.)     # how far out from the AI line
HEIGHTS = (3., 6.)      # over the ground
SEE_THROUGH = {'OutWire', 'Fence001', 'ConeOrange', 'OutGlass'}

# the rooms cameras stand in, at a standing man's eyes: (name, x0, x1, z0, z1, eye height)
ROOMS = [('Race control', -49.8, -44.2, 77.7, 83.3, 4.8 + 1.6),
         ("Judges' tower", -169.6, -167., 158.5, 161.1, 5.71 + 1.5)]
ROOM_REACH = 90.        # a room's camera is preferred for karts this near it
ROOM_BONUS = 1.
GRID_BONUS = (1.2, .9)  # the gantry's camera, one beside the grid
LONG_LENS = 60.         # the rooms and the gantry watch from afar: their distance counts less

_models = {}


def model(fn):
    if fn not in _models:
        _models[fn] = kn5io.Model(os.path.join(AC, fn))
    return _models[fn]


def layout_files(lay):
    c = configparser.ConfigParser()
    c.read(os.path.join(AC, 'models_%s.ini' % lay))
    return [c[s]['FILE'] for s in c.sections()]


def colour(mat):
    """A plain colour for the check pictures, from the material's name."""
    n = mat.lower()
    for keys, rgb in ((('euca', 'leaf', 'tree', 'palm', 'bamboo', 'shrub'), (.16, .32, .14)),
                      (('grass', 'ground', 'sat', 'far'), (.42, .58, .3)),
                      (('tarmac', 'asphalt', 'road', 'paver', 'path'), (.36, .36, .38)),
                      (('rumble', 'kerb'), (.75, .25, .22)),
                      (('tyre', 'tire', 'rubber'), (.08, .08, .08)),
                      (('dirt',), (.66, .55, .38)),
                      (('out', 'wall', 'roof', 'pillar', 'bld', 'plaster'), (.78, .52, .45))):
        if any(k in n for k in keys):
            return rgb
    return (.6, .6, .6)


class Scene:
    """What stands round a layout, to cast rays at: everything seen (occluders), the ground under
    the cameras, and the roads they must not stand over."""

    def __init__(s, files):
        occ_v, occ_f, occ_c, gr_v, gr_f, rd_v, rd_f = [], [], [], [], [], [], []
        n_occ = n_gr = n_rd = 0
        s.dummies = {}
        for fn in files:
            m = model(fn)
            s.dummies.update(m.dummies)
            for x in m.meshes:
                name, mat = x['name'], m.materials[x['mat']]
                v, f = x['v'][:, :3], x['f'].astype(np.int64)
                if not len(f):
                    continue
                if name[:1].isdigit() and not name.startswith('1WALL'):
                    gr_v.append(v); gr_f.append(f + n_gr); n_gr += len(v)
                    if name.startswith(('1ROAD', '1KERB')) and not name.startswith(('1ROAD_PADDOCK', '1ROAD_PATH')):
                        rd_v.append(v); rd_f.append(f + n_rd); n_rd += len(v)
                if not (x['visible'] and x['renderable']) or mat['blend'] or mat['name'] in SEE_THROUGH:
                    continue
                if 0 < x['lod_out'] <= 60:      # a near copy of something with a far one
                    continue
                occ_v.append(v); occ_f.append(f + n_occ); n_occ += len(v)
                occ_c.append(np.tile(colour(mat['name']), (len(f), 1)))
        cat = lambda a: np.concatenate(a)
        s.occ_v, s.occ_f, s.occ_c = cat(occ_v), cat(occ_f), cat(occ_c)
        s.occ = BVHTree.FromPolygons(s.occ_v.tolist(), s.occ_f.tolist(), all_triangles=True)
        s.ground = BVHTree.FromPolygons(cat(gr_v).tolist(), cat(gr_f).tolist(), all_triangles=True)
        s.roads = BVHTree.FromPolygons(cat(rd_v).tolist(), cat(rd_f).tolist(), all_triangles=True)

    def ground_y(s, x, z):
        hit = s.ground.ray_cast(Vector((x, 300., z)), Vector((0., -1., 0.)), 600.)
        return None if hit[0] is None else hit[0].y

    def over_road(s, x, z, r=2.):
        down = Vector((0., -1., 0.))
        return any(s.roads.ray_cast(Vector((x + dx, 300., z + dz)), down, 600.)[0] is not None
                   for dx, dz in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)))

    def free(s, p, clear=1.):
        """Room round a camera, and open sky over it (not under a tree or a roof)."""
        p = Vector(p)
        if s.occ.find_nearest(p, clear)[0] is not None:
            return False
        return s.occ.ray_cast(p, Vector((0., 1., 0.)), 40.)[0] is None

    def sees(s, p, targets):
        """Which targets a camera at p sees, and how far they are."""
        d = np.linalg.norm(targets - p, axis=-1)
        seen = np.zeros(len(targets), bool)
        P = Vector(p)
        for i in np.nonzero(d < MAX_DIST)[0]:
            t = targets[i]
            for h in KART:
                q = Vector((t[0], t[1] + h, t[2]))
                dv = q - P
                L = dv.length
                if s.occ.ray_cast(P, dv / L, L - .3)[0] is None:
                    seen[i] = True
                    break
        return seen, d


def ai_line(lay):
    b = open(os.path.join(SOURCE, lay, 'ai', 'fast_lane.ai'), 'rb').read()
    n = int.from_bytes(b[4:8], 'little')
    d = np.frombuffer(b, dtype='<f4', count=n * 5, offset=16).reshape(-1, 5)
    p, dist = d[:, :3].astype(float), d[:, 3].astype(float)
    total = dist[-1] + np.linalg.norm(p[0] - p[-1])
    return p, dist / total, total


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


def fov(d):
    return math.degrees(2 * math.atan(FRAME / 2 / max(d, 1.)))


class Cam:
    def __init__(s, name, pos, kind='lap', bonus=None):
        s.name, s.pos, s.kind = name, np.asarray(pos, float), kind
        s.bonus = bonus         # (lap sample mask, how much it is preferred there)


def candidates(sc, line, fwd, lay_corners, grid_zone, layout_dummies):
    cams = []
    # the rooms: their middle and toward each wall
    for name, x0, x1, z0, z1, y in ROOMS:
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        for fx, fz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            cams.append(Cam(name, (cx + fx * ((x1 - x0) / 2 - .6), y, cz + fz * ((z1 - z0) / 2 - .6)), 'room'))
    # the start gantries, over the road, a little over their tops, and beside the grid's front
    for fn in ('velopark_gantry_1500.kn5', 'velopark_gantry_1000.kn5'):
        v = np.concatenate([x['v'][:, :3] for x in model(fn).meshes])
        lo, hi = v.min(0), v.max(0)
        for t in (.3, .5, .7):
            cams.append(Cam('Start grid', (lo[0] + t * (hi[0] - lo[0]), hi[1] + .6, (lo[2] + hi[2]) / 2), 'gantry'))
    front = layout_dummies['AC_START_0'][3, :3]
    k = np.linalg.norm(line - front, axis=1).argmin()
    for ahead in (8., 15.):
        a = line[(k + int(ahead / 2)) % len(line)]
        side = unit(np.cross(fwd[k], (0, 1, 0)))
        for sd in (-1, 1):
            for off in (7., 11.):
                for h in (4., 6.):
                    x, z = a[0] + sd * off * side[0], a[2] + sd * off * side[2]
                    g = sc.ground_y(x, z)
                    if g is not None and not sc.over_road(x, z):
                        cams.append(Cam('Start grid', (x, g + h, z), 'grid'))
    # round the lap, off the road on either side
    n = len(line)
    seg = np.linalg.norm(np.roll(line, -1, 0) - line, axis=1)
    station = np.cumsum(seg) // STATIONS
    picks = sorted(set(np.searchsorted(station, np.unique(station))))
    for i in picks:
        side = unit(np.cross(fwd[i], (0, 1, 0)))
        for sd in (-1, 1):
            for off in SIDES:
                x, z = line[i, 0] + sd * off * side[0], line[i, 2] + sd * off * side[2]
                g = sc.ground_y(x, z)
                if g is None or sc.over_road(x, z):
                    continue
                for h in HEIGHTS:
                    cams.append(Cam('lap', (x, g + h, z)))
    # keep those with room round them and sky over them (the rooms and gantries are placed by hand)
    return [c for c in cams if c.kind in ('room', 'gantry', 'grid') or sc.free(c.pos)]


def costs(cams, line, fwd, seen, dist):
    """The cost of each camera showing each lap sample: near and from in front is best."""
    C, N = len(cams), len(line)
    cost = np.full((C, N), np.inf)
    for c, cam in enumerate(cams):
        to_cam = cam.pos - line
        horiz = np.linalg.norm(to_cam[:, [0, 2]], axis=1)
        facing = np.einsum('ij,ij->i', unit(to_cam * (1, 0, 1)), unit(fwd * (1, 0, 1)))
        elev = np.degrees(np.arctan2(to_cam[:, 1], horiz))
        d = dist[c]
        k = ((d - 22.) / (30. if cam.kind == 'lap' else LONG_LENS)) ** 2 + .2
        k += 3. * (d < 5.)
        k += .5 * np.clip(-facing - .2, 0, None)        # seen from behind
        k += .3 * (elev < 2.) + .5 * (elev > 45.)
        if cam.bonus is not None:
            k -= cam.bonus[1] * cam.bonus[0]
        cost[c, seen[c]] = k[seen[c]]
    return cost


def choose(cost):
    """The camera for each lap sample: cheapest with a cut costing SWITCH (Viterbi)."""
    C, N = cost.shape
    big = 1e6
    c = np.where(np.isfinite(cost), cost, big)
    dp = c[:, 0].copy()
    back = np.zeros((N, C), np.int32)
    for i in range(1, N):
        best = dp.argmin()
        stay = dp
        move = dp[best] + SWITCH
        take = move < stay
        back[i] = np.where(take, best, np.arange(C))
        dp = np.where(take, move, stay) + c[:, i]
    path = np.empty(N, np.int32)
    path[-1] = dp.argmin()
    for i in range(N - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def runs(path):
    cut = np.r_[0, np.nonzero(np.diff(path))[0] + 1, len(path)]
    return [[int(path[a]), int(a), int(b)] for a, b in zip(cut[:-1], cut[1:])]


def merge_short(rs, cost):
    """Runs shorter than MIN_RUN go to a neighbour that sees them."""
    changed = True
    while changed and len(rs) > 1:
        changed = False
        for j, (c, a, b) in enumerate(rs):
            if b - a >= MIN_RUN:
                continue
            opts = []
            for nb in (j - 1, j + 1):
                if 0 <= nb < len(rs):
                    k = rs[nb][0]
                    tot = cost[k, a:b].sum()
                    if np.isfinite(tot):
                        opts.append((tot, nb))
            if not opts:
                continue
            nb = min(opts)[1]
            if nb < j:
                rs[nb][2] = b
            else:
                rs[nb][1] = a
            del rs[j]
            changed = True
            break
    return rs


def section(i, cam, run, line, pos, dist):
    c, a, b, out = run
    d = dist[c][a:b]
    k = a + int(d.argmin())
    f = unit(line[k] + (0, KART[0], 0) - cam.pos)
    right = unit(np.cross(f, (0, 1, 0)))
    up = np.cross(right, f)
    lo = max(3., min(60., fov(d.max())))
    hi = max(lo + 1., min(70., fov(d.min())))
    fl = lambda v: ', '.join('%.3f' % x for x in v)
    return '\n'.join([
        '[CAMERA_%d]' % i, 'NAME=%s' % cam.name, 'POSITION=%s' % fl(cam.pos), 'FORWARD=%s' % fl(f),
        'UP=%s' % fl(up), 'MIN_FOV=%.1f' % lo, 'MAX_FOV=%.1f' % hi,
        'IN_POINT=%.4f' % pos[a], 'OUT_POINT=%.4f' % out,
        'SHADOW_SPLIT0=50', 'SHADOW_SPLIT1=200', 'SHADOW_SPLIT2=600',
        'NEAR_PLANE=%.1f' % (.1 if cam.kind == 'room' else .3), 'FAR_PLANE=3000',
        'MIN_EXPOSURE=0', 'MAX_EXPOSURE=10000', 'DOF_FACTOR=10', 'DOF_RANGE=10000', 'DOF_FOCUS=0',
        'DOF_MANUAL=0', 'SPLINE=', 'SPLINE_ROTATION=0', 'FOV_GAMMA=0', 'SPLINE_ANIMATION_LENGTH=0',
        'IS_FIXED=0', ''])


def build(lay, render=False):
    files = layout_files(lay)
    sc = Scene(files)
    full, pos_full, total = ai_line(lay)
    idx = np.unique(np.searchsorted(pos_full, np.arange(0, 1, STEP / total)))
    idx = idx[idx < len(full)]
    line, pos = full[idx], pos_full[idx]
    fwd = unit(np.roll(line, -1, 0) - np.roll(line, 1, 0))
    N = len(line)
    # the corners, numbered from the start line
    lay_corners = [(int(np.searchsorted(idx, a)), int(np.searchsorted(idx, b)), ang) for a, b, ang in corners(full)]
    # the grid, as many slots as the layout has karts: from its last slot to the line
    karts = int(json.load(open(os.path.join(SOURCE, 'ui', lay, 'ui_track.json'), encoding='utf-8'))['pitboxes'])
    slots = sorted((k for k in sc.dummies if k.startswith('AC_START_')), key=lambda k: int(k.split('_')[-1]))[:karts]
    ks = [int(np.linalg.norm(line - sc.dummies[k][3, :3], axis=1).argmin()) for k in slots]
    gate = int(np.linalg.norm(line - sc.dummies['AC_TIME_0_L'][3, :3], axis=1).argmin())
    last = max(ks, key=lambda k: (gate - k) % N)
    grid_zone = np.zeros(N, bool)
    for j in range((gate - last) % N + 3):
        grid_zone[(last + j) % N] = True
    cams = candidates(sc, line, fwd, lay_corners, grid_zone, sc.dummies)
    for cam in cams:
        if cam.kind == 'room':
            near = np.linalg.norm(line - cam.pos, axis=1) < ROOM_REACH
            cam.bonus = (near, ROOM_BONUS)
        elif cam.kind in ('gantry', 'grid'):
            cam.bonus = (grid_zone, GRID_BONUS[cam.kind == 'grid'])
    seen, dist = [], []
    for cam in cams:
        s_, d_ = sc.sees(cam.pos, line)
        seen.append(s_)
        dist.append(d_)
    seen, dist = np.array(seen), np.array(dist)
    # one place in each room and on each gantry: the one that sees most of what it is there for
    best = {}
    for c, cam in enumerate(cams):
        if cam.kind in ('room', 'gantry'):
            key = (cam.name, round(float(cam.pos[2]) / 20))
            score = int((seen[c] & cam.bonus[0]).sum())
            if key not in best or score > best[key][0]:
                best[key] = (score, c)
    keep = {c for _, c in best.values()}
    for c, cam in enumerate(cams):
        if cam.kind in ('room', 'gantry') and c not in keep:
            seen[c] = False
    cost = costs(cams, line, fwd, seen, dist)
    blind = ~np.isfinite(cost).any(0)
    path = choose(cost)
    rs = merge_short(runs(path), cost)
    # the corners' names for the cameras that show them
    for c, a, b in rs:
        cam = cams[c]
        if cam.kind == 'lap':
            mid = [j + 1 for j, (ca, cb, _) in enumerate(lay_corners) if a <= (ca + cb) // 2 < b]
            if mid:
                cam.name = 'Turn ' + ' and '.join(map(str, mid))
            else:
                nxt = [j + 1 for j, (ca, cb, _) in enumerate(lay_corners) if ca >= a]
                cam.name = 'Before turn %d' % nxt[0] if nxt else 'Last straight'
    outs = [pos[rs[j + 1][1]] for j in range(len(rs) - 1)] + [1.]
    rs = [r + [o] for r, o in zip(rs, outs)]
    rs[0][1] = 0
    text = '[HEADER]\nVERSION=3\nCAMERA_COUNT=%d\nSET_NAME=TV\n\n' % len(rs)
    text += '\n'.join(section(i, cams[r[0]], r, line, np.r_[0., pos[1:]], dist) for i, r in enumerate(rs))
    out = os.path.join(SOURCE, lay, 'data', 'cameras.ini')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', newline='\n').write(text)
    kinds = {}
    for c, *_ in rs:
        kinds[cams[c].kind] = kinds.get(cams[c].kind, 0) + 1
    print('%-22s %4d m, %3d corners, %4d places tried, %2d cameras (%s), %d m unseen by any: %s'
          % (lay, total, len(lay_corners), len(cams), len(rs), ', '.join('%d %s' % (v, k) for k, v in sorted(kinds.items())),
             int(blind.sum() * total / N), os.path.relpath(out, ROOT)))
    if render:
        draw(lay, sc, cams, rs, line, dist, total)


# ---- Pictures to check by eye -----------------------------------------------------------------------
def draw(lay, sc, cams, rs, line, dist, total):
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(OUT, exist_ok=True)
    # the map: the lap coloured by camera, each camera a dot and a line to its stretch
    xz = line[:, [0, 2]]
    camxz = np.array([cams[c].pos[[0, 2]] for c, *_ in rs])
    lo = np.minimum(xz.min(0), camxz.min(0)) - 15
    hi = np.maximum(xz.max(0), camxz.max(0)) + 15
    S = 1100 / (hi - lo).max()
    W, H = ((hi - lo) * S).astype(int) + 40
    im = Image.new('RGB', (int(W), int(H)), (248, 247, 242))
    d = ImageDraw.Draw(im)
    P = lambda q: (float(W - 20 - (q[0] - lo[0]) * S), float(H - 20 - (q[1] - lo[1]) * S))   # x runs left, z up
    f = ImageFont.truetype('arial.ttf', 13)
    pal = [(214, 69, 65), (49, 112, 204), (46, 150, 80), (230, 140, 30), (140, 70, 180), (20, 150, 160), (190, 60, 140)]
    for j, (c, a, b, _) in enumerate(rs):
        col = pal[j % len(pal)]
        pts = [P(q) for q in xz[a:b + 1 if b < len(xz) else b]]
        if len(pts) > 1:
            d.line(pts, fill=col, width=5)
        cp = P(cams[c].pos[[0, 2]])
        mid = P(xz[(a + b) // 2])
        d.line([cp, mid], fill=col, width=1)
        d.ellipse((cp[0] - 5, cp[1] - 5, cp[0] + 5, cp[1] + 5), fill=col, outline=(0, 0, 0))
        d.text((cp[0] + 7, cp[1] - 7), '%d %s' % (j, cams[c].name), font=f, fill=(20, 20, 20))
    s0 = P(xz[0])
    d.rectangle((s0[0] - 4, s0[1] - 4, s0[0] + 4, s0[1] + 4), fill=(0, 0, 0))
    im.save(os.path.join(OUT, 'cameras_%s.png' % lay))
    # a picture from each camera
    shots = render(sc, [(cams[c], line[a + int(dist[c][a:b].argmin())], line[(a + b) // 2], dist[c][a:b])
                        for c, a, b, _ in rs])
    cols = 4
    tw, th = shots[0].size
    sheet = Image.new('RGB', (cols * tw, -(-len(shots) // cols) * (th + 18)), (255, 255, 255))
    sd = ImageDraw.Draw(sheet)
    for j, (sh, (c, a, b, _)) in enumerate(zip(shots, rs)):
        x, y = (j % cols) * tw, (j // cols) * (th + 18)
        sheet.paste(sh, (x, y + 18))
        sd.text((x + 4, y + 2), '%d %s (%d m)' % (j, cams[c].name, (b - a) * total / len(line)), font=f, fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, 'views_%s.png' % lay))


_scene_obj = {}


def render(sc, shots):
    from PIL import Image
    key = id(sc)
    scn = bpy.context.scene
    if key not in _scene_obj:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scn = bpy.context.scene
        me = bpy.data.meshes.new('track')
        v = sc.occ_v * (1, 1, 1)
        # AC's y up to Blender's z up: (x, y, z) -> (x, -z, y)
        me.from_pydata(np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1).tolist(), [], sc.occ_f.tolist())
        attr = me.color_attributes.new('col', 'FLOAT_COLOR', 'CORNER')
        cols = np.repeat(np.c_[sc.occ_c, np.ones(len(sc.occ_c))], 3, axis=0)
        attr.data.foreach_set('color', cols.ravel().astype(np.float32))
        ob = bpy.data.objects.new('track', me)
        scn.collection.objects.link(ob)
        kart = bpy.data.meshes.new('kart')
        kv = [(x, y, z) for x in (-.7, .7) for y in (-.9, .9) for z in (0, 1.)]
        kart.from_pydata(kv, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
        ka = kart.color_attributes.new('col', 'FLOAT_COLOR', 'CORNER')
        ka.data.foreach_set('color', [1., 0., 1., 1.] * 24)
        ko = bpy.data.objects.new('kart', kart)
        scn.collection.objects.link(ko)
        cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
        scn.collection.objects.link(cam)
        scn.camera = cam
        scn.render.engine = 'BLENDER_WORKBENCH'
        scn.display.shading.light = 'STUDIO'
        scn.display.shading.color_type = 'VERTEX'
        scn.world = bpy.data.worlds.new('w')
        scn.display.shading.background_type = 'WORLD'
        scn.world.color = (.55, .7, .9)
        scn.render.resolution_x, scn.render.resolution_y = 384, 216
        scn.render.film_transparent = False
        _scene_obj.clear()
        _scene_obj[key] = (ko, cam)
    ko, cam = _scene_obj[key]
    out = []
    bl = lambda p: Vector((p[0], -p[2], p[1]))
    for c, near, mid, d in shots:
        ko.location = bl(mid)
        cam.location = bl(c.pos)
        look = bl(mid + (0, .5, 0)) - cam.location
        cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
        dm = float(np.linalg.norm(mid - c.pos))
        cam.data.angle = math.radians(max(5., min(70., fov(dm))))
        cam.data.clip_start = .1
        cam.data.clip_end = 3000
        path = os.path.join(OUT, '_shot.png')
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(Image.open(path).convert('RGB'))
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    lays = [a for a in args if not a.startswith('--')] or sorted(
        x[len('models_'):-4] for x in os.listdir(AC) if x.startswith('models_') and x.endswith('.ini'))
    for lay in lays:
        build(lay, render='--render' in args)
