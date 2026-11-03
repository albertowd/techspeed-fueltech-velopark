"""Low-poly kerbs and pole, plus the lowered start gantry. Reads the track, never writes to it.

Usage: python build_simple.py <tex|gantry|concrete|floor|lod|pole|all> [more]
"""
import math, os, struct, sys

import numpy as np
from PIL import Image

import build_preview as bp
from build_preview import BLOCK, TILE, RIBS, A_DEPTH, B_WIDTH, B_T0, B_T1, B_HALF, B_DEPTH, POLE_H, TEX, TRACK, OUT, REF

from kn5_scan import read, bbox

KN5_OUT = os.path.join(OUT, 'kn5')
GANTRY_BOARD_BOTTOM = 2.8     # above the road, measured off the start panel photo


# ---- Meshes in track space ---------------------------------------------------
def chord_place(radius, s0, s1):
    """Curve the kerb in straight block-long chords. Every vertex stays on its chord,
    so the fine rib edge and the coarse ramp edge meet without cracks."""
    n0, n1 = int(round(s0 / BLOCK)), int(round(s1 / BLOCK))

    def node(k, t):
        a = k * BLOCK / radius
        r = radius - t
        return r * np.sin(a), radius - r * np.cos(a)

    def place(s, t, z):
        k = np.clip(np.floor(s / BLOCK), n0, n1 - 1)
        f = s / BLOCK - k
        x0, y0 = node(k, t)
        x1, y1 = node(k + 1, t)
        return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, z
    return place


class Soup:
    """Quads in track space, with every corner its own (s, t, z)."""
    def __init__(s):
        s.q = []

    def quad(s, *c):
        s.q.append(c)

    def build(s, name, mat, place, depth, angle):
        p = np.array(s.q, float).reshape(-1, 3)
        # weld so smooth shading works across quads
        key = np.round(p, 5)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        X, Y, Z = place(uniq[:, 0], uniq[:, 1], uniq[:, 2])
        v = np.stack([X, Y, Z], 1)
        uv = np.stack([uniq[:, 0] / TILE, 1 - uniq[:, 1] / depth], 1)
        # a repeated corner makes a triangle, ribs clipped at the strip ends collapse to nothing
        f = [list(dict.fromkeys(q.tolist())) for q in inv.reshape(-1, 4)]
        f = [q for q in f if len(q) >= 3]
        bp.add_mesh(name, v, f, mat, uv=uv, smooth_angle=angle)
        return sum(len(q) - 2 for q in f)


def concrete_mesh(s0, s1):
    q = Soup()
    prof = bp.a_profile
    rows = [0, .05, .20, 1.03, 1.09]
    for b in range(int(round(s0 / BLOCK)), int(round(s1 / BLOCK))):
        x0, x1 = b * BLOCK, (b + 1) * BLOCK
        for t0, t1 in zip(rows, rows[1:]):
            q.quad((x0, t0, prof(t0)), (x1, t0, prof(t0)), (x1, t1, prof(t1)), (x0, t1, prof(t1)))
    pitch = TILE / RIBS
    t0, t2 = 1.09, 2.0
    a, e = float(prof(t0)), float(prof(t2))
    # the grooves run out at the back edge, where the kerb meets the ground
    da, de = a - bp.A_GROOVE, max(e - bp.A_GROOVE, 0.)
    clip = lambda x: min(max(x, s0), s1)
    for k in range(int(math.floor(s0 / pitch)) - 1, int(math.ceil(s1 / pitch)) + 1):
        # V grooves: two walls meeting at the bottom, closed by a triangle at the top end
        c1, cm, c4 = [clip(k * pitch + d) for d in (.143, .2085, .274)]
        land = clip((k + 1) * pitch + .143)
        q.quad((c4, t0, a), (land, t0, a), (land, t2, e), (c4, t2, e))
        q.quad((c1, t0, a), (cm, t0, da), (cm, t2, de), (c1, t2, e))
        q.quad((cm, t0, da), (c4, t0, a), (c4, t2, e), (cm, t2, de))
        q.quad((c1, t0, a), (c4, t0, a), (cm, t0, da), (cm, t0, da))
    return q


def concrete_lod(s0, s1):
    q = Soup()
    prof = bp.a_profile
    rows = [0, .20, 1.03, 2.0]
    for b in range(int(round(s0 / BLOCK)), int(round(s1 / BLOCK))):
        x0, x1 = b * BLOCK, (b + 1) * BLOCK
        for t0, t1 in zip(rows, rows[1:]):
            q.quad((x0, t0, prof(t0)), (x1, t0, prof(t0)), (x1, t1, prof(t1)), (x0, t1, prof(t1)))
    return q


def floor_mesh(s0, s1):
    q = Soup()
    h = B_HALF
    for b in range(int(round(s0 / BLOCK)), int(round(s1 / BLOCK))):
        x0, x1 = b * BLOCK, (b + 1) * BLOCK
        n = 5 if b % 2 else 4
        q.quad((x0, 0, 0), (x1, 0, 0), (x1, B_T0, 0), (x0, B_T0, 0))
        q.quad((x0, B_T1, 0), (x1, B_T1, 0), (x1, B_WIDTH, 0), (x0, B_WIDTH, 0))
        edge = x0
        for k in range(n):
            c = x0 + BLOCK * (k + .5) / n
            q.quad((edge, B_T0, 0), (c - h, B_T0, 0), (c - h, B_T1, 0), (edge, B_T1, 0))
            edge = c + h
            top = [(c - h, B_T0, 0), (c + h, B_T0, 0), (c + h, B_T1, 0), (c - h, B_T1, 0)]
            a, b = (c, B_T0 + h, -B_DEPTH), (c, B_T1 - h, -B_DEPTH)
            q.quad(top[0], top[1], a, a)
            q.quad(top[1], top[2], b, a)
            q.quad(top[2], top[3], b, b)
            q.quad(top[3], top[0], a, b)
        q.quad((edge, B_T0, 0), (x1, B_T0, 0), (x1, B_T1, 0), (edge, B_T1, 0))
    return q


def floor_lod(s0, s1):
    q = Soup()
    for b in range(int(round(s0 / BLOCK)), int(round(s1 / BLOCK))):
        x0, x1 = b * BLOCK, (b + 1) * BLOCK
        q.quad((x0, 0, 0), (x1, 0, 0), (x1, B_WIDTH, 0), (x0, B_WIDTH, 0))
    return q


# ---- Textures ----------------------------------------------------------------
def normal_map(h, px_m, name):
    dy, dx = np.gradient(h, 1 / px_m)
    n = np.stack([-dx, dy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    bp.save_rgb((n * .5 + .5) * 255, name)


def read16(name):
    return np.asarray(Image.open(os.path.join(TEX, name)), float) / 65535


def make_textures():
    # concrete: paint and pores only, the ribs stay real geometry
    h = read16('concrete_height.png')
    W = h.shape[1]
    normal_map(h * .004, W / TILE, 'concrete_normal.png')
    normal_map(read16('ramp_height.png') * .004, W / TILE, 'ramp_normal.png')

    # floor: smooth shading rounds the grooves, this adds paint and pores
    h = read16('floor_height.png')
    normal_map(h * .006, h.shape[1] / TILE, 'floor_normal.png')

    # far versions: flat meshes, so the grooves are painted in as shade
    alb = np.asarray(Image.open(os.path.join(TEX, 'concrete_albedo.png')).convert('RGB'), float)
    H, W = alb.shape[:2]
    t, s = np.mgrid[0:H, 0:W].astype(float)
    s = s / W * TILE
    t = t / H * A_DEPTH
    g = np.interp(np.mod(s, TILE / RIBS), [0, .143, .155, .262, .274, TILE / RIBS], [0, 0, 1, 1, 0, 0]) * bp.smooth(1.09, 1.12, t)
    Image.fromarray(np.clip(alb * (1 - .62 * g)[..., None], 0, 255).astype(np.uint8)).resize((2048, H * 2048 // W), Image.LANCZOS)\
        .save(os.path.join(TEX, 'concrete_far.png'))
    alb = np.asarray(Image.open(os.path.join(TEX, 'floor_albedo.png')).convert('RGB'), float)
    H, W = alb.shape[:2]
    t, s = np.mgrid[0:H, 0:W].astype(float)
    s = s / W * TILE
    t = t / H * B_WIDTH
    g = bp.b_mask(s + .012, t)
    Image.fromarray(np.clip(alb * (1 - .4 * g)[..., None], 0, 255).astype(np.uint8)).resize((2048, H * 2048 // W), Image.LANCZOS)\
        .save(os.path.join(TEX, 'floor_far.png'))


def mat_nm(name, albedo, normal=None, rough=.8, spec=.4):
    bpy = bp.bpy_mod()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']
    p.inputs['Roughness'].default_value = rough
    p.inputs['Specular IOR Level'].default_value = spec
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = bpy.data.images.load(os.path.join(TEX, albedo), check_existing=True)
    nt.links.new(t.outputs['Color'], p.inputs['Base Color'])
    if normal:
        n = nt.nodes.new('ShaderNodeTexImage')
        n.image = bpy.data.images.load(os.path.join(TEX, normal), check_existing=True)
        n.image.colorspace_settings.name = 'Non-Color'
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nm.inputs['Strength'].default_value = 1
        nt.links.new(n.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], p.inputs['Normal'])
    return m


# ---- Kerb scenes -------------------------------------------------------------
SKY_A = (bp.lin((120, 150, 190)), bp.lin((190, 200, 212)), bp.lin((232, 226, 214)))
SKY_B = (bp.lin((130, 150, 180)), bp.lin((196, 200, 208)), bp.lin((228, 222, 210)))


def ground_a(place, s0, s1):
    m = bp.mats_common()
    fl = np.linspace(s0, s1, 90)
    bp.strip('road', fl, np.array([-9., 0]), lambda s, t: np.zeros_like(s), m['tar'], bp.uv_plane(4, 8), place)
    bp.strip('verge', fl, np.linspace(1.9, 12, 6), lambda s, t: np.full_like(s, -.006), m['grass'], bp.uv_plane(5, 5), place)


def shot_concrete():
    bpy, sc = bp.reset(exposure=.55)
    bp.world(*SKY_A, .6)
    s0, s1 = -6., 8.4
    s0 = BLOCK * round(s0 / BLOCK)
    place = chord_place(45., s0, s1)
    ground_a(place, s0, s1)
    tris = concrete_mesh(s0, s1).build('kerb', mat_nm('c', 'concrete_albedo.png', 'concrete_normal.png', .82, .35), place, A_DEPTH, 38)
    bp.sun(186, 15, 3.6)
    dens = lambda s, t: np.clip((t - 1.78) * 2.2, 0, 1) + .012 * (np.sin(s * 3.1) > .4) * (t > 1.2)
    bp.grass_blades('blades', 1500000, (s0, s1), (1.25, 4.0), (.05, .2), place, 3, dens, w_rng=(.002, .006))
    px, py, _ = place(np.array([1.0]), np.array([.95]), np.zeros(1))
    bp.camera((px[0] - .05, py[0] + 1.75, 2.28), (px[0], py[0] + .22, 0), lens=25)
    bp.render(sc, 'simple_concrete')
    bp.camera((px[0] - 1, py[0] + 1.2, .48), (px[0] + 1.4, py[0] + .15, .04), lens=24)
    bp.render(sc, 'simple_concrete_low')
    print('concrete tris/m', round(tris / (s1 - s0), 1), flush=True)


def ground_b(s0, s1):
    m = bp.mats_common()
    fl = np.array([s0, s1])
    bp.strip('road', fl, np.array([-8., 0]), lambda s, t: np.zeros_like(s), m['tar'], bp.uv_plane(4, 8))
    bp.strip('line', fl, np.array([-1.62, -1.42]), lambda s, t: np.full_like(s, .002),
             bp.mat_flat('line', bp.lin((205, 205, 200)), rough=.8), bp.uv_plane(4, 8))
    bp.strip('shoulder', fl, np.array([B_WIDTH, B_WIDTH + 1.5]), lambda s, t: np.zeros_like(s),
             bp.mat_img('tar2', os.path.join(TEX, 'tarmac.png'), rough=.85, tint=(.6, .46, .36)), bp.uv_plane(4, 8))
    bp.strip('verge', fl, np.array([B_WIDTH + 1.5, 20]), lambda s, t: np.full_like(s, -.004), m['grass'], bp.uv_plane(5, 5))


def shot_floor():
    bpy, sc = bp.reset(exposure=.5)
    bp.world(*SKY_B, .7)
    s0, s1 = -3.6, 6.
    ground_b(s0, s1)
    tris = floor_mesh(s0, s1).build('kerb', mat_nm('f', 'floor_albedo.png', 'floor_normal.png', .72, .4),
                                    lambda s, t, z: (s, t, z), B_WIDTH, 20)
    bp.sun(35, 33, 2.6, color=(1, .92, .8))
    bp.camera((1.05, -1.5, 2.6), (1.05, -.55, 0), lens=24)
    bp.render(sc, 'simple_floor')
    bp.camera((1.5, -.55, .34), (.9, .42, 0), lens=30)
    bp.render(sc, 'simple_floor_close')
    print('floor tris/m', round(tris / (s1 - s0), 1), flush=True)


def shot_lod():
    """Near and far versions at a distance, near version on top."""
    for name, far in (('simple_lod_near', False), ('simple_lod_far', True)):
        bpy, sc = bp.reset((1600, 500), exposure=.5)
        bp.world(*SKY_B, .7)
        m = bp.mats_common()
        s0, s1 = -48., 96.
        fl = np.array([s0, s1])
        bp.strip('road', fl, np.array([-30., 40.]), lambda s, t: np.zeros_like(s), m['tar'], bp.uv_plane(4, 8))
        id_ = lambda s, t, z: (s, t, z)
        if far:
            concrete_lod(s0, s1).build('a', mat_nm('c', 'concrete_far.png', None, .82), id_, A_DEPTH, 38)
            floor_lod(s0, s1).build('b', mat_nm('f', 'floor_far.png', None, .72),
                                    lambda s, t, z: (s, t - 3.5, z + .002), B_WIDTH, 20)
        else:
            concrete_mesh(s0, s1).build('a', mat_nm('c', 'concrete_albedo.png', 'concrete_normal.png', .82), id_, A_DEPTH, 38)
            floor_mesh(s0, s1).build('b', mat_nm('f', 'floor_albedo.png', 'floor_normal.png', .72),
                                     lambda s, t, z: (s, t - 3.5, z + .002), B_WIDTH, 20)
        bp.sun(160, 26, 3.0, color=(1, .9, .76))
        bp.camera((0, -44, 3.4), (34, 0, 0), lens=70)
        bp.render(sc, name)


# ---- Pole --------------------------------------------------------------------
class Lean(bp.Part):
    """Round tubes without end caps: their ends are never seen from the ground."""
    def tube(s, p0, p1, w=.06, h=None):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        R = bp.frame_z(p1 - p0)
        an = np.arange(6) * np.pi / 3
        c = np.stack([np.cos(an), np.sin(an)], 1) * w * .56
        v = [p + R @ np.array([x, y, 0]) for p in (p0, p1) for x, y in c]
        s.add(np.array(v), [(i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6) for i in range(6)])


def dish(frame, glass, base, face, r=.30, post=.50):
    base = np.array(base, float)
    c = base + (0, 0, post + r * .75)
    face = np.array(face, float)
    face /= np.linalg.norm(face)
    R = bp.frame_z(face)
    frame.lathe([(0, -.17), (.11, -.16), (.2, -.1), (r * .9, -.02), (r, .05)], 28, at=c, axis=face)
    glass.lathe([(r, .05), (0, .03)], 28, at=c, axis=face)
    side = R[:, 0]
    if abs(side[2]) > .7:
        side = R[:, 1]
    a, b = c + side * (r + .025), c - side * (r + .025)
    y0 = base + (0, 0, post * .55)
    frame.tube(a, y0 + side * (r + .025), .03)
    frame.tube(b, y0 - side * (r + .025), .03)
    frame.tube(y0 + side * (r + .025), y0 - side * (r + .025), .034)
    frame.tube(base, y0, .05)


def pole_parts(height=POLE_H):
    """Mast, frame, lens and glass as plain geometry. The lamps face -Y."""
    z = height - .55
    mast, frame, lens, glass = Lean(), Lean(), Lean(), Lean()
    mast.lathe([(.165, 0), (.120, z * .5), (.108, z * .5), (.092, z * .75), (.075, z)], 20)
    P = np.c_[np.array([(-.74, .62), (-.25, .70), (.27, .58), (.60, .24), (.60, -.20),
                        (.12, -.20), (.0, .16), (-.60, .22), (-.64, .05)]), np.full(9, z)]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 8), (8, 7)):
        frame.tube(P[a], P[b], .045)
    hub = np.array([0, 0, z])
    frame.tube(P[7], hub, .045)
    frame.tube(hub, P[2], .05)
    frame.tube(hub, P[5], .05)
    frame.box((0, 0, z - .02), (.20, .20, .16))
    dish(frame, glass, (-.62, .64, z), (-.12, -.72, -.68))
    dish(frame, glass, (.12, .64, z), (.05, -.72, -.68))
    dish(frame, glass, P[3] + (.2, -.04, 0), (.55, -.6, -.58))
    dish(frame, lens, (-.28, .05, z - .55), (-.72, -.62, -.3), r=.22, post=.26)
    # rectangular flood
    base = P[4] + (.05, .05, -.28)
    face = np.array((.72, -.58, -.38))
    face /= np.linalg.norm(face)
    c = base + (0, 0, .26)
    frame.box(c, (.46, .34, .16), axis=face, taper=.55)
    lens.box(c + face * .082, (.41, .29, .006), axis=face)
    frame.tube(base, base + (0, 0, .16), .05)
    return dict(mast=mast, frame=frame, lens=lens, glass=glass)


SMOOTH = dict(mast=60, frame=50, lens=60, glass=60)


def pole_simple(height=POLE_H):
    mats = dict(mast=bp.mat_img('mast', os.path.join(TEX, 'mast.png'), rough=.62, metal=.25),
                frame=bp.mat_flat('frame', (.016, .016, .018), rough=.55),
                lens=bp.mat_flat('lens', (.5, .52, .54), rough=.22, spec=.9),
                glass=bp.mat_flat('glass', (.035, .037, .04), rough=.14, spec=1.))
    tris = 0
    for name, part in pole_parts(height).items():
        m, a = mats[name], SMOOTH[name]
        tris += part.build(name, m, smooth_angle=a)[1]
    return tris


def shot_pole():
    bpy, sc = bp.reset((900, 1600))
    bp.world(bp.lin((26, 66, 128)), bp.lin((88, 150, 205)), bp.lin((205, 222, 232)), 1.0)
    m = bp.mats_common()
    tris = pole_simple()
    bp.tyres_scene([(-2.1, 30, -3.6)], ((0, 0),))
    bp.build_ctx_ground(m)
    bp.sun(160, 9, 3.0, color=(1, .8, .56))
    bp.backdrop()
    f = 1600 / (2 * math.tan(math.radians(32.5)))
    pitch = math.atan((1020 - 800) / f)
    bp.camera((1.55, -19.5, 1.2), (1.55, -9.5, 1.2 + 10 * math.tan(pitch)), lens=f / 1600 * 36, fit='VERTICAL')
    bp.render(sc, 'simple_pole')
    sc.render.resolution_x, sc.render.resolution_y = 1200, 900
    z = POLE_H - .4
    bp.camera((-.5, -13.5, z - 11.0), (.05, .1, z + .1), lens=100)
    bp.render(sc, 'simple_pole_head')
    print('pole tris', tris, flush=True)


# ---- Gantry ------------------------------------------------------------------
MOVED = ('GANTRY_BOARD', 'GANTRY_HOUSING', 'GANTRY_FLAG', 'KS_START_LIGHT')


def road_level(x, z):
    """Height of the road under the gantry, from the asphalt mesh."""
    b, ver, tex, mats, nodes = read(os.path.join(TRACK, 'velopark.kn5'))
    best = []
    for n in nodes:
        if not n['name'].startswith('1ROAD') or mats[n['mat']]['name'] == 'Rumble1':
            continue
        d = np.frombuffer(b, dtype='<f4', count=n['vc'] * 11, offset=n['vp']).reshape(-1, 11)[:, :3]
        near = d[(np.abs(d[:, 0] - x) < 3) & (np.abs(d[:, 2] - z) < 3)]
        if len(near):
            best.append(near[:, 1])
    return float(np.median(np.concatenate(best)))


def rotation(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + a @ b)


def components(idx, count):
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, c in idx:
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    return np.array([find(i) for i in range(count)])


def patch_gantry(fn):
    b, ver, tex, mats, nodes = read(os.path.join(TRACK, fn))
    out = bytearray(b)
    board = next(n for n in nodes if n['name'] == 'GANTRY_BOARD')
    lo, hi = bbox(b, board)
    road = road_level((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
    shift = lo[1] - (road + GANTRY_BOARD_BOTTOM)
    assert shift > 0, shift

    def verts(n):
        return np.frombuffer(b, dtype='<f4', count=n['vc'] * 11, offset=n['vp']).reshape(-1, 11).copy()

    def store(n, d):
        out[n['vp']:n['vp'] + n['vc'] * 44] = d.astype('<f4').tobytes()

    def move_sphere(n, dy):
        c = n['ip'] + n['ic'] * 2 + 16 + 4
        struct.pack_into('<f', out, c, struct.unpack_from('<f', out, c)[0] - dy)

    for n in nodes:
        m = n['m']
        assert abs(m[5] - 1) < 1e-6 and abs(m[1]) + abs(m[4]) + abs(m[6]) + abs(m[9]) < 1e-6, n['name']
        if n['name'].startswith(MOVED):
            d = verts(n)
            d[:, 1] -= shift
            store(n, d)
            move_sphere(n, shift)

    # the board's own trim moves with it, the four long pieces are the legs
    steel = next(n for n in nodes if n['name'] == 'GANTRY_STEEL')
    idx = np.frombuffer(b, dtype='<u2', count=steel['ic'], offset=steel['ip']).reshape(-1, 3)
    d = verts(steel)
    comp = components(idx, steel['vc'])

    # panel ends, from the panel itself
    bv = verts(board)[:, [0, 2]]
    centre = (bv.min(0) + bv.max(0)) / 2
    axis = np.linalg.svd(bv - bv.mean(0), full_matrices=False)[2][0]
    along = (bv - centre) @ axis
    ends = {+1: bv[along > along.max() - .2].mean(0), -1: bv[along < along.min() + .2].mean(0)}
    normal = np.array([-axis[1], axis[0]])
    board_top = verts(board)[:, 1].max() - shift

    legs = []
    for c in np.unique(comp):
        k = comp == c
        if k.sum() <= 4:
            d[k, 1] -= shift
            continue
        y = d[k, 1]
        foot = d[k][y < y.min() + .1, :3].mean(0)
        ring = d[k][y > y.max() - .1, :3]
        top = ring.mean(0)
        radius = np.linalg.norm(ring - top, axis=1).max()
        side = 1 if (top[[0, 2]] - centre) @ axis > 0 else -1
        lean = 1 if (top[[0, 2]] - ends[side]) @ normal > 0 else -1
        spot = ends[side] + axis * side * (radius + .03) + normal * lean * .14
        new_top = np.array([spot[0], board_top - .35, spot[1]])
        rot = rotation(top - foot, new_top - foot)
        dir_ = (new_top - foot) / np.linalg.norm(new_top - foot)
        scale = np.linalg.norm(new_top - foot) / np.linalg.norm(top - foot)
        q = (d[k, :3] - foot) @ rot.T
        q += np.outer((q @ dir_) * (scale - 1), dir_)
        d[k, :3] = q + foot
        d[k, 3:6] = d[k, 3:6] @ rot.T
        d[k, 8:11] = d[k, 8:11] @ rot.T
        legs.append((foot, rot))
    assert len(legs) == 4, len(legs)
    store(steel, d)
    move_sphere(steel, shift * .5)

    # the tyres on each foot follow their leg
    tyres = next(n for n in nodes if n['name'] == 'GANTRY_TYRES')
    idx = np.frombuffer(b, dtype='<u2', count=tyres['ic'], offset=tyres['ip']).reshape(-1, 3)
    d = verts(tyres)
    comp = components(idx, tyres['vc'])
    for c in np.unique(comp):
        k = comp == c
        mid = d[k, :3].mean(0)
        foot, rot = min(legs, key=lambda l: np.linalg.norm((l[0] - mid)[[0, 2]]))
        d[k, :3] = (d[k, :3] - foot) @ rot.T + foot
        d[k, 3:6] = d[k, 3:6] @ rot.T
        d[k, 8:11] = d[k, 8:11] @ rot.T
    store(tyres, d)

    os.makedirs(KN5_OUT, exist_ok=True)
    path = os.path.join(KN5_OUT, fn)
    open(path, 'wb').write(out)
    print(fn, 'road', round(road, 2), 'lowered', round(shift, 2), 'board', round(lo[1] - shift - road, 2), 'to', round(hi[1] - shift - road, 2), flush=True)
    return path, road


def add_gantry(path, road):
    bpy = bp.bpy_mod()
    b, ver, tex, mats, nodes = read(path)
    t = next(x for x in tex if x[0] == 'GantryBoard.dds')
    dds = os.path.join(TEX, 'gantry_board.dds')
    open(dds, 'wb').write(b[t[2]:t[2] + t[1]])
    png = os.path.join(TEX, 'gantry_board.png')
    Image.open(dds).convert('RGB').save(png)
    made = dict(GantryBoard=bp.mat_img('gb', png, rough=.5), GantrySteel=bp.mat_flat('gs', (.5, .52, .54), rough=.45, metal=.6),
                StartLamp=bp.mat_flat('sl', (.02, .02, .02), rough=.3), GantryRubber=bp.mat_flat('gr', (.015, .015, .016), rough=.6),
                GantryFlag=bp.mat_flat('gf', (.02, .3, .03), rough=.3), TyreBarrier=bp.mat_flat('tb', (.018, .018, .019), rough=.62))
    for n in nodes:
        if n['name'].startswith('1WALL'):
            continue
        d = np.frombuffer(b, dtype='<f4', count=n['vc'] * 11, offset=n['vp']).reshape(-1, 11)
        q = np.stack([d[:, 0] + n['m'][12], -(d[:, 2] + n['m'][14]), d[:, 1] + n['m'][13] - road], 1)
        uv = np.stack([d[:, 6], 1 - d[:, 7]], 1)
        f = np.frombuffer(b, dtype='<u2', count=n['ic'], offset=n['ip']).reshape(-1, 3)
        bp.add_mesh(n['name'], q, f, made[mats[n['mat']]['name']], uv=uv, smooth_angle=50)


def shot_gantry():
    path, road = patch_gantry('velopark_gantry_1500.kn5')
    patch_gantry('velopark_gantry_1000.kn5')
    for name, src in (('gantry_current', os.path.join(TRACK, 'velopark_gantry_1500.kn5')), ('gantry_lowered', path)):
        bpy, sc = bp.reset((1570, 883), exposure=.2)
        bp.world(bp.lin((110, 160, 215)), bp.lin((180, 205, 225)), bp.lin((228, 228, 222)), .8)
        m = bp.mats_common()
        bp.strip('g', np.array([-200., 200]), np.array([-300., 100]), lambda s, t: np.full_like(s, -.03), m['grass'], bp.uv_plane(7, 7))
        bp.strip('r', np.array([-10.5, -.5]), np.array([-300., 100]), lambda s, t: np.zeros_like(s), m['tar'], bp.uv_plane(4, 8))
        add_gantry(src, road)
        bp.sun(120, 25, 3.0, color=(1, .9, .74))
        bp.camera((-5.3, -114 + 34, 1.6), (-5.3, -114, 2.6), lens=42)
        bp.render(sc, name)
        if name == 'gantry_lowered':
            bp.camera((9, -114 + 15, 1.6), (0, -113.5, 2.6), lens=32)
            bp.render(sc, 'gantry_lowered_end')


if __name__ == '__main__':
    os.makedirs(TEX, exist_ok=True)
    args = sys.argv[1:] or ['all']
    jobs = dict(tex=make_textures, concrete=shot_concrete, floor=shot_floor, lod=shot_lod, pole=shot_pole, gantry=shot_gantry)
    for k, f in jobs.items():
        if k in args or 'all' in args:
            f()
