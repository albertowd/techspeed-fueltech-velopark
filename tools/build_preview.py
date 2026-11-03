"""Preview renders of the new kerbs and pole. Reads the track, never writes to it.

Usage: python build_preview.py <shot|tex|all> [more shots]
"""
import math, os, struct, sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'tools')
TEX = os.path.join(OUT, 'textures')
ASSETS = os.path.join(OUT, 'reference')
TRACK = os.path.join(ROOT, 'base', 'velopark-0.5.0', 'content', 'tracks', 'techspeed_velopark')
REF = ASSETS        # photos and textures taken from the old track
SAMPLES = int(os.environ.get('VP_SAMPLES', 160))

# ---- Dimensions, metres ------------------------------------------------------
BLOCK = 1.2          # painted block length along the track
TILE = 4 * BLOCK     # texture repeat
RIBS = 17            # concrete ribs per tile, about 0.28 m pitch
A_DEPTH = 2.05       # concrete kerb, track edge to grass
A_RAMP = 0.112       # concrete kerb, crest height
A_GROOVE = 0.055     # concrete kerb, groove depth
B_WIDTH = 0.92       # floor kerb, across the track
B_DEPTH = 0.032      # floor kerb, groove depth
B_HALF = 0.068       # floor kerb, groove half width
B_T0, B_T1 = 0.09, 0.83
POLE_H = 17.0


def smooth(a, b, x):
    k = np.clip((x - a) / (b - a), 0, 1)
    return k * k * (3 - 2 * k)


def fbm(h, w, beta=2.0, seed=0, ax=1.0, ay=1.0):
    """Tileable noise in 0..1. ax/ay stretch it: a large ax gives streaks along y."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal((h, w)))
    fy, fx = np.meshgrid(np.fft.fftfreq(h) * h / w, np.fft.fftfreq(w), indexing='ij')
    r = np.sqrt((fx * ax) ** 2 + (fy * ay) ** 2)
    r[0, 0] = 1
    a = r ** (-beta / 2)
    a[0, 0] = 0
    n = np.fft.ifft2(f * a).real
    return np.clip((n - n.mean()) / n.std() * 0.2 + 0.5, 0, 1)


def crackle(h, w, cells, seed, width=1.3):
    """Tileable crazed paint: 1 on the cracks."""
    rng = np.random.default_rng(seed)
    pts = rng.random((cells, 2)) * (h, w)
    tree = cKDTree(pts, boxsize=(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    wob = (fbm(h, w, 2.6, seed + 1) - 0.5) * h * 0.012
    q = np.stack([(yy + wob).ravel() % h, (xx - wob).ravel() % w], 1)
    d, _ = tree.query(q, k=2, workers=-1)
    return (1 - smooth(0, width, (d[:, 1] - d[:, 0]))).reshape(h, w)


def lerp(a, b, k):
    return a + (b - a) * k[..., None]


def rgb(*c):
    return np.array(c, float)


def save_rgb(a, name):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(TEX, name))


def save_height(a, name):
    a = (np.clip(a, 0, 1) * 65535).astype(np.uint16)
    Image.fromarray(a).save(os.path.join(TEX, name))


# ---- Concrete kerb -----------------------------------------------------------
def a_profile(t):
    return np.interp(t, [0, .02, .18, .20, .95, 1.00, 1.03, 1.05, 1.08, 2.00, 2.05],
                     [0, .012, .020, .024, .098, .108, A_RAMP, .104, .095, .000, -.030])


def a_groove(s):
    """1 inside a groove between two concrete ribs."""
    pitch = TILE / RIBS
    x = np.mod(s, pitch)
    return smooth(.012, 0, np.abs(x - .155)) * 0 + ((x > .155) * (x < .275)) * 1.0 \
        + (x > .143) * (x <= .155) * ((x - .143) / .012) + (x >= .275) * (x < .2824) * 0


def a_height(s, t):
    pitch = TILE / RIBS
    x = np.mod(s, pitch)
    g = np.interp(x, [0, .143, .155, .262, .274, pitch], [0, 0, 1, 1, 0, 0])
    return a_profile(t) - A_GROOVE * g * smooth(1.09, 1.12, t)


def a_s_lines(s0, s1):
    pitch = TILE / RIBS
    k0, k1 = int(s0 // pitch), int(s1 // pitch) + 1
    out = [s0, s1]
    for k in range(k0, k1 + 1):
        out += [k * pitch + d for d in (0, .143, .155, .262, .274)]
    return np.unique(np.clip(out, s0, s1))


A_T_LINES = np.array([0, .02, .18, .20, .55, .95, 1.0, 1.03, 1.05, 1.08, 1.09, 1.12, 1.5, 2.0, 2.05])


def tex_concrete(name='concrete', paint_out=1.085, warp=None):
    """Paint runs from the road edge over the ramp and down its back to paint_out. warp,
    (profile, built) distances from the road edge, is how the kerb is built narrower or
    wider in parts: the texture is laid out as built, its detail the same size in metres."""
    warp = warp or ([0, A_DEPTH], [0, A_DEPTH])
    depth = warp[1][-1]
    W, H = 4096, int(4096 * depth / TILE) // 4 * 4
    t, s = np.mgrid[0:H, 0:W].astype(float)
    s = s / W * TILE
    t = np.interp(t / H * depth, warp[1], warp[0])
    n_lo = fbm(H, W, 2.4, 1)
    n_mid = fbm(H, W, 1.7, 2)
    n_fine = fbm(H, W, 0.9, 3)
    pores = smooth(.74, .80, fbm(H, W, 0.7, 4))
    streak = fbm(H, W, 1.3, 5, ax=8, ay=.6)       # roller marks across the kerb
    scuff = fbm(H, W, 1.8, 6, ax=.35, ay=7)       # tyre rubber along the track
    wob = (fbm(H, W, 2.2, 7) - .5)

    # bare concrete
    col = rgb(200, 193, 176) * (0.78 + 0.42 * n_mid)[..., None]
    col = lerp(col, rgb(120, 118, 112), smooth(.52, .72, n_lo) * .55)
    col = lerp(col, rgb(70, 72, 62), smooth(.56, .72, fbm(H, W, 2.2, 8)) * smooth(1.25, 2.0, t) * .75)
    col = lerp(col, rgb(85, 82, 76), pores * .7)

    # paint, from the road edge to the back of the ramp
    edge_out = paint_out + wob * .01
    paint = smooth(edge_out + .004, edge_out - .004, t)
    block = np.floor((s + wob * .03) / BLOCK).astype(int) % 2
    red = rgb(186, 88, 92) * (.74 + .5 * streak)[..., None]
    red = lerp(red, rgb(140, 60, 72), smooth(.5, .7, n_lo) * .6)
    white = rgb(226, 232, 238) * (.82 + .34 * streak)[..., None]
    white = lerp(white, rgb(172, 182, 192), smooth(.5, .68, n_mid) * .5)
    pcol = np.where(block[..., None] == 1, white, red)
    # concrete grinning through the roller marks
    thin = smooth(.60, .74, streak * .7 + n_fine * .3) * .35
    pcol = lerp(pcol, rgb(150, 145, 135), thin)
    # rubber on the track side of the ramp
    fall = (1 - smooth(.20, .62, t + wob * .1)) ** 1.5
    rub = smooth(.36, .78, scuff) * fall * np.where(block == 1, 1.0, .75)
    pcol = lerp(pcol, rgb(38, 44, 66), rub * .9)
    # a few tyre arcs
    for k, (a, b, c) in enumerate([(2.8, .62, -.12)]):
        line = b + c * np.sin((s - a) * 1.3) + (wob * .02)
        pcol = lerp(pcol, rgb(70, 40, 46), smooth(.03, .004, np.abs(t - line)) * .16)
    col = lerp(col, pcol, paint)

    # slab joints through the ribbed part
    for j in (.32, 3.05):
        js = j + (t - 1.05) * .05 + wob * .02
        joint = smooth(.009, .003, np.abs(s - js)) * smooth(paint_out, paint_out + .01, t)
        col = lerp(col, rgb(28, 26, 24), joint)

    save_rgb(col, name + '_albedo.png')
    h = (.5 + (n_mid - .5) * .35 - pores * .25) * (1 - paint) \
        + (.55 + (streak - .5) * .9 + (n_fine - .5) * .25) * paint
    bead = smooth(.02, 0, np.abs(t - edge_out))
    save_height(h + bead * .4, name + '_height.png')


def tex_ramp():
    """The ramp kerb without concrete behind it: painted all the way to its outer edge."""
    tex_concrete('ramp', 1.26)


# ---- Floor kerb --------------------------------------------------------------
def b_grooves(s):
    """Distance to the nearest groove centre. Red blocks have 4, white blocks 5."""
    block = np.floor(s / BLOCK)
    n = np.where(np.mod(block, 2) == 1, 5, 4)
    u = s - block * BLOCK
    pitch = BLOCK / n
    return np.abs(np.mod(u, pitch) - pitch / 2)


def b_mask(s, t):
    across = 1 - b_grooves(s) / B_HALF
    along = np.minimum(t - B_T0, B_T1 - t) / B_HALF
    return np.clip(np.minimum(across, along), 0, 1)


def b_height(s, t):
    return -B_DEPTH * b_mask(s, t)


def b_s_lines(s0, s1):
    out = [s0, s1]
    for b in range(int(s0 // BLOCK), int(s1 // BLOCK) + 1):
        n = 5 if b % 2 else 4
        out += [b * BLOCK]
        for k in range(n):
            c = b * BLOCK + BLOCK * (k + .5) / n
            for d in (0, .4, .7, .88, 1, 1.03):
                out += [c - d * B_HALF, c + d * B_HALF]
    return np.unique(np.clip(out, s0, s1))


_ends = B_HALF * np.array([0, .03, .12, .3, .6, 1])
B_T_LINES = np.array([0, *(B_T0 + _ends), *(B_T1 - _ends), .35, .58, B_WIDTH])
B_T_LINES = np.unique(B_T_LINES)


def tex_floor():
    W, H = 4096, int(4096 * B_WIDTH / TILE)
    t, s = np.mgrid[0:H, 0:W].astype(float)
    s = s / W * TILE
    t = t / H * B_WIDTH
    n_lo = fbm(H, W, 2.4, 11)
    n_mid = fbm(H, W, 1.7, 12)
    wob = fbm(H, W, 2.2, 13) - .5
    grain = fbm(H, W, .25, 14)                    # aggregate tips
    grain2 = fbm(H, W, .9, 15)
    block = np.floor(s / BLOCK).astype(int) % 2

    rim_s = np.minimum(np.mod(s, BLOCK), BLOCK - np.mod(s, BLOCK))
    rim = 1 - smooth(.035, .06, np.minimum(np.minimum(t, B_WIDTH - t), rim_s) + wob * .03)

    cover_flat = np.where(block == 1, .50, .68) + (n_mid - .5) * .35
    tips = smooth(1 - cover_flat - .05, 1 - cover_flat + .05, grain * .75 + grain2 * .25)
    dark = np.where(block[..., None] == 1, rgb(58, 58, 60), rgb(70, 44, 44))
    flat_paint = np.where(block[..., None] == 1, rgb(238, 238, 236), rgb(178, 92, 90))
    col = lerp(dark, flat_paint, tips)
    # border band: thick paint on the rim
    rim_paint = np.where(block[..., None] == 1, rgb(226, 226, 224), rgb(172, 90, 90))
    rim_cov = smooth(.28, .42, grain * .5 + grain2 * .5)
    col = lerp(col, lerp(dark, rim_paint, rim_cov), rim)

    # inside the grooves: smooth, thick, brighter paint
    mask = b_mask(s, t)
    inside = smooth(.02, .3, mask)
    g_paint = np.where(block[..., None] == 1, rgb(232, 230, 224), rgb(232, 150, 128))
    g_paint = g_paint * (.86 + .3 * n_mid)[..., None]
    # rubber dragged across the white grooves and red staining
    wear = smooth(.5, .66, n_lo) * smooth(.45, .9, mask)
    g_paint = lerp(g_paint, np.where(block[..., None] == 1, rgb(120, 118, 112), rgb(120, 58, 50)), wear * .55)
    col = lerp(col, g_paint, inside)

    # shared rust and rubber stains
    stain = smooth(.55, .72, fbm(H, W, 2.3, 16))
    col = lerp(col, rgb(52, 46, 46), stain * .28)

    # crazed paint
    cr = crackle(H, W, 7000, 17, 1.0)
    amount = np.maximum(rim * .8, inside)
    col = lerp(col, rgb(52, 46, 44), cr * amount * .55)

    save_rgb(col, 'floor_albedo.png')
    h = (.25 + (grain * .75 + grain2 * .25) * .6) * (1 - inside) + (.7 + (n_mid - .5) * .2) * inside
    h = h * (1 - rim * .35) + rim * .35 * (.4 + grain * .3)
    save_height(h - cr * amount * .3, 'floor_height.png')


# ---- Context textures --------------------------------------------------------
def tex_context():
    H, W = 1024, 1024
    a = fbm(H, W, 1.8, 21)
    b = fbm(H, W, 2.6, 22)
    c = fbm(H, W, .6, 23)
    g = lerp(rgb(126, 136, 48), rgb(196, 178, 76), a)
    g = lerp(g, rgb(150, 136, 70), smooth(.5, .7, b) * .6) * (.8 + .4 * c)[..., None]
    save_rgb(g, 'grass.png')
    # galvanised mast, weathered in long vertical runs
    H, W = 2048, 512
    a = fbm(H, W, 1.9, 24, ax=.25, ay=6)
    b = fbm(H, W, 1.6, 25)
    m = lerp(rgb(150, 156, 160), rgb(104, 106, 108), a)
    m = lerp(m, rgb(182, 186, 188), smooth(.55, .75, b) * .5)
    save_rgb(m, 'mast.png')
    # rust-free dark paint, with the lens glass on its own material
    tmac = Image.open(os.path.join(REF, 'TarmacDark.dds')).convert('RGB')
    tmac.save(os.path.join(TEX, 'tarmac.png'))


TEX_FUNCS = (tex_concrete, tex_ramp, tex_floor, tex_context)


# ---- Blender -----------------------------------------------------------------
def bpy_mod():
    import bpy
    return bpy


def reset(res=(1600, 900), exposure=0.0):
    bpy = bpy_mod()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import addon_utils
    addon_utils.enable('cycles')
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == 'OPTIX'
    sc.cycles.device = 'GPU'
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.image_settings.file_format = 'PNG'
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = exposure
    return bpy, sc


def world(top, mid, horizon, strength=1.0):
    bpy = bpy_mod()
    w = bpy.data.worlds.new('sky')
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes['Background']
    bg.inputs['Strength'].default_value = strength
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sp = nt.nodes.new('ShaderNodeSeparateXYZ')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    nt.links.new(tc.outputs['Generated'], sp.inputs[0])
    nt.links.new(sp.outputs['Z'], ramp.inputs['Fac'])
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (*horizon, 1)
    cr.elements[1].position = 0.75
    cr.elements[1].color = (*top, 1)
    e = cr.elements.new(0.22)
    e.color = (*mid, 1)
    nt.links.new(ramp.outputs[0], bg.inputs['Color'])
    bpy.context.scene.world = w


def sun(azimuth, elevation, power, color=(1, .86, .68), angle=1.0):
    """Azimuth in degrees: 0 means the sun sits on +X, 90 on +Y."""
    bpy = bpy_mod()
    from mathutils import Vector
    l = bpy.data.lights.new('sun', 'SUN')
    l.energy = power
    l.color = color
    l.angle = math.radians(angle)
    o = bpy.data.objects.new('sun', l)
    bpy.context.collection.objects.link(o)
    a, e = math.radians(azimuth), math.radians(elevation)
    to_sun = Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = (-to_sun).to_track_quat('-Z', 'Y')


def camera(loc, look, lens=28, fit='AUTO'):
    bpy = bpy_mod()
    from mathutils import Vector
    c = bpy.data.cameras.new('cam')
    c.lens = lens
    c.sensor_fit = fit
    c.sensor_height = 36
    c.clip_end = 2000
    o = bpy.data.objects.new('cam', c)
    bpy.context.collection.objects.link(o)
    o.location = loc
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = (Vector(look) - Vector(loc)).to_track_quat('-Z', 'Y')
    bpy.context.scene.camera = o


def mat_img(name, img, height=None, bump=.003, rough=.8, spec=.4, metal=0., tint=None):
    bpy = bpy_mod()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    p.inputs['Specular IOR Level'].default_value = spec
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = bpy.data.images.load(img, check_existing=True)
    out = t.outputs['Color']
    if tint is not None:
        mx = nt.nodes.new('ShaderNodeMixRGB')
        mx.blend_type = 'MULTIPLY'
        mx.inputs[0].default_value = 1
        mx.inputs[2].default_value = (*tint, 1)
        nt.links.new(out, mx.inputs[1])
        out = mx.outputs[0]
    nt.links.new(out, p.inputs['Base Color'])
    if height:
        h = nt.nodes.new('ShaderNodeTexImage')
        h.image = bpy.data.images.load(height, check_existing=True)
        h.image.colorspace_settings.name = 'Non-Color'
        b = nt.nodes.new('ShaderNodeBump')
        b.inputs['Distance'].default_value = bump
        b.inputs['Strength'].default_value = 1
        nt.links.new(h.outputs['Color'], b.inputs['Height'])
        nt.links.new(b.outputs['Normal'], p.inputs['Normal'])
    return m


def mat_flat(name, c, rough=.5, metal=0., spec=.5, emit=None):
    bpy = bpy_mod()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*c, 1)
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    p.inputs['Specular IOR Level'].default_value = spec
    if emit:
        p.inputs['Emission Color'].default_value = (*emit[0], 1)
        p.inputs['Emission Strength'].default_value = emit[1]
    return m


def add_mesh(name, v, f, mat, uv=None, smooth_angle=None):
    bpy = bpy_mod()
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in v], [], [tuple(q) for q in f])
    me.update()
    if uv is not None:
        lay = me.uv_layers.new(name='UV')
        idx = np.zeros(len(me.loops), int)
        me.loops.foreach_get('vertex_index', idx)
        lay.data.foreach_set('uv', uv[idx].ravel())
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    if smooth_angle is not None:
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        es = o.modifiers.new('es', 'EDGE_SPLIT')
        es.split_angle = math.radians(smooth_angle)
    return o


def strip(name, s_lines, t_lines, hf, mat, uvf, place=None, smooth_angle=None):
    """A grid in track space: s along the track, t from the edge to the infield."""
    S, T = np.meshgrid(s_lines, t_lines, indexing='ij')
    Z = hf(S, T)
    if place is None:
        place = lambda s, t, z: (s, t, z)
    X, Y, Z2 = place(S, T, Z)
    v = np.stack([X, Y, Z2], -1).reshape(-1, 3)
    n, m = len(s_lines), len(t_lines)
    i, j = np.mgrid[0:n - 1, 0:m - 1]
    a = (i * m + j).ravel()
    f = np.stack([a, a + m, a + m + 1, a + 1], 1)
    return add_mesh(name, v, f, mat, uv=uvf(S.ravel(), T.ravel()), smooth_angle=smooth_angle), len(f) * 2


def bend(radius):
    def place(s, t, z):
        a = s / radius
        r = radius - t
        return r * np.sin(a), radius - r * np.cos(a), z
    return place


def grass_blades(name, n, s_rng, t_rng, h_rng, place, seed, density=None, z=0., w_rng=(.003, .008)):
    """Loose blades as single triangles with vertex colours."""
    bpy = bpy_mod()
    rng = np.random.default_rng(seed)
    s = rng.uniform(*s_rng, n * 2)
    t = rng.uniform(*t_rng, n * 2)
    if density is not None:
        keep = rng.random(n * 2) < density(s, t)
        s, t = s[keep], t[keep]
    n = len(s)
    h = rng.uniform(*h_rng, n)
    az = rng.uniform(0, 6.283, n)
    lean = rng.uniform(.1, .7, n) * h
    w = rng.uniform(*w_rng, n)
    px, py, _ = place(s, t, np.zeros(n))
    v = np.zeros((n * 3, 3))
    v[0::3] = np.stack([px - np.sin(az) * w, py + np.cos(az) * w, np.full(n, z)], 1)
    v[1::3] = np.stack([px + np.sin(az) * w, py - np.cos(az) * w, np.full(n, z)], 1)
    v[2::3] = np.stack([px + np.cos(az) * lean, py + np.sin(az) * lean, z + h], 1)
    f = np.arange(n * 3).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.vertices.add(n * 3)
    me.vertices.foreach_set('co', v.ravel())
    me.loops.add(n * 3)
    me.loops.foreach_set('vertex_index', f.ravel())
    me.polygons.add(n)
    me.polygons.foreach_set('loop_start', np.arange(n) * 3)
    me.update(calc_edges=True)
    base = rng.choice(4, n)
    pal = np.array([[.10, .16, .02], [.20, .26, .03], [.30, .32, .05], [.34, .30, .10]])[base]
    col = np.zeros((n * 3, 4))
    col[..., 3] = 1
    col[0::3, :3] = pal * .45
    col[1::3, :3] = pal * .45
    col[2::3, :3] = pal * 1.25
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', col.ravel())
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']
    p.inputs['Roughness'].default_value = .5
    p.inputs['Transmission Weight'].default_value = 0
    at = nt.nodes.new('ShaderNodeVertexColor')
    at.layer_name = 'Col'
    nt.links.new(at.outputs['Color'], p.inputs['Base Color'])
    me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)


# ---- Parts for the pole ------------------------------------------------------
def frame_z(z):
    z = np.array(z, float)
    z = z / np.linalg.norm(z)
    a = np.array([0, 0, 1.]) if abs(z[2]) < .9 else np.array([1., 0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


class Part:
    def __init__(s):
        s.v, s.f, s.uv, s.n = [], [], [], 0

    def add(s, v, f, uv=None):
        s.v.append(v)
        s.f.append(np.array(f) + s.n)
        s.uv.append(np.zeros((len(v), 2)) if uv is None else uv)
        s.n += len(v)

    def lathe(s, prof, seg=24, at=(0, 0, 0), axis=(0, 0, 1)):
        prof = np.array(prof, float)
        rows = len(prof)
        an = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        ring = np.stack([np.cos(an), np.sin(an)], 1)
        v = np.zeros((rows * seg, 3))
        for j in range(rows):
            v[j * seg:(j + 1) * seg, :2] = ring * prof[j, 0]
            v[j * seg:(j + 1) * seg, 2] = prof[j, 1]
        uv = np.stack([np.tile(np.arange(seg) / seg, rows),
                       np.repeat(prof[:, 1] / max(prof[:, 1].max(), 1e-6), seg)], 1)
        v = v @ frame_z(axis).T + np.array(at)
        f = []
        for j in range(rows - 1):
            for i in range(seg):
                i2 = (i + 1) % seg
                f.append((j * seg + i, j * seg + i2, (j + 1) * seg + i2, (j + 1) * seg + i))
        s.add(v, f, uv)

    def tube(s, p0, p1, w=.06, h=None):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        h = h or w
        R = frame_z(p1 - p0)
        c = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
        v = []
        for p in (p0, p1):
            for x, y in c:
                v.append(p + R @ np.array([x, y, 0]))
        f = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (3, 2, 1, 0), (4, 5, 6, 7)]
        s.add(np.array(v), f)

    def box(s, c, size, axis=(0, 0, 1), taper=1.0):
        R = frame_z(axis)
        w, h, d = np.array(size) / 2
        v = []
        for z, k in ((-d, taper), (d, 1)):
            for x, y in ((-w, -h), (w, -h), (w, h), (-w, h)):
                v.append(np.array(c, float) + R @ np.array([x * k, y * k, z]))
        f = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (3, 2, 1, 0), (4, 5, 6, 7)]
        s.add(np.array(v), f)

    def build(s, name, mat, smooth_angle=None):
        v = np.concatenate(s.v)
        uv = np.concatenate(s.uv)
        f = np.concatenate([np.array(x) for x in s.f])
        return add_mesh(name, v, f, mat, uv=uv, smooth_angle=smooth_angle), len(f) * 2


def dish_at(frame, lens, base, face, r=.30, post=.50):
    """Round flood on a yoke. base is on the ring, face is where it throws light."""
    base = np.array(base, float)
    c = base + (0, 0, post + r * .75)
    face = np.array(face, float)
    face = face / np.linalg.norm(face)
    R = frame_z(face)
    prof = [(0, -.17), (.07, -.165), (.15, -.13), (.21, -.07), (r - .01, 0), (r, .03), (r, .05), (r - .015, .055)]
    frame.lathe(prof, 28, at=c, axis=face)
    frame.lathe([(.04, -.175), (.045, -.22), (0, -.225)], 10, at=c, axis=face)
    lens.lathe([(r - .012, .046), (r * .5, .03), (0, .024)], 28, at=c, axis=face)
    # yoke round the housing
    side = R[:, 0]
    if abs(side[2]) > .7:
        side = R[:, 1]
    a, b = c + side * (r + .025), c - side * (r + .025)
    y0 = base + (0, 0, post * .55)
    frame.tube(a, y0 + side * (r + .025), .028)
    frame.tube(b, y0 - side * (r + .025), .028)
    frame.tube(y0 + side * (r + .025), y0 - side * (r + .025), .034)
    frame.tube(base, y0, .05)


def flood_at(frame, lens, base, face, post=.16):
    """Rectangular flood, the lower lamp in the photo."""
    base = np.array(base, float)
    face = np.array(face, float)
    face = face / np.linalg.norm(face)
    c = base + (0, 0, post + .1)
    frame.box(c, (.46, .34, .16), axis=face, taper=.55)
    # cooling fins on the back
    R = frame_z(face)
    for k in np.linspace(-.13, .13, 6):
        frame.box(c - R[:, 2] * .10 + R[:, 0] * k, (.012, .17, .05), axis=face)
    lens.box(c + R[:, 2] * .082, (.41, .29, .006), axis=face)
    frame.tube(base, base + (0, 0, post), .05)


def build_pole(at=(0, 0), height=POLE_H, ground=0.):
    """Mast, service ring and lamps. Layout read off the photo, seen from below."""
    bpy = bpy_mod()
    mast_m = mat_img('mast', os.path.join(TEX, 'mast.png'), rough=.62, metal=.25)
    frame_m = mat_flat('frame', (.016, .016, .018), rough=.55)
    lens_m = mat_flat('lens', (.5, .52, .54), rough=.22, spec=.9)
    x0, y0 = at
    ring_z = height - .55
    glass_m = mat_flat('glass', (.035, .037, .04), rough=.14, spec=1.)
    mast, frame, lens, glass = Part(), Part(), Part(), Part()

    n = 12
    zs = [0, 4.9, 4.9, 5.05, 5.05, 9.9, 9.9, 10.03, 10.03, ring_z - .02]
    rs = []
    for k, z in enumerate(zs):
        r = .165 + (.075 - .165) * z / ring_z
        if k in (2, 3, 6, 7):
            r += .012
        rs.append(r)
    mast.lathe(list(zip(rs, zs)) + [(0, ring_z - .02)], n * 2)
    # conduit up the mast
    frame.tube((.0, -.17, 1), (.0, -.075, ring_z - .1), .025)
    # service ring, horizontal, a little over a metre across
    pts = np.array([(-.74, .62), (-.25, .70), (.27, .58), (.60, .24), (.60, -.20),
                    (.12, -.20), (.0, .16), (-.60, .22), (-.64, .05)])
    P = np.c_[pts, np.full(len(pts), ring_z)]
    chain = [0, 1, 2, 3, 4, 5]
    for a, b in zip(chain, chain[1:]):
        frame.tube(P[a], P[b], .045)
    frame.tube(P[0], P[8], .045)
    frame.tube(P[8], P[7], .045)
    frame.tube(P[7], (0, 0, ring_z), .045)
    frame.tube((0, 0, ring_z), P[2], .05)
    frame.tube((0, 0, ring_z), P[5], .05)
    frame.tube((0, 0, ring_z), P[4] * (.6, .6, 1), .05)
    frame.box((0, 0, ring_z - .02), (.20, .20, .16))        # collar
    frame.box((.26, .07, ring_z - .12), (.20, .14, .22))    # ballast box

    dish_at(frame, glass, (-.62, .64, ring_z), (-.12, -.72, -.68))
    dish_at(frame, glass, (.12, .64, ring_z), (.05, -.72, -.68))
    dish_at(frame, glass, P[3] + (.2, -.04, 0), (.55, -.6, -.58))
    flood_at(frame, lens, P[4] + (.05, .05, -.28), (.72, -.58, -.38))
    dish_at(frame, lens, (-.28, .05, ring_z - .55), (-.72, -.62, -.3), r=.22, post=.26)

    objs = []
    tris = 0
    for part, name, m, a in ((mast, 'mast', mast_m, 50), (frame, 'frame', frame_m, 40), (lens, 'lens', lens_m, 60), (glass, 'glass', glass_m, 60)):
        o, c = part.build(name, m, smooth_angle=a)
        o.location = (x0, y0, ground)
        tris += c
    return tris


def tyre(part, c, od=.60, id_=.36, w=.19, seg=18, tilt=(0, 0, 1)):
    ro, ri = od / 2, id_ / 2
    prof = [(ri, -w * .36), (ri + .03, -w * .47), (ro - .05, -w * .50), (ro - .012, -w * .42), (ro, -w * .22),
            (ro, w * .22), (ro - .012, w * .42), (ro - .05, w * .50), (ri + .03, w * .47), (ri, w * .36), (ri, -w * .36)]
    part.lathe(prof, seg, at=c, axis=tilt)


def tyres_scene(walls, pole_at=((0, 0),)):
    t = Part()
    rng = np.random.default_rng(5)
    for (x, y) in pole_at:
        for k in range(5):
            d = rng.uniform(-.015, .015, 2)
            tyre(t, (x + d[0], y + d[1], .095 + k * .19), seg=24)
    for x0, x1, y in walls:
        for x in np.arange(x0, x1, .62):
            for lane in (0, .55):
                for k in range(2):
                    d = rng.uniform(-.02, .02, 2)
                    tyre(t, (x + d[0] + lane * .5, y + lane + d[1], .095 + k * .19), seg=14)
    return t.build('tyres', mat_flat('rubber', (.018, .018, .019), rough=.62, spec=.35), smooth_angle=50)


def backdrop(distance=190., height=46.):
    """Tree line cut out of the photo, with the near mast painted over."""
    bpy = bpy_mod()
    im = np.asarray(Image.open(os.path.join(ASSETS, 'pole.jpeg')).convert('RGB'))[820:1022].astype(float)
    im[:, 348:392] = im[:, 296:340]
    lum = im.mean(2)
    blue = im[..., 2] - im[..., 0]
    a = np.clip((70 - blue) / 40, 0, 1) * np.clip((165 - lum) / 45, 0, 1)
    a[-14:] = 1
    rgba = np.dstack([im, a * 255]).astype(np.uint8)
    p = os.path.join(TEX, 'treeline.png')
    Image.fromarray(rgba).save(p)
    W = height * im.shape[1] / im.shape[0]
    v = [(-W / 2, distance, -.5), (W / 2, distance, -.5), (W / 2, distance, height), (-W / 2, distance, height)]
    m = bpy.data.materials.new('tree')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    tx = nt.nodes.new('ShaderNodeTexImage')
    tx.image = bpy.data.images.load(p)
    tx.extension = 'CLIP'
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mx = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(tx.outputs['Color'], em.inputs['Color'])
    nt.links.new(tx.outputs['Alpha'], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs['Surface'])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
    add_mesh('tree', v, [(0, 1, 2, 3)], m, uv=uv)


# ---- Current track meshes ----------------------------------------------------
def kn5_mesh(name_filter):
    """Pull one mesh out of velopark.kn5, as positions, uvs and triangles."""
    from kn5_scan import read
    b, ver, tex, mats, nodes = read(os.path.join(TRACK, 'velopark.kn5'))
    n = next(x for x in nodes if x['name'] == name_filter)
    d = np.frombuffer(b, dtype='<f4', count=n['vc'] * 11, offset=n['vp']).reshape(-1, 11).copy()
    idx = np.frombuffer(b, dtype='<u2', count=n['ic'], offset=n['ip']).reshape(-1, 3)
    return d[:, :3], d[:, 6:8], idx


def old_kerb(place_at):
    bpy = bpy_mod()
    p, uv, f = kn5_mesh('1ROAD_87')
    q = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)
    q -= q.mean(0)
    u, s, vt = np.linalg.svd(q[:, :2], full_matrices=False)
    q[:, :2] = q[:, :2] @ vt.T
    if np.sign(q[:, 2] @ q[:, 1]) > 0:
        q[:, 1] *= -1
        f = f[:, ::-1]
    q[:, 2] -= np.percentile(q[:, 2], 2)
    q += place_at
    uv2 = np.stack([uv[:, 0], 1 - uv[:, 1]], 1)
    m = mat_img('rumble_old', os.path.join(REF, 'Rumble1.dds.png'), rough=.8)
    return add_mesh('old_kerb', q, f, m, uv=uv2), q


def old_pole(at):
    bpy = bpy_mod()
    p, uv, f = kn5_mesh('OBJ66')
    q = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)
    q[:, :2] -= q[q[:, 2] < np.percentile(q[:, 2], 30)][:, :2].mean(0)
    q[:, 2] -= q[:, 2].min()
    q += (at[0], at[1], 0)
    uv2 = np.stack([uv[:, 0], 1 - uv[:, 1]], 1)
    m = mat_img('pole_old', os.path.join(REF, 'track_light.dds.png'), rough=.5)
    return add_mesh('old_pole', q, f, m, uv=uv2, smooth_angle=45)


# ---- Scenes ------------------------------------------------------------------
def uv_tile(depth):
    return lambda s, t: np.stack([s / TILE, 1 - t / depth], 1)


def uv_plane(sx, sy):
    return lambda s, t: np.stack([s / sx, t / sy], 1)


def lin(c):
    c = np.array(c, float) / 255
    return tuple(np.where(c < .04, c / 12.92, ((c + .055) / 1.055) ** 2.4))


def mats_common():
    return dict(
        conc=mat_img('conc', os.path.join(TEX, 'concrete_albedo.png'), os.path.join(TEX, 'concrete_height.png'), bump=.004, rough=.82, spec=.35),
        floor=mat_img('floor', os.path.join(TEX, 'floor_albedo.png'), os.path.join(TEX, 'floor_height.png'), bump=.006, rough=.72, spec=.4),
        tar=mat_img('tar', os.path.join(TEX, 'tarmac.png'), rough=.85, spec=.35, tint=(1.15, 1.08, 1.0)),
        grass=mat_img('grass', os.path.join(TEX, 'grass.png'), rough=.9, spec=.2),
    )


def base_a(s0=-6, s1=8, R=45.):
    m = mats_common()
    place = bend(R)
    tris = 0
    o, c = strip('kerb_a', a_s_lines(s0, s1), A_T_LINES, a_height, m['conc'], uv_tile(A_DEPTH), place, smooth_angle=38)
    fl = np.linspace(s0, s1, 90)
    strip('road', fl, np.array([-9., 0]), lambda s, t: np.zeros_like(s), m['tar'], uv_plane(4, 8), place)
    strip('verge', fl, np.linspace(1.9, 12, 6), lambda s, t: np.full_like(s, -.006), m['grass'], uv_plane(5, 5), place)
    return place, c


def shot_concrete_photo():
    bpy, sc = reset(exposure=.55)
    world(lin((120, 150, 190)), lin((190, 200, 212)), lin((232, 226, 214)), .6)
    place, c = base_a()
    sun(186, 15, 3.6)
    dens = lambda s, t: np.clip((t - 1.78) * 2.2, 0, 1) + .012 * (np.sin(s * 3.1) > .4) * (t > 1.2)
    grass_blades('blades', 1500000, (-6, 8), (1.25, 4.0), (.05, .2), place, 3, dens, w_rng=(.002, .006))
    px, py, _ = place(np.array([1.0]), np.array([.95]), np.zeros(1))
    camera((px[0] - .05, py[0] + 1.75, 2.28), (px[0], py[0] + .22, 0), lens=25)
    render(sc, 'concrete_photo')
    return c


def shot_concrete_driver():
    bpy, sc = reset(exposure=.45)
    world(lin((120, 150, 190)), lin((190, 200, 212)), lin((232, 226, 214)), .6)
    place, c = base_a(-4, 24)
    sun(150, 22, 3.4)
    dens = lambda s, t: np.clip((t - 1.78) * 2.2, 0, 1)
    grass_blades('blades', 2500000, (-4, 24), (1.25, 7), (.05, .18), place, 4, dens, w_rng=(.002, .006))
    a = place(np.array([.2]), np.array([-1.5]), np.zeros(1))
    b = place(np.array([8.]), np.array([.9]), np.zeros(1))
    camera((a[0][0], a[1][0], .42), (b[0][0], b[1][0], .05), lens=34)
    render(sc, 'concrete_driver')


def shot_concrete_low():
    bpy, sc = reset(exposure=.45)
    world(lin((120, 150, 190)), lin((190, 200, 212)), lin((232, 226, 214)), .6)
    place, c = base_a(-3, 14)
    sun(186, 15, 3.6)
    dens = lambda s, t: np.clip((t - 1.82) * 2.5, 0, 1)
    grass_blades('blades', 1500000, (-3, 14), (1.25, 4.0), (.05, .16), place, 5, dens, w_rng=(.002, .006))
    a = place(np.array([.0]), np.array([2.15]), np.zeros(1))
    b = place(np.array([2.4]), np.array([1.1]), np.zeros(1))
    camera((a[0][0], a[1][0], .48), (b[0][0], b[1][0], .04), lens=24)
    render(sc, 'concrete_low')


def base_b(s0=-3.5, s1=6):
    m = mats_common()
    o, c = strip('kerb_b', b_s_lines(s0, s1), B_T_LINES, b_height, m['floor'], uv_tile(B_WIDTH), smooth_angle=48)
    fl = np.array([s0, s1])
    strip('road', fl, np.array([-8., 0]), lambda s, t: np.zeros_like(s), m['tar'], uv_plane(4, 8))
    strip('line', fl, np.array([-1.62, -1.42]), lambda s, t: np.full_like(s, .002),
          mat_flat('line', lin((205, 205, 200)), rough=.8), uv_plane(4, 8))
    strip('shoulder', fl, np.array([B_WIDTH, B_WIDTH + 1.5]), lambda s, t: np.zeros_like(s),
          mat_img('tar2', os.path.join(TEX, 'tarmac.png'), rough=.85, tint=(.6, .46, .36)), uv_plane(4, 8))
    strip('verge', fl, np.array([B_WIDTH + 1.5, 20]), lambda s, t: np.full_like(s, -.004), m['grass'], uv_plane(5, 5))
    return c


def shot_floor_photo():
    bpy, sc = reset(exposure=.5)
    world(lin((130, 150, 180)), lin((196, 200, 208)), lin((228, 222, 210)), .7)
    c = base_b()
    sun(35, 33, 2.6, color=(1, .92, .8))
    camera((1.05, -1.5, 2.6), (1.05, -.55, 0), lens=24)
    render(sc, 'floor_photo')
    return c


def shot_floor_close():
    bpy, sc = reset(exposure=.5)
    world(lin((130, 150, 180)), lin((196, 200, 208)), lin((228, 222, 210)), .7)
    base_b()
    sun(10, 14, 3.0, color=(1, .88, .72))
    camera((1.5, -.55, .34), (.9, .42, 0), lens=30)
    render(sc, 'floor_close')


def shot_pole_photo():
    bpy, sc = reset((900, 1600))
    world(lin((26, 66, 128)), lin((88, 150, 205)), lin((205, 222, 232)), 1.0)
    m = mats_common()
    tris = build_pole((0, 0))
    tyres_scene([(-2.1, 30, -3.6)], ((0, 0),))
    build_ctx_ground(m)
    sun(160, 9, 3.0, color=(1, .8, .56))
    backdrop()
    f = 1600 / (2 * math.tan(math.radians(32.5)))
    cam_y = -19.5
    pitch = math.atan((1020 - 800) / f)
    camera((1.55, cam_y, 1.2), (1.55, cam_y + 10, 1.2 + 10 * math.tan(pitch)), lens=f / 1600 * 36, fit='VERTICAL')
    render(sc, 'pole_photo')
    return tris


def build_ctx_ground(m, road=(-15.3, -7.3)):
    def place(s, t, z):
        return s, road[1] + t, z + .012
    strip('kerb_b', b_s_lines(-45, 45), B_T_LINES, b_height, m['floor'], uv_tile(B_WIDTH), place, smooth_angle=48)
    fl = np.array([-300., 300])
    strip('road', fl, np.array(road), lambda s, t: np.full_like(s, .012), m['tar'], uv_plane(4, 8))
    strip('near', fl, np.array([-60., road[0]]), lambda s, t: np.zeros_like(s), m['grass'], uv_plane(7, 7))
    strip('far', fl, np.array([road[1] + B_WIDTH, 200.]), lambda s, t: np.zeros_like(s), m['grass'], uv_plane(7, 7))
    plain = lambda s, t, z: (s, t, z)
    grass_blades('near_blades', 2500000, (-8, 10), (-20, road[0]), (.04, .13), plain, 7, w_rng=(.002, .005))
    grass_blades('far_blades', 3000000, (-22, 30), (road[1] + B_WIDTH, 4.), (.05, .16), plain, 8, w_rng=(.006, .014))


def shot_pole_head():
    bpy, sc = reset((1200, 900))
    world(lin((26, 66, 128)), lin((88, 150, 205)), lin((205, 222, 232)), 1.0)
    build_pole((0, 0))
    sun(170, 18, 3.2, color=(1, .8, .58))
    z = POLE_H - .4
    camera((-.5, -13.5, z - 11.0), (.05, .1, z + .1), lens=100)
    render(sc, 'pole_head_below')
    camera((3.2, -3.0, z + 1.4), (0, .1, z), lens=50)
    render(sc, 'pole_head_side')


def shot_compare_pole():
    bpy, sc = reset((1300, 1300))
    world(lin((60, 110, 175)), lin((140, 180, 215)), lin((222, 228, 228)), .9)
    m = mats_common()
    strip('g', np.array([-200., 200]), np.array([-200., 200]), lambda s, t: np.zeros_like(s), m['grass'], uv_plane(7, 7))
    old_pole((-3.2, 0))
    build_pole((3.2, 0))
    tyres_scene([], ((3.2, 0), (-3.2, 0)))
    sun(140, 32, 3.0, color=(1, .92, .8))
    camera((0, -27, 1.7), (0, 0, 8.6), lens=31)
    render(sc, 'compare_pole')


def shot_compare_kerb():
    bpy, sc = reset()
    world(lin((96, 140, 190)), lin((180, 198, 215)), lin((222, 220, 212)), .75)
    m = mats_common()
    strip('road', np.array([-30., 30]), np.array([-12., 40.]), lambda s, t: np.zeros_like(s), m['tar'], uv_plane(4, 8))
    old_kerb(np.array([0., 3.0, .005]))
    # new concrete kerb across the frame, under the old one, floor kerb on the near side
    strip('a', a_s_lines(-7, 7), A_T_LINES, a_height, m['conc'], uv_tile(A_DEPTH),
          lambda s, t, z: (s, t - 1.15, z + .002), smooth_angle=38)
    strip('b', b_s_lines(-7, 7), B_T_LINES, b_height, m['floor'], uv_tile(B_WIDTH),
          lambda s, t, z: (s, t - 3.6, z + .002), smooth_angle=48)
    sun(160, 26, 3.0, color=(1, .9, .76))
    camera((.3, -6.6, 7.4), (.3, -.5, 0), lens=30)
    render(sc, 'compare_kerb')


def render(sc, name):
    import bpy
    sc.render.filepath = os.path.join(OUT, 'render', name + '.png')
    bpy.ops.render.render(write_still=True)
    print('RENDERED', name, flush=True)


SHOTS = dict(concrete_photo=shot_concrete_photo, concrete_driver=shot_concrete_driver, concrete_low=shot_concrete_low, floor_photo=shot_floor_photo,
             floor_close=shot_floor_close, pole_photo=shot_pole_photo, pole_head=shot_pole_head,
             compare_pole=shot_compare_pole, compare_kerb=shot_compare_kerb)


if __name__ == '__main__':
    os.makedirs(TEX, exist_ok=True)
    os.makedirs(os.path.join(OUT, 'render'), exist_ok=True)
    args = sys.argv[1:] or ['all']
    if 'tex' in args or 'all' in args:
        for fn in TEX_FUNCS:
            fn()
            print('TEX', fn.__name__, flush=True)
    for k, f in SHOTS.items():
        if k in args or 'all' in args:
            print('tris', k, f(), flush=True)
