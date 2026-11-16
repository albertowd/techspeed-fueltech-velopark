"""The objects' catalogue: tools/reference/templates/models.json, each model a template of its own there
(<model>.npz), what tools/place_objects.py lays wherever objects.json puts one, and the road editor
(top-race-editor) lists, describes and draws.

    python tools/templates.py bake [MODEL ...]   (the light pole, the cone, the board, the timing table taken from the
                                                 models as they are; only those named, if any)
    python tools/templates.py dress      (the light pole mapped in metres, its lamps' glass texture made)

A template: its parts, each '<part>_verts' (points: x, y, z, normal, u, v, tangent; in the object's own frame:
x to its right, y up from the floor, z forward along its heading, its middle at 0, 0), '<part>_faces' and
'<part>_material' (the material's name in the models). models.json: for each model, its file, a label and a
description for the editor, its size (width, height, depth), how it may lie along a curve (fits: 'place', 'bend'),
the collisions it may have (a single one: 'none', 'static', 'dynamic'; along a curve: 'none', 'wall'), its hull
(a dynamic one's physics box: its part 'hull'), and its lights (the Custom Shaders Patch's: each its series (the
DESCRIPTION of a LIGHT_SERIES of ext_config.ini), where it is ('at': right, up, forward) and where it faces
('facing', the same frame), written there for each object of it).
"""
import json
import math
import os
import re
import sys

import numpy as np

import build_paths as bp
import gltf_source

DIR = os.path.join(bp.ROOT, 'tools', 'reference', 'templates')
CATALOGUE = os.path.join(DIR, 'models.json')
# the road editor's own models (top-race-editor/models: <id>.glb, its catalogue.json), laid as the catalogue's are
EDITOR_MODELS = os.environ.get('EDITOR_MODELS', os.path.join(os.path.dirname(bp.ROOT), 'top-race-editor', 'models'))
# what the editor's kinds of model may do, as the catalogue says of its own: how along a curve, what collisions
EDITOR_KINDS = {'fences': (['bend', 'place'], ['none', 'wall']), 'trees': (['place'], ['none', 'static']),
                'sheds': (['place'], ['none', 'static']), 'gantries': (['place'], ['none', 'static']),
                'karts': (['place'], ['none', 'static', 'dynamic'])}


def catalogue():
    """The catalogue: models.json's, and the editor's models it has not (theirs: from their glb, 'editor': its path)."""
    out = json.load(open(CATALOGUE, encoding='utf-8'))
    ed = os.path.join(EDITOR_MODELS, 'catalogue.json')
    if os.path.exists(ed):
        items = json.load(open(ed, encoding='utf-8'))
        items = items.get('models', items)
        for key, m in (items.items() if isinstance(items, dict) else [(x.get('id'), x) for x in items]):
            if key in out or not isinstance(m, dict) or not os.path.exists(os.path.join(EDITOR_MODELS, key + '.glb')):
                continue
            fits, coll = EDITOR_KINDS.get(m.get('kind'), (['place'], ['none', 'static']))
            out[key] = dict(editor=os.path.join(EDITOR_MODELS, key + '.glb'), label=m.get('label', key), size=m.get('size'),
                            fits=fits, collision=coll, lights=[], kind=m.get('kind'))
    return out


_EDITOR = {}


def editor_model(model):
    """An editor's model as a kn5 (its materials, its textures as they are in its glb)."""
    if model not in _EDITOR:
        doc, binary = gltf_source.read_glb(catalogue()[model]['editor'])
        _EDITOR[model] = gltf_source.from_gltf(doc, binary, lambda uri: open(uri, 'rb').read())
    return _EDITOR[model]


def load(model):
    """A model's parts: [(part, points (n, 11), faces (m, 3), material)]."""
    meta = catalogue()[model]
    if 'editor' in meta:
        k = editor_model(model)
        out = []
        for n, m in bp.meshes(k):
            if len(n['faces']):
                V = gltf_source_world(n, m)
                W = np.zeros((len(V), 11))
                W[:, :min(11, V.shape[1])] = V[:, :11]
                out.append((n['name'].lower(), W, n['faces'].astype(int).reshape(-1, 3), k['materials'][n['material']]['name']))
        return out
    t = np.load(os.path.join(DIR, meta['file']))
    parts = sorted({k[:-len('_verts')] for k in t.files if k.endswith('_verts')})
    return [(p, t[p + '_verts'].astype(float), t[p + '_faces'].astype(int), str(t[p + '_material'])) for p in parts]


def to_local(V, c, y0, h):
    """Points of the model (n, 11) into an object's own frame: its middle c (x, z), its floor y0, its heading h."""
    s, co = math.sin(h), math.cos(h)
    R = np.array([[co, -s], [s, co]])                 # (local x, z) @ R: the world's, turned to h
    out = V.copy()
    out[:, [0, 2]] = (V[:, [0, 2]] - c) @ R.T
    out[:, 1] = V[:, 1] - y0
    out[:, [3, 5]] = V[:, [3, 5]] @ R.T
    if V.shape[1] > 10:
        out[:, [8, 10]] = V[:, [8, 10]] @ R.T
    return out


def world(V, c, y0, h):
    """An object's own points (n, 11) into the world, as to_local's other way."""
    s, co = math.sin(h), math.cos(h)
    R = np.array([[co, -s], [s, co]])
    out = V.copy()
    out[:, [0, 2]] = V[:, [0, 2]] @ R + c
    out[:, 1] = V[:, 1] + y0
    out[:, [3, 5]] = V[:, [3, 5]] @ R
    if V.shape[1] > 10:
        out[:, [8, 10]] = V[:, [8, 10]] @ R
    return out


def save(model, parts):
    np.savez(os.path.join(DIR, model + '.npz'),
             **{k: v for p, V, F, mat in parts
                for k, v in ((p + '_verts', V.astype(np.float32)), (p + '_faces', F.astype(np.int32)), (p + '_material', np.array(mat)))})


def size(parts):
    P = np.concatenate([V[:, :3] for p, V, F, m in parts if p != 'hull'])
    return [round(float(x), 3) for x in np.ptp(P, 0)]


def bake(which=None):
    """The models named (all if none) taken as templates from the models as they are: the pole (POLE_0 of
    velopark_poles.glb: its mast, frame, lenses and glass, its lights in ext_config.ini), the cone and the board (the
    kids layout's first cone and its BOARD_0, with their physics boxes), the timing table (by the hotlap's
    building); the others' as the catalogue has them."""
    import place_objects as po
    out = json.load(open(CATALOGUE, encoding='utf-8')) if os.path.exists(CATALOGUE) else {}
    do = lambda m: which is None or m in which
    # the pole
    if do('light_pole'):
        bake_pole(po, out)
    # the cone and the board: under a layout's physics object (AC_POBJECT_n: its box, and the thing under it)
    if do('cone') or do('board'):
        bake_pobjects(po, out, [m for m in ('cone', 'board') if do(m)])
    if do('timing_table'):
        bake_table(po, out)
    if do('marshal_post') or do('chair'):
        bake_marshal(po, out)
    if do('shed_passage'):
        bake_passage(out)
    if do('open_passage'):
        bake_open_passage(out)
    if do('tyre_stack'):
        stack = load_file('tyre_stack.npz')
        out['tyre_stack'] = dict(file='tyre_stack.npz', label='Tyre stack',
                                 description="Three tyres stacked, the tyre walls' stack: laid along a curve, a tyre wall.",
                                 size=size(stack), fits=['place'], collision=['none', 'static', 'wall'], lights=[],
                                 lods={'tyres': [0., 45.], 'fill': [45., 2000.]})     # (the tyres near, a plain stand-in for them far)
    json.dump(dict(sorted(out.items())), open(CATALOGUE, 'w', encoding='utf-8'), indent=2)
    return {m: out[m] for m in out if do(m)}


def bake_pole(po, out):
    k = gltf_source.load_kn5(po.POLES)
    parts = po.pole_parts(k)['POLE_0']
    c, h, lens = po.where_pole(parts)
    y0 = float(po.floor(c[0], c[1])[0])
    pole = [(n['name'].split('_')[-1].lower(), to_local(n['verts'].astype(float), c, y0, h), n['faces'].astype(int).reshape(-1, 3),
             k['materials'][n['material']]['name']) for n, m in parts if not n['name'].endswith('_TYRES') and len(n['faces'])]
    save('light_pole', pole)
    dress()
    lights = []
    text = open(po.EXT, encoding='utf-8').read().replace('\r', '')
    s_, co = math.sin(h), math.cos(h)
    right, fwd = np.array([co, -s_]), np.array([s_, co])
    for m in re.finditer(r'\[LIGHT_SERIES_\d+\]\n(.*?)(?=\n\[|\Z)', text, re.S):
        body = m.group(1)
        desc = re.search(r'^DESCRIPTION = (.*)$', body, re.M).group(1).strip()
        common = re.search(r'^DIRECTION = (.*)$', body, re.M)
        for i, p in re.findall(r'^POSITION_(\d+) = (.*)$', body, re.M):
            P = np.array([float(v) for v in p.split(',')])
            if np.hypot(*(P[[0, 2]] - lens)) >= po.LIGHT_NEAR:
                continue
            d = re.search(r'^DIRECTION_%s = (.*)$' % i, body, re.M) or common
            D = np.array([float(v) for v in d.group(1).split(',')]) if d else np.array([0., -1., 0.])
            q = P[[0, 2]] - c
            lights.append({'series': desc, 'at': [round(float(q @ right), 3), round(float(P[1] - y0), 3), round(float(q @ fwd), 3)],
                           'facing': [round(float(D[[0, 2]] @ right), 3), round(float(D[1]), 3), round(float(D[[0, 2]] @ fwd), 3)]})
    out['light_pole'] = dict(file='light_pole.npz', label='Light pole',
                       description='A light pole, %.0f m high, with %d night lights at its top (Custom Shaders Patch): '
                                   'a flood beam facing forward along its heading and down onto the track, and a spill '
                                   'straight down round it. Its tyres are a tyre stack of its own.' % (size(pole)[1], len(lights)),
                       size=size(pole), fits=['place'], collision=['none'], lights=lights)


def bake_pobjects(po, out, models):
    k = gltf_source.load_kn5(os.path.join(bp.MODELS, 'vp_kids.glb'))
    for model, child, label, desc, heading in (
            ('cone', 'OBJ', 'Cone', 'A traffic cone, orange.', 0.),
            ('board', 'BOARD_', 'Marker board', 'A marker board on a stand: its face toward its heading.', None)):
        if model not in models:
            continue
        top = next(n for n in k['root']['children'] if n['name'].startswith('AC_POBJECT_')
                   and any(c_['name'].startswith(child) for c_ in n.get('children', [])))
        vis_name = next(c_['name'] for c_ in top['children'] if c_['name'].startswith(child))
        all_ = bp.meshes(k)
        hull_n, hull_m = next((n, m) for n, m in all_ if n['name'] == top['name'])
        vis_n, vis_m = next((n, m) for n, m in all_ if n['name'] == vis_name)
        Wv = gltf_source_world(vis_n, vis_m)
        Wh = gltf_source_world(hull_n, hull_m)
        c = Wv[:, [0, 2]].mean(0)
        h = heading if heading is not None else po.heading_of(np.array(top['matrix'], float).reshape(4, 4))
        y0 = float(Wv[:, 1].min())
        parts = [(model, to_local(Wv, c, y0, h), vis_n['faces'].astype(int).reshape(-1, 3), k['materials'][vis_n['material']]['name']),
                 ('hull', to_local(Wh, c, y0, h), hull_n['faces'].astype(int).reshape(-1, 3), k['materials'][hull_n['material']]['name'])]
        save(model, parts)
        out[model] = dict(file=model + '.npz', label=label, description=desc, size=size(parts), fits=['place'],
                          collision=['none', 'static', 'dynamic', 'wall'], hull='hull', lights=[])


def bake_table(po, out):
    """The timing table (tools/build_outside.py's, by the hotlap's building): its table, its TVs and their two screens
    (the Custom Shaders Patch's displays, by their materials), its front the way the screens face."""
    k = gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_outside.glb'))
    parts = [(n, m) for n, m in bp.meshes(k) if n['name'].startswith('TIMING_') and len(n['faces'])]
    screen = next(gltf_source_world(n, m) for n, m in parts if n['name'].startswith('TIMING_SCREEN'))
    Wt = np.concatenate([gltf_source_world(n, m) for n, m in parts if n['name'] == 'TIMING_TABLE'])
    c = Wt[:, [0, 2]].mean(0)
    h = math.atan2(*screen[:, [3, 5]].mean(0))
    y0 = float(po.floor(c[0], c[1])[0])
    table = [(n['name'][len('TIMING_'):].lower(), to_local(gltf_source_world(n, m), c, y0, h), n['faces'].astype(int).reshape(-1, 3),
              k['materials'][n['material']]['name']) for n, m in parts]
    save('timing_table', table)
    out['timing_table'] = dict(file='timing_table.npz', label='Timing table',
                               description="A table with two TVs, their screens forward: the session's times and the HotLap "
                                           "Velopark site's page (Custom Shaders Patch displays, as on the table by the hotlap's building).",
                               size=size(table), fits=['place'], collision=['none', 'static'], lights=[])



MARSHAL_AT = (-87.4, 96.4, 1.)   # the vp_1000's flag marshal's post (tools/build_outside.py's marshal: its middle, its track's side)


def bake_marshal(po, out):
    """The flag marshal's post (tools/build_outside.py's, by the vp_1000's start) as two models: its shelter (its four
    poles, the struts from its back, its canopy; its front to the track) and its chair (seat, back and legs; its front
    the way one sits in it)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    k = gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_outside.glb'))
    cx, cz, side = MARSHAL_AT
    y0 = float(po.floor(cx, cz)[0])
    mesh = {n['name']: (n, m) for n, m in bp.meshes(k) if n['name'].endswith('_1000') and n['name'].startswith('MARSHAL_')}
    n, m = mesh['MARSHAL_POLES_1000']
    V, F = gltf_source_world(n, m), n['faces'].astype(int).reshape(-1, 3)
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
    # (the chair: the pieces within its seat's square, under its back's top)
    chair_piece = {c for c in np.unique(lab) if np.all(np.abs(V[lab == c, 0] - cx) < .3) and np.all(np.abs(V[lab == c, 2] - cz) < .3)
                   and V[lab == c, 1].max() < y0 + 1.}
    in_chair = np.isin(lab[F[:, 0]], list(chair_piece))
    mat = k['materials'][n['material']]['name']
    h_shelter = math.atan2(side, 0.)                       # (its front to the track's side)
    h_chair = math.atan2(-side, 0.)                        # (its back toward the track's side: one sits facing away)
    c = np.array([cx, cz])

    def part(name, Fs, h, mat_):
        idx = np.unique(Fs)
        remap = np.full(len(V), -1)
        remap[idx] = np.arange(len(idx))
        return (name, to_local(V[idx], c, y0, h), remap[Fs], mat_)
    tn, tm = mesh['MARSHAL_TOP_1000']
    shelter = [part('poles', F[~in_chair], h_shelter, mat),
               ('top', to_local(gltf_source_world(tn, tm), c, y0, h_shelter), tn['faces'].astype(int).reshape(-1, 3), k['materials'][tn['material']]['name'])]
    chair = [part('chair', F[in_chair], h_chair, mat)]
    save('marshal_post', shelter)
    save('chair', chair)
    out['marshal_post'] = dict(file='marshal_post.npz', label='Marshal shelter',
                               description='A flag marshal\'s shelter: a white canopy on four poles, braced from its back; its front '
                                           'to the track. Its chair and its cones are objects of their own.',
                               size=size(shelter), fits=['place'], collision=['none', 'static'], lights=[])
    out['chair'] = dict(file='chair.npz', label='Chair', description='A plain steel chair: its front the way one sits in it.',
                        size=size(chair), fits=['place'], collision=['none', 'static'], lights=[])


def bake_passage(out):
    """The editor's white shed as an open passage: its end walls (and its door) out below its eaves, so one walks through
    it from end to end; its gables, its roof, its side walls and windows, its floor and its lamps as they are."""
    parts = []
    for p, V, F, mat in load('shed_white'):
        if p == 'shed_openings':
            continue
        if p in ('shed_walls', 'shed_inside'):
            T = V[F]
            end = np.all(np.abs(np.abs(T[:, :, 2]) - 3.) < .05, axis=1) & (T[:, :, 1].max(1) <= 2.61)
            F = F[~end]
        parts.append((p.replace('shed_', ''), V, F, mat))
    save('shed_passage', parts)
    out['shed_passage'] = dict(file='shed_passage.npz', label='Open shed',
                               description='The white shed open at both ends, a covered passage: one walks through it from end to end.',
                               size=size(parts), fits=['place'], collision=['none', 'static'], lights=[])



PASSAGE = dict(width=6.4, depth=7.4, wall=.25, high=2.7, slab=.25, over=.2)   # metres: across its walls, along it, a wall's
                                                                               # thickness, the slab's underside, its thickness, past the walls


def box_mesh(lo, hi, metres=1.):
    """A box (lo, hi: its corners) as points (24, 11) and faces (12, 3): each side its own points, facing out, mapped
    flat on it, a texture every metres."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V, F = [], []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3); n[axis] = sign
            u_ax, v_ax = [a for a in range(3) if a != axis]
            pts = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.where(sign > 0, hi, lo).astype(float).copy()
                p[u_ax] = (lo, hi)[a][u_ax]
                p[v_ax] = (lo, hi)[b][v_ax]
                pts.append(p)
            base = len(V)
            for p in pts:
                row = np.zeros(11)
                row[:3], row[3:6] = p, n
                row[6], row[7] = p[u_ax] / metres, p[v_ax] / metres if axis != 1 else p[v_ax] / metres
                V.append(row)
            f = [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            # (each face facing out: its winding turned where its normal points the other way)
            P0, P1, P2 = pts[0], pts[1], pts[2]
            if np.dot(np.cross(P1 - P0, P2 - P0), n) < 0:
                f = [(a, c, b) for a, b, c in f]
            F += f
    V = np.array(V)
    return V, np.array(F)


def bake_open_passage(out):
    """A walk-through passage: a concrete floor, two side walls (white), a straight concrete slab for its roof; open at
    both ends (its front and its back: one walks through it along z)."""
    p = PASSAGE
    w, d, t, h, s, o = p['width'], p['depth'], p['wall'], p['high'], p['slab'], p['over']
    parts = []
    walls = [box_mesh((-w / 2, -.3, -d / 2), (-w / 2 + t, h, d / 2)), box_mesh((w / 2 - t, -.3, -d / 2), (w / 2, h, d / 2))]
    V = np.concatenate([walls[0][0], walls[1][0]]); F = np.concatenate([walls[0][1], walls[1][1] + len(walls[0][0])])
    parts.append(('walls', tangents(V, F), F, 'OutWhite'))
    V, F = box_mesh((-w / 2 - o, h, -d / 2 - o), (w / 2 + o, h + s, d / 2 + o), 2.)
    parts.append(('roof', tangents(V, F), F, 'OutConcrete'))
    V, F = box_mesh((-w / 2 + t, -.1, -d / 2), (w / 2 - t, .03, d / 2), 2.)
    parts.append(('floor', tangents(V, F), F, 'OutConcrete'))
    save('open_passage', parts)
    out['open_passage'] = dict(file='open_passage.npz', label='Open passage',
                               description='A walk-through passage: two side walls under a straight concrete slab, open at its '
                                           'front and its back (one walks through it front to back), %.1f m across, %.1f m long.' % (w, d),
                               size=size(parts), fits=['place'], collision=['none', 'static'], lights=[])


MAST_R = .165        # metres: the mast's radius at its foot (its texture round it at that)
MAST_M = 2.          # metres a texture of the masts' concrete covers (tools/scan_materials.py's scan)
STEEL_M = 1.         # metres a texture of the frames' steel covers (tools/scan_materials.py's scan)
LENS_PX = 512        # the lamps' glass texture's side, pixels


def tangents(V, F):
    """Each point's tangent (along its u), from its faces' uv: V (n, 11), its columns 8-10 set."""
    T = np.zeros((len(V), 3))
    P, UV = V[:, :3], V[:, 6:8]
    for f in F:
        e1, e2 = P[f[1]] - P[f[0]], P[f[2]] - P[f[0]]
        d1, d2 = UV[f[1]] - UV[f[0]], UV[f[2]] - UV[f[0]]
        det = d1[0] * d2[1] - d2[0] * d1[1]
        if abs(det) > 1e-12:
            T[f] += (e1 * d2[1] - e2 * d1[1]) / det
    N = V[:, 3:6]
    T -= N * np.sum(T * N, 1, keepdims=True)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    V[:, 8:11] = T
    return V


def unshared(V, F):
    """Each face its own three points (so each may take its own uv): points, faces."""
    return V[F.reshape(-1)].copy(), np.arange(F.size).reshape(-1, 3)


def mapped(part, V, F):
    """A light pole's part mapped in metres, its steel STEEL_M a texture: the mast round it (u round at its foot's
    radius, v up), the frame each face on its plane most facing it (as a box); each lamp's glass (a piece of the
    lens or glass) flat across its face, its middle the texture's middle, its rim the texture's circle."""
    V, F = unshared(V, F)
    if part == 'mast':
        for f in F:
            a = np.arctan2(V[f, 0], V[f, 2])
            a = np.where(a - a[0] > np.pi, a - 2 * np.pi, np.where(a[0] - a > np.pi, a + 2 * np.pi, a))
            V[f, 6] = a * MAST_R / MAST_M
            V[f, 7] = V[f, 1] / MAST_M
    elif part == 'frame':
        for f in F:
            n = np.abs(np.cross(V[f[1], :3] - V[f[0], :3], V[f[2], :3] - V[f[0], :3]))
            i, j = [(1, 2), (0, 2), (0, 1)][int(np.argmax(n))]
            V[f, 6], V[f, 7] = V[f, i] / STEEL_M, V[f, j] / STEEL_M
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        # (each lamp its own piece: points at one place joined)
        key = np.unique(np.round(V[:, :3], 4), axis=0, return_inverse=True)[1].ravel()
        e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]
        _, lab = connected_components(coo_matrix((np.ones(len(e)), (key[e[:, 0]], key[e[:, 1]])), shape=(key.max() + 1,) * 2), directed=False)
        piece = lab[key]
        for c in np.unique(piece):
            at = piece == c
            P = V[at, :3]
            mid = P.mean(0)
            _, _, W = np.linalg.svd(P - mid)
            a, b = W[0], W[1]                                    # (its face's plane: its two widest ways)
            up = np.array([0., 1., 0.])
            if abs(b @ up) < abs(a @ up):
                a, b = b, a                                      # (v the way most up)
            r = np.abs(np.c_[(P - mid) @ a, (P - mid) @ b]).max()
            V[at, 6] = .5 + (P - mid) @ a / (2 * r)
            V[at, 7] = .5 - (P - mid) @ b / (2 * r)
    return tangents(V, F), F


def dress():
    """The light pole's template mapped (mapped) and its lamps' glass texture made (lens_texture, into
    source/textures/pole_lens.dds): run again, the same."""
    import kn5io
    from PIL import Image
    save('light_pole', [(p,) + mapped(p, V, F) + (mat,) for p, V, F, mat in load_file('light_pole.npz')])
    img = Image.fromarray((lens_texture() * 255 + .5).astype(np.uint8), 'RGB')
    open(os.path.join(bp.ROOT, 'source', 'textures', 'pole_lens.dds'), 'wb').write(kn5io.dds(img, 'DXT1'))


def lens_texture():
    """The lamps' glass: a floodlight's front, a grid of small prisms over a bright reflector, the bulb's spot at
    its middle, a dark bezel at its rim (the texture's circle). Its light at night the Custom Shaders Patch's
    emissive of PoleLens (ext_config.ini) through it. RGB, LENS_PX a side."""
    n = LENS_PX
    y, x = np.mgrid[0:n, 0:n] / n
    r = np.hypot(x - .5, y - .5) / .5
    cell = 1. / 24
    fx, fy = (x % cell) / cell, (y % cell) / cell
    prism = .82 + .18 * (1 - .5 * (fx + fy)) - .1 * (np.minimum(fx, fy) < .06)          # (each lit from its corner, its edges dark)
    reflector = .62 + .38 * np.exp(-(r / .55) ** 2) + .25 * np.exp(-(r / .12) ** 2)       # (the bulb's spot, the reflector round it)
    v = np.clip(prism * reflector, 0, 1.15)
    v = np.where(r > .93, .12 + .08 * (r < .98), v)                                        # (the bezel)
    rgb = np.stack([v * .95, v * .97, v * 1.], -1)
    return np.clip(rgb, 0, 1)


def load_file(name):
    t = np.load(os.path.join(DIR, name))
    parts = sorted({k[:-len('_verts')] for k in t.files if k.endswith('_verts')})
    return [(p, t[p + '_verts'].astype(float), t[p + '_faces'].astype(int), str(t[p + '_material'])) for p in parts]


def gltf_source_world(n, m):
    """A mesh's points (n, 11) in the world: its node's matrix applied (points, normals, tangents)."""
    V = n['verts'].astype(float).copy()
    M = np.asarray(m, float).reshape(4, 4)                      # (row points: [p, 1] @ M, as build_paths.triangles)
    V[:, :3] = V[:, :3] @ M[:3, :3] + M[3, :3]
    V[:, 3:6] = V[:, 3:6] @ M[:3, :3]
    if V.shape[1] > 10:
        V[:, 8:11] = V[:, 8:11] @ M[:3, :3]
    return V


if __name__ == '__main__':
    if sys.argv[1:] == ['dress']:
        dress()
        print("light pole mapped, its lamps' glass made")
    elif sys.argv[1:2] == ['bake']:
        for name, m in bake(sys.argv[2:] or None).items():
            print('%-11s %-13s size %s, %d lights' % (name, m['label'], m['size'], len(m['lights'])))
    else:
        print(__doc__)
