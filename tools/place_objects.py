"""What stands on the map placed as tools/reference/objects.json says (the road editor's, top-race-editor):
each layout's grid slots, pit boxes, hotlap start, timing gates, cones and boards; the light poles, their
night lights with them; and the paint marks drawn on the roads. Nothing here is driven on: none of it goes
into the floor's merge or its checks, it stands on the floor the merge made.

    python tools/place_objects.py [--objects PATH]       (PATH: another objects file, to try one)

No objects.json (the editor writes it at its first save): nothing changed, the models' own as they are.

Version 3 of the file: the layouts' spots and gates, the marks, and its objects; nothing else. An object is a model
of the catalogue (tools/templates.py: tools/reference/templates/models.json) laid alone (one node: at it, turned
to its heading) or along its curve ('place': copies spaced evenly, no farther apart than its spacing, as a tyre
wall's stacks; 'bend': copies end to end, bent along it and stretched to fill it whole). Its collision: none; a
fixed box round each copy ('static'); a band along its curve ('wall'); or, alone, an AC physics object
('dynamic', AC_POBJECT_n: its hull, its model under it). Its layouts: none named, every layout's (in SHARED, the
dynamic ones in each layout's model); named, only theirs (in each one's model). The old models' tyre walls, poles,
cones and boards out; the light poles' lights (their model's: in its light series, the catalogue's 'lights')
written to places.json's `lights`, one each, as each pole stands. The old lists below (version 2) until then:

- A spot (grid slot, pit box, hotlap start: AC_START_n, AC_PIT_n, AC_HOTLAP_START_0 of vp_<layout>.glb)
  at its x, z, turned to its heading (radians from +z toward +x: the node's +z), as high over the floor
  as it was. More of them than the layout has: copies of its last; fewer: the rest out, and the layout's
  pit boxes (ui_track.json's pitboxes) as many as its pits.
- A timing gate's two ends (AC_TIME_n_L, _R) at theirs; a cone or a board (AC_POBJECT_n, an OBJ or a
  BOARD_ under it) at its x, z (a board turned to its heading), on the floor as it was.
- A pole (POLE_n_* of velopark_poles.glb, its parts as one: its mast, frame and lights; the old model's tyres
  round it out, a tyre wall of the file's in their place) moved and turned about its mast's middle from
  where it stands to its x, z and heading (from the mast to the lenses), as high over the floor as it
  was; its lights in the Custom Shaders Patch's config (any light series' POSITION_k within LIGHT_NEAR of
  its lenses) moved and turned with it, their DIRECTION_k too.
- A mark: a strip of paint WIDTH wide along its heading from p, its length, white (LinePaint) or yellow
  (PitLinePaint), solid or in dashes of dash with gaps of gap; LINE_LIFT over the road under it, and only
  where there is road under it (paint never on the grass); a mesh MARK_n of velopark_lines.glb.
- A start line (START_LINES: the layouts whose start the models have none of): squares CHECKER a side, white
  and the road's own by turns, CHECKER_ROWS of them behind its first timing gate (on its grid's side), across
  its track's road from CHECKER_IN in from one edge to as far from the other; a mesh START_<layout> of
  velopark_lines.glb, LINE_LIFT over the road.
- A tyre wall: one stack (tools/reference/templates/tyre_stack.npz: three tyres and their fill) every spacing
  metres along its curve (Bezier nodes as a road's; closed: back to its first), turned along it, rows to its
  right, each SINK into the floor; its meshes TYRES_<name> and TYRES_<name>F, and its collision (unless
  collision is false) a band either side, 1WALL_TYRE_<name>, not drawn. Every tyre wall of the model is
  laid so: those the file does not have are gone.
"""
import copy
import json
import math
import os
import re
import shutil
import sys

import numpy as np

import build_paths as bp
import gltf_source
import kn5tree
import roads as rd

OBJECTS = os.path.join(bp.ROOT, 'tools', 'reference', 'objects.json')
LINES = os.path.join(bp.MODELS, 'velopark_lines.glb')
POLES = os.path.join(bp.MODELS, 'velopark_poles.glb')
EXT = os.path.join(bp.ROOT, 'source', 'track', 'extension', 'ext_config.ini')
LIGHTS = []          # (the placed objects' lights, in the world: places.json's `lights`)
TRACK_DIR = bp.AC_MADE                # (the layouts' model lists the steps make: track/ac)
TERRAIN = ('velopark.glb', 'velopark_kerbs.glb', 'velopark_kids.glb', 'velopark_paddock.glb', 'velopark_runoffs.glb',
           'velopark_outside.glb')         # (and the floors laid in the buildings, the tent's: what stands in them on them)
STEP = .25           # metres between a mark's rows along it (it follows the road)
LIGHT_NEAR = 3.      # metres (x, z) from a pole's lenses its lights are
MOVED = 1e-4         # metres, radians: less than this, not moved
MERGE_MAX = 60000    # points of a merged mesh at most (a model's meshes index theirs in 16 bits)
MERGE_CELL = 20.     # metres a side of the squares a part drawn only so near or so far is merged in
SPOT_LIFT = .5       # metres over the floor every spot and gate stands in places.json (a game drops its car from there)
# a layout's start line painted, checkered, on its track's road: the layout, its road (not the pit lane's, which
# may lie over it there); the start line its first timing gate (AC_TIME_0)
START_LINES = {'vp_kids': '1ROAD_KIDS_0'}
CHECKER, CHECKER_ROWS, CHECKER_IN = .3, 2, .3   # metres a square, its rows behind the line, metres in from the road's edges


# ---- the floor everything stands on

_FLOOR = []


def floor(x, z):
    """The floor's top (the terrain's: the merge's meshes) at x, z (arrays); its nearest point's where none."""
    if not _FLOOR:
        from scipy.spatial import cKDTree
        T = []
        for f in TERRAIN:
            p = os.path.join(bp.MODELS, f)
            if not os.path.exists(p):
                continue
            for n, m in bp.meshes(gltf_source.load_kn5(p)):
                if len(n['faces']) and n['name'].upper().startswith(('1ROAD', '1KERB', '1GRASS', 'GRASS', 'KERB')):
                    T.append(bp.triangles(n, m))
        T = np.concatenate(T)
        V = T.reshape(-1, 3)
        _FLOOR.extend([bp.Surface(T), V, cKDTree(V[:, [0, 2]])])
    S, V, tree = _FLOOR
    q = np.c_[np.atleast_1d(x), np.atleast_1d(z)].astype(float)
    y = S.ys(q)
    gone = np.isnan(y)
    if gone.any():
        y[gone] = V[tree.query(q[gone])[1], 1]
    return y


def turned(heading):
    """A node's axes (rows) turned to heading: its +z along (sin, 0, cos)."""
    s, c = math.sin(heading), math.cos(heading)
    return np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])


def heading_of(m):
    return math.atan2(m[2, 0], m[2, 2])


# ---- the layouts' spots, gates, cones and boards

def dummies(k):
    """A model's nodes with a matrix of their own (kind 1), by name, with the node holding each."""
    out = {}

    def walk(n, parent):
        for c in n.get('children', []):
            if c['kind'] == 1:
                out[c['name']] = (c, n)
            walk(c, n)
    walk(k['root'], None)
    return out


def set_spot(node, x, z, heading=None):
    """A node set at x, z (as high over the floor as it was), turned to heading if given. Whether it moved."""
    M = np.array(node['matrix'], float).reshape(4, 4)
    ox, oy, oz = M[3, :3]
    if abs(ox - x) < MOVED and abs(oz - z) < MOVED and (heading is None or abs(math.remainder(heading_of(M) - heading, 2 * math.pi)) < MOVED):
        return False
    up = oy - float(floor(ox, oz)[0])
    M[3, 0], M[3, 2] = x, z
    M[3, 1] = float(floor(x, z)[0]) + up
    if heading is not None:
        M[:3, :3] = turned(heading)
    node['matrix'] = M.ravel().tolist()
    return True


def numbered(d, prefix):
    """The nodes named prefix + a number (n), in its order: [(n, node, holder)]."""
    out = []
    for name, (node, holder) in d.items():
        m = re.fullmatch(re.escape(prefix) + r'(\d+)', name)
        if m:
            out.append((int(m.group(1)), node, holder))
    return sorted(out, key=lambda t: t[0])


def renamed(node, old, new):
    """A copy of a node and all under it, every name with old in it as new."""
    c = copy.deepcopy(node)

    def walk(n):
        n['name'] = n['name'].replace(old, new)
        for ch in n.get('children', []):
            walk(ch)
    walk(c)
    return c


def as_many(d, prefix, want):
    """The layout's prefix_n nodes made as many as want: copies of its last, or the rest out. The nodes."""
    have = numbered(d, prefix)
    if not have:
        return []
    while len(have) < want:
        n, node, holder = have[-1]
        c = renamed(node, '%s%d' % (prefix, n), '%s%d' % (prefix, n + 1))
        holder['children'].append(c)
        have.append((n + 1, c, holder))
    for n, node, holder in have[want:]:
        holder['children'].remove(node)
    return [node for _, node, _ in have[:want]]


def pitboxes(layout, n):
    """The layout's pit boxes in its ui_track.json, n."""
    p = os.path.join(TRACK_DIR, layout, 'ui', 'ui_track.json')
    if not os.path.exists(p):
        return
    raw = open(p, 'rb').read()
    text = raw.decode('utf-8-sig')
    new = re.sub(r'("pitboxes"\s*:\s*")\d+(")', lambda m: '%s%d%s' % (m.group(1), n, m.group(2)), text)
    if new != text:
        open(p, 'w', encoding='utf-8', newline='').write(new)


MARKERS = re.compile(r'^AC_(START_\d+|PIT_\d+|HOTLAP_START_\d+|TIME_\d+_[LR])$')
PLACES = os.path.join(bp.MODELS, 'places.json')     # (version 3: the layouts' places, for each game's build to make its own)


def layouts(data, places=None):
    """Each layout's spots, gates, cones and boards set as the file says. How many moved. With places (a dict): its
    spots and gates written there, each where it stands (x, y, z, heading), and taken out of the layout's model
    (each game's build makes its own: AC's AC_START_0 and so on), with the number of its pit boxes."""
    moved = 0
    for layout, L in (data.get('layouts') or {}).items():
        path = os.path.join(bp.MODELS, layout + '.glb')
        if not os.path.exists(path):
            continue
        k = gltf_source.load_kn5(path)
        d = dummies(k)
        before = moved
        for prefix, spots in (('AC_START_', L.get('grid')), ('AC_PIT_', L.get('pits'))):
            if spots is None:
                continue
            nodes = as_many(d, prefix, len(spots))
            for node, s in zip(nodes, spots):
                moved += set_spot(node, s['x'], s['z'], s.get('heading'))
            if prefix == 'AC_PIT_' and places is None:
                pitboxes(layout, len(spots))
        h = L.get('hotlap')
        if h is not None and 'AC_HOTLAP_START_0' in d:
            moved += set_spot(d['AC_HOTLAP_START_0'][0], h['x'], h['z'], h.get('heading'))
        for i, g in enumerate(L.get('timing') or []):
            for side, key in (('L', 'left'), ('R', 'right')):
                name = 'AC_TIME_%d_%s' % (i, side)
                if name in d and g.get(key):
                    moved += set_spot(d[name][0], g[key]['x'], g[key]['z'])
        # (the placed objects: a cone (an OBJ under it) or a board (a BOARD_ under it), in their order)
        objs = numbered(d, 'AC_POBJECT_')
        cones = [node for _, node, _ in objs if any(c['name'].startswith('OBJ') for c in node.get('children', []))]
        boards = [node for _, node, _ in objs if any(c['name'].startswith('BOARD_') for c in node.get('children', []))]
        for node, c in zip(cones, L.get('cones') or []):
            moved += set_spot(node, c['x'], c['z'])
        by_board = {}
        for node in boards:
            for c in node.get('children', []):
                if c['name'].startswith('BOARD_'):
                    by_board[c['name']] = node
        for b in L.get('boards') or []:
            node = by_board.get(b.get('name'))
            if node is not None:
                moved += set_spot(node, b['x'], b['z'], b.get('heading'))
        if places is not None:
            # (the spots and gates as they now stand, in the model's order; out of it)
            here = places.setdefault(layout, dict(spots=[], pits=len(L.get('pits') or []), objects=[]))
            keep = []
            for c in k['root']['children']:
                if c['kind'] == 1 and MARKERS.match(c['name']):
                    M = np.array(c['matrix'], float).reshape(4, 4)
                    x, z = float(M[3, 0]), float(M[3, 2])
                    here['spots'].append(dict(name=c['name'][len('AC_'):], x=x, y=float(floor(x, z)[0]) + SPOT_LIFT, z=z,
                                              heading=float(heading_of(M))))
                elif not c['name'].startswith('AC_POBJECT_'):
                    keep.append(c)
            k['root']['children'] = keep
            bp.save(k, path)
        elif moved != before:
            bp.save(k, path)
    return moved


# ---- the poles, their lights with them

def pole_parts(k):
    """Each pole's meshes (POLE_n_*), by pole."""
    out = {}
    for n, m in bp.meshes(k):
        g = re.fullmatch(r'(POLE_[A-Z0-9]+?)_(MAST|FRAME|LENS|GLASS|TYRES)', n['name'])
        if g and len(n['faces']):
            out.setdefault(g.group(1), []).append((n, m))
    return out


def where_pole(parts):
    """A pole's mast's middle (x, z) and heading (from it to its lenses' middle)."""
    mast = next((n for n, m in parts if n['name'].endswith('_MAST')), None)
    lens = next((n for n, m in parts if n['name'].endswith('_LENS')), None)
    c = mast['verts'][:, [0, 2]].astype(float).mean(0)
    l_ = lens['verts'][:, [0, 2]].astype(float).mean(0) if lens is not None else c + (0., 1.)
    return c, math.atan2(l_[0] - c[0], l_[1] - c[1]), (lens['verts'][:, [0, 2]].astype(float).mean(0) if lens is not None else c)


def poles(data):
    """The poles placed as the file says, their lights with them. How many moved."""
    want = {p['name']: p for p in data.get('poles') or []}
    if not want or not os.path.exists(POLES):
        return 0
    k = gltf_source.load_kn5(POLES)
    lights = []
    moved = 0
    for name, parts in pole_parts(k).items():
        p = want.get(name)
        if p is None:
            continue
        c, h0, lens = where_pole(parts)
        dh = math.remainder(p.get('heading', h0) - h0, 2 * math.pi)
        dx, dz = p['x'] - c[0], p['z'] - c[1]
        if abs(dx) < MOVED and abs(dz) < MOVED and abs(dh) < MOVED:
            continue
        dy = float(floor(p['x'], p['z'])[0] - floor(c[0], c[1])[0])
        # (turned by dh about its mast's middle, then moved there: its points, normals and tangents)
        s, co = math.sin(dh), math.cos(dh)
        R = np.array([[co, -s], [s, co]])          # ((x, z) @ R: turned from +z toward +x by dh, as the walls' stacks)

        def place(xz, c=c, R=R, dx=dx, dz=dz):                    # (this pole's own: bound now, not the loop's last)
            return (xz - c) @ R + c + (dx, dz)
        for n, m in parts:
            V = n['verts'].astype(float)
            V[:, [0, 2]] = place(V[:, [0, 2]])
            V[:, 1] += dy
            V[:, [3, 5]] = V[:, [3, 5]] @ R
            if V.shape[1] > 10:
                V[:, [8, 10]] = V[:, [8, 10]] @ R
            n['verts'] = V.astype(n['verts'].dtype)
            bp.sphere(n)
        lights.append((lens, place, R, dy))
        moved += 1
    gone = no_pole_tyres(k)
    remade = poles_from_template(k, want)
    if moved or gone or remade:
        bp.save(k, POLES)
    if moved:
        lights_moved(lights)
    return moved


def poles_from_template(k, want):
    """Each pole's parts made again from the catalogue's light pole (tools/templates.py: its parts mapped in metres,
    its lamps' glass), at its place, turned to its heading, on the floor there. How many."""
    import templates
    tpl = {p: (V, F) for p, V, F, m in templates.load('light_pole')}
    made = 0
    for name, parts in pole_parts(k).items():
        p = want.get(name)
        if p is None:
            continue
        x, z = float(p['x']), float(p['z'])
        y = float(floor(x, z)[0])
        for n, m in parts:
            part = n['name'].split('_')[-1].lower()
            if part not in tpl or not np.allclose(m, np.eye(4)):
                continue
            V, F = tpl[part]
            W = np.zeros((len(V), n['verts'].shape[1]))
            w = min(W.shape[1], V.shape[1])
            W[:, :w] = templates.world(V, np.array([x, z]), y, float(p.get('heading', 0.)))[:, :w]
            n['verts'] = W.astype(n['verts'].dtype)
            n['faces'] = F.astype('<u2' if len(W) < 65536 else '<u4')
            bp.sphere(n)
        made += 1
    return made


def no_pole_tyres(k):
    """Each pole only its pole: its tyres (POLE_n_TYRES, the old model's) out. A pole's tyres are a tyre wall of
    the file's now (one stack round its mast, its collision as the file says). How many."""
    out = 0
    for n, m in bp.meshes(k):
        if n['name'].startswith('POLE_') and n['name'].endswith('_TYRES') and len(n['faces']):
            n['faces'] = n['faces'][:0]
            out += 1
    return out


def lights_moved(moves):
    """The Custom Shaders Patch's lights by the poles moved, each with its pole: POSITION_k within
    LIGHT_NEAR of a moved pole's lenses (as they were), and its DIRECTION_k turned the same."""
    if not os.path.exists(EXT):
        return
    raw = open(EXT, 'rb').read()
    crlf = b'\r\n' in raw
    lines = raw.decode('utf-8').replace('\r\n', '\n').split('\n')
    pos = re.compile(r'^(POSITION_(\d+)\s*=\s*)([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*$')
    dirn = re.compile(r'^(DIRECTION_(\d+)\s*=\s*)([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*$')
    section, turn = None, {}
    for i, line in enumerate(lines):
        if line.startswith('['):
            section, turn = line, {}
            continue
        m = pos.match(line)
        if m:
            x, y, z = map(float, m.group(3, 4, 5))
            for lens, place, R, dy in moves:
                if np.hypot(x - lens[0], z - lens[1]) < LIGHT_NEAR:
                    nx, nz = place(np.array([x, z]))
                    lines[i] = '%s%.3f, %.3f, %.3f' % (m.group(1), nx, y + dy, nz)
                    turn[m.group(2)] = R
                    break
            continue
        m = dirn.match(line)
        if m and m.group(2) in turn:
            x, y, z = map(float, m.group(3, 4, 5))
            nx, nz = np.array([x, z]) @ turn[m.group(2)]
            lines[i] = '%s%.3f, %.3f, %.3f' % (m.group(1), nx, y, nz)
    text = '\n'.join(lines)
    open(EXT, 'wb').write((text.replace('\n', '\r\n') if crlf else text).encode('utf-8'))


# ---- the marks: paint drawn on the roads

def marks(data):
    """The marks painted (see the module's note), each a mesh of the lines' model. How many."""
    want = data.get('marks') or []
    if not os.path.exists(LINES):
        return 0
    L = gltf_source.load_kn5(LINES)
    # (the marks laid before out: each build lays them all again)
    L['root']['children'] = [c for c in L['root']['children'] if not c['name'].startswith('MARK_')]
    if not want:
        bp.save(L, LINES)
        return 0
    import paint_roads as pr
    from shapely import contains_xy
    made = 0
    for mk in want:
        tpl = rd.paint_mesh(L, mk.get('colour'))
        p = np.array(mk['p'], float)
        h, length, w = float(mk['heading']), float(mk['length']), float(mk.get('width', rd.LINE_W))
        t = np.array([math.sin(h), math.cos(h)])
        side = np.array([t[1], -t[0]])
        if mk.get('style') == 'dotted':
            dash, gap = float(mk.get('dash', .5)), float(mk.get('gap', .5))
            runs = [(a, min(a + dash, length)) for a in np.arange(0., length, dash + gap)]
        else:
            runs = [(0., length)]
        R = pr.road_union((p[0] - length - 2, p[1] - length - 2, p[0] + length + 2, p[1] + length + 2))
        P, F = [], []
        for a, b in runs:
            S = np.r_[np.arange(a, b, STEP), b]
            mid = p + np.outer(S, t)
            on = contains_xy(R, mid[:, 0], mid[:, 1])          # (only where there is road under it)
            for j in range(len(S)):
                for sgn in (-.5, .5):
                    q = mid[j] + side * w * sgn
                    P.append((q[0], 0., q[1]))
            base = len(P) - 2 * len(S)
            for j in range(len(S) - 1):
                if on[j] and on[j + 1]:
                    i0 = base + 2 * j
                    F += [(i0, i0 + 2, i0 + 1), (i0 + 1, i0 + 2, i0 + 3)]
        if not F:
            continue
        P = np.array(P)
        P[:, 1] = floor(P[:, 0], P[:, 2]) + rd.LINE_LIFT
        F = np.array(F)
        up = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 1]
        F = np.where((up < 0)[:, None], F[:, ::-1], F)
        V = np.zeros((len(P), tpl['verts'].shape[1]))
        V[:, :3] = P
        V[:, 4] = 1.
        V[:, 6], V[:, 7] = P[:, 0] / 10., P[:, 2] / 20.
        if V.shape[1] > 8:
            V[:, 8] = 1.
        n = dict(tpl, name=mk.get('name', 'MARK_%d' % (made + 1)), children=[], verts=V.astype(tpl['verts'].dtype),
                 faces=F.astype('<u2' if len(V) < 65536 else '<u4'))
        bp.sphere(n)
        L['root']['children'].append(n)
        made += 1
    bp.save(L, LINES)
    return made


def start_lines(data):
    """The start lines painted (see the module's note). Which layouts'."""
    from shapely.geometry import LineString, Polygon
    if not os.path.exists(LINES):
        return []
    L = gltf_source.load_kn5(LINES)
    L['root']['children'] = [c for c in L['root']['children'] if not c['name'].startswith('START_')]
    roads = rd.load()
    made = []
    for layout, road in START_LINES.items():
        lay = (data.get('layouts') or {}).get(layout) or {}
        gate = next((t for t in lay.get('timing') or [] if t.get('name') == 'AC_TIME_0'), None)
        if gate is None or road not in roads:
            continue
        a = np.array([gate['left']['x'], gate['left']['z']])
        b = np.array([gate['right']['x'], gate['right']['z']])
        u = (b - a) / np.linalg.norm(b - a)
        back = np.array([-u[1], u[0]])
        grid = lay.get('grid') or []
        if grid and np.dot(np.array([grid[0]['x'], grid[0]['z']]) - (a + b) / 2, back) < 0:
            back = -back                                            # (behind the line: the grid's side)
        own = rd.outline(roads[road], roads)
        across = LineString([a - u * 20, b + u * 20]).intersection(own)
        if across.is_empty:
            continue
        seg = max(getattr(across, 'geoms', [across]), key=lambda g: g.length)
        s = sorted(float(np.dot(np.array(c) - a, u)) for c in seg.coords)
        s0, s1 = s[0] + CHECKER_IN, s[-1] - CHECKER_IN
        n = int((s1 - s0) // CHECKER)
        s0 += ((s1 - s0) - n * CHECKER) / 2                         # (whole squares, as far in from both edges)
        P, F = [], []
        for i in range(n):
            for r_ in range(CHECKER_ROWS):
                if (i + r_) % 2:
                    continue
                c0 = a + u * (s0 + i * CHECKER) + back * (r_ * CHECKER)
                q = [c0, c0 + u * CHECKER, c0 + u * CHECKER + back * CHECKER, c0 + back * CHECKER]
                if not own.contains(Polygon(q)):
                    continue
                k0 = len(P)
                P += [(x, 0., z) for x, z in q]
                F += [(k0, k0 + 1, k0 + 2), (k0, k0 + 2, k0 + 3)]
        if not F:
            continue
        P, F = np.array(P), np.array(F)
        P[:, 1] = floor(P[:, 0], P[:, 2]) + rd.LINE_LIFT
        up = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 1]
        F = np.where((up < 0)[:, None], F[:, ::-1], F)
        tpl = rd.paint_mesh(L, 'white')
        V = np.zeros((len(P), tpl['verts'].shape[1]))
        V[:, :3] = P
        V[:, 4] = 1.
        V[:, 6], V[:, 7] = P[:, 0] / 10., P[:, 2] / 20.
        if V.shape[1] > 8:
            V[:, 8] = 1.
        m = dict(tpl, name='START_' + layout.upper(), verts=V.astype(tpl['verts'].dtype), faces=F.astype('<u2'))
        bp.sphere(m)
        L['root']['children'].append(m)
        made.append(layout)
    bp.save(L, LINES)
    return made


# ---- the tyre walls: one stack (TEMPLATE) laid along each wall's curve

TYRES = os.path.join(bp.MODELS, 'velopark_tyres.glb')
TEMPLATE = os.path.join(bp.ROOT, 'tools', 'reference', 'templates', 'tyre_stack.npz')
SINK = .03           # metres a stack stands into the floor (as tools/drape_track.py has them)
WALL_HIGH, WALL_DOWN = .7, .05   # metres: the collision wall's top over the floor, its foot under it
STACK_HALF = .31     # metres: a stack's half width (the collision wall that far out from its middle)


def wall_stacks(w):
    """A wall's stacks: [(x, z, heading)], along its curve evenly, as many as make them no farther apart than
    spacing (so every gap the same, none wider, the last too): an open one from its first node to its last, a
    closed one round to its first again (its last gap back to its first stack as the others); rows to its right
    (roads.py's: (tz, -tx) of the way it goes) each a spacing further out."""
    nodes = [dict(n, left=0., right=0.) for n in w.get('curve') or []]
    if not nodes:
        return []
    if len(nodes) == 1:
        p = nodes[0]['p']
        return [(float(p[0]), float(p[1]), float(w.get('heading', 0.)))]
    if w.get('closed'):
        nodes = nodes + [dict(nodes[0])]
    C, _ = rd.bezier(nodes, .02)
    P = C[:, :2]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    sp = max(float(w.get('spacing', .62)), .05)
    n = max(int(np.ceil(s[-1] / sp - 1e-9)), 1)                     # (gaps: none wider than sp)
    S = np.arange(n) * (s[-1] / n) if w.get('closed') else np.linspace(0., s[-1], n + 1)
    X, Z = np.interp(S, s, P[:, 0]), np.interp(S, s, P[:, 1])
    T = np.gradient(P, axis=0)
    tx, tz = np.interp(S, s, T[:, 0]), np.interp(S, s, T[:, 1])
    nrm = np.hypot(tx, tz) + 1e-12
    tx, tz = tx / nrm, tz / nrm
    out = []
    for r in range(max(int(w.get('rows', 1)), 1)):
        off = r * sp
        for x, z, a, b in zip(X, Z, tx, tz):
            out.append((x + b * off, z - a * off, math.atan2(a, b)))
    return out


def wall_band(w, stacks, half=STACK_HALF, high=WALL_HIGH):
    """A wall's collision: a band either side of its stacks (half out from the outermost rows), from
    WALL_DOWN under the floor to high over it: points, faces."""
    rows = max(int(w.get('rows', 1)), 1)
    first = stacks[:len(stacks) // rows] if rows > 1 else stacks
    if len(first) < 2:
        x, z, h = first[0] if first else (0., 0., 0.)
        line = np.array([[x - math.sin(h) * half, z - math.cos(h) * half],
                         [x + math.sin(h) * half, z + math.cos(h) * half]])
    else:
        line = np.array([(x, z) for x, z, h in first])
    if w.get('closed') and len(line) > 2:
        line = np.r_[line, line[:1]]
    t = np.gradient(line, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    right = np.c_[t[:, 1], -t[:, 0]]
    sp = float(w.get('spacing', .62))
    P, F = [], []
    for off in (-half, (rows - 1) * sp + half):
        Q = line + right * off
        g = floor(Q[:, 0], Q[:, 1])
        base = len(P)
        for (x, z), y in zip(Q, g):
            P += [(x, y - WALL_DOWN, z), (x, y + high, z)]
        for j in range(len(Q) - 1):
            a = base + 2 * j
            # (both faces: hit from either side)
            F += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3), (a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
    return np.array(P, float), np.array(F, int)


def walls(data):
    """Every tyre wall laid from the file (see the module's note), in place of the model's: (walls, stacks)."""
    want = data.get('tyre_walls')
    if not want or not os.path.exists(TYRES) or not os.path.exists(TEMPLATE):
        return 0, 0
    tp = np.load(TEMPLATE)
    k = gltf_source.load_kn5(TYRES)
    nodes = [n for n, m in bp.meshes(k)]
    tpl_t = next((n for n in nodes if n['name'].startswith('TYRES_') and not n['name'].endswith('F') and len(n['faces'])), None)
    tpl_f = next((n for n in nodes if n['name'].startswith('TYRES_') and n['name'].endswith('F') and len(n['faces'])), tpl_t)
    tpl_w = next((n for n in nodes if n['name'].startswith('1WALL_TYRE') and len(n['faces'])), None)
    if tpl_t is None:
        return 0, 0

    def prune(n):
        n['children'] = [c for c in n.get('children', [])
                         if not (c.get('kind') == 2 and c['name'].startswith(('TYRES_', '1WALL_TYRE')))]
        for c in n['children']:
            prune(c)
    prune(k['root'])
    mats = [m_['name'] for m_ in k['materials']]
    made = 0
    for w in want:
        stacks = wall_stacks(w)
        if not stacks:
            continue
        g = floor(np.array([x for x, z, h in stacks]), np.array([z for x, z, h in stacks]))
        for part, tpl, suffix in (('tyres', tpl_t, ''), ('fill', tpl_f, 'F')):
            V0, F0 = tp[part + '_verts'].astype(float), tp[part + '_faces'].astype(int)
            Vs, Fs, base = [], [], 0
            for (x, z, h), y in zip(stacks, g):
                s, c = math.sin(h), math.cos(h)
                R = np.array([[c, -s], [s, c]])            # (x, z) turned to heading (its +z to (sin, cos))
                V = V0.copy()
                V[:, [0, 2]] = V0[:, [0, 2]] @ R + (x, z)
                V[:, 1] = V0[:, 1] + y - SINK
                V[:, [3, 5]] = V0[:, [3, 5]] @ R
                if V.shape[1] > 10:
                    V[:, [8, 10]] = V0[:, [8, 10]] @ R
                Fs.append(F0 + base)
                base += len(V)
                Vs.append(V)
            V, F = np.concatenate(Vs), np.concatenate(Fs)
            mat = str(tp[part + '_material'])
            n = dict(tpl, name='%s%s' % (w['name'], suffix), children=[], material=mats.index(mat) if mat in mats else tpl['material'],
                     verts=V.astype(tpl['verts'].dtype), faces=F.astype('<u2' if len(V) < 65536 else '<u4'))
            bp.sphere(n)
            k['root']['children'].append(n)
        if w.get('collision', True) and tpl_w is not None:
            P, F = wall_band(w, stacks)
            if len(F):
                V = np.zeros((len(P), tpl_w['verts'].shape[1]))
                V[:, :3] = P
                V[:, 4] = 1.
                name = '1WALL_TYRE_' + re.sub(r'^TYRES_', '', w['name'])
                n = dict(tpl_w, name=name, children=[], verts=V.astype(tpl_w['verts'].dtype),
                         faces=F.astype('<u2' if len(V) < 65536 else '<u4'))
                bp.sphere(n)
                k['root']['children'].append(n)
        made += len(stacks)
    bp.save(k, TYRES)
    return len(want), made


# ---- the objects (objects.json version 3): each a model of the catalogue (tools/templates.py), alone or along a curve

SHARED = os.path.join(bp.MODELS, 'velopark_objects.glb')   # (the objects every layout has, but those knocked about)
SINKS = {'tyre_stack': SINK}                                # metres a model stands into the floor (the others: none)
COLLISION_OVER = .08                                        # metres a wall's or a box's collision stands over its model


def layout_names():
    return sorted(f[:-4] for f in os.listdir(bp.MODELS) if f.startswith('vp_') and f.endswith('.glb'))


def object_places(o, size):
    """Where an object's copies stand ('place'): [(x, z, heading)], as a tyre wall's stacks (a single one: at its
    node, turned to its heading); spaced no farther apart than its spacing (its model's depth unless it says)."""
    return wall_stacks(dict(o, spacing=float(o.get('spacing') or size[2])))


def object_bent(o, parts, size):
    """An object's model bent along its curve ('bend'): as many copies end to end as fit it whole, each stretched
    to fill its share, its points across it out to its right, on the floor there. [(part, points (n, 11), faces,
    material)] in the world."""
    nodes = [dict(n, left=0., right=0.) for n in o['curve']]
    if o.get('closed'):
        nodes = nodes + [dict(nodes[0])]
    C, _ = rd.bezier(nodes, .02)
    Pc = C[:, :2]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
    T = np.gradient(Pc, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    vis = [(p, V, F, m) for p, V, F, m in parts if p != 'hull']
    z0 = min(V[:, 2].min() for p, V, F, m in vis)
    Lm = max(max(V[:, 2].max() for p, V, F, m in vis) - z0, 1e-6)
    n = max(int(round(s[-1] / Lm)), 1)
    k_ = s[-1] / (n * Lm)
    sink = SINKS.get(o['model'], 0.)
    out = []
    for p, V0, F0, mat in vis:
        Vs, Fs, base = [], [], 0
        for r in range(max(int(o.get('rows', 1)), 1)):
            for i in range(n):
                at = i * s[-1] / n + (V0[:, 2] - z0) * k_
                fx, fz = np.interp(at, s, T[:, 0]), np.interp(at, s, T[:, 1])
                rx, rz = fz, -fx                                    # (its right: (cos h, -sin h) of forward (sin h, cos h))
                off = V0[:, 0] + r * float(o.get('spacing') or size[0])
                V = V0.copy()
                V[:, 0] = np.interp(at, s, Pc[:, 0]) + rx * off
                V[:, 2] = np.interp(at, s, Pc[:, 1]) + rz * off
                V[:, 1] = floor(V[:, 0], V[:, 2]) + V0[:, 1] - sink
                for a, b in ((3, 5), (8, 10)) if V.shape[1] > 10 else ((3, 5),):
                    V[:, a], V[:, b] = rx * V0[:, a] + fx * V0[:, b], rz * V0[:, a] + fz * V0[:, b]
                Vs.append(V)
                Fs.append(F0 + base)
                base += len(V)
        out.append((p, np.concatenate(Vs), np.concatenate(Fs), mat))
    return out


def shaped(o, meta, parts, size):
    """An object's model as it lays it: scaled (its 'scale': across, up, deep; its normals and tangents kept square to
    it) and, bent along a curve, turned so its length lies along it (a model longer across than deep, as a fence
    panel, along its width). Its parts and its size."""
    s = np.array(o.get('scale') or [1., 1., 1.], float)
    turn = o.get('fit') == 'bend' and len(o.get('curve') or []) > 1 and size[0] > size[2]
    out = []
    for p, V, F, mat in parts:
        V = V.copy()
        V[:, :3] *= s
        V[:, 3:6] /= s
        if V.shape[1] > 10:
            V[:, 8:11] *= s
        if turn:                                                    # (x along: its x to z, its z to -x)
            for a, b in ((0, 2), (3, 5), (8, 10)) if V.shape[1] > 10 else ((0, 2), (3, 5)):
                V[:, a], V[:, b] = -V[:, b].copy(), V[:, a].copy()
        for a in (3, 8) if V.shape[1] > 10 else (3,):
            V[:, a:a + 3] /= np.maximum(np.linalg.norm(V[:, a:a + 3], axis=1, keepdims=True), 1e-12)
        out.append((p, V, F, mat))
    size = [float(v) for v in np.array(size, float) * s]
    if turn:
        size = [size[2], size[1], size[0]]
    return out, size


BASE = .1            # metres up from a model's lowest point its base reaches (what it stands on: wheels, legs, a mast's foot)
EDGE_IN = .9         # of the way out from its base's middle to its corners, where it is taken to touch


def feet(parts):
    """How many pieces a model's base is in: its faces with every corner within BASE of its lowest point, those
    sharing a corner (where it is, to the millimetre) one piece."""
    Vs = [V[:, :3] for p, V, F, m in parts if p != 'hull'] or [V[:, :3] for p, V, F, m in parts]
    low = min(V[:, 1].min() for V in Vs) + BASE
    parent = {}

    def root(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for p, V, F, m in parts:
        if p == 'hull' and any(q != 'hull' for q, *_ in parts):
            continue
        key = [tuple(k) for k in np.round(V[:, :3] * 1000).astype(np.int64).tolist()]
        for f in np.asarray(F).reshape(-1, 3):
            if all(V[i, 1] < low for i in f):
                a = root(key[f[0]])
                for i in f[1:]:
                    b = root(key[i])
                    if a != b:
                        parent[b] = a
    return len({root(a) for a in list(parent)})


def rest_heights(parts, places):
    """Where each copy of a model rests on the floor: a model on feet (its base in three pieces or more: wheels,
    legs) on the highest of the floor under its base's middle and corners, as such a thing rests on the highest it
    touches, no wheel sunk into a step; one on a base of a piece (a tyre stack's, a cone's, a mast's foot) on the
    floor under its middle, as it beds down on it. (places: [(x, z, heading)])"""
    import templates
    P = np.concatenate([V[:, :3] for p, V, F, m in parts if p != 'hull'] or [V[:, :3] for p, V, F, m in parts])
    P = P[P[:, 1] < P[:, 1].min() + BASE]
    if feet(parts) < 3:
        return floor(np.array([x for x, z, h in places]), np.array([z for x, z, h in places]))
    lo, hi = P.min(0), P.max(0)
    mid = (lo + hi) / 2
    Q = np.zeros((5, 11))
    Q[:, 0] = [mid[0]] + [mid[0] + (c - mid[0]) * EDGE_IN for c in (lo[0], lo[0], hi[0], hi[0])]
    Q[:, 2] = [mid[2]] + [mid[2] + (c - mid[2]) * EDGE_IN for c in (lo[2], hi[2], lo[2], hi[2])]
    W = np.concatenate([templates.world(Q, np.array([x, z]), 0., h) for x, z, h in places])
    return floor(W[:, 0], W[:, 2]).reshape(len(places), 5).max(1)


def object_placed(o, parts, places):
    """An object's model at each of its places ('place'): [(part, points (n, 11), faces, material)] in the world,
    each copy resting on the floor (rest_heights; its hull out: a dynamic one's alone)."""
    import templates
    sink = SINKS.get(o['model'], 0.)
    ys = rest_heights(parts, places)
    out = []
    for p, V0, F0, mat in parts:
        if p == 'hull':
            continue
        Vs = [templates.world(V0, np.array([x, z]), y - sink, h) for (x, z, h), y in zip(places, ys)]
        out.append((p, np.concatenate(Vs), np.concatenate([F0 + i * len(V0) for i in range(len(Vs))]), mat))
    return out


_MATERIAL_SOURCES = {}
_MODEL = None         # (the model being laid: an editor's model brings its own materials where the track has none)


def material_of(k, name):
    """The index of material name in model k: brought in, with its textures, from the models that have it if k has
    none of it; else from the editor's model being laid, its textures written into source/textures (named after it)."""
    names = [m['name'] for m in k['materials']]
    if name in names:
        return names.index(name)
    import glob
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        if f not in _MATERIAL_SOURCES:
            _MATERIAL_SOURCES[f] = gltf_source.load_kn5(f)
        src = _MATERIAL_SOURCES[f]
        m = next((m for m in src['materials'] if m['name'] == name), None)
        held = {t['name'] for t in src['textures']}
        # (from a model that holds its textures: one may name a library material whose textures another holds)
        if m is None or any(t['name'] not in held for t in m.get('textures', [])):
            continue
        have = {t['name'] for t in k['textures']}
        for t in m.get('textures', []):
            if t['name'] not in have:
                k['textures'].append(copy.deepcopy(next(x for x in src['textures'] if x['name'] == t['name'])))
                have.add(t['name'])
        k['materials'].append(copy.deepcopy(m))
        return len(k['materials']) - 1
    import templates
    if _MODEL and 'editor' in templates.catalogue().get(_MODEL, {}):
        src = templates.editor_model(_MODEL)
        m = next((m for m in src['materials'] if m['name'] == name), None)
        if m is not None:
            m = copy.deepcopy(m)
            have = {t['name'] for t in k['textures']}
            for t in m.get('textures', []):
                data = next(x['data'] for x in src['textures'] if x['name'] == t['name'])
                ext = '.png' if data[1:4] == b'PNG' else '.jpg' if data[:3] == bytes((255, 216, 255)) else '.dds'
                new = '%s_%s%s' % (_MODEL, t['name'], ext)
                if new not in have:
                    path = os.path.join(bp.ROOT, 'source', 'textures', new)
                    if not os.path.exists(path) or open(path, 'rb').read() != data:
                        open(path, 'wb').write(data)
                    k['textures'].append(dict(name=new, active=1, data=data))
                    have.add(new)
                t['name'] = new
            k['materials'].append(m)
            return len(k['materials']) - 1
    raise SystemExit('no model has the material %s' % name)


def mesh(k, name, V, F, mat, renderable=True, lod=(0., 0.)):
    """A mesh node for model k: V (n, 11) points, F faces, in material mat (its name); seen from lod[0] to lod[1]
    metres away (0, 0: always)."""
    V = np.asarray(V, float)
    W = np.zeros((len(V), 11))
    W[:, :min(V.shape[1], 11)] = V[:, :11]
    n = dict(kind=2, name=name, active=1, children=[], verts=W.astype(np.float32),
             faces=np.asarray(F).reshape(-1, 3).astype('<u2' if len(W) < 65536 else '<u4'),
             material=material_of(k, mat), cast_shadows=int(renderable), visible=1, transparent=0, layer=0,
             lod_in=float(lod[0]), lod_out=float(lod[1]), renderable=int(renderable), sphere=[0., 0., 0., 0.])
    bp.sphere(n)
    return n


def part_name(name, part, parts):
    """A mesh's name: its object's, and its part's after it where its model has more than one (POLE_3_MAST)."""
    return name if parts == 1 else '%s_%s' % (name, part.upper())


def box(lo, hi):
    """A box's points (8, 3) and faces (12, 3), facing out."""
    P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    F = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                  (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return P, F


def objects(data, placed=None):
    """Every object of the file laid (see the module's note), in place of the old models' tyre walls, poles, cones and
    boards: the shared ones in SHARED, the others in their layouts' models, a dynamic one as each layout's physics
    object; the light poles' lights written. How many, by model."""
    import templates
    cat = templates.catalogue()
    want = data.get('objects') or []
    layouts_all = layout_names()
    # (the old models' out: the tyre walls, the poles, the layouts' cones and boards)
    k = gltf_source.load_kn5(TYRES)
    for n, m in bp.meshes(k):
        if n['name'].startswith(('TYRES_', '1WALL_TYRE')):
            n['faces'] = n['faces'][:0]
    bp.save(k, TYRES)
    k = gltf_source.load_kn5(POLES)
    for n, m in bp.meshes(k):
        if n['name'].startswith(('POLE_', '1WALL_POLE')):
            n['faces'] = n['faces'][:0]
    bp.save(k, POLES)
    files = {}
    for lay in layouts_all:
        k = gltf_source.load_kn5(os.path.join(bp.MODELS, lay + '.glb'))
        k['root']['children'] = [c for c in k['root']['children'] if not c['name'].startswith('AC_POBJECT_')]
        files[lay] = k
    if not os.path.exists(SHARED):
        shutil.copyfile(TYRES, SHARED)                           # (its materials and textures to start from)
    shared = gltf_source.load_kn5(SHARED)
    shared['root']['children'] = []
    files[None] = shared
    count, pobj, lit, solid = {}, {lay: 0 for lay in layouts_all}, [], []
    merged = {}           # (file, model, part, material, lod): [(points, faces)], the copies of what has nothing to hit
    for o in want:
        meta = cat.get(o.get('model'))
        if meta is None or not o.get('curve'):
            continue
        parts, size = shaped(o, meta, templates.load(o['model']), meta['size'])
        lods = meta.get('lods') or {}                               # (a part seen only so near, or so far: its stand-in)
        draw = meta.get('draw')                                     # (and the whole model's, for the parts lods does not name)
        whole = (float(draw[0] or 0.), float(draw[1] or 0.)) if draw else (0., 0.)
        global _MODEL
        _MODEL = o['model']
        single = len(o['curve']) == 1
        coll = o.get('collision') or 'none'
        lays = [lay for lay in (o.get('layouts') or layouts_all) if lay in files]
        targets = [None] if o.get('layouts') is None and coll != 'dynamic' else lays
        name = o['name']
        if single:
            p = o['curve'][0]['p']
            places = [(float(p[0]), float(p[1]), float(o.get('heading', 0.)))]
        else:
            places = object_places(o, size)
        if coll == 'dynamic' and single and placed is not None:
            # (a layout's physics object: in its places, each game's build makes its own, AC's AC_POBJECT_n)
            x, z, h = places[0]
            y = float(floor(x, z)[0]) - SINKS.get(o['model'], 0.)
            for lay in targets:
                placed.setdefault(lay, dict(spots=[], pits=0, objects=[]))['objects'].append(
                    dict(name=name, model=o['model'], x=x, y=y, z=z, heading=h))
        elif coll == 'dynamic' and single:
            x, z, h = places[0]
            y = float(floor(x, z)[0]) - SINKS.get(o['model'], 0.)
            hull = next(((V, F, m) for pp, V, F, m in parts if pp == 'hull'), None)
            vis = [(pp, V, F, m) for pp, V, F, m in parts if pp != 'hull']
            for lay in targets:
                k = files[lay]
                M = np.eye(4)
                M[:3, :3] = turned(h)
                M[3, :3] = x, y, z
                top = dict(kind=1, name='AC_POBJECT_%d' % pobj[lay], active=1, matrix=[float(v) for v in M.ravel()], children=[])
                if hull is not None:
                    top['children'].append(mesh(k, top['name'], hull[0], hull[1], hull[2], renderable=False))
                node = dict(kind=1, name=name, active=1, matrix=[float(v) for v in np.eye(4).ravel()], children=[])
                for pp, V, F, mat in vis:
                    node['children'].append(mesh(k, part_name(name, pp, len(vis)), V, F, mat, lod=lods.get(pp, whole)))
                top['children'].append(node)
                k['root']['children'].append(top)
                pobj[lay] += 1
        else:
            laid = object_bent(o, parts, size) if o.get('fit') == 'bend' and not single else object_placed(o, parts, places)
            for lay in targets:
                k = files[lay]
                for pp, V, F, mat in laid:
                    if coll == 'none':
                        # (nothing to hit: one mesh of all the copies of this part of the model in this file, drawn
                        # at once, as many objects as there are; those drawn only so near or so far, of each MERGE_CELL
                        # square, as a game measures that from the mesh's middle)
                        lod = tuple(lods.get(pp, whole))
                        cell = (0, 0) if lod == (0., 0.) else tuple(int(v // MERGE_CELL) for v in places[0][:2])
                        merged.setdefault((lay, o['model'], pp, mat, lod, cell), []).append((V, F))
                    else:
                        k['root']['children'].append(mesh(k, part_name(name, pp, len(laid)), V, F, mat, lod=lods.get(pp, whole)))
                if coll == 'wall' and not single:
                    P, F = wall_band(o, places, half=size[0] / 2, high=size[1] + COLLISION_OVER)
                    if lay == targets[0]:
                        import shapely
                        solid.append(shapely.LineString([(x, z) for x, z, h in places]).buffer(size[0] / 2))
                    W = np.zeros((len(P), 11))
                    W[:, :3] = P
                    W[:, 4] = 1.
                    k['root']['children'].append(mesh(k, '1WALL_' + name, W, F, laid[0][3], renderable=False))
                elif coll in ('static', 'wall'):
                    hull = next((V for pp, V, F, m in parts if pp == 'hull'), None)
                    P0 = (hull if hull is not None else np.concatenate([V for pp, V, F, m in parts]))[:, :3]
                    lo, hi = P0.min(0), P0.max(0)
                    hi[1] += COLLISION_OVER
                    ys = rest_heights(parts, places)
                    Bs, BF = [], []
                    for j, ((x, z, h), y) in enumerate(zip(places, ys)):
                        B, F = box(lo, hi)
                        W = np.zeros((8, 11))
                        W[:, :3] = B
                        W[:, 4] = 1.
                        Bs.append(templates.world(W, np.array([x, z]), y, h))
                        BF.append(F + 8 * j)
                        if lay == targets[0]:
                            import shapely
                            solid.append(shapely.Polygon(Bs[-1][[0, 1, 5, 4], :][:, [0, 2]]).convex_hull)
                    k['root']['children'].append(mesh(k, '1WALL_' + name, np.concatenate(Bs), np.concatenate(BF), laid[0][3], renderable=False))
        if meta.get('lights'):
            lit += [(meta['lights'], q) for q in places]
        count[o['model']] = count.get(o['model'], 0) + 1
    for (lay, model, pp, mat, lod, cell), copies in sorted(merged.items(), key=lambda x: str(x[0])):
        k = files[lay]
        runs, run, n = [], [], 0
        for V, F in copies:                                         # (in meshes of fewer than 65536 points)
            if run and n + len(V) > MERGE_MAX:
                runs.append(run)
                run, n = [], 0
            run.append((V, F))
            n += len(V)
        runs.append(run)
        for i, run in enumerate(runs):
            off = np.cumsum([0] + [len(V) for V, F in run[:-1]])
            V = np.concatenate([V for V, F in run])
            F = np.concatenate([F + o_ for (V_, F), o_ in zip(run, off)])
            name = '%s_%s' % (model.upper(), pp.upper()) + ('' if lod == (0., 0.) else '_%d_%d' % cell) + ('_%d' % i if i else '')
            k['root']['children'].append(mesh(k, name, V, F, mat, lod=lod))
    for lay, k in files.items():
        bp.save(k, SHARED if lay is None else os.path.join(bp.MODELS, lay + '.glb'))
    object_lights(lit)
    cleared = clear_trees(solid)
    if cleared:
        count['trees cleared'] = cleared
    return count


TREES = os.path.join(bp.MODELS, 'velopark_trees.glb')
TREE_CLEAR = .5      # metres round a solid object (a static box, a wall) no tree stands
TREE_REACH = 6.      # metres from its trunk a piece of a tree (a branch, a card of leaves) is its


def clear_trees(areas):
    """The trees (velopark_trees.glb, its list velopark_trees.json) whose trunks stand in areas (the solid objects'
    ground, TREE_CLEAR round it) out: every piece of the trees' meshes nearest such a trunk, within TREE_REACH of it.
    How many trees."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    listed = os.path.splitext(TREES)[0] + '.json'
    if not areas or not os.path.exists(TREES) or not os.path.exists(listed):
        return 0
    data = json.load(open(listed, encoding='utf-8'))
    at = np.array([[t[1], t[3]] for t in data['trees']], float)
    zone = shapely.union_all(areas).buffer(TREE_CLEAR)
    gone = shapely.contains_xy(zone, at[:, 0], at[:, 1])
    if not gone.any():
        return 0
    nearest = cKDTree(at)
    k = gltf_source.load_kn5(TREES)
    for n, m in bp.meshes(k):
        if not len(n['faces']):
            continue
        F = n['faces'].astype(int).reshape(-1, 3)
        V = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
        e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]
        _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
        cnt = np.bincount(lab)
        mid = np.c_[np.bincount(lab, V[:, 0]) / cnt, np.bincount(lab, V[:, 2]) / cnt]
        d, i = nearest.query(mid)
        drop = gone[i] & (d < TREE_REACH)
        keep = ~drop[lab[F[:, 0]]]
        if not keep.all():
            n['faces'] = F[keep].astype(n['faces'].dtype)
            bp.sphere(n)
    bp.save(k, TREES)
    data['trees'] = [t for t, g in zip(data['trees'], gone) if not g]
    json.dump(data, open(listed, 'w', encoding='utf-8'))
    return int(gone.sum())


def object_lights(lit):
    """The lights of the objects that have them (the light poles'), in the world (TRACK_FORMAT.md, "Lights and
    screens"): each of a model's lights (its series, where it is and where it faces, in the model's own frame), its
    object turned to its heading, on the floor there. They go to places.json's `lights`."""
    out = []
    for lights, (x, z, h) in lit:
        y = float(floor(x, z)[0])
        s, c = math.sin(h), math.cos(h)
        right, fwd = np.array([c, -s]), np.array([s, c])
        for L in lights:
            a, f = L['at'], L['facing']
            p = np.array([x, z]) + right * a[0] + fwd * a[2]
            d = right * f[0] + fwd * f[2]
            out.append(dict(series=L['series'], at=[round(float(v), 3) for v in (p[0], y + a[1], p[1])],
                            facing=[round(float(v), 3) + 0. for v in (d[0], f[1], d[1])]))
    LIGHTS[:] = sorted(out, key=lambda L: L['series'])     # (by series, each's in the objects' order)
    return len(out)



def main():
    global OBJECTS
    if '--objects' in sys.argv:
        OBJECTS = sys.argv[sys.argv.index('--objects') + 1]
        data = json.load(open(OBJECTS, encoding='utf-8')) if os.path.exists(OBJECTS) else None
    else:
        import track_data                       # (track.json's objects, or objects.json while they are not in it)
        data, base = track_data.part('objects')
        if data is not None and base == track_data.root():
            data = dict(data, version=data.get('version', 3))     # (in track.json: of its format, version 3's body)
    if data is None:
        print('no objects (the road editor writes them at its first save): nothing placed, the models as they are')
        return
    if int(data.get('version') or 0) >= 3:
        # (version 3: the layouts' spots and gates; everything else that stands there an object of the catalogue; the
        # layouts' places and physics objects in PLACES, for each game's build to make its own)
        placed = {}
        print('%d spots and gates moved in the layouts' % layouts(data, placed))
        print('%d marks painted on the roads' % marks(data))
        print('start lines painted: %s' % (', '.join(start_lines(data)) or 'none'))
        laid = objects(data, placed)
        print('objects laid: %s' % (', '.join('%d %s' % (n, m) for m, n in sorted(laid.items())) or 'none'))
        json.dump(dict(version=1, layouts=placed, lights=LIGHTS), open(PLACES, 'w', encoding='utf-8'), indent=1)
        print('places of %d layouts: %d spots and gates, %d physics objects' % (
            len(placed), sum(len(v['spots']) for v in placed.values()), sum(len(v['objects']) for v in placed.values())))
        return
    print('%d spots, gates, cones and boards moved in the layouts' % layouts(data))
    print('%d poles moved, their lights with them' % poles(data))
    print('%d marks painted on the roads' % marks(data))
    print('start lines painted: %s' % (', '.join(start_lines(data)) or 'none'))
    print('%d tyre walls laid, %d stacks in them' % walls(data))


if __name__ == '__main__':
    main()
