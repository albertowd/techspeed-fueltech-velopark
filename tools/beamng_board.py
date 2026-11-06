"""The start gantry's board and lamps for the BeamNG.drive level, as the Custom Shaders Patch
script in the track's ext_config.ini draws them in Assetto Corsa.

BeamNG draws no text on a texture without a web page, so the board is made the way its own
fuel price signs and drag strip timing boards are: a small shape for each character, glowing
red like LEDs, which the level's Lua (beamng/lua/ge/extensions/techspeedVelopark/board.lua)
swaps as the times change. This writes those shapes, one per character, and board.json: where
each character of each row stands on each face of each gantry, laid out as the patch script
lays out its text on the board's texture (DISPLAY_POS and the row positions there), carried
onto the board through its mesh's texture coordinates.

The start lamps and the flag lamp are shapes of their own with a material whose glow BeamNG
multiplies by the object's instance colour, as its traffic signals are: the Lua lights them by
setting that colour.
"""
import json, os, re

import numpy as np
from PIL import Image

# the patch script's board: its display on the texture, and the text on the display, in pixels
TEXTURE = (2048, 320)
DISPLAY_POS = (306, 8)
LETTER = (30, 60)
CLOCK_LETTER = (48, 88)
CLOCK_POS = (609, 146)
CLOCK_CELLS = 5
RANK_X = 0
NUMBER_X = LETTER[0] * 2.2
TIME_X = LETTER[0] * 5.2
CELLS = {'rank': 2, 'number': 2, 'time': 8}
# how far the characters stand out of the board, metres
LIFT_OFF = .004
LAMPS = re.compile(r'^(KS|NO)_START_LIGHT_(\d+)$')
FLAGS = re.compile(r'^(GANTRY|STATIC)_FLAG$')
BOARD = 'GANTRY_BOARD'
GLYPHS = '0123456789:.'
SHAPES = 'board'


def row_at(rank):
    """Where the patch script writes a row, on its display: places 1 to 4 on the left, 5 and 6
    in the middle, 7 to 10 on the right."""
    if rank <= 4:
        return 10, 8 + (rank - 1) * 74
    if rank <= 6:
        return 490, 8 + (rank - 5) * 74
    return 968, 8 + (rank - 7) * 74


def glyph_name(ch):
    return {':': 'colon', '.': 'dot'}.get(ch, ch)


def _segments(ch):
    """The lit rectangles of a character, (x0, z0, x1, z1), a character being 1 high and 0.5
    wide, centred: seven segments as an LED display's, a little slanted as the patch's font."""
    w, h, t, g = .19, .43, .075, .012
    seg = {
        'a': (-w + g, h - t, w - g, h), 'd': (-w + g, -h, w - g, -h + t), 'g': (-w + g, -t / 2, w - g, t / 2),
        'f': (-w, g, -w + t, h - g), 'b': (w - t, g, w, h - g),
        'e': (-w, -h + g, -w + t, -g), 'c': (w - t, -h + g, w, -g),
    }
    lit = {'0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg', '5': 'acdfg', '6': 'acdefg',
           '7': 'abc', '8': 'abcdefg', '9': 'abcdfg'}
    if ch in lit:
        return [seg[s] for s in lit[ch]]
    d = t * .9
    if ch == ':':
        return [(-d / 2, .16 - d / 2, d / 2, .16 + d / 2), (-d / 2, -.16 - d / 2, d / 2, -.16 + d / 2)]
    if ch == '.':
        return [(-d / 2, -h, d / 2, -h + d)]
    raise ValueError(ch)


def glyph_mesh(ch, slant=.08):
    """A character's shape: (P, N, U, F) in its own axes, x across, z up, facing -y."""
    P, F = [], []
    for x0, z0, x1, z1 in _segments(ch):
        k = len(P)
        for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
            P.append((x + z * slant, 0, z))
        # seen from -y: x to the right, z up
        F += [(k, k + 2, k + 1), (k, k + 3, k + 2)]
    P = np.array(P, float)
    return P, np.tile([0, -1., 0], (len(P), 1)), np.zeros((len(P), 2)), np.array(F)


def materials(level):
    """The characters' glow and the lamps' (lit by their instance colour)."""
    white = '/levels/%s/art/shapes/%s/white.png' % (level, SHAPES)
    base = {'class': 'Material', 'version': 1.5, 'castShadows': False}
    led = dict(base, name='velopark_led', mapTo='velopark_led', doubleSided=True, persistentId='6f2b1e50-6d1a-4f83-9a55-0b1d7c3e2a01',
               Stages=[{'baseColorFactor': [.02, 0, 0, 1], 'roughnessFactor': 1., 'metallicFactor': 0., 'emissive': True,
                        'emissiveFactor': [1, .06, .04], 'emissiveMap': white, 'emissiveIntensityNits': 2500}, {}, {}, {}])
    lamp = dict(base, name='velopark_lamp', mapTo='velopark_lamp', persistentId='6f2b1e50-6d1a-4f83-9a55-0b1d7c3e2a02',
                Stages=[{'baseColorFactor': [.06, .06, .06, 1], 'roughnessFactor': .4, 'metallicFactor': 0., 'emissive': True,
                         'emissiveFactor': [1, 1, 1], 'emissiveMap': white, 'emissiveIntensityNits': 5000,
                         'instanceEmissive': True}, {}, {}, {}])
    return {'velopark_led': led, 'velopark_lamp': lamp}


def write_shapes(shapes, write_dae):
    """The characters' shapes, and the plain map the glow reads."""
    d = os.path.join(shapes, SHAPES)
    os.makedirs(d, exist_ok=True)
    Image.new('RGB', (4, 4), 'white').save(os.path.join(d, 'white.png'))
    for ch in GLYPHS:
        write_dae(os.path.join(d, 'glyph_%s.dae' % glyph_name(ch)), [('glyph', 'velopark_led') + glyph_mesh(ch)], sizes=[1])


class Face:
    """A face of the board: from a point of its texture to where it is, in BeamNG's axes."""

    def __init__(s, tris, uvs):
        s.tris, s.uvs = tris, uvs

    def at(s, u, v):
        for t, q in zip(s.tris, s.uvs):
            a, b, c = q
            m = np.array([b - a, c - a]).T
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            l1, l2 = np.linalg.solve(m, np.array([u, v]) - a)
            if l1 >= -1e-6 and l2 >= -1e-6 and l1 + l2 <= 1 + 1e-6:
                return t[0] + l1 * (t[1] - t[0]) + l2 * (t[2] - t[0])
        return None

    def cell(s, x, y, size):
        """A character's cell of size (w, h) pixels with its top left at (x, y) on the display."""
        tx, ty = DISPLAY_POS[0] + x + size[0] / 2, DISPLAY_POS[1] + y + size[1] / 2
        u, v = tx / TEXTURE[0], ty / TEXTURE[1]
        du, dv = .5 / TEXTURE[0], .5 / TEXTURE[1]
        p = s.at(u, v)
        right = s.at(u + du, v) - s.at(u - du, v)
        up = s.at(u, v - dv) - s.at(u, v + dv)
        px = np.linalg.norm(up) / (2 * dv) / TEXTURE[1]       # metres per pixel up the board
        right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
        out = np.cross(right, up)
        return {'pos': [float(c) for c in p + out * LIFT_OFF], 'right': [float(c) for c in right],
                'up': [float(c) for c in up], 'size': float(size[1] * px)}


def faces(model, to_beamng, lift):
    """The board's faces in a gantry model: its triangles grouped by which way they face."""
    m = model.mesh(BOARD)
    v = m['v']
    groups = {}
    for f in m['f']:
        p = to_beamng(v[f, :3]) + (0, 0, lift)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        key = tuple(np.round(n / np.linalg.norm(n), 1))
        groups.setdefault(key, ([], []))
        groups[key][0].append(p)
        groups[key][1].append(v[f, 6:8].astype(float))
    return [Face(np.array(t), np.array(q)) for t, q in groups.values()]


def gantry(name, model, to_beamng, lift, lamps, flags):
    """A gantry's entry in board.json: its lamps, and the cells of each face's rows and clock."""
    out = []
    for face in faces(model, to_beamng, lift):
        rows = []
        for rank in range(1, 11):
            x, y = row_at(rank)
            cells = []
            for col, x0 in (('rank', RANK_X), ('number', NUMBER_X), ('time', TIME_X)):
                cells += [face.cell(x + x0 + k * LETTER[0], y, LETTER) for k in range(CELLS[col])]
            rows.append(cells)
        clock = [face.cell(CLOCK_POS[0] + k * CLOCK_LETTER[0], CLOCK_POS[1], CLOCK_LETTER) for k in range(CLOCK_CELLS)]
        out.append({'rows': rows, 'clock': clock})
    board = to_beamng(model.mesh(BOARD)['v'][:, :3])
    centre = (board.min(0) + board.max(0)) / 2 + (0, 0, lift)
    return {'name': name, 'centre': [float(c) for c in centre], 'lamps': lamps, 'flags': flags, 'faces': out}


def write_config(level_dir, gantries, races):
    """board.json: the gantries, each with the layouts whose race starts under it, the one it
    shows before any is raced first. races: {layout: start position}."""
    for g in gantries:
        c = np.array(g['centre'])
        near = sorted(lay for lay, at in races.items() if np.linalg.norm(np.asarray(at)[:2] - c[:2]) < 40)
        own = 'vp_' + re.sub(r'^velopark_gantry_(\d+).*$', r'\1', g['name'])
        g['layouts'] = near
        g['default'] = own if own in near else (near[0] if near else None)
    json.dump({'cells': CELLS, 'gantries': gantries}, open(os.path.join(level_dir, 'board.json'), 'w'))
