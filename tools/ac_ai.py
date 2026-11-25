"""Assetto Corsa's AI lines (ai/fast_lane.ai, ai/pit_lane.ai, data/ideal_line.ai), read and written, as its build
makes them from the layouts' racing lines (TRACK_FORMAT.md, "Racing lines").

A file (version 7): its points, each where it is (x, y, z), how far along (metres) and its number; then for each the
18 values AC drives by (EXTRA: its speed, throttle and brake, the radius it turns, the road to either side, its camber,
which way it turns, the floor's normal, its forward, its grade); then, for a lap's line, the grid AC finds the line's
points by: GRID_MARGIN metres round them in squares of GRID_CELL, each square the GRID_NEAR points nearest its middle,
nearest first.
"""
import struct

import numpy as np

VERSION = 7
EXTRA = ('speed', 'gas', 'brake', 'obsolete_lat_g', 'radius', 'side_left', 'side_right', 'camber', 'direction',
         'normal_x', 'normal_y', 'normal_z', 'length', 'forward_x', 'forward_y', 'forward_z', 'tag', 'grade')
GRID_MARGIN = 350.
GRID_CELL = 10.
GRID_NEAR = 10


def read(path):
    """{'points': (n, 3), 'along': (n,), 'extra': (n, 18) or None, 'grid': dict or None} of an AI file."""
    b = open(path, 'rb').read()
    version, n, _, _ = struct.unpack_from('<4i', b, 0)
    P = np.frombuffer(b, '<f4', n * 5, 16).reshape(-1, 5)
    out = dict(version=version, points=P[:, :3].astype(float), along=P[:, 3].astype(float), extra=None, grid=None)
    p = 16 + n * 20
    if len(b) >= p + 4 and struct.unpack_from('<i', b, p)[0] == n:
        out['extra'] = np.frombuffer(b, '<f4', n * 18, p + 4).reshape(-1, 18).astype(float)
        p += 4 + n * 72
    if len(b) >= p + 4 and struct.unpack_from('<i', b, p)[0]:
        X1, Y1, Z1, X0, Y0, Z0 = struct.unpack_from('<6f', b, p + 4)
        near, cell, nx = struct.unpack_from('<ifi', b, p + 28)
        ints = np.frombuffer(b, '<i4', offset=p + 40).reshape(nx, -1)
        nz = int(ints[0, 0])
        out['grid'] = dict(hi=(X1, Y1, Z1), lo=(X0, Y0, Z0), near=near, cell=cell,
                           cells=ints[:, 1:].reshape(nx, nz, near + 1)[:, :, 1:])
    return out


def grid(points):
    """The grid of a lap's line: its bounds GRID_MARGIN round its points (y 0), and in each GRID_CELL square, by
    columns of x then rows of z, the GRID_NEAR points nearest its middle, nearest first."""
    from scipy.spatial import cKDTree
    P = np.asarray(points, float)
    lo = np.array([P[:, 0].min() - GRID_MARGIN, 0., P[:, 2].min() - GRID_MARGIN], '<f4')
    hi = np.array([P[:, 0].max() + GRID_MARGIN, 0., P[:, 2].max() + GRID_MARGIN], '<f4')
    nx = int((float(hi[0]) - float(lo[0])) / GRID_CELL)
    nz = int((float(hi[2]) - float(lo[2])) / GRID_CELL)
    xs = float(lo[0]) + (np.arange(nx) + .5) * GRID_CELL
    zs = float(lo[2]) + (np.arange(nz) + .5) * GRID_CELL
    C = np.stack(np.meshgrid(xs, zs, indexing='ij'), -1).reshape(-1, 2)
    P2 = P[:, [0, 2]].astype('<f4').astype(float)
    k = min(GRID_NEAR, len(P2))
    d = np.hypot(P2[None, :, 0] - C[:, None, 0], P2[None, :, 1] - C[:, None, 1]) if len(C) * len(P2) < 2e7 else None
    if d is not None:
        idx = np.argsort(d, axis=1, kind='stable')[:, :k]
    else:
        tree = cKDTree(P2)
        idx = np.empty((len(C), k), int)
        for s in range(0, len(C), 2000):
            _, cand = tree.query(C[s:s + 2000], k=min(len(P2), k + 20))
            for r, (c, row) in enumerate(zip(C[s:s + 2000], cand)):
                dd = np.hypot(*(P2[row] - c).T)
                idx[s + r] = row[np.lexsort((row, dd))][:k]
    return dict(hi=tuple(hi.tolist()), lo=tuple(lo.tolist()), near=k, cell=GRID_CELL, cells=idx.reshape(nx, nz, k))


def write(path, points, along, extra=None, with_grid=False, g=None):
    """An AI file: its points (n, 3), how far along each is, its extra (n, 18), and for a lap's line its grid."""
    P = np.asarray(points, float)
    n = len(P)
    rows = np.zeros((n, 5), '<f4')
    rows[:, :3] = P
    rows[:, 3] = along
    rows[:, 4] = np.arange(n, dtype='<i4').view('<f4')
    out = [struct.pack('<4i', VERSION, n, 0, 0), rows.tobytes()]
    if extra is not None:
        out += [struct.pack('<i', n), np.asarray(extra, '<f4').tobytes()]
    if with_grid or g is not None:
        g = g or grid(P)
        cells = np.asarray(g['cells'], '<i4')
        nx, nz, k = cells.shape
        out.append(struct.pack('<i6fifi', 1, *g['hi'], *g['lo'], k, g['cell'], nx))
        col = np.concatenate([np.full((nx, nz, 1), k, '<i4'), cells], 2).reshape(nx, -1)
        out.append(np.concatenate([np.full((nx, 1), nz, '<i4'), col], 1).astype('<i4').tobytes())
    else:
        out.append(struct.pack('<i', 0))
    open(path, 'wb').write(b''.join(out))

