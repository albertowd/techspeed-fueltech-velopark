"""The car parks south of the pit apron by the hotlap's pit lane, as the ground photo shows them, for
tools/restyle_paddock.py (their asphalt, before the drape) and tools/paint_pits.py (their bays' lines,
after it).

- the car park south of the apron (restyle_paddock's PARK): asphalt a step under the apron
  (tools/drape_track.py), but for the grass round its trees (PARK_ISLANDS); its bays in rows square
  to the road south of it (PARK_ROWS);
- the car park east of it (EAST), past the road down the paddock's east side, which it opens off
  (its north west corner), but for the grass round its trees (EAST_ISLAND); its bays either side of
  two lines down it (EAST_SPINES), as the photo shows them;
- the road south of them (the paddock's) carried on west from its end (SOUTH_ON), between the trees,
  to the dirt road there.

Bays BAY_W wide and BAY_D deep, their lines BAY_LINE wide, white.
"""
import os

import numpy as np

import build_paths as bp

# the grass round the trees in the car park south of the apron: ellipses ((x, z), half across x, half across z)
PARK_ISLANDS = (((130., -50.5), 8., 9.), ((97.4, -79.8), 8.2, 6.9))
# its rows of bays along its south edge (restyle_paddock.PARK's from its third corner to its second):
# each a line that far in from it (metres) and how deep its bays are north of it and south of it: a
# row from the edge in, then two rows back to back about a line, aisles between
PARK_ROWS = ((0., 5., 0.), (16., 5., 5.), (32., 5., 5.))
# the car park east of it (x, z round it), the grass round its trees in it, and the lines its bays
# are either side of (each from, to: x, z)
EAST = ((14.6, -67.8), (-39.6, -79.6), (-30.4, -120.2), (23.1, -107.9))
EAST_ISLAND = ((6.5, -86.), (-2.5, -86.5), (-2.5, -110.), (1., -114.5), (6., -111.))
EAST_SPINES = (((-14.5, -80.), (-12., -105.)), ((-30.5, -84.), (-27.5, -109.5)))
EAST_MESH = '1ROAD_PADDOCK_PARK_EAST'
# the road south carried on: from its end (x, z, going that way) to the dirt road (x, z, going that way), as wide
SOUTH_ON = (((118.5, -88.), (.986, .168)), ((196., -72.5), (.997, .077)))
SOUTH_ON_W, SOUTH_ON_MESH = 8.2, '1ROAD_PADDOCK_SOUTH_ON'
# the road down between the car parks, from the paddock to the road south: two lanes round a long
# island of grass (LANES_ISLAND, x, z round it), the east one along the car park east's edge; its
# asphalt (LANES, x, z round it) laid where the paddock's is not (LANES_MESH)
LANES = ((34.25, -65.), (21.5, -65.5), (14.6, -67.8), (23.1, -107.9), (40., -105.5), (39.25, -97.))
LANES_ISLAND = ((27.5, -68.5), (24.5, -70.), (25.5, -80.), (28., -90.), (29.5, -97.5), (32.5, -98.8), (35.5, -97.5),
                (34., -88.), (31.5, -75.), (29.5, -69.))
LANES_MESH = '1ROAD_PADDOCK_LANES'
SOUTH_ON_TREES = 2.          # metres clear of its edges: no tree nearer
BAY_W, BAY_D, BAY_LINE = 2.5, 5., .1
BAYS_MESH = 'LINE_CAR_PARKS'
PARK_LINES = 'PARKING_LINES'     # (the track's own bays' lines in the car park south of the apron)


def ramp_shape():
    """The ramp from the car park up to the apron (tools/drape_track.py's PARK_RAMP along the apron's
    edge), x, z: no lines on it."""
    import drape_track as dt
    import restyle_paddock as rp
    from shapely.geometry import LineString, box
    edge = LineString(rp.APRON_SOUTH)
    (x0, x1), d = dt.PARK_RAMP, dt.PARK_RAMP_LONG + 4.         # (and the bays' ends by it)
    return edge.buffer(d, cap_style=2).intersection(box(x0 - .5, -200., x1 + .5, 200.))


def park_south_fill(pad):
    """South of the car park's drawn edge (restyle_paddock.PARK's third corner to its second) to the
    roads there: SOUTH_FILL metres of it, less the paddock's asphalt there and the road south carried
    on, x, z (the ground no more between them)."""
    import restyle_paddock as rp
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    a, b = np.array(rp.PARK[2], float), np.array(rp.PARK[1], float)
    u = (b - a) / np.linalg.norm(b - a)
    s = np.array([u[1], -u[0]])
    if s[1] > 0:
        s = -s                                                  # (south, away from it)
    strip = Polygon([tuple(a), tuple(b), tuple(b + s * SOUTH_FILL), tuple(a + s * SOUTH_FILL)])
    tris = []
    for n, m in bp.meshes(pad):
        if not n['name'].startswith(bp.ASPHALT) or not len(n['faces']):
            continue
        T = bp.triangles(n, m)[:, :, [0, 2]]
        c = T.mean(1)
        import shapely
        inside = shapely.contains_xy(strip.buffer(1.), c[:, 0], c[:, 1]) & ~shapely.contains_xy(Polygon(rp.PARK), c[:, 0], c[:, 1])
        tris += [Polygon(t).buffer(1e-4) for t in T[inside] if Polygon(t).area > 1e-8]
    have = unary_union(tris) if tris else Polygon()
    return strip.difference(have.buffer(.01)).difference(south_on_shape()).buffer(-.01).buffer(.01)
SOUTH_FILL = 3.


def ellipse(c, rx, rz, k=48):
    from shapely.geometry import Polygon
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return Polygon(np.c_[c[0] + rx * np.cos(a), c[1] + rz * np.sin(a)])


def park_islands():
    from shapely.ops import unary_union
    return unary_union([ellipse(*e) for e in PARK_ISLANDS])


def east_shape():
    """The car park east, x, z, less the grass round its trees."""
    from shapely.geometry import Polygon
    return Polygon(EAST).buffer(0).difference(Polygon(EAST_ISLAND).buffer(.8).buffer(-.8)).buffer(0)


def lanes_island():
    from shapely.geometry import Polygon
    return Polygon(LANES_ISLAND).buffer(1.).buffer(-1.)


def lanes_shape():
    """The road down between the car parks, x, z, less its island."""
    from shapely.geometry import Polygon
    return Polygon(LANES).buffer(0).difference(lanes_island()).buffer(0)


def south_on_line(k=40):
    """The middle of the road south carried on, (k, 2) x, z."""
    (a, ta), (b, tb) = (np.array(p, float) for p in SOUTH_ON[0]), (np.array(p, float) for p in SOUTH_ON[1])
    ta, tb = ta / np.linalg.norm(ta), tb / np.linalg.norm(tb)
    L = np.linalg.norm(b - a)
    u = np.linspace(0., 1., k)[:, None]
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * L * ta
            + (-2 * u ** 3 + 3 * u ** 2) * b + (u ** 3 - u ** 2) * L * tb)


def south_on_shape():
    """The road south carried on, x, z: from a little back into its end, flat across."""
    from shapely.geometry import LineString
    P = south_on_line()
    t0 = np.array(SOUTH_ON[0][1], float)
    P = np.r_[[P[0] - t0 / np.linalg.norm(t0) * 1.5], P]
    return LineString(P).buffer(SOUTH_ON_W / 2, cap_style=2, join_style=1)


def bays(lot, spines_rows):
    """The lines of rows of bays, LineStrings: spines_rows, each (a spine's from, to, x, z; how deep its
    bays either side (0 for none on that side)), the lines across every BAY_W along it, and the
    spine itself where there are bays both sides."""
    from shapely.geometry import LineString
    out = []
    for (a, b), (left, right) in spines_rows:
        a, b = np.array(a, float), np.array(b, float)
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        for s in np.arange(0., L + 1e-6, BAY_W):
            p = a + u * s
            out.append(LineString([tuple(p - n * right), tuple(p + n * left)]))
        if left and right:
            out.append(LineString([tuple(a), tuple(b)]))
    return [g for g in (x.intersection(lot) for x in out) if not g.is_empty and g.length > .3]


def park_bays(lot):
    """The bays of the car park south of the apron (its shape as laid, lot, x, z): rows along its south edge."""
    import restyle_paddock as rp
    a, b = np.array(rp.PARK[2], float), np.array(rp.PARK[1], float)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])                                 # (bays() puts left on this side)
    s_ = 1. if n[1] > 0 else -1.                                # (north, into it)
    rows = [((a - u * 10. + n * s_ * o, b + u * 10. + n * s_ * o), (north, south) if s_ > 0 else (south, north))
            for o, north, south in PARK_ROWS]
    return bays(lot.buffer(-.3), rows)


def east_bays():
    lot = east_shape().buffer(-.3)
    return bays(lot, [((a, b), (BAY_D, BAY_D)) for a, b in EAST_SPINES])


def lay_flat(track, pad, shape, name):
    """Flat asphalt over shape (x, z), a mesh of the paddock's named name, as its asphalt is mapped (from
    above), on the ground as it is. How many faces."""
    import gltf_source
    import restyle_paddock as rp
    if shape.is_empty or any(n['name'] == name for n, _ in bp.meshes(pad)):
        return 0
    base = next(n for n, _ in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                        if n['name'].startswith((bp.GROUND, bp.FILLER)) and len(n['faces'])]))
    P = np.concatenate(rp.fill(shape.segmentize(1.), most=4.))
    y = ground.ys(P)
    if np.isnan(y).any():
        # (no ground under it: the roads round it, as high, the nearest's where none is under it)
        from scipy.spatial import cKDTree
        R = np.concatenate([bp.triangles(n, m).reshape(-1, 3) for k_ in (track, pad) for n, m in bp.meshes(k_)
                            if n['name'].startswith('1ROAD') and len(n['faces'])])
        x0, z0, x1, z1 = shape.buffer(5.).bounds
        R = R[(R[:, 0] > x0) & (R[:, 0] < x1) & (R[:, 2] > z0) & (R[:, 2] < z1)]
        if len(R):
            gone = np.isnan(y)
            y[gone] = R[cKDTree(R[:, [0, 2]]).query(P[gone])[1], 1] - .02
    y = np.where(np.isnan(y), np.nanmedian(y), y) + .02
    V = np.zeros((len(P), base['verts'].shape[1]))
    V[:, 0], V[:, 1], V[:, 2] = P[:, 0], y, P[:, 1]
    V[:, 4] = 1.
    F = np.arange(len(P)).reshape(-1, 3)
    up = np.cross(V[F[:, 1], :3] - V[F[:, 0], :3], V[F[:, 2], :3] - V[F[:, 0], :3])[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)
    ref = dict(base)
    rp.flat_map(ref)
    sc = np.median(np.linalg.norm(ref['verts'][:, 6:8].astype(float), axis=1) / np.maximum(np.linalg.norm(ref['verts'][:, [0, 2]].astype(float), axis=1), 1e-6))
    V[:, 6:8] = V[:, [0, 2]] * sc
    if V.shape[1] > 8:
        V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
    node = dict(base, name=name, children=[], verts=V.astype(base['verts'].dtype), faces=F.astype('<u2'))
    bp.sphere(node)
    pad['root']['children'].append(node)
    return len(F)


def clear_trees(shape):
    """The trees standing within shape (x, z) out of the trees' model and the forest. How many. None while the trees
    are not planted yet (tools/place_trees.py plants them after this: track.json's trees have these out already)."""
    import json
    import shapely
    import build_trees as bt
    import gltf_source
    import plant_outside as po
    if not os.path.exists(bt.FOREST):
        return 0
    forest = json.load(open(bt.FOREST))
    S = np.array([[s[1], s[3]] for s in forest['trees']])
    gone = shapely.contains_xy(shape, S[:, 0], S[:, 1])
    if not gone.any():
        return 0
    extra = gltf_source.load_kn5(bt.EXTRA)
    po.take_out(extra, forest, gone, tags=None)
    bt.save(extra, bt.EXTRA)
    json.dump(forest, open(bt.FOREST, 'w'), indent=0)
    return int(gone.sum())
