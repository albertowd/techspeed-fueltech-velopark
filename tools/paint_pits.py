"""The pit lanes' lines, pit boxes and racing lines, laid after tools/drape_track.py on the roads as
they lie (so a change to them needs no drape again).

    python tools/paint_pits.py

- the pit lanes' lines, yellow, along both sides of each: the 1000's along its lane as it is now,
  the 1500's entry from the track to its boxes, the hotlap's from its fork to the fence, straight
  along the building by its pit lane from a white line across its start, a line between its boxes
  and the lane; white lines along the road south from the 1500's entry to the car park;
- the vp_1000's pit boxes and racing line past them along its west yellow line;
- the hotlap's and the 2500's pit boxes in a row in front of that building, their pit lanes along
  the middle of the lane past them, the boxes' paint moved with them.

Each part does nothing when it is done already.
"""
import os, sys

import numpy as np

import build_paths as bp
import gltf_source
import kn5io
from restyle_paddock import (HOTLAP_EDGE, HOTLAP_AI, HOTLAP_BOX_ROAD, HOTLAP_LANE_Z, HOTLAP_LAYOUTS, HOTLAP_MESH, HOTLAP_ROADS, HOTLAP_ROUND, HOTLAP_START, HOTLAP_START_MESH, MOUTH_BOX, PIT_FROM, PIT_IN, PIT_LAYOUT, PIT_PAINT, clip_in, fill, garages, hotlap_row, lane_shape, roads_shape, wall_line)

TEX = os.path.join(bp.ROOT, 'source', 'textures')
PAINT_UP = .006           # metres paint lies over the road it is moved onto
LINE_STEP = .5            # metres between a line's points along it
# where the hotlap's pit lane splits off the track (HOTLAP_SPLIT, z: the road wider than the track
# south of it), and the track's edge between there and the nose where the track's own line begins
# again (HOTLAP_DIVIDE, x, z): east of it the lane, its outer line yellow from the split (the old
# white line along that edge, LINE_0, out south of it), its inner line from the nose
HOTLAP_SPLIT, HOTLAP_DIVIDE, HOTLAP_EDGE_LINE = 51., ((-18.2, 51.), (-17.8, 36.7)), 'LINE_0'
HOTLAP_IN = .13            # metres its lines lie in from its edges: as the track's white line along it before the split


# and its pit lane's racing line past them, PIT_PASS in from the walls' line, from z to z (easing
# to it over PIT_PASS_EASE metres at both ends)
PIT_PASS, PIT_PASS_Z, PIT_PASS_EASE = 3.8, (96., 172.), 4.


# the pit lanes' lines, yellow (PIT_PAINT_COLOUR of the white paint's texture), along both sides of
# each, PIT_LINE_IN in from its edges, PIT_LINE_W wide, from its mouth (PIT_LINE_CLEAR short of the
# track and the roads joining it) to its pit boxes: the 1000's (its lane, 1ROAD_24, from z
# PIT_LINE_FROM north), and the 1500's entry (PIT_1500_ROAD in PIT_1500_AREA, x, z, up to its boxes)
PIT_PAINT_MAT, PIT_PAINT_TEX, PIT_PAINT_COLOUR = 'PitLinePaint', 'PitLinePaint.dds', (1., .78, .1)


PIT_LINE_IN, PIT_LINE_W, PIT_LINE_CLEAR, PIT_LINE_FROM, PIT_LINE_MESH = .25, .12, .6, 128.5, 'LINE_VP_1000_PIT'


PIT_1500_ROAD, PIT_1500_AREA, PIT_1500_MESH = '1ROAD_25', ((-46., -8.), (55., 104.)), 'LINE_VP_1500_PIT'


# the road south from the 1500's pit entry to the car park, white lines along both its sides
SOUTH_ROAD, SOUTH_ROAD_AREA, SOUTH_ROAD_MESH = '1ROAD_26', ((-60., 40.), (-44., 63.)), 'LINE_ROAD_26'


# the 1000's lane's lines short of the track at its mouth by PIT_1000_CLEAR, so they end along it, not curling
PIT_1000_CLEAR = 1.6


# its pit boxes along its west yellow line, PIT_BOX_CLEAR in from it, where that is further in than
# PIT_IN from the walls' line; its racing line PIT_PASS - PIT_IN beyond them
PIT_BOX_CLEAR = 1.3


PIT_LINE_SHORTEST = 2.     # metres: no line shorter than that, where a lane opens onto the track


# a yellow line carried on from where a white line ends, if one of its ends is within JOIN_REACH of
# it, going on as the white line goes and easing into its own way over JOIN_EASE (metres)
JOIN_REACH, JOIN_EASE = 8., 2.5
# the 1500's entry's outer line ends turning east into the race control's ground: where it passes
# z EAST_TURN_Z (east of x EAST_TURN_X), over EAST_TURN_R metres, to x EAST_TURN_TO
EAST_TURN_Z, EAST_TURN_X, EAST_TURN_R, EAST_TURN_TO = 85.5, -37.5, .8, -43.
# the track's white lines either side of the 1000's pit lane's mouth: its east end's, its west end's
MOUTH_LINES = ('LINE_23', 'LINE_33')
CRACK = .005               # metres: a gap narrower than twice this between roads' faces is road (to paint on; under tools/check_holes.py's PAINT_NEAR)
HUG_MAX = 5.               # metres a white line is cut back at most, to where its road opens out
HUG = .1                   # metres: a white line farther than this from the grass it ran along, the road there opens out


# past the hairpin (HAIRPIN_OUT) the track's outer line (FORK_FROM) forks in two (both in FORK_LINE):
# FORK_LINE out in FORK_BOX (x, z), both its branches laid again from FORK_FROM's end as FORK_MESH
FORK_FROM, FORK_LINE, FORK_BOX, FORK_MESH = 'LINE_4', 'LINE_2_SOLID', ((14., 21.5), (-23.5, -18.5)), 'LINE_2_SOLID_FORK'


PIT_LINE_FROM_WIDE = 1.5  # metres: a lane's lines begin where it is this wide (where it forks off the track)


# the track's edge line past both 1500 pit entries, unbroken: TRACK_LINE_FROM (its line) cut at
# TRACK_LINE_CUT (z), where it bent off into the 1500's entry, and carried straight on to the top of
# TRACK_LINE_TO, as TRACK_LINE_MESH, in the white paint
TRACK_LINE_FROM, TRACK_LINE_CUT, TRACK_LINE_TO, TRACK_LINE_MESH = 'LINE_32', 75., 'LINE_0', 'LINE_32_0'


HOTLAP_FENCE = (110., -8.)


# the fence the hotlap's pit lane joins the track through, its line (x, z, west to east... as built):
# its lines end on it, straight
HOTLAP_FENCE_LINE = ((124.43, -9.33), (198.5, 8.6))


HOTLAP_FORK = (42., 66.)   # z: where it forks off, its lines beginning as it widens


HOTLAP_NOSE = 36.7         # z: the nose of the fork (where the track's line on its other side begins), its inner line from there


HOTLAP_ROW_MESH = 'LINE_VP_1500_HOTLAP_BOXES'


def lane_bands(lane, trk, within, sides=None, fork=None, smooth=.6, bumps=.5, clear=PIT_LINE_CLEAR, inset=PIT_LINE_IN, lines=False):
    """A pit lane's lines, x, z: along both its sides (lane), PIT_LINE_IN in, short of the track and the
    roads joining it (trk) and, at its fork off the track (fork, x, z, if given), of where it is
    narrower than PIT_LINE_FROM_WIDE, within; the longest sides of them, if given, else every one
    PIT_LINE_SHORTEST or longer. (With lines, their middles, LineStrings, not the bands.)"""
    from shapely.ops import unary_union
    inner = lane.buffer(-inset, join_style=2)
    inner = inner.buffer(-bumps).buffer(bumps).buffer(smooth).buffer(-smooth)   # (its edges smoothed: no notches, no spikes)
    ring = unary_union([g.exterior for g in getattr(inner, 'geoms', [inner])])
    h = PIT_LINE_FROM_WIDE / 2
    thin = lane.difference(lane.buffer(-h).buffer(h + .01))             # (where it forks off, still narrow)
    thin = thin.intersection(fork) if fork is not None else thin.difference(thin)
    keep = within.buffer(-.3, join_style=2)                            # (not along the ends of the lane cut off)
    found = ring.difference(trk.buffer(clear + inset + .2)).difference(thin.buffer(PIT_LINE_FROM_WIDE)).intersection(keep)
    found = sorted([g for g in getattr(found, 'geoms', [found]) if g.length >= PIT_LINE_SHORTEST], key=lambda g: -g.length)
    found = found[:sides] if sides else found
    if lines:
        return found
    return unary_union([g.buffer(PIT_LINE_W / 2, cap_style=2, join_style=2) for g in found]).buffer(0) if found else None


def white_end(L, name, north=None, west=None):
    """Where a line of paint (mesh name of L) ends, its north end or its south (or its west end or its
    east): the middle of its end, x, z, and the way it runs out of it there."""
    n = next((n for n, _ in bp.meshes(L) if n['name'] == name and len(n['faces'])), None)
    if n is None:
        return None
    V = n['verts'][np.unique(n['faces'].astype(int))][:, [0, 2]].astype(float)
    if west is not None:
        tip = V[np.argmax(V[:, 0]) if west else np.argmin(V[:, 0])]
    else:
        tip = V[np.argmax(V[:, 1]) if north else np.argmin(V[:, 1])]
    near = V[np.linalg.norm(V - tip, axis=1) < 3.]
    t = np.linalg.svd(near - near.mean(0))[2][0]
    if (tip - near.mean(0)) @ t < 0:
        t = -t
    a = (near - near.mean(0)) @ t
    e = near[a > a.max() - .03].mean(0)                         # (the middle of its end's edge)
    return e, t


def hermite(a, ta, b, tb, k=24):
    """A smooth curve from a, going ta, to b, arriving going tb: (k, 2) points."""
    L = np.linalg.norm(b - a)
    u = np.linspace(0., 1., k)[:, None]
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * L * ta
            + (-2 * u ** 3 + 3 * u ** 2) * b + (u ** 3 - u ** 2) * L * tb)


def carry_on(lines, ends, reach=None, ease=None, whites=None, roads=None, trim=None):
    """Lines' middles (LineStrings) carried on from where white lines end (ends: each (its end, the way
    it runs out of it[, a fork])): the line with an end nearest each (within reach) begins there
    instead, going on as the white line goes and easing into its own way over ease metres. A fork
    (where the roads part at a corner): the white line carried straight on to where it meets the
    line's way (that carried back straight from its end, if need be), its point, both lines going
    from there each its own way (the white's piece added to whites)."""
    from shapely.geometry import LineString
    from shapely.ops import substring
    reach = JOIN_REACH if reach is None else reach
    ease = JOIN_EASE if ease is None else ease
    lines = list(lines)
    for end in ends:
        if end is None:
            continue
        e, t = end[0], end[1]
        sharp = len(end) > 2 and end[2]
        if sharp and roads is not None and trim is not None and len(end) > 3:
            # (the white line back to where the road beside it begins to open out from it: the fork there)
            c = hugs_from(e, t, roads, reach)
            if c is not None and np.linalg.norm(c - e) > HUG_MAX:
                c = e - t * HUG_MAX                             # (no more than that: a road's edge a little off its grass's is no opening)
            if c is not None and np.linalg.norm(c - e) > .2:
                trim(end[3], c, t)
                e = c
        best = None
        for i, g in enumerate(lines):
            for at_start in (True, False):
                q = np.array(g.coords[0] if at_start else g.coords[-1])
                d = np.linalg.norm(q - e)
                if d < reach and (best is None or d < best[0]):
                    best = (d, i, at_start)
        if best is None:
            continue
        _, i, at_start = best
        g = lines[i] if at_start else LineString(lines[i].coords[::-1])
        a = min(ease, g.length / 2)
        p1 = np.array(g.interpolate(a).coords[0])
        t1 = np.array(g.interpolate(a + .1).coords[0]) - np.array(g.interpolate(max(a - .1, 0.)).coords[0])
        t1 /= max(np.linalg.norm(t1), 1e-9)
        rest = substring(g, a, g.length)
        if sharp and roads is not None:
            fork = fork_round(e, g, a, roads, reach, end[4] if len(end) > 4 else None)
            if fork is not None:
                white_, rest_ = fork
                if whites is not None and white_ is not None:
                    whites.append(white_)
                lines[i] = rest_
                continue
        if sharp:
            fork = fork_point(e, t, g, reach)
            if fork is not None:
                A, rest_ = fork
                if whites is not None and np.linalg.norm(A - e) > .02:
                    whites.append(LineString([tuple(e), tuple(A)]))
                lines[i] = rest_
                continue
            t = (p1 - e) / max(np.linalg.norm(p1 - e), 1e-9)          # (straight out of the point, not on past it)
        lines[i] = LineString([tuple(p) for p in hermite(e, t, p1, t1)[:-1]] + list(rest.coords))
    return lines


def trimmer(L):
    """A white line of L (by its name) cut back to c, all of it on past c the way it went (t)."""
    def trim(name, c, t):
        from shapely.geometry import LineString
        n = next((n for n, _ in bp.meshes(L) if n['name'] == name and len(n['faces'])), None)
        if n is not None:
            clip_in(n, LineString([tuple(c), tuple(c + t * 30.)]).buffer(.6, cap_style=2))
    return trim


def hugs_from(e, t, roads, reach):
    """Back along a white line ending at e (going t there, straight) the last point where it runs as
    near the grass beside it as it does before (within HUG of it): where the road beside it begins
    to open out. None if there is no grass by it."""
    from shapely.geometry import Point
    grass = Point(*e).buffer(reach * 2).difference(roads).buffer(-.01).buffer(.01)
    if grass.is_empty:
        return None
    s = np.arange(0., reach * 2, .1)
    d = np.array([grass.distance(Point(*(e - t * s_))) for s_ in s])
    base = np.median(d[len(d) // 2:])
    if not np.isfinite(base) or base > 1.5:
        return None
    near = np.nonzero((d <= base + HUG) & (d >= .08))[0]          # (and clear of it by half its width at least)
    return e - t * s[near[0]] if len(near) else None


def east_turn(lines):
    """The 1500's entry's outer line (east of EAST_TURN_X where it passes EAST_TURN_Z) ending there,
    turning east into the race control's ground over EAST_TURN_R to x EAST_TURN_TO, as the photos show
    it; the rest as they are."""
    from shapely.geometry import LineString
    out = []
    for g in lines:
        c = np.array(g.coords)
        cross = np.nonzero((c[:-1, 1] - EAST_TURN_Z) * (c[1:, 1] - EAST_TURN_Z) <= 0)[0]
        hit = [i for i in cross if min(c[i, 0], c[i + 1, 0]) < EAST_TURN_X]
        if not hit:
            out.append(g)
            continue
        i = hit[0]
        f = (EAST_TURN_Z - c[i, 1]) / max(c[i + 1, 1] - c[i, 1], 1e-9)
        p = c[i] + (c[i + 1] - c[i]) * f
        north = c[i + 1, 1] > c[i, 1]
        keep = list(map(tuple, c[:i + 1])) + [tuple(p)] if north else [tuple(p)] + list(map(tuple, c[i + 1:]))
        t = (c[i + 1] - c[i]) if north else (c[i] - c[i + 1])
        t = t / max(np.linalg.norm(t), 1e-9)
        q = np.array([EAST_TURN_TO, p[1] + EAST_TURN_R])
        bend = [tuple(v) for v in hermite(p, t, q, np.array([-1., 0.]), 16)[1:]]
        out.append(LineString(keep + bend if north else bend[::-1] + keep))
    return out


def fork_round(e, g, ease, roads, reach, tip=None):
    """Where a white line ends at e by a point of grass between two roads (roads, x, z: what is not
    them is grass) and line g goes on along that grass's other side: both along the grass's edge,
    as far out from it as the white line is, to the point of that (beyond the grass's point, on the
    road), the white from e to it, g from it to where it is ease along, then as it goes. (The white's
    piece, g so carried on; None if there is no such grass by e.)"""
    from shapely.geometry import LineString, Point, box
    from shapely.ops import substring
    P0 = Point(*e)
    area = P0.buffer(reach)
    grass = area.difference(roads).buffer(-.01).buffer(.01)
    parts = [q for q in getattr(grass, 'geoms', [grass]) if q.area > .05]
    if not parts:
        return None
    nose = min(parts, key=lambda q: q.distance(P0))
    d = nose.distance(P0)
    if d > 1.5:
        return None
    ring = nose.buffer(max(d, .05), join_style=2, mitre_limit=20.).exterior
    L = ring.length
    a_ = min(ease, g.length / 2)
    p1 = g.interpolate(a_)
    se, s1 = ring.project(P0), ring.project(p1)
    if ring.distance(p1) > .6:
        return None
    # (the shorter way round from e's to p1's, the point of it between: farthest from the grass)
    fwd = (s1 - se) % L
    path = substring(ring, se, se + fwd) if se + fwd <= L else LineString(list(substring(ring, se, L).coords) + list(substring(ring, 0., se + fwd - L).coords)[1:])
    if fwd > L / 2:
        back = L - fwd
        path = substring(ring, s1, s1 + back) if s1 + back <= L else LineString(list(substring(ring, s1, L).coords) + list(substring(ring, 0., s1 + back - L).coords)[1:])
        path = LineString(path.coords[::-1])
    path = path.segmentize(.05)
    C = np.array(path.coords)
    k = int(np.argmax(C @ np.asarray(tip, float))) if tip is not None else int(np.argmax([nose.distance(Point(*c)) for c in C]))
    white_ = LineString(C[:k + 1]) if k >= 1 else None
    rest = substring(g, a_, g.length)
    yellow = LineString([tuple(c) for c in C[k:]] + list(rest.coords)[1:])
    return white_, yellow


def fork_point(e, t, g, reach):
    """Where a white line ending at e (going t) carried straight on meets line g (from its start, carried
    back straight from it some), within reach: the point, and g from there. None if they do not meet."""
    from shapely.geometry import LineString, Point
    from shapely.ops import substring
    c = np.array(g.coords)
    t0 = c[1] - c[0]
    t0 /= max(np.linalg.norm(t0), 1e-9)
    ext = LineString([tuple(c[0] - t0 * reach)] + [tuple(p) for p in c])
    ray = LineString([tuple(e), tuple(e + t * reach)])
    hit = ray.intersection(ext)
    if hit.is_empty:
        return None
    pts = [np.array(q.coords[0]) for q in getattr(hit, 'geoms', [hit]) if q.geom_type == 'Point']
    if not pts:
        return None
    A = min(pts, key=lambda q: np.linalg.norm(q - e))
    return A, substring(ext, ext.project(Point(*A)), ext.length)


def bands_of(lines, w=None):
    """Lines' middles as bands of paint w wide (PIT_LINE_W), x, z."""
    from shapely.ops import unary_union
    w = PIT_LINE_W if w is None else w
    return unary_union([g.buffer(w / 2, cap_style=2, join_style=1) for g in lines]).buffer(0) if lines else None


_ROADS = []


def road_union(bounds):
    """The roads and kerbs (1ROAD, 1KERB) of every model under bounds (x0, z0, x1, z1), as one shape, x, z."""
    import glob
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if not _ROADS:
        T = []
        for f in glob.glob(os.path.join(bp.MODELS, 'velopark*.glb')):
            for n, m in bp.meshes(gltf_source.load_kn5(f)):
                if n['name'].startswith(('1ROAD', '1KERB')) and len(n['faces']):
                    T.append(bp.triangles(n, m)[:, :, [0, 2]])
        _ROADS.append(np.concatenate(T))
    T = _ROADS[0]
    x0, z0, x1, z1 = bounds
    s_ = (T[:, :, 0].max(1) >= x0) & (T[:, :, 0].min(1) <= x1) & (T[:, :, 1].max(1) >= z0) & (T[:, :, 1].min(1) <= z1)
    U = unary_union([Polygon(t) for t in T[s_] if Polygon(t).area > 1e-4])     # (no slivers: a flat triangle's edge is no road)
    return U.buffer(CRACK).buffer(-CRACK)                       # (the cracks between its faces closed: road there)


_FLOOR = []


def floor_surface():
    """Every road's and the ground's top in the track's model, as it is (once)."""
    if not _FLOOR:
        tr = gltf_source.load_kn5(bp.TRACK)
        _FLOOR.append(bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(tr)
                                                 if n['name'].startswith(('1ROAD', '1GRASS')) and len(n['faces'])])))
    return _FLOOR[0]


def line_mesh(L, template, mi, band, surface, name):
    """A line of paint (band, x, z) laid on surface, as mesh name of L in material mi. How many faces."""
    if band is None or band.is_empty:
        return 0
    # (paint on the road only: never over the grass beside it)
    band = band.intersection(road_union(band.bounds)).buffer(0)
    if band.is_empty:
        return 0
    # (points along its edges every LINE_STEP, so it follows the road between them, not a long
    # straight face over its rises and dips)
    tris = fill(band.segmentize(LINE_STEP), most=.05)
    if not tris:
        return 0
    P = np.concatenate(tris)
    y = surface.ys(P) + .006
    if np.isnan(y).any():
        y[np.isnan(y)] = floor_surface().ys(P[np.isnan(y)]) + .006
    y = np.where(np.isnan(y), np.nanmedian(y), y)
    V = np.zeros((len(P), template['verts'].shape[1]))
    V[:, 0], V[:, 1], V[:, 2] = P[:, 0], y, P[:, 1]
    V[:, 4] = 1.
    V[:, 6], V[:, 7] = P[:, 0] / 10., P[:, 1] / 20.
    if V.shape[1] > 8:
        V[:, 8] = 1.
    F = np.arange(len(P)).reshape(-1, 3)
    a_, b_, c_ = V[F[:, 0], :3], V[F[:, 1], :3], V[F[:, 2], :3]
    up = np.cross(b_ - a_, c_ - a_)[:, 1]
    F = np.where((up < 0)[:, None], F[:, ::-1], F)[np.abs(up) > 1e-8]
    node = dict(template, name=name, children=[], material=mi, verts=V.astype('<f4'), faces=F.astype('<u2'))
    bp.sphere(node)
    L['root']['children'].append(node)
    return len(F)


def line_middle(n):
    """A line of paint's middle, (x, z) points along it (its points pair across it), and its width."""
    V = n['verts'][:, [0, 2]].astype(float)
    return (V[0::2] + V[1::2]) / 2, float(np.median(np.linalg.norm(V[1::2] - V[0::2], axis=1)))


def track_line_join(L):
    """The track's edge line past the 1500's pit entries, as it runs: TRACK_LINE_FROM cut where it bent
    into the entry, its middle carried straight on to TRACK_LINE_TO's top. (The join's band, x, z, and
    the whole line's middle, (x, z) points north to south.)"""
    from shapely.geometry import LineString
    a = next(n for n, _ in bp.meshes(L) if n['name'] == TRACK_LINE_FROM)
    b = next(n for n, _ in bp.meshes(L) if n['name'] == TRACK_LINE_TO)
    (ma, w), (mb, _) = line_middle(a), line_middle(b)
    ma, mb = ma[np.argsort(-ma[:, 1])], mb[np.argsort(-mb[:, 1])]
    up = ma[ma[:, 1] >= TRACK_LINE_CUT]
    d = np.polyfit(up[-4:, 1], up[-4:, 0], 1)
    start = np.array([np.polyval(d, TRACK_LINE_CUT), TRACK_LINE_CUT])
    band = LineString([tuple(start), tuple(mb[0])]).buffer(w / 2, cap_style=2)
    return band, np.r_[up, [start], mb]


def beyond_fence(shift=0.):
    """The side of the hotlap's fence's line (see HOTLAP_FENCE_LINE) beyond it, shift further on, x, z."""
    from shapely.geometry import Polygon
    a, b = (np.array(q, float) for q in HOTLAP_FENCE_LINE)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])                                         # (its north side)
    if n[1] < 0:
        n = -n
    a, b = a + n * shift - t * 200., b + n * shift + t * 200.
    from shapely.geometry import box
    return Polygon([tuple(a), tuple(b), tuple(b + n * 300.), tuple(a + n * 300.)]).intersection(box(HOTLAP_FENCE[0], -200., 400., 400.))


def beyond_lane(track, pad):
    """The side of the pit lane along the building beyond its outer edge (HOTLAP_EDGE out from the
    building's front), along the building and past its ends, x, z: the apron the lane runs beside."""
    from shapely.geometry import Polygon
    _, p0, d, nrm, xa, xb = garages(pad, track)
    a0, a1 = (xa - 40. - p0[0]) / d[0], (xb + 6. - p0[0]) / d[0]
    o0, o1 = HOTLAP_EDGE, HOTLAP_EDGE + 60.
    return Polygon([tuple(p0 + d * a + nrm * o) for a, o in ((a0, o0), (a1, o0), (a1, o1), (a0, o1))])


def beside_track():
    """East of the track's edge from where the hotlap's lane splits off it to the nose (HOTLAP_DIVIDE),
    x, z: the lane's side."""
    from shapely.geometry import Polygon
    (ax, az), (bx, bz) = HOTLAP_DIVIDE
    ext = 30.
    return Polygon([(ax, az + ext), (ax, az), (bx, bz), (bx, bz - ext), (bx - 60., bz - ext), (ax - 60., az + ext)]).buffer(0)


def road_surface(track, names):
    """The tops of the roads named, as one surface: what a line along them is laid on."""
    return bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track) if n['name'] in names and len(n['faces'])]))


def hotlap_lane(track, middle, pad=None):
    """The hotlap's pit lane, x, z, from its fork off the track to the fence (see HOTLAP_LANE_Z): on the
    track's mesh outside its edge line (middle, its points north to south), then on its own roads."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    z0 = HOTLAP_LANE_Z[0]
    big = ((-80., 200.), (-80., 120.))
    on_track = roads_shape(track, big, lambda n: n == '1ROAD_3').intersection(beside_track()).intersection(box(-80., z0, 200., HOTLAP_SPLIT + 5.))
    own = roads_shape(track, big, lambda n: n in HOTLAP_ROADS)
    if pad is not None:
        own = own.difference(beyond_lane(track, pad))                   # (not out over the apron)
    lane = unary_union([on_track, own]).buffer(.05).buffer(-.05)
    # (cut well past the fence's line: its lines are cut on it, straight)
    return lane.difference(beyond_fence(4.)).buffer(0)


def pit_lines(track, pad=None):
    """The pit lanes' lines, yellow, along both sides of each from the track to its pit boxes: the
    1000's along its lane as it is now (widened, its mouth rounded), the 1500's along its entry;
    the track's old lines across their mouths taken out. How many faces painted."""
    import io, copy, urllib.parse
    from PIL import Image
    from shapely.geometry import box
    from shapely.ops import unary_union
    path = os.path.join(bp.MODELS, 'velopark_lines.glb')
    L = gltf_source.load_kn5(path)
    if any(n['name'] == PIT_LINE_MESH for n, _ in bp.meshes(L)):
        return 0
    pad = pad if pad is not None else gltf_source.load_kn5(bp.PADDOCK)       # (the paddock as restyled so far)
    # the yellow paint: the white's texture coloured, its alpha as it is
    white = next(m for m in L['materials'] if m['name'] == 'LinePaint')
    tex = next(t for t in L['textures'] if t['name'] == white['textures'][0]['name'])
    img = Image.open(io.BytesIO(tex['data'])).convert('RGBA')
    a = np.asarray(img).astype(float)
    a[..., :3] *= np.array(PIT_PAINT_COLOUR)
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8), 'RGBA')
    data = kn5io.dds(img, 'DXT5')
    open(os.path.join(TEX, PIT_PAINT_TEX), 'wb').write(data)
    L['textures'].append(dict(name=PIT_PAINT_TEX, active=1, data=data))
    yellow = copy.deepcopy(white)
    yellow['name'] = PIT_PAINT_MAT
    for t in yellow['textures']:
        t['name'] = PIT_PAINT_TEX
    L['materials'].append(yellow)
    mi = len(L['materials']) - 1
    others = [gltf_source.load_kn5(os.path.join(bp.MODELS, f)) for f in ('velopark_kerbs.glb', 'velopark_runoffs.glb')
              if os.path.exists(os.path.join(bp.MODELS, f))]
    template = next(n for n, _ in bp.meshes(L) if n['name'].startswith('LINE_') and len(n['verts']))
    made = 0
    road_top = bp.Surface(np.concatenate([bp.triangles(n, m) for k_ in [track, pad] + others for n, m in bp.meshes(k_)
                                          if n['name'].startswith(('1ROAD', '1KERB')) and len(n['faces'])]))
    # the car parks' bays, white, on their asphalt as it lies
    import car_parks as cp
    from restyle_paddock import PARK_MESH
    _area = ((-60., 200.), (-140., -20.))
    lots = [n for n, _ in bp.meshes(pad) if n['name'].startswith(PARK_MESH) and len(n['faces'])]
    for n, _ in list(bp.meshes(L)):
        if n['name'] == cp.PARK_LINES and len(n['faces']):
            # (off its grass, its ramp and the apron over its step)
            import restyle_paddock as _rp
            clip_in(n, cp.park_islands().buffer(.5).union(cp.ramp_shape()).union(_rp.apron_shape(track, pad).buffer(.3)))
    if lots:
        lines_ = cp.east_bays()
        top = bp.Surface(np.concatenate([bp.triangles(n, np.eye(4)) for n in lots]))
        made += line_mesh(L, template, next(i for i, m in enumerate(L['materials']) if m['name'] == 'LinePaint'),
                          bands_of(lines_, cp.BAY_LINE), top, cp.BAYS_MESH)
    # the 1000's lane as it is now (widened, its mouth rounded), and the track by its mouth; the
    # track's old line across its mouth out (not the pit boxes' paint on the lane)
    lane = lane_shape(track, every=True)[1].intersection(box(-120., PIT_LINE_FROM - 1., 0., 260.))
    trk = unary_union([roads_shape(track, MOUTH_BOX, lambda n: n != '1ROAD_24')] + [roads_shape(k_, MOUTH_BOX) for k_ in others])
    within = box(-120., PIT_LINE_FROM, 0., 260.)
    mouth = lane.buffer(.35).intersection(within)
    for n, _ in list(bp.meshes(L)):
        if n['name'].startswith('LINE_') and len(n['faces']):
            clip_in(n, mouth)
    lines_1000 = lane_bands(lane, trk, within, sides=2, smooth=1.2, clear=PIT_1000_CLEAR, lines=True)
    ends_1000 = [(e_[0], e_[1], True, nm) for e_, nm in ((white_end(L, MOUTH_LINES[0], west=False), MOUTH_LINES[0]),
                                                          (white_end(L, MOUTH_LINES[1], west=True), MOUTH_LINES[1])) if e_]
    whites_1000 = []
    roads_1000 = unary_union([roads_shape(track, MOUTH_BOX)] + [roads_shape(k_, MOUTH_BOX) for k_ in others])
    lines_1000 = carry_on(lines_1000, ends_1000, whites=whites_1000, roads=roads_1000, trim=trimmer(L))
    top_1000 = road_surface(track, ('1ROAD_24', '1ROAD_17'))
    made += line_mesh(L, template, mi, bands_of(lines_1000), top_1000, PIT_LINE_MESH)
    made += line_mesh(L, template, next(i for i, m_ in enumerate(L['materials']) if m_['name'] == 'LinePaint'),
                      bands_of(whites_1000, .12), top_1000, PIT_LINE_MESH + '_FORKS')
    # the track's edge line where it bent into the 1500's pit entry out (no white line across it)
    middle = None                                               # (the track's edge line by the 1500's entry: as it is)
    white = next(i for i, m in enumerate(L['materials']) if m['name'] == 'LinePaint')
    clip_in(next(n for n, _ in bp.meshes(L) if n['name'] == TRACK_LINE_FROM), box(-120., -500., 200., TRACK_LINE_CUT))
    # the road south from the 1500's entry to the car park, white
    (x0, x1), (z0, z1) = SOUTH_ROAD_AREA
    near = ((x0 - 5., x1 + 5.), (z0 - 5., z1 + 5.))
    lane = roads_shape(track, near, lambda n: n == SOUTH_ROAD).intersection(box(x0, z0 - 1., x1, z1 + 1.))
    trk = unary_union([roads_shape(track, near, lambda n: n != SOUTH_ROAD)] + [roads_shape(k_, near) for k_ in others + [pad]])
    south_lines = lane_bands(lane, trk, box(x0, z0, x1, z1), smooth=1.2, lines=True)
    made += line_mesh(L, template, mi, bands_of(south_lines), road_surface(track, (SOUTH_ROAD,)), SOUTH_ROAD_MESH)
    # the 1500's entry from the track to its pit boxes: its yellow lines carried on from the track's
    # white lines' ends either side of its mouth, and from the road south's where they meet it
    (x0, x1), (z0, z1) = PIT_1500_AREA
    near = ((x0 - 5., x1 + 5.), (z0 - 5., z1 + 5.))
    lane = roads_shape(track, near, lambda n: n == PIT_1500_ROAD).intersection(box(x0, z0, x1, z1 + 1.))
    trk = unary_union([roads_shape(track, near, lambda n: n == '1ROAD_3')] + [roads_shape(k_, near) for k_ in others])
    entry = lane_bands(lane, trk, box(x0, z0, x1, z1), smooth=1.2, clear=.15, lines=True)
    ends = [white_end(L, TRACK_LINE_FROM, north=False), white_end(L, TRACK_LINE_TO, north=True)]
    ends = [e_ and (e_[0], e_[1], True, nm) for e_, nm in zip(ends, (TRACK_LINE_FROM, TRACK_LINE_TO))]
    side_ends = []
    for g in south_lines:
        c = np.array(g.coords)
        j = int(np.argmax(c[:, 1]))
        t_ = c[j] - c[j - 1 if j == len(c) - 1 else j + 1]
        side_ends.append((c[j], t_ / max(np.linalg.norm(t_), 1e-9), True))
    whites, side_yellow = [], []
    entry_top = road_surface(track, (PIT_1500_ROAD, '1ROAD_3', SOUTH_ROAD))
    fork_roads = unary_union([roads_shape(track, near)] + [roads_shape(k_, near) for k_ in others + [pad]])
    entry = carry_on(entry, ends, whites=whites, roads=fork_roads, trim=trimmer(L))
    entry = carry_on(entry, side_ends, whites=side_yellow, roads=fork_roads) + side_yellow
    entry = east_turn(entry)
    made += line_mesh(L, template, mi, bands_of(entry), entry_top, PIT_1500_MESH)
    made += line_mesh(L, template, white, bands_of(whites, .12), entry_top, PIT_1500_MESH + '_FORKS')
    # the hotlap's, from its fork off the track to the fence
    lane = hotlap_lane(track, middle, pad)
    # (the track beside it, its lines from the split and the nose; none along the opening between)
    from shapely.geometry import LineString
    trk = roads_shape(track, ((-80., 200.), (20., 120.)), lambda n: n == '1ROAD_3').difference(beside_track())
    within = box(-80., -120., 200., HOTLAP_SPLIT + .3).difference(beyond_fence())      # (its lines on from the white line's end)
    within = within.difference(LineString(HOTLAP_DIVIDE).buffer(HOTLAP_IN + .1, cap_style=2))
    fork = None
    for n, _ in list(bp.meshes(L)):
        if n['name'] == HOTLAP_EDGE_LINE and len(n['faces']):
            clip_in(n, box(-120., -500., 200., HOTLAP_SPLIT))           # (white till the split, yellow on)
    # in front of the building: straight, from the white line across at its west end to its east end
    from shapely.geometry import LineString, Polygon
    _, p0, d, nrm, xa, xb = garages(pad, track)
    sa, sb = (HOTLAP_START - p0[0]) / d[0], (xb - p0[0]) / d[0]
    edge, row_at = hotlap_row(track, pad)                              # (the road's outer edge, the row's far side)
    # (the lane there exactly from the building's front to the road's edge, straight, so its lines
    # carry on from the curves into the straight ones as one)
    across = lambda a0, a1, o0, o1: Polygon([tuple(p0 + d * a0 + nrm * o0), tuple(p0 + d * a1 + nrm * o0),
                                             tuple(p0 + d * a1 + nrm * o1), tuple(p0 + d * a0 + nrm * o1)])
    lane = lane.difference(across(sa, sb, -30., 30.)).union(across(sa - .01, sb, 0., edge)).buffer(.01).buffer(-.01)
    found = lane_bands(lane, trk, within, fork=fork, smooth=.6, bumps=.05, clear=-(HOTLAP_IN + .15), inset=HOTLAP_IN, lines=True)
    # (carried on from the white lines where they end: the track's edge line at the split, its line at the nose)
    nose = white_end(L, 'LINE_4', north=True)
    nose_whites = []
    nose_roads = unary_union([roads_shape(track, ((-40., 10.), (20., 60.)))] + [roads_shape(k_, ((-40., 10.), (20., 60.))) for k_ in others])
    bands = bands_of(carry_on(found, [white_end(L, HOTLAP_EDGE_LINE, north=False), nose and (nose[0], nose[1], True, 'LINE_4', (0., 1.))],
                              whites=nose_whites, roads=nose_roads, trim=trimmer(L)))
    lane_top = road_surface(track, ('1ROAD_3',) + tuple(HOTLAP_ROADS))
    made += line_mesh(L, template, mi, bands, lane_top, HOTLAP_MESH)
    made += line_mesh(L, template, white, bands_of(nose_whites, .12), lane_top, HOTLAP_MESH + '_FORK')
    row = LineString([tuple(p0 + d * sa + nrm * row_at), tuple(p0 + d * sb + nrm * row_at)]).buffer(PIT_LINE_W / 2, cap_style=2)
    made += line_mesh(L, template, mi, row, lane_top, HOTLAP_ROW_MESH)
    across = LineString([tuple(p0 + d * sa + nrm * (PIT_LINE_IN - PIT_LINE_W / 2)),
                         tuple(p0 + d * sa + nrm * (edge - PIT_LINE_IN + PIT_LINE_W / 2))]).buffer(PIT_LINE_W, cap_style=2)
    made += line_mesh(L, template, white, across, lane_top, HOTLAP_START_MESH)
    made += hairpin_fork(L, template, white, track, others)
    print('  %d lines of paint cut back to the roads under them' % paint_on_roads(L))
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(L, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)
    return made


def paint_on_roads(L):
    """Every line of paint of L, the track's own too, cut to the roads under it (none of it over the
    grass beside them). How many lines were cut."""
    from shapely.geometry import box
    cut = 0
    for n, m in list(bp.meshes(L)):
        if not len(n['faces']) or not np.allclose(m, np.eye(4)):
            continue
        T = bp.triangles(n, m)[:, :, [0, 2]]
        if not len(T):
            continue
        (x0, z0), (x1, z1) = T.min((0, 1)) - 1., T.max((0, 1)) + 1.
        before = len(n['faces'])
        # (only a line with some of it off the road; what is off it, the road's hairline cracks not)
        c = T.mean(1)
        Pc = np.concatenate([c] + [T[:, i] + (c - T[:, i]) * .2 for i in range(3)])     # (where tools/check_holes.py looks)
        R = road_union((x0, z0, x1, z1))
        import shapely
        if shapely.contains_xy(R.buffer(CRACK), Pc[:, 0], Pc[:, 1]).all():
            continue
        out = box(x0, z0, x1, z1).difference(R).buffer(-CRACK / 2).buffer(CRACK / 2)
        if out.is_empty:
            continue
        clip_in(n, out)
        cut += len(n['faces']) != before
    # (and what is left of it off the road by tools/check_holes.py's own measure out, face by face)
    import check_holes as ch
    ch._ROADS.clear()
    for n, m in list(bp.meshes(L)):
        if not len(n['faces']) or not np.allclose(m, np.eye(4)):
            continue
        F = n['faces'].astype(int)
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        T = n['verts'][:, :3].astype(float)[F]
        off = ch.off_road(ch.paint_samples(T)[:, [0, 2]]).reshape(4, -1).any(0)
        if off.any():
            n['faces'] = F[~off].astype('<u2')
            bp.sphere(n)
            cut += 1
    return cut


def hairpin_fork(L, template, white, track, others):
    """The track's outer line past the hairpin (FORK_FROM, from its south end) forking in two (FORK_LINE:
    along the kerb, and along the road beyond it), both leaving it as it goes and easing into their
    own ways: FORK_LINE out in FORK_BOX, laid again from there as FORK_MESH. How many faces."""
    from shapely.geometry import box
    n = next((n for n, _ in bp.meshes(L) if n['name'] == FORK_LINE and len(n['faces'])), None)
    end = white_end(L, FORK_FROM, north=False)
    if n is None or end is None:
        return 0
    (x0, x1), (z0, z1) = FORK_BOX
    clip_in(n, box(x0, z0, x1, z1))
    # (each branch where it is cut, and the way it goes on from there)
    V = n['verts'][np.unique(n['faces'].astype(int))][:, [0, 2]].astype(float)
    cut = V[(V[:, 0] > x1 - .05) & (V[:, 0] < x1 + 2.5) & (V[:, 1] > z0) & (V[:, 1] < z1)]
    if not len(cut):
        return 0
    zs = np.sort(cut[:, 1])
    gap = int(np.argmax(np.diff(zs))) if len(zs) > 1 else 0
    split = (zs[gap] + zs[gap + 1]) / 2 if len(zs) > 1 else zs[0]
    e, t = end
    from shapely.geometry import LineString
    lines = []
    for side in (cut[:, 1] < split, cut[:, 1] >= split):
        Q = cut[side]
        if len(Q) < 3:
            continue
        at = Q[Q[:, 0] < x1 + .1].mean(0)
        d = np.linalg.svd(Q - Q.mean(0))[2][0]
        d = d if d[0] > 0 else -d
        lines.append(LineString([tuple(p) for p in hermite(e, t, at, d)]))
    top = bp.Surface(np.concatenate([bp.triangles(n_, m) for k_ in [track] + others for n_, m in bp.meshes(k_)
                                     if n_['name'].startswith(('1ROAD', '1KERB')) and len(n_['faces'])]))
    w = .12                                                     # (the track's white lines' width)
    return line_mesh(L, template, white, bands_of(lines, w), top, FORK_MESH)


def pit_paint(before, after):
    """The pit boxes' paint (GRID_PITS of the lines' model) by the boxes before (x, z) moved with them
    to after, piece by piece."""
    import urllib.parse
    from scipy.spatial import cKDTree
    from grade_paths import pieces as parts
    path = os.path.join(bp.MODELS, 'velopark_lines.glb')
    L = gltf_source.load_kn5(path)
    n = next((n for n, _ in bp.meshes(L) if n['name'] == 'GRID_PITS'), None)
    if n is None:
        return 0
    V = n['verts'].astype(float)
    lab = parts(n['faces'].astype(int), len(V))
    tree = cKDTree(before)
    shifted = 0
    road = bp.Surface(np.concatenate([bp.triangles(n_, m_) for n_, m_ in bp.meshes(gltf_source.load_kn5(bp.TRACK))
                                      if n_['name'].startswith('1ROAD') and len(n_['faces'])]))
    for i in np.unique(lab):
        sel = lab == i
        dist, j = tree.query(V[sel][:, [0, 2]].mean(0))
        if dist > 2.5:
            continue                                                    # (not a hotlap box's)
        V[sel, 0] += after[j, 0] - before[j, 0]
        V[sel, 2] += after[j, 1] - before[j, 1]
        y = road.ys(V[sel][:, [0, 2]])
        V[sel, 1] = np.where(np.isnan(y), V[sel, 1], y + PAINT_UP)
        shifted += 1
    if not shifted:
        return 0
    n['verts'] = V.astype(n['verts'].dtype)
    bp.sphere(n)
    doc, _ = gltf_source.read_glb(path)
    uris = {im['name']: urllib.parse.unquote(im['uri']) for im in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(L, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)
    return shifted


def hotlap_pits(track, pad):
    """The hotlap's and the 2500's pit boxes in the middle of the pit boxes' row in front of the building
    (see HOTLAP_BOX_ROW), and their pit lanes' racing lines along the middle of the pit lane past them
    (see HOTLAP_AI), not out over the car park. How many boxes moved."""
    import struct
    from shapely.geometry import LineString, Point, box
    road13 = roads_shape(track, ((-80., 200.), (-80., 120.)), lambda n: n in HOTLAP_ROADS[:1])
    road14 = roads_shape(track, ((-80., 200.), (-80., 120.)), lambda n: n == HOTLAP_BOX_ROAD).difference(beyond_lane(track, pad))
    top = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                     if n['name'].startswith('1ROAD') and len(n['faces'])]))

    def middle(shape, p, across):
        """The middle of shape across p (along across, both ways), or None."""
        cut = LineString([p - across * 12., p + across * 12.]).intersection(shape)
        parts = sorted([g for g in getattr(cut, 'geoms', [cut]) if not g.is_empty and g.length > .5], key=lambda g: g.distance(Point(p)))
        if not parts or parts[0].distance(Point(p)) > 3.:
            return None
        c = np.array(parts[0].coords)
        return c.mean(0)
    inside, p0, d, nrm, xa, xb = garages(pad, track)
    edge, row_at = hotlap_row(track, pad)
    moved = 0
    for lay in HOTLAP_LAYOUTS:
        glb = os.path.join(bp.MODELS, lay + '.glb')
        doc, binary = gltf_source.read_glb(glb)
        nodes = [n for n in doc['nodes'] if n['name'].startswith('AC_PIT_') and 'matrix' in n]
        if not nodes:
            continue
        P = np.array([[n['matrix'][12], n['matrix'][14]] for n in nodes])
        along = (P - p0) @ d
        out = p0 + np.outer(along, d) + nrm * ((PIT_LINE_IN + row_at) / 2)      # (between the front's line and the row's)
        if np.abs(out - P).max() < .05:
            continue                                                    # (in their row already)
        for n, q in zip(nodes, out):
            n['matrix'][12], n['matrix'][14] = float(q[0]), float(q[1])
        gltf_source.write_glb(glb, doc, binary)
        moved += len(nodes)
        if lay == HOTLAP_LAYOUTS[0]:
            pit_paint(P, out)
        # its pit lane's racing line along the pit lane's middle
        path = os.path.join(bp.ROOT, 'source', 'track', lay, 'ai', 'pit_lane.ai')
        b = bytearray(open(path, 'rb').read())
        n_ = struct.unpack('<i', b[4:8])[0]
        A = np.frombuffer(bytes(b[16:16 + n_ * 20]), '<f4').reshape(n_, 5).astype(float)
        i0, i1 = HOTLAP_AI
        pts = [A[i0, [0, 2]]]
        # down its road, as it ran, kept to its middle
        for i in range(i0 + 1, i1):
            q = A[i, [0, 2]]
            if not road13.buffer(.3).contains(Point(*q)):
                break
            t = A[i + 1, [0, 2]] - A[i - 1, [0, 2]]
            t /= np.linalg.norm(t)
            m = middle(road13, q, np.array([-t[1], t[0]]))
            if m is not None:
                pts.append(m)
        # in front of the building, the middle of the lane beyond the boxes' row
        from shapely.geometry import Polygon as _Poly
        a_, b_ = p0 + d * ((xa - 10. - p0[0]) / d[0]), p0 + d * ((xb + 10. - p0[0]) / d[0])
        behind = _Poly([tuple(a_ + nrm * row_at), tuple(b_ + nrm * row_at), tuple(b_ - nrm * 20.), tuple(a_ - nrm * 20.)])
        front = road14.difference(behind)
        for s_ in np.arange((xa - p0[0]) / d[0], (xb - p0[0]) / d[0] + .01, 1.):
            q = p0 + d * s_ + nrm * (row_at + 1.)
            m = middle(front, q, nrm)
            if m is not None:
                pts.append(m)
        # and up the road from their east end to the fence
        zf = p0[1] + d[1] * ((xb - p0[0]) / d[0])                        # (the garages' front at their east end)
        east = road14.difference(inside).intersection(box(xb - 8., zf + 1.5, 400., HOTLAP_FENCE[1] + 3.))
        for z in np.arange(zf + 2., A[i1, 2], 1.):
            m = middle(east, np.array([np.mean(east.bounds[::2]), z]), np.array([1., 0.]))
            if m is not None:
                pts.append(m)
        pts.append(A[i1, [0, 2]])
        Q = np.array(pts)
        # its corners rounded: resampled finely, each point the mean of those within HOTLAP_ROUND / 2
        dq = np.r_[0., np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        u = np.arange(0., dq[-1], .25)
        Q = np.c_[np.interp(u, dq, Q[:, 0]), np.interp(u, dq, Q[:, 1])]
        k = int(HOTLAP_ROUND / .25) | 1
        for _ in range(2):
            Q = np.c_[np.convolve(np.pad(Q[:, 0], k // 2, mode='edge'), np.ones(k) / k, 'valid'),
                      np.convolve(np.pad(Q[:, 1], k // 2, mode='edge'), np.ones(k) / k, 'valid')]
        dq = np.r_[0., np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        u = np.linspace(0., dq[-1], i1 - i0 + 1)
        new = np.c_[np.interp(u, dq, Q[:, 0]), np.interp(u, dq, Q[:, 1])]
        # (eased in from the line as it ran at either end)
        e = np.clip(np.minimum(np.arange(len(new)), np.arange(len(new))[::-1]) / 8., 0., 1.)
        e = (e * e * (3 - 2 * e))[:, None]
        new = A[i0:i1 + 1][:, [0, 2]] * (1 - e) + new * e
        y = top.ys(new)
        y = np.where(np.isnan(y), np.interp(np.arange(len(new)), [0, len(new) - 1], [A[i0, 1], A[i1, 1]]), y)
        A[i0:i1 + 1, 0], A[i0:i1 + 1, 1], A[i0:i1 + 1, 2] = new[:, 0], y, new[:, 1]
        A[:, 3] = np.r_[0., np.cumsum(np.linalg.norm(np.diff(A[:, [0, 2]], axis=0), axis=1))]
        b[16:16 + n_ * 20] = A.astype('<f4').tobytes()
        open(path, 'wb').write(bytes(b))
    return moved


def box_x(z, lane=None):
    """Where the vp_1000's pit boxes stand across its lane (x) at z: PIT_IN in from the walls' line, or,
    where its west yellow line runs further in, PIT_BOX_CLEAR in from that."""
    from shapely.geometry import LineString
    z = np.atleast_1d(np.asarray(z, float))
    x = wall_line(z) - PIT_IN
    if lane is None:
        lane = lane_shape(gltf_source.load_kn5(bp.TRACK), every=True)[1]
    for i, zz in enumerate(z):
        if zz < PIT_LINE_FROM:
            continue
        cut = LineString([(-80., zz), (-30., zz)]).intersection(lane)
        if cut.is_empty:
            continue
        west = max(q[0] for g in getattr(cut, 'geoms', [cut]) for q in g.coords)
        x[i] = min(x[i], west - PIT_LINE_IN - PIT_BOX_CLEAR)
    return x


def pit_boxes():
    """The vp_1000's pit boxes, and their painted boxes, moved to PIT_IN in from the walls' line, in
    one row. How many."""
    import kn5tree
    path = os.path.join(bp.MODELS, PIT_LAYOUT + '.glb')
    k = gltf_source.load_kn5(path)
    moves = []
    lane = lane_shape(gltf_source.load_kn5(bp.TRACK), every=True)[1]
    for n, m in kn5tree.walk(k['root']):
        if n['kind'] == 1 and n['name'].startswith('AC_PIT_') and m[3, 2] > PIT_FROM:
            dx = float(box_x(m[3, 2], lane)[0]) - m[3, 0]
            if abs(dx) < .05:
                continue
            M4 = np.array(n['matrix'], float).reshape(4, 4)
            M4[3, 0] += dx                              # (the layout's dummies stand under its root as they are)
            n['matrix'] = list(M4.ravel())
            moves.append((m[3, 0], m[3, 2], dx))
    if not moves:
        return 0
    bp.save(k, path)
    lines = os.path.join(bp.MODELS, 'velopark_lines.glb')
    L = gltf_source.load_kn5(lines)
    at = np.array([(x, z) for x, z, _ in moves])
    road = bp.Surface(np.concatenate([bp.triangles(n_, m_) for n_, m_ in bp.meshes(gltf_source.load_kn5(bp.TRACK))
                                      if n_['name'].startswith('1ROAD') and len(n_['faces'])]))
    for n, m in bp.meshes(L):
        if n['name'] != 'GRID_PITS' or not len(n['faces']):
            continue
        from grade_paths import pieces as parts
        V = n['verts'][:, :3].astype(float)
        lab = parts(n['faces'].astype(int), len(V))
        for piece in np.unique(lab):
            idx = np.nonzero(lab == piece)[0]
            c = V[idx][:, [0, 2]].mean(0)
            d = np.hypot(*(at - c).T)
            j = int(np.argmin(d))
            if d[j] < PIT_PAINT:
                n['verts'][idx, 0] = (V[idx, 0] + moves[j][2]).astype(n['verts'].dtype)
                y = road.ys(np.c_[V[idx, 0] + moves[j][2], V[idx, 2]])
                n['verts'][idx, 1] = np.where(np.isnan(y), V[idx, 1], y + PAINT_UP).astype(n['verts'].dtype)
        bp.sphere(n)
    bp.save(L, lines)
    return len(moves)


def pit_pass():
    """The vp_1000's pit lane's racing line moved to PIT_PASS in from the walls' line, past its pit
    boxes, within PIT_PASS_Z. How many points moved."""
    import struct
    path = os.path.join(bp.ROOT, 'source', 'track', PIT_LAYOUT, 'ai', 'pit_lane.ai')
    b = bytearray(open(path, 'rb').read())
    n = struct.unpack('<i', b[4:8])[0]
    P = np.frombuffer(bytes(b[16:16 + n * 20]), '<f4').reshape(n, 5).astype(float)
    z0, z1 = PIT_PASS_Z
    w = np.clip(np.minimum(P[:, 2] - z0, z1 - P[:, 2]) / PIT_PASS_EASE, 0., 1.)
    w = w * w * (3 - 2 * w) * ((P[:, 0] > -62.) & (P[:, 0] < -40.))
    x = P[:, 0] + w * (box_x(P[:, 2]) - (PIT_PASS - PIT_IN) - P[:, 0])
    moved = np.abs(x - P[:, 0]) > .02
    if not moved.any():
        return 0
    for i in np.nonzero(moved)[0]:
        b[16 + i * 20:16 + i * 20 + 4] = struct.pack('<f', x[i])
    open(path, 'wb').write(bytes(b))
    return int(moved.sum())


def main():
    track = gltf_source.load_kn5(bp.TRACK)
    pad = gltf_source.load_kn5(bp.PADDOCK)
    painted = pit_lines(track, pad)
    if painted:
        print("the pit lanes' lines, %d faces" % painted)
    passed = pit_pass()
    if passed:
        print("%d points of the %s's pit lane racing line moved past its pit boxes" % (passed, PIT_LAYOUT))
    boxes = hotlap_pits(gltf_source.load_kn5(bp.TRACK), pad)
    if boxes:
        print("%d of the hotlap's and the 2500's pit boxes in a row in front of the building, their pit lanes past them" % boxes)
    moved = pit_boxes()
    if moved:
        print("%d of the %s's pit boxes along its yellow line, with their paint" % (moved, PIT_LAYOUT))
    if not (painted or passed or boxes or moved):
        print('the pits are painted already')


if __name__ == '__main__':
    main()
