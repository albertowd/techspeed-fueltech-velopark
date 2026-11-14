"""The track draped over the real ground, as Race Track Builder made it, in every source.

    python tools/drape_track.py [--dry-run]

The real ground is tools/far_terrain.py's, from every height there is, between its 10 m points a
smooth (cubic) surface. On it:

- the real ground has the middle buildings' platform on it (PLATFORM: its heights' filter took
  them out with the buildings'), the pit lanes ramping down from it to both tracks;
- the near ground (1GRASS_0 and 1GRASS_PATHS of source/models/velopark.glb) is the real ground,
  every point of it, under the roads too; its faces within NEAR of the roads cut by a grid of CUT
  metres first, so it follows the real ground between its points there; and so is every other
  piece of grass (the kerbs' beds, islands and run-offs);
- the roads (every 1ROAD_ mesh of asphalt: the track, the paths, the paddock's and the car
  park's asphalt) and the paving lie ROAD over it, every point of their
  tops, their skirts the same depth under their tops as they were;
- the kerbs, the paint and the rest driven on move as the roads round them moved (the change
  over the roads, carried out from them to the nearest), so they keep their shapes on them;
- the layouts' AI lines and start points move so too;
- everything else standing on the ground moves by what the ground's top (it and the roads) moved
  under it, the trees of BeamNG's forest each to the ground where it stands, the Custom Shaders
  Patch's lights with their poles.

So the grass never comes over a road, and every edge drops the road's own height onto the grass,
which pulls a kart off as a tyre goes over, as the real track's does, and nowhere more. Run
tools/far_terrain.py after, to meet the near ground's edge, and tools/build_outside.py, built on
the ground as it is (velopark_outside.glb is left to it). Marked done in
tools/reference/elevation/draped.json.
"""
import datetime, glob, json, os, re, struct, sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

import build_paths as bp
from grade_paths import pieces
import build_trees as bt
import far_terrain as ft
import gltf_source
import join_paddock as jp
import level_ground as lg
import settle_objects as so

DONE = os.path.join(lg.ev.OUT, 'draped.json')
ROAD = .02               # metres a road lies over the real ground: its own height
RAMP = .12               # the slope from an edge of what is driven on down to the grass, metres a metre
RAMP_W = (.12, 1.)       # the narrowest and widest a ramp is, metres
PIECE = 4.               # metres: a piece of a thing standing on the ground smaller than this across moves as one
# the middle buildings stand on a platform, a metre over the 1500's side, the pit lanes ramping down
# from it to both tracks and the 1000's pit lane up to it from the storage building's end: the real
# ground's filter took its heights out with the buildings' (there are no roofs among them), and the
# ground round filled it in, half a metre lower. In this box (x, z) the raw heights, smoothed over
# PLATFORM_SMOOTH, raise the real ground where they are higher, easing out over PLATFORM_EDGE
# (from the race control tower north: south of it the paddock building's roof is among the heights);
# never more than PLATFORM_MAX, a platform's height, so no roof or tree top gets in
PLATFORM = ((-88., -22.), (62., 165.))
WORK = ((-110., 10.), (-60., 185.))     # the grid the platform is worked out on, x and z
PLATFORM_SMOOTH, PLATFORM_EDGE, PLATFORM_MAX = 4., 8., .7
# the pits and the shed north of them stand PIT_RISE over the 1000's pit lane at the foot of its
# ramp (screenshots 01 and 22 to 24 in tools/reference/site): the lane runs along the storage
# building on the ground as it is, then up a smooth ramp from the building's south end (RAMP_FOOT,
# z) to the pit building's north end (RAMP_TOP); on its west the shed's floor (the kart platform
# under the tent's arches) at the pits' height, from FACE_IN in from LANE_EDGE (x, z at both ends;
# x grows west), the line the pit building's and the storage building's east walls stand on, beside
# the lane, the pit boxes in a row along it, north to SHED_NORTH (z), ending in steep concrete faces
# down to the lane and to the grass north of it (tools/build_outside.py makes them, and the
# floor's concrete over the edge). PIT_AREA is the pits' (both lanes under the roof, and the
# workshop), x, z round it; the ground eases down from it over PIT_EASE, from the shed's floor
# over SHED_EDGE to its faces, and over PIT_EASE west of SHED_LONG (x), where the 1500's pit exit
# and the road beside the storage building run down from it. Within TRACK_KEEP of any layout's racing
# line (and of the pit lanes south of KEEP_PITS_SOUTH) the ground is as it was, easing up to that
# over TRACK_EASE more, so the tracks race as before
PIT_RISE = 1.
RAMP_FOOT, RAMP_TOP = 141.5, 126.
LANE_EDGE = ((-48.78, 126.), (-47.26, 142.))     # (the pit building's and the storage building's east walls' line)
FACE_IN, LANE_EASE = 1.2, 8.
SHED_NORTH = 133.7          # (its concrete, with the face over its edge, to z 135.5)
PIT_AREA = [(-28.5, 124.5), (-30.9, 94.5), (-62.6, 97.), (-60.2, 126.5)]
SHED_WEST = -35.6
PIT_EASE, SHED_EDGE, SHED_LONG = 16., .5, -37.
TRACK_KEEP, TRACK_EASE = 5., 6.
KEEP_PITS_SOUTH = 60.    # z: the layouts' pit lanes south of it (the hotlap's and the 2500's, by the stands) are kept as racing lines are
# the raised ground goes on south of the pits, past the race control tower, the north paddock
# building and the bar to the two wooden stands by the 1500's straight (MIDDLE_AREA, x, z round
# it), falling MIDDLE_FALL a metre south from the pits' height at PIT_SOUTH (z), as the ground
# round does; easing down from it as from the pits
MIDDLE_AREA = [(-27., 94.5), (-63., 97.), (-68., 60.), (-67., 30.), (-62., 14.), (-45., -5.), (-38., -40.),
               (-10., -42.), (-20., 0.), (-27., 15.)]
PIT_SOUTH, MIDDLE_FALL = 94.5, .009
NEAR = 15.               # metres from a road the ground's faces are cut
CUT = 3.                 # metres, the grid they are cut along
ROAD_CUT = 2.            # metres, the grid the roads' faces are cut along, so they follow the real ground
SINK = .03               # metres a tyre stack stands into the ground
IN_TYRE = .3             # of a tyre's width either way of its middle: a piece with its middle there is of its stack
STACK = .1               # metres apart the middles of the tyres of one stack may be (more, and the stacks of a wall chain into one)
PAINT = ('LinePaint', 'GridNumbers', 'PitLinePaint')   # the paint's materials, laid on the roads as they are
GRID = .25               # metres, the grid the changes are worked out on
CARRY = 12.              # metres from the roads their change is carried out to
SMOOTH = .5              # metres the carried change is smoothed over


ROADS = ('Tarmac Dark', 'Tarmac Light', 'Pavers')


def is_road(name, material):
    """A surface laid over the ground at ROAD: a road's or the paving's (by its material: some
    kerbs are 1ROAD_ meshes too, of the kerbs' materials)."""
    return material in ROADS and name.startswith(('1ROAD_', '1KERB_'))


def is_grass(name):
    """A piece of the ground: the near ground's, or the grass the kerbs' model lays in their beds,
    islands and run-offs (not the far ground, GRASS_1, which tools/far_terrain.py makes)."""
    return name.startswith('1GRASS')


def driven(name):
    """What is driven on or lies on it (kerbs, paint), moved with the roads."""
    return so.driven(name) and 'GRASS' not in name or name.startswith(('KERB', 'EDGE', 'PAINT', 'LINE')) or 'KERB' in name


def platform(real):
    """The real ground with the middle buildings' platform on it, as PLATFORM says."""
    from scipy.interpolate import griddata
    x, z, h = ft.heights()
    h = h - json.load(open(lg.DONE))['offset_m']
    (xa, xb), (za, zb) = PLATFORM
    m = 20.
    sel = (x > xa - m) & (x < xb + m) & (z > za - m) & (z < zb + m)
    X, Z = np.arange(WORK[0][0], WORK[0][1] + .1, 1.), np.arange(WORK[1][0], WORK[1][1] + .1, 1.)
    XX, ZZ = np.meshgrid(X, Z)
    R = griddata(np.c_[x[sel], z[sel]], h[sel], (XX, ZZ), method='linear')
    R = np.where(np.isnan(R), real(XX, ZZ), R)
    R = gaussian_filter(R, PLATFORM_SMOOTH)
    inside = np.minimum.reduce([XX - xa, xb - XX, ZZ - za, zb - ZZ])
    t = np.clip(inside / PLATFORM_EDGE, 0, 1)
    bump = np.clip(R - real(XX, ZZ), 0, PLATFORM_MAX) * t * t * (3 - 2 * t)
    # the 1000's pit lane down to the ground as it is north of its ramp, east of the face
    (xa_, za_), (xb_, zb_) = LANE_EDGE
    face = xa_ + (xb_ - xa_) * (ZZ - za_) / (zb_ - za_) + FACE_IN
    smooth = lambda u: (lambda c: c * c * (3 - 2 * c))(np.clip(u, 0., 1.))
    down = smooth((ZZ - RAMP_TOP) / (RAMP_FOOT - RAMP_TOP)) * (1 - smooth((face - XX) / LANE_EASE)) * (XX < face)
    bump = bump * (1 - down)
    base = real(XX, ZZ) + bump
    # the pits and the shed PIT_RISE over the lane at the ramp's foot
    fx = lambda zz: xa_ + (xb_ - xa_) * (zz - za_) / (zb_ - za_) + FACE_IN
    H = float(real(np.array([fx(RAMP_FOOT) - FACE_IN - 1.8]), np.array([RAMP_FOOT]))[0]) + PIT_RISE      # (on the lane's middle)
    import shapely
    from shapely.geometry import Polygon
    in_pits = shapely.contains_xy(Polygon(PIT_AREA), XX, ZZ)
    shed_poly = Polygon([(fx(RAMP_TOP - 1.5), RAMP_TOP - 1.5), (fx(SHED_NORTH), SHED_NORTH),
                         (SHED_WEST, SHED_NORTH), (SHED_WEST, RAMP_TOP - 1.5)])
    in_shed = shapely.contains_xy(shed_poly, XX, ZZ)
    w_pits = smooth(1 - distance_transform_edt(~in_pits) / PIT_EASE)
    d_shed = distance_transform_edt(~in_shed)
    w_shed = np.maximum(smooth(1 - d_shed / SHED_EDGE), smooth(1 - d_shed / PIT_EASE) * smooth((XX - SHED_LONG) / 2.))
    w_shed = w_shed * ~((XX < face) & (ZZ > RAMP_TOP - 1.))         # (not over the face onto the lane)
    # the racing lines kept as they are
    lines = []
    for f in glob.glob(os.path.join(bp.ROOT, 'source', 'track', '*', 'ai', '*_lane.ai')):
        b = open(f, 'rb').read()
        n = int.from_bytes(b[4:8], 'little')
        p = np.frombuffer(b, dtype='<f4', count=n * 5, offset=16).reshape(-1, 5)[:, [0, 2]].astype(float)
        if os.path.basename(f) == 'pit_lane.ai':
            p = p[p[:, 1] < KEEP_PITS_SOUTH]       # (the pit lanes by the stands, not those up to the pits)
        lines.append(p)
    d_line = cKDTree(np.concatenate(lines)).query(np.c_[XX.ravel(), ZZ.ravel()])[0].reshape(XX.shape)
    keep = smooth((d_line - TRACK_KEEP) / TRACK_EASE)
    lift = np.clip(H - base, 0, None) * np.maximum(w_pits, w_shed) * keep
    # and on south of them
    in_mid = shapely.contains_xy(Polygon(MIDDLE_AREA), XX, ZZ)
    w_mid = smooth(1 - distance_transform_edt(~in_mid) / PIT_EASE)
    Hs = H - MIDDLE_FALL * np.clip(PIT_SOUTH - ZZ, 0, None)
    lift = np.maximum(lift, np.clip(Hs - base, 0, None) * w_mid * keep)
    bump = bump + lift
    print("the middle buildings' platform: the ground raised up to %.2f m there; the pits and the shed at %.2f m, "
          "%.2f m over the 1000's pit lane at its ramp's foot" % (bump.max(), H, PIT_RISE))

    def ground(xx, zz):
        xx, zz = np.asarray(xx, float), np.asarray(zz, float)
        return real(xx, zz) + map_coordinates(bump, [zz - Z[0], xx - X[0]], order=1, cval=0.)
    return ground


def main():
    dry = '--dry-run' in sys.argv
    if os.path.exists(DONE):
        print('the track is draped already (%s)' % os.path.relpath(DONE, bp.ROOT))
        return
    real = platform(ft.real_ground(json.load(open(os.path.join(ft.SURROUNDINGS, 'surroundings.json'))), order=3))
    files = sorted(glob.glob(os.path.join(bp.MODELS, '*.glb')))
    # (and the parked karts, which tools/place_karts.py stands on the ground as it is after)
    files = [f for f in files if os.path.basename(f) not in so.SKIP + ('velopark_karts.glb',)]
    models = {f: gltf_source.load_kn5(f) for f in files}
    track = models[bp.TRACK]

    def mat(k, n):
        return k['materials'][n['material']]['name']
    # the roads' tops as they are, and the ground's top
    roads = [(f, n, m) for f, k in models.items() for n, m in bp.meshes(k)
             if len(n['faces']) and is_road(n['name'], mat(k, n)) and n['visible']]
    tops = []
    for f, n, m in roads:
        t = bp.top_of(n) if n['name'].startswith('1ROAD_PATH_') else bp.triangles(n, m)
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ln = np.linalg.norm(nn, axis=1)
        tops.append(t[(ln > 1e-9) & (np.abs(nn[:, 1]) > .5 * np.maximum(ln, 1e-9))])
    tops = np.concatenate(tops)
    old_road = bp.Surface(tops)
    grounds = [n for n, m in bp.meshes(track) if n['name'].startswith((bp.GROUND, bp.FILLER)) and len(n['faces'])]
    old_ground = np.concatenate([bp.triangles(n, np.eye(4)) for n in grounds])

    # the change over the roads, on a grid, carried out from them to the nearest, smoothed a little
    lo = tops[:, :, [0, 2]].min((0, 1)) - CARRY - NEAR
    hi = tops[:, :, [0, 2]].max((0, 1)) + CARRY + NEAR
    g = jp.Grid(lo, hi)
    g_top = g.top(tops)
    on = ~np.isnan(g_top)
    XX, ZZ = g.X, g.Z
    change = np.where(on, real(XX, ZZ) + ROAD - np.where(on, g_top, 0.), np.nan)
    dist, (jn, i_n) = distance_transform_edt(~on, return_indices=True)
    dist *= jp.R
    carried = change[jn, i_n]
    w = gaussian_filter(np.ones_like(carried), SMOOTH / jp.R)
    carried = gaussian_filter(carried, SMOOTH / jp.R) / w
    print('the roads move %+.2f to %+.2f m (nine in ten %+.2f to %+.2f)'
          % (np.nanmin(change), np.nanmax(change), *np.nanpercentile(change, [5, 95])))

    def road_change(x, z):
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        return map_coordinates(carried, [(z - g.z0) / jp.R, (x - g.x0) / jp.R], order=1, mode='nearest')

    def near_road(P):
        i = np.round((P[:, 0] - g.x0) / jp.R).astype(int)
        j = np.round((P[:, -1] - g.z0) / jp.R).astype(int)
        ok = (i >= 0) & (i < g.nx) & (j >= 0) & (j < g.nz)
        out = np.zeros(len(P), bool)
        out[ok] = dist[j[ok], i[ok]] <= NEAR
        return out

    # the near ground: the real ground, every point, its faces near the roads cut first
    # (all its meshes as one: cut one by one, a cut point on an edge two of them share would be in
    # one only, and a crack would open along it)
    jp.GRID = CUT
    root = track['root']
    Vs, Fs, off = [], [], 0
    for n in grounds:
        Vs.append(n['verts'].astype(float))
        Fs.append(n['faces'].astype(int) + off)
        off += len(n['verts'])
    first = grounds[0]
    for n in grounds[1:]:
        root['children'] = [c for c in root['children'] if c is not n]
    first['verts'], first['faces'] = np.concatenate(Vs), np.concatenate(Fs)
    first['name'] = bp.GROUND
    grounds = [first]
    for n in grounds:
        V, F, new, cut, Fc = jp.refine(n['verts'].astype(float), n['faces'].astype(int), near_road)
        y = real(V[:, 0], V[:, 2])
        print('%s: %d faces cut into %d; %d points, moved %+.2f to %+.2f m (nine in ten %+.2f to %+.2f)'
              % (n['name'], cut.sum(), len(Fc), len(V), (y - V[:, 1]).min(), (y - V[:, 1]).max(), *np.percentile(y - V[:, 1], [5, 95])))
        V[:, 1] = y
        f = np.cross(V[F[:, 1], :3] - V[F[:, 0], :3], V[F[:, 2], :3] - V[F[:, 0], :3])
        f[f[:, 1] < 0] *= -1
        N = np.zeros((len(V), 3))
        for k in range(3):
            np.add.at(N, F[:, k], f)
        V[:, 3:6] = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)
        parts = jp.chunks(V, F)
        i = next(k for k, c in enumerate(root['children']) if c is n)
        nodes = []
        for k, (v, f_) in enumerate(parts):
            node = dict(n)
            node.update(name=n['name'] if k == 0 else '%s_%d' % (n['name'], k + 1), children=[],
                        verts=v.astype('<f4'), faces=f_.astype('<u2'))
            bp.sphere(node)
            nodes.append(node)
        root['children'][i:i + 1] = nodes
    # the far ground made again round the near as it is now, as tools/far_terrain.py makes it, so
    # what stands on it moves with it too
    info = json.load(open(os.path.join(ft.SURROUNDINGS, 'surroundings.json')))
    far, fm = next((n, m) for n, m in bp.meshes(track) if n['name'] == ft.FAR)
    old_far = bp.triangles(far, fm)
    fV, fN, fU, fF, diff = ft.build(track, info, ft.real_ground(info))
    Vl = fV - fm[3, :3]
    far['verts'] = np.c_[Vl, fN, fU, gltf_source.tangents(Vl, fN, fU, fF)].astype('<f4')
    far['faces'] = fF.astype('<u2')
    bp.sphere(far)
    print("%s made again: %d points; at the near ground's edge the real ground differs by %+.2f to %+.2f m"
          % (ft.FAR, len(fV), diff.min(), diff.max()))
    if dry:
        return
    new_ground = np.concatenate([bp.triangles(n, np.eye(4)) for n, _ in bp.meshes(track)
                                 if n['name'].startswith((bp.GROUND, bp.FILLER)) and len(n['faces'])])

    # every road: its faces cut by a grid of ROAD_CUT, so it follows the real ground between its
    # points, then every point moved by the change over the roads, which takes the top of them ROAD
    # over the real ground: the roads lying over others (patches, run-offs) as far over them as
    # they were, their skirts as deep under them
    jp.GRID = ROAD_CUT
    cut_roads = []
    for f, n, m in roads:
        V, F, new, cut, Fc = jp.refine(n['verts'].astype(float), n['faces'].astype(int), lambda P: np.ones(len(P), bool))
        W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
        W[:, 1] += road_change(W[:, 0], W[:, 2])
        V[:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(m))[:, :3]
        V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)
        parts = [(v, f_) for v, f_ in jp.chunks(V, F) if len(f_)]
        if not parts:
            # (all its faces had no area: as it was)
            cut_roads.append((f, n, m))
            continue
        nodes = []
        for i, (v, f_) in enumerate(parts):
            node = n if i == 0 else dict(n, name='%s_%d' % (n['name'], i + 1), children=[])
            node['verts'], node['faces'] = v.astype('<f4'), f_.astype('<u2')
            bp.sphere(node)
            nodes.append(node)
        if len(nodes) > 1:
            k = models[f]
            def put(c):
                for j, ch in enumerate(c['children']):
                    if ch is n:
                        c['children'][j + 1:j + 1] = nodes[1:]
                        return True
                    if put(ch):
                        return True
                return False
            put(k['root'])
        cut_roads += [(f, x, m) for x in nodes]
    roads = cut_roads
    road_ids = {id(n) for _, n, _ in roads}
    print('  %d road meshes, cut and moved' % len(roads))
    new_tops = []
    for f, n, m in roads:
        t = bp.top_of(n) if n['name'].startswith('1ROAD_PATH_') else bp.triangles(n, m)
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ln = np.linalg.norm(nn, axis=1)
        new_tops.append(t[(ln > 1e-9) & (np.abs(nn[:, 1]) > .5 * np.maximum(ln, 1e-9))])
    new_road = bp.Surface(np.concatenate(new_tops))
    # (and the other pieces of grass, the kerbs' model's patches in the near ground's holes, as
    # they will lie: on the real ground)
    patches = [bp.triangles(n, m) for k in models.values() if k is not track for n, m in bp.meshes(k)
               if is_grass(n['name']) and len(n['faces'])]
    old_patch = np.concatenate(patches) if patches else np.zeros((0, 3, 3))
    new_patch = old_patch.copy()
    if len(new_patch):
        P3 = new_patch.reshape(-1, 3)
        new_patch[..., 1] = real(P3[:, 0], P3[:, 2]).reshape(-1, 3)
    new_top = bp.Surface(np.concatenate([new_ground, bp.triangles(far, fm), new_patch, np.concatenate(new_tops)]))
    old_top = bp.Surface(np.concatenate([old_ground, old_far, old_patch, tops]))

    # what the top moved, under what stands on it
    (xa, xb), (za, zb) = so.EXTENT
    X, Z = np.arange(xa, xb + .1, 1.), np.arange(za, zb + .1, 1.)
    XG, ZG = np.meshgrid(X, Z)
    Q = np.c_[XG.ravel(), ZG.ravel()]
    D = (new_top.ys(Q) - old_top.ys(Q)).reshape(XG.shape)
    D = np.where(np.isnan(D), 0., D)

    def top_change(x, z):
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        return map_coordinates(D, [(z - Z[0]), (x - X[0])], order=1, cval=0.)
    print('the top moved %+.2f to %+.2f m under what stands on it' % (D.min(), D.max()))

    # the trees, each on the ground where it stands
    forest = json.load(open(lg.FOREST))
    S = np.array([[s[1], s[3]] for s in forest['trees']])
    gy = new_top.ys(S)
    shift = np.where(np.isnan(gy), 0., gy - np.array([s[2] for s in forest['trees']]))
    for s, v in zip(forest['trees'], shift):
        s[2] += float(v)
    json.dump(forest, open(lg.FOREST, 'w'), indent=0)
    stands = cKDTree(S)

    def on_ground(W, F):
        """Each tyre stack stood on the top as it is now, as one: not bent, but turned to the
        slope of the ground under it and sunk till no part of its bottom tyre's rim is over the
        ground, SINK under it. A stack is the tyres (each a piece of the mesh) whose middles stand
        within STACK of each other, one over another, or a piece's middle within the middle IN_TYRE
        of a tyre's width (the halves of a tyre's tread, a leaning stack's tyres, as the gantry's
        legs'); a bigger piece moves point by point by the top's change. Returns the points where
        they go, each one's turn (3 by 3), and which."""
        I3 = np.eye(3)
        if not len(F):
            up = top_change(W[:, 0], W[:, 2])
            return W + np.c_[np.zeros(len(W)), up, np.zeros(len(W))], I3[None], np.zeros(len(W), int)
        lab = pieces(F, len(W))
        ids, inv = np.unique(lab, return_inverse=True)
        lo = np.full((len(ids), 3), np.inf)
        hi = np.full((len(ids), 3), -np.inf)
        np.minimum.at(lo, inv, W)
        np.maximum.at(hi, inv, W)
        c = (lo + hi)[:, [0, 2]] / 2
        st = np.arange(len(ids))
        tree = cKDTree(c)
        size = (hi - lo)[:, [0, 2]]
        Q = tree.query_pairs(.3, output_type='ndarray').reshape(-1, 2)
        d = np.abs(c[Q[:, 0]] - c[Q[:, 1]])
        within = np.zeros(len(Q), bool)
        for b_ in (0, 1):
            within |= (size[Q[:, b_]].max(1) < 1.) & (d <= IN_TYRE * size[Q[:, b_]]).all(1)
        for a_, b_ in np.r_[tree.query_pairs(STACK, output_type='ndarray').reshape(-1, 2), Q[within]]:
            ra, rb = a_, b_
            while st[ra] != ra:
                ra = st[ra]
            while st[rb] != rb:
                rb = st[rb]
            st[max(ra, rb)] = min(ra, rb)
        for i_ in range(len(st)):
            r_ = i_
            while st[r_] != r_:
                r_ = st[r_]
            st[i_] = r_
        sid, sinv = np.unique(st, return_inverse=True)
        which = sinv[inv]                      # each point's stack
        out = W.copy()
        turns = [I3]
        turn_of = np.zeros(len(W), int)
        order = np.argsort(which)
        bounds = np.searchsorted(which[order], np.arange(len(sid) + 1))
        for s_ in range(len(sid)):
            idx = order[bounds[s_]:bounds[s_ + 1]]
            q = W[idx]
            q0, q1 = q.min(0), q.max(0)
            if (q1 - q0)[[0, 2]].max() >= PIECE:
                out[idx, 1] += top_change(q[:, 0], q[:, 2])
                continue
            mid = (q0 + q1) / 2
            r = max((q1 - q0)[[0, 2]].max() / 2, .2)
            # the ground's slope under it: a plane through it under the middle and the rim
            ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            S = np.r_[[mid[[0, 2]]], mid[[0, 2]] + r * np.c_[np.cos(ang), np.sin(ang)]]
            g = new_top.ys(S)
            ok = ~np.isnan(g)
            if ok.sum() < 4:
                out[idx, 1] += top_change(q[:, 0], q[:, 2])
                continue
            A = np.c_[S[ok], np.ones(ok.sum())]
            (gx, gz, gd), *_ = np.linalg.lstsq(A, g[ok], rcond=None)
            nrm = np.array([-gx, 1., -gz])
            nrm /= np.linalg.norm(nrm)
            # the turn taking up to the slope's normal, about the stack's foot
            v = np.cross([0., 1., 0.], nrm)
            sn, cs = np.linalg.norm(v), nrm[1]
            if sn > 1e-6:
                k = v / sn
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                Rm = I3 + sn * K + (1 - cs) * K @ K
            else:
                Rm = I3
            foot = np.array([mid[0], q0[1], mid[2]])
            p = (q - foot) @ Rm.T + foot
            # sunk till its bottom tyre's rim (its points in the lowest 5 cm) is all SINK under
            low = (q[:, 1] - q0[1]) < .05
            gl = new_top.ys(p[low][:, [0, 2]])
            okl = ~np.isnan(gl)
            dy = (np.min(gl[okl] - p[low][okl, 1]) if okl.any() else float(g[ok][0] - q0[1])) - SINK
            p[:, 1] += dy
            out[idx] = p
            turns.append(Rm)
            turn_of[idx] = len(turns) - 1
        return out, np.array(turns), turn_of

    def as_pieces(W, F, change):
        """What the points move by: each piece smaller than PIECE across (a tyre stack, a post,
        a board) as one, by the change at its middle, so a stack by a road's edge is not sheared;
        the bigger, point by point."""
        up = change(W[:, 0], W[:, 2])
        if not len(F):
            return up
        lab = pieces(F, len(W))
        ids, inv = np.unique(lab, return_inverse=True)
        lo = np.full((len(ids), 2), np.inf)
        hi = np.full((len(ids), 2), -np.inf)
        np.minimum.at(lo, inv, W[:, [0, 2]])
        np.maximum.at(hi, inv, W[:, [0, 2]])
        small = (hi - lo).max(1) < PIECE
        mid = (lo + hi) / 2
        whole = change(mid[:, 0], mid[:, 1])
        return np.where(small[inv], whole[inv], up)

    # everything else: what is driven on or lies on it as the roads moved, the rest as the top
    def is_tyre(n):
        # (the gantries' tyres lean round their legs, and move as the legs do)
        return 'TYRE' in n['name'].upper() and not n['name'].startswith(('1WALL', 'GANTRY'))

    for f, k in models.items():
        moved = [0]
        # the tyre stacks of all the model's tyre meshes at once: a tyre's tread and its sides may
        # lie in different meshes (TYRES_n and TYRES_nF), and must turn and move as one
        placed = {}
        tyres = [(n, m) for n, m in bp.meshes(k) if is_tyre(n) and len(n['verts']) and len(n['faces'])]
        if tyres:
            Ws, Fs, off = [], [], 0
            for n, m in tyres:
                Ws.append((np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3])
                Fs.append(n['faces'].astype(int) + off)
                off += len(n['verts'])
            Wn, turns, turn_of = on_ground(np.concatenate(Ws), np.concatenate(Fs))
            off = 0
            for (n, m), W in zip(tyres, Ws):
                placed[id(n)] = (Wn[off:off + len(W)], turn_of[off:off + len(W)])
                off += len(W)

        def walk(n, parent):
            if n['kind'] == 1:
                M4 = np.array(n['matrix'], float).reshape(4, 4)
                m = M4 @ parent
                if not np.allclose(M4, np.eye(4)):
                    o = m[3, :3]
                    by = road_change if n['name'].startswith('AC_') else top_change
                    M4[3, :3] += np.array([0., float(by([o[0]], [o[2]])[0]), 0.]) @ np.linalg.inv(parent[:3, :3])
                    n['matrix'] = list(M4.ravel())
                    moved[0] += 1
                    return
                for c in n['children']:
                    walk(c, m)
            elif n['kind'] == 2 and len(n['verts']) and id(n) not in road_ids and n['name'] != 'GRASS_1'                     and not (k is track and n['name'].startswith((bp.GROUND, bp.FILLER))):
                V = n['verts']
                W = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ parent)[:, :3]
                if is_grass(n['name']):
                    up = real(W[:, 0], W[:, 2]) - W[:, 1]
                elif k['materials'][n['material']]['name'] in PAINT:
                    # on the road as it is now, as far over it as it was (a little at least)
                    was, now = old_road.ys(W[:, [0, 2]]), new_road.ys(W[:, [0, 2]])
                    over = np.maximum(np.where(np.isnan(was), .004, W[:, 1] - was), .004)
                    up = np.where(np.isnan(now) | np.isnan(was), road_change(W[:, 0], W[:, 2]), now + over - W[:, 1])
                elif id(n) in placed:
                    # the tyre stacks, each turned and moved as one; their normals turned with them
                    Wn, turn_of = placed[id(n)]
                    R3 = parent[:3, :3]
                    iR3 = np.linalg.inv(R3)
                    for c0 in (3, 8):
                        Nw = V[:, c0:c0 + 3].astype(float) @ R3
                        Nw = np.einsum('nij,nj->ni', turns[turn_of], Nw)
                        V[:, c0:c0 + 3] = (Nw @ iR3).astype(V.dtype)
                    up = Wn[:, 1] - W[:, 1]
                    W[:, [0, 2]] = Wn[:, [0, 2]]
                elif driven(n['name']) or n['name'].startswith('AC_'):
                    up = road_change(W[:, 0], W[:, 2])
                elif n['name'].startswith((bt.PREFIX, '1WALL_' + bt.PREFIX)):
                    dd, i = stands.query(W[:, [0, 2]])
                    up = np.where(dd < so.TREE_NEAR, shift[i], top_change(W[:, 0], W[:, 2]))
                else:
                    up = as_pieces(W, n['faces'].astype(int), top_change)
                if np.abs(up).max() < 1e-4:
                    return
                W[:, 1] += up
                V[:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(parent))[:, :3].astype(V.dtype)
                bp.sphere(n)
                moved[0] += len(V)
        walk(k['root'], np.eye(4))
        if os.path.basename(f) == 'velopark_trees.glb':
            bt.save(k, f)
        else:
            bp.save(k, f)
        print('  %-30s %7d points' % (os.path.basename(f), moved[0]))
    ramps(models, track, real)
    bp.save(track, bp.TRACK)
    # the Custom Shaders Patch's grass (and BeamNG's, which reads it) on every mesh of the near
    # ground: it is cut into meshes of fewer than 65536 points
    names = [n['name'] for n, _ in bp.meshes(track) if n['name'] == bp.GROUND or re.fullmatch(re.escape(bp.GROUND) + r'(_\d+)+', n['name'])]
    text = open(lg.CSP, newline='').read()
    m_ = re.search(r'(?m)^GRASS_MESHES *= *(.*)$', text)
    have = [x.strip() for x in m_.group(1).split(',') if x.strip()]
    keep = [x for x in have if not (x == bp.FILLER or re.fullmatch(re.escape(bp.GROUND) + r'(_\d+)*', x))]
    text = text[:m_.start(1)] + ', '.join(names + keep) + text[m_.end(1):]
    open(lg.CSP, 'w', newline='').write(text)
    print('  grass on %d meshes of the near ground' % len(names))
    # the AI lines, as the roads under them
    ai = sorted(glob.glob(os.path.join(lg.TRACK, '*', 'ai', '*.ai')) + glob.glob(os.path.join(lg.TRACK, '*', 'data', '*.ai')))
    for path in ai:
        lg.move_ai(path, road_change)
    print('  %d AI lines' % len(ai))
    print('  %d lights' % lg.move_csp(lg.CSP, top_change))
    print('  %d trees stood on the ground, by %+.2f to %+.2f m' % (len(shift), shift.min(), shift.max()))
    json.dump(dict(draped=datetime.date.today().isoformat(), road_m=ROAD, cut_m=CUT, near_m=NEAR), open(DONE, 'w'), indent=1)


def ramps(models, track, real):
    """A ramp of grass along every edge of what is driven on (the roads, the paving, the kerbs),
    from its edge down to the grass, RAMP a metre, its outer end just under the grass, so a tyre
    leaving or meeting the track rolls over it, not off a step: 1GRASS_EDGE_RAMP, in the ground's
    material, mapped as the ground photo is."""
    import bpy  # noqa  (mathutils)
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    tops = []
    for k in models.values():
        for n, m in bp.meshes(k):
            if not (len(n['faces']) and n['visible'] and n['name'].startswith(('1ROAD', '1KERB'))):
                continue
            t = bp.top_of(n) if n['name'].startswith('1ROAD_PATH_') else bp.triangles(n, m)
            nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            ln = np.linalg.norm(nn, axis=1)
            tops.append(t[(ln > 1e-9) & (np.abs(nn[:, 1]) > .5 * np.maximum(ln, 1e-9))])
    T = np.concatenate(tops)
    bvh = BVHTree.FromPolygons(T.reshape(-1, 3).tolist(), np.arange(len(T) * 3).reshape(-1, 3).tolist(), all_triangles=True)
    # each edge of each face: an edge of what is driven on where nothing driven on goes on past it
    # (tested there, not by counting the faces an edge belongs to: some surfaces lie on others)
    key = np.round(T[:, :, [0, 2]] / .01).astype(np.int64)
    kid = key[..., 0] * 10 ** 9 + key[..., 1]
    down = Vector((0., -1., 0.))
    pos, acc, edges, seen = {}, {}, [], set()
    for f in range(len(T)):
        c = T[f][:, [0, 2]].mean(0)
        for i in range(3):
            ka, kb = kid[f, i], kid[f, (i + 1) % 3]
            if ka == kb or (min(ka, kb), max(ka, kb)) in seen:
                continue
            A, B = T[f, i], T[f, (i + 1) % 3]
            d = (B - A)[[0, 2]]
            L = np.linalg.norm(d)
            if L < .01:
                continue
            nrm = np.array([d[1], -d[0]]) / L
            mid = (A + B) / 2
            if nrm @ (mid[[0, 2]] - c) < 0:
                nrm = -nrm
            q = mid[[0, 2]] + nrm * .08
            hit = bvh.ray_cast(Vector((q[0], mid[1] + .5, q[1])), down, 1.)
            if hit[0] is not None:
                continue
            seen.add((min(ka, kb), max(ka, kb)))
            for kk, Pt in ((ka, A), (kb, B)):
                pos[kk] = Pt if kk not in pos or Pt[1] > pos[kk][1] else pos[kk]
                acc[kk] = acc.get(kk, 0.) + nrm
            edges.append((ka, kb))
    out = {}
    for kk, P in pos.items():
        n2 = acc[kk] / max(np.linalg.norm(acc[kk]), 1e-9)
        g = float(real(np.array([P[0]]), np.array([P[2]]))[0])
        w = float(np.clip((P[1] - g) / RAMP, *RAMP_W))
        q = P[[0, 2]] + n2 * w
        out[kk] = np.array([q[0], float(real(np.array([q[0]]), np.array([q[1]]))[0]) - .004, q[1]])
    # the ground photo's mapping, from the near ground's points
    g = next(n for n, _ in bp.meshes(track) if n['name'] == bp.GROUND)
    Vg = g['verts'].astype(float)
    M, *_ = np.linalg.lstsq(np.c_[Vg[:, 0], Vg[:, 2], np.ones(len(Vg))], Vg[:, 6:8], rcond=None)
    V, F = [], []
    for ka, kb in edges:
        quad = np.array([pos[ka], pos[kb], out[kb], out[ka]])
        nv = len(V)
        nn = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        if nn[1] < 0:
            F += [(nv, nv + 2, nv + 1), (nv, nv + 3, nv + 2)]
        else:
            F += [(nv, nv + 1, nv + 2), (nv, nv + 2, nv + 3)]
        V += list(quad)
    V = np.array(V, float)
    F = np.array(F, int)
    # (none lying on a road: an edge whose next road begins a little past where it was tested)
    c = V.reshape(-1, 4, 3).mean(1)
    y = bp.Surface(T).ys(c[:, [0, 2]])
    on = np.isfinite(y) & (np.abs(c[:, 1] - y) < .5)
    F = F.reshape(-1, 2, 3)[~on].reshape(-1, 3)
    edges = [e for e, o in zip(edges, on) if not o]
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    U = np.c_[V[:, 0], V[:, 2], np.ones(len(V))] @ M
    Tg = gltf_source.tangents(V, N, U, F)
    root = track['root']
    root['children'] = [c for c in root['children'] if not c['name'].startswith('1GRASS_EDGE_RAMP')]
    mat = g['material']
    for i in range(0, len(F), 20000):
        f = F[i:i + 20000]
        used = np.unique(f)
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        node = dict(g, name='1GRASS_EDGE_RAMP' + ('' if i == 0 else '_%d' % (i // 20000)), children=[], material=mat,
                    verts=np.c_[V[used], N[used], U[used], Tg[used]].astype('<f4'), faces=remap[f].astype('<u2'))
        bp.sphere(node)
        root['children'].append(node)
    print('  %d edges ramped down to the grass, %d triangles' % (len(edges), len(F)))


if __name__ == '__main__':
    main()
