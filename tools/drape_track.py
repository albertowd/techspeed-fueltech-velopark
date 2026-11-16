"""The track draped over the real ground, as Race Track Builder made it, in every source.

    python tools/drape_track.py [--dry-run]

The real ground is tools/terrain.py's ground: the real ground's heights filtered, a smooth (cubic)
surface between its 10 m points, its edits on it (the middle buildings' platform, the pits and the
shed raised). On it:

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

import build_paths as bp
from grade_paths import pieces
import build_trees as bt
import far_terrain as ft
import gltf_source
import join_paddock as jp
import level_ground as lg
import settle_objects as so
import terrain

DONE = os.path.join(lg.ev.OUT, 'draped.json')
ROAD = .02               # metres a road lies over the real ground: its own height
RAMP = .12               # the slope from an edge of what is driven on down to the grass, metres a metre
RAMP_UNDER = .01         # metres a ramp's outer edge lies under the near ground's mesh
RAMP_W = (.12, 1.)       # the narrowest and widest a ramp is, metres
PIECE = 4.               # metres: a piece of a thing standing on the ground smaller than this across moves as one
# where the pits' shed stands (tools/build_outside.py and tools/restyle_paddock.py build its floor and its
# concrete faces on them; the ground under them, raised, is tools/terrain.py's paddock_platform edit): the
# line the pit building's and the storage building's east walls stand on (x, z at both ends; x grows
# west), the shed's floor FACE_IN in from it, north to SHED_NORTH (z), its edge SHED_EDGE
LANE_EDGE = ((-48.78, 126.), (-47.26, 142.))
FACE_IN = 1.2
SHED_NORTH = 133.7          # (its concrete, with the face over its edge, to z 135.5)
SHED_EDGE = .5
NEAR = 15.               # metres from a road the ground's faces are cut
CUT = 3.                 # metres, the grid they are cut along
ROAD_CUT = 2.            # metres, the grid the roads' faces are cut along, so they follow the real ground
SINK = .03               # metres a tyre stack stands into the ground
IN_TYRE = .3             # of a tyre's width either way of its middle: a piece with its middle there is of its stack
STACK = .1               # metres apart the middles of the tyres of one stack may be (more, and the stacks of a wall chain into one)
# the car park south of the pit apron by the hotlap's pit lane: PARK_DROP under the apron, a step
# along its edge (tools/restyle_paddock.py's APRON_SOUTH and APRON_EAST), but where a ramp joins them
# (PARK_RAMP, x, the step sloping down over PARK_RAMP_LONG) and beside the road south (within
# PARK_SIDE of it, rising to it); its asphalt (PARK_MESH)
PARK_DROP, PARK_RAMP, PARK_RAMP_LONG, PARK_SIDE, PARK_MESH = .10, (113., 122.), 2.5, 3., '1ROAD_PADDOCK_PARK'
# the step down from the apron flat for PARK_FLAT metres from its edge, easing back up to the ground
# as it is over PARK_EASE more (the road south of the car park and the car park east of it as they are)
PARK_FLAT, PARK_EASE = 8., 10.
GANTRY = 'velopark_gantry'  # (the start gantries' models: each moved as one)
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


def park_drop(track):
    """How far the car park lies under the pit apron at points (x, z): a function of them, 0 elsewhere."""
    import restyle_paddock as rp
    import shapely
    from shapely.geometry import LineString, Polygon
    from scipy.spatial import cKDTree
    park = Polygon(rp.PARK).buffer(.05).union(Polygon(rp.PARK).buffer(3.5, join_style=2))
    edge = LineString(rp.APRON_SOUTH)
    side = np.concatenate([n['verts'][:, [0, 2]].astype(float) for n, m in bp.meshes(track) if n['name'] == '1ROAD_26'])
    tree = cKDTree(side)

    def drop(x, z):
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        under = shapely.contains_xy(park, x, z)
        x, z = x.copy(), z.copy()
        f = np.ones(len(x))
        ramp = (x > PARK_RAMP[0]) & (x < PARK_RAMP[1])
        if ramp.any():
            f[ramp] = np.clip(np.array([edge.distance(shapely.geometry.Point(a, b)) for a, b in zip(x[ramp], z[ramp])]) / PARK_RAMP_LONG, 0., 1.)
        f *= np.clip(tree.query(np.c_[x, z])[0] / PARK_SIDE, 0., 1.)
        # (from the apron's edge out over PARK_EASE: the road south of it as high as it is)
        d_edge = np.array([edge.distance(shapely.geometry.Point(a, b)) for a, b in zip(x, z)]) if under.any() else np.zeros(len(x))
        f *= np.clip(1. - (d_edge - PARK_FLAT) / PARK_EASE, 0., 1.)
        return np.where(under, -PARK_DROP * f, 0.)
    return drop


def main():
    dry = '--dry-run' in sys.argv
    if os.path.exists(DONE):
        print('the track is draped already (%s)' % os.path.relpath(DONE, bp.ROOT))
        return
    real = terrain.ground(3)                                   # (the ground, its edits on it: tools/terrain.py)
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
    drop = park_drop(track)
    cut_roads = []
    # (a road described with its heights (tools/reference/roads.json) lies as described: not moved)
    import roads as rd_
    fixed = {k_ for k_, r_ in rd_.load().items() if r_.get('heights')}
    kept = [(f, n, m) for f, n, m in roads if n['name'] in fixed]
    roads = [(f, n, m) for f, n, m in roads if n['name'] not in fixed]
    if kept:
        # (each as high as its heights say, where its points are; the roads in their order, each point a
        # road shares with one above it (where it was cut on its edge) as high as that one's there)
        from scipy.spatial import cKDTree
        all_ = rd_.load()
        rank = {x: i for i, x in enumerate(rd_.priority(all_))}
        kept.sort(key=lambda fnm: rank.get(fnm[1]['name'], 1e9))
        laid = {}
        for f, n, m in kept:
            road = all_[n['name']]
            if 'along' in road:
                road = rd_.along_road(road, all_)
            Pd, _, Fd = rd_.mesh(road)
            Sd = bp.Surface(Pd[Fd])
            V = n['verts'].astype(float)
            W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
            y = Sd.ys(W[:, [0, 2]])
            # (a point on one of its description's points as high as that point: at its edges and its end
            # rows too, where the surface's test can miss, so two roads joined end to end meet exactly)
            dd_, jj_ = cKDTree(Pd[:, [0, 2]]).query(W[:, [0, 2]])
            y = np.where(dd_ < 1e-3, Pd[jj_, 1], y)
            W[:, 1] = np.where(np.isfinite(y), y, W[:, 1] + road_change(W[:, 0], W[:, 2]))
            import shapely
            for x in rd_.above(all_[n['name']], all_):
                if x not in laid:
                    continue
                on = shapely.distance(rd_.outline(all_[x], all_).boundary, shapely.points(W[:, [0, 2]])) < .002
                if on.any():
                    d_, j_ = laid[x][0].query(W[on][:, [0, 2]], k=2)
                    w_ = 1. / np.maximum(d_, 1e-6)
                    W[on, 1] = (laid[x][1][j_] * w_).sum(1) / w_.sum(1)
            laid.setdefault(n['name'], [None, None])
            prev = laid[n['name']]
            pts = W[:, [0, 2]] if prev[0] is None else np.r_[prev[0].data, W[:, [0, 2]]]
            ys = W[:, 1] if prev[1] is None else np.r_[prev[1], W[:, 1]]
            laid[n['name']] = [cKDTree(pts), ys]
            V[:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(m))[:, :3]
            n['verts'] = V.astype(n['verts'].dtype)
            bp.sphere(n)
        print('  %d roads as described, as high as their heights say: %s' % (len(kept), ', '.join(n['name'] for _, n, _ in kept)))
    for f, n, m in roads:
        V, F, new, cut, Fc = jp.refine(n['verts'].astype(float), n['faces'].astype(int), lambda P: np.ones(len(P), bool))
        W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
        W[:, 1] += road_change(W[:, 0], W[:, 2])
        if n['name'].startswith(PARK_MESH):
            W[:, 1] += drop(W[:, 0], W[:, 2])                   # (the car park a step under the apron)
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
    roads = cut_roads + kept
    # the car park's ramp up to the apron: from the apron's own height at its edge (it is not quite
    # ROAD over the ground) down to the car park's over PARK_RAMP_LONG
    import restyle_paddock as rp
    from shapely.geometry import LineString as _LS, Point as _Pt
    ap = [bp.triangles(n, m) for f, n, m in roads if n['name'].startswith(rp.HOTLAP_BOX_ROAD) and len(n['faces'])]
    if ap:
        apron_top = bp.Surface(np.concatenate(ap))
        edge = _LS(rp.APRON_SOUTH)
        a_, b_ = np.array(rp.APRON_SOUTH[0], float), np.array(rp.APRON_SOUTH[-1], float)
        u_ = (b_ - a_) / np.linalg.norm(b_ - a_)
        out_ = np.array([u_[1], -u_[0]]) if u_[0] > 0 else np.array([-u_[1], u_[0]])
        if out_[1] > 0:
            out_ = -out_                                            # (into the car park, south)
        for f, n, m in roads:
            if not n['name'].startswith(PARK_MESH) or not len(n['faces']):
                continue
            V = n['verts'].astype(float)
            W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
            sel = np.where((W[:, 0] > PARK_RAMP[0]) & (W[:, 0] < PARK_RAMP[1]))[0]
            for i in sel:
                d = edge.distance(_Pt(W[i, 0], W[i, 2]))
                if d >= PARK_RAMP_LONG:
                    continue
                q = np.array(edge.interpolate(edge.project(_Pt(W[i, 0], W[i, 2]))).coords[0])
                A = apron_top.y(*(q - out_ * .15))
                if np.isfinite(A):
                    s_ = d / PARK_RAMP_LONG
                    W[i, 1] = A * (1 - s_) + W[i, 1] * s_
            V[:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(m))[:, :3]
            n['verts'] = V.astype(n['verts'].dtype)
            bp.sphere(n)
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
        # a start gantry moves as one, by a plane through what the top moved under its feet (each
        # of its legs moved by its own foot's, its board by the road's under it, they came apart)
        plane = None
        if os.path.basename(f).startswith(GANTRY):
            feet = np.array([(np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3].mean(0)
                             for n, m in bp.meshes(k) if n['name'].startswith('1WALL_GANTRY') and len(n['verts'])])
            if len(feet) >= 3:
                A_ = np.c_[np.ones(len(feet)), feet[:, 0], feet[:, 2]]
                cf = np.linalg.lstsq(A_, top_change(feet[:, 0], feet[:, 2]), rcond=None)[0]
                plane = lambda x, z, cf=cf: cf[0] + cf[1] * np.asarray(x, float) + cf[2] * np.asarray(z, float)
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
                    by = plane if plane is not None else road_change if n['name'].startswith('AC_') else top_change
                    M4[3, :3] += np.array([0., float(by([o[0]], [o[2]])[0]), 0.]) @ np.linalg.inv(parent[:3, :3])
                    n['matrix'] = list(M4.ravel())
                    moved[0] += 1
                    return
                for c in n['children']:
                    walk(c, m)
            elif n['kind'] == 2 and len(n['verts']) and id(n) not in road_ids and n['name'] != 'GRASS_1'                     and not (k is track and n['name'].startswith((bp.GROUND, bp.FILLER))):
                V = n['verts']
                W = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ parent)[:, :3]
                if plane is not None:
                    up = plane(W[:, 0], W[:, 2])
                elif is_grass(n['name']):
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
    # (each edge's point out on the grass: all of them at once)
    ks = list(pos)
    Pk = np.array([pos[kk] for kk in ks])
    Nk = np.array([acc[kk] for kk in ks])
    Nk /= np.maximum(np.linalg.norm(Nk, axis=1, keepdims=True), 1e-9)
    g = np.asarray(real(Pk[:, 0], Pk[:, 2]), float)
    w = np.clip((Pk[:, 1] - g) / RAMP, *RAMP_W)
    Q = Pk[:, [0, 2]] + Nk * w[:, None]
    yq = np.asarray(real(Q[:, 0], Q[:, 1]), float) - .004
    # (just under the near ground as its mesh lies there, which between its points is not quite the
    # real ground: else the ramp's edge stands over it, open)
    gr = [bp.triangles(n, m) for n, m in bp.meshes(track) if n['name'].startswith(bp.GROUND) and len(n['faces'])]
    if gr:
        gy = bp.Surface(np.concatenate(gr)).ys(Q)
        yq = np.where(np.isfinite(gy), np.minimum(yq, gy - RAMP_UNDER), yq)
    out = {kk: np.array([q[0], y, q[1]]) for kk, q, y in zip(ks, Q, yq)}
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
    # (none lying on a road: an edge whose next road begins a little past where it was tested; each
    # of its two triangles by itself, one may reach over a road the other does not)
    c = V[F].mean(1)
    y = bp.Surface(T).ys(c[:, [0, 2]])
    on = np.isfinite(y) & (np.abs(c[:, 1] - y) < .5)
    pair = on.reshape(-1, 2).any(1)
    F = F[~on]
    edges = [e for e, o in zip(edges, pair) if not o]
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
