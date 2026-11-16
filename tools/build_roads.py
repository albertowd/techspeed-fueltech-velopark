"""The roads described in tools/reference/roads.json built again from their descriptions (see ROADS.md),
in place of their meshes in the track's model.

    python tools/build_roads.py

Each road's mesh (its name's, in source/models/velopark.glb; all its pieces, where it was too big for
one) is made again by tools/roads.py: its points, faces and mapping; its material and the rest as they
were. Its points as high as the road was there (its old top, or its patches' where it had a hole they
filled; the nearest point of it where neither is): tools/drape_track.py lays it as its heights say, the
kerbs and the paint on it moved by as much as it. The patches made for it (1ROAD_PATCH_..._<its name>,
in velopark_runoffs.glb: pieces of it laid where it had holes) out, as it covers them now.

The roads first, then the branches (road['joins']: a road that forks off another and merges into one
again), each cut where it lies over the roads it joins (the part of it outside them left, its edge on
theirs), then the run-offs (road['along']), onto the edges of the roads they lie along.

Their edge lines (road['lines']) laid along them into the lines they take the place of, forked where a
branch leaves and joins: a road's line out where it runs inside a branch's lines (the mouth), a branch's
out where it runs inside its roads' (until it parts from their line, and from where it meets it again).
First, the old meshes the editor's track replaced (tools/reference/retired.json) out of every model.
Done again each time from the models as they were before it (tools/build_models.py's chain).
"""
import os

import numpy as np

import build_paths as bp
import gltf_source
import roads as rd

RUNOFFS = os.path.join(bp.MODELS, 'velopark_runoffs.glb')
EDGE_CLEAR = .15      # metres round a road's new edge line the line it takes the place of is cleared
CLEAR_IN, CLEAR_OUT = 2., .3   # metres in from a described road's edge and past it its old edge lines are cleared
SNAP = .05            # metres: a run-off's point this near its road's edge is put on it
SNAP_NEAR = .05       # metres: a road's edge point this near the edge of a road above it is put on that edge
RETIRED = os.path.join(bp.ROOT, 'tools', 'reference', 'retired.json')   # (the old meshes the editor's track replaced)
JOINED = .5           # metres in from the edges of the roads a branch joins its old lines are left (theirs)


def idx(F, n_points):
    """Faces as the indices a mesh of n_points takes: 16 bits where they do, 32 where not (the model's
    loader cuts such a mesh in pieces a kn5 mesh holds)."""
    return np.asarray(F).astype('<u2' if n_points < 65536 else '<u4')


def keeps(road, roads):
    """Where each of a road's lines is kept (see the module's note): a function (line) -> keep(x, z)."""
    from shapely import contains_xy
    branch = road.get('joins')
    if branch:
        # (a branch: out where inside its roads' lines, grown by half a line's width)
        theirs = [rd.line_area(roads[x], rd.LINE_W / 2) for x in set(branch) if x and x in roads]
        return lambda x, z: ~np.any([contains_xy(a, x, z) for a in theirs], axis=0)
    mine = [rd.line_area(b, -rd.LINE_W / 2) for b in roads.values() if road['name'] in (b.get('joins') or ())]
    if not mine:
        return None
    return lambda x, z: ~np.any([contains_xy(a, x, z) for a in mine], axis=0)


def paint_lines(roads, tops):
    """Each described road's edge lines (road['lines']) laid along it (tools/roads.py's line_strip), each
    into the line of paint it takes the place of (its name's, in the lines' model): that line out along
    it first (EDGE_CLEAR round the new one), the rest of that line as it was; forked where branches
    leave and join (keeps). How many."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import restyle_paddock as rp
    path = os.path.join(bp.MODELS, 'velopark_lines.glb')
    if not os.path.exists(path):
        return 0
    L = gltf_source.load_kn5(path)
    by = {n['name']: n for n, _ in bp.meshes(L)}
    made = 0
    # (every old edge line along the sides of the roads with lines of their own out first: all of their
    # paint from CLEAR_IN in from each edge to CLEAR_OUT past it, the whole road along; a branch's not
    # over the roads it joins, but within JOINED of their edges)
    for road in roads.values():
        if not road.get('lines'):
            continue
        mid, left, wl, wr, at = rd.fine(road)
        inside = None
        if road.get('joins'):
            inside = unary_union([Polygon(np.r_[m + l * a[:, None], (m - l * b[:, None])[::-1]]).buffer(0).buffer(-JOINED)
                                  for m, l, a, b, _ in (rd.fine(roads[x]) for x in set(road['joins']) if x and x in roads)])
        for sgn, w in ((1., wl), (-1., wr)):
            A = mid + left * (sgn * (w - CLEAR_IN))[:, None]
            B = mid + left * (sgn * (w + CLEAR_OUT))[:, None]
            strip = Polygon(np.r_[A, B[::-1]]).buffer(0)
            if inside is not None:
                strip = strip.difference(inside)
            for x, _ in bp.meshes(L):
                if x['name'].startswith('LINE_') and len(x['faces']):
                    rp.clip_in(x, strip)
    laid = []
    for road in roads.values():
        keep = keeps(road, roads)
        for line0 in road.get('lines', []):
            n = by.get(line0['name'])
            if n is None:
                # (a line the track had none of there: a mesh of its own, in the white paint, or the
                # pits' yellow)
                n = dict(rd.paint_mesh(L, line0.get('colour')), name=line0['name'])
                L['root']['children'].append(n)
                by[line0['name']] = n
            runs = rd.line_runs(road, line0, keep) if keep else [dict(line0, **{'from': line0.get('from') or 0., 'to': line0['to'] if line0.get('to') is not None else rd.fine(road)[4][-1]})]
            for line in rd.dashes(line0, runs):
                P, F = rd.line_strip(road, line, tops.get(road['name']))
                band = unary_union([Polygon(P[f][:, [0, 2]]) for f in F if Polygon(P[f][:, [0, 2]]).area > 1e-8]).buffer(EDGE_CLEAR)
                laid.append((n, P, F, band))
    # (the old paint round every new line out, then the new lines laid: a road's line never cuts another's
    # where they meet, as at a joint)
    for n, P, F, band in laid:
        if len(n['faces']) and not n.get('_new'):
            rp.clip_in(n, band)
    for n, P, F, band in laid:
        if True:
            if True:
                n['_new'] = True
                V = np.zeros((len(P), n['verts'].shape[1]))
                V[:, :3] = P
                V[:, 4] = 1.
                V[:, 6], V[:, 7] = P[:, 0] / 10., P[:, 2] / 20.
                if V.shape[1] > 8:
                    V[:, 8] = 1.
                base = len(n['verts'])
                n['verts'] = np.r_[n['verts'], V.astype(n['verts'].dtype)]
                n['faces'] = idx(np.r_[n['faces'].astype(int).reshape(-1, 3), F + base], len(n['verts']))
                bp.sphere(n)
                made += 1
    for n, _ in bp.meshes(L):
        n.pop('_new', None)
    bp.save(L, path)
    return made


def retire():
    """The meshes tools/reference/retired.json names taken out of every model (an old kerb with its parts).
    Which, by model."""
    import glob, json
    import build_kerbs as bk
    if not os.path.exists(RETIRED):
        return {}
    names = json.load(open(RETIRED, encoding='utf-8')).get('meshes', [])
    kerbs = [x[len('1ROAD_KERB_'):] for x in names if x.startswith('1ROAD_KERB_')]
    out = {}
    for f in sorted(glob.glob(os.path.join(bp.MODELS, 'velopark*.glb'))):
        k = gltf_source.load_kn5(f)
        gone = [n for n, _ in bp.meshes(k) if len(n['faces']) and (n['name'] in names or any(bk.parts_of(b, n['name']) for b in kerbs))]
        if gone:
            for n in gone:
                n['faces'] = n['faces'][:0]
            bp.save(k, f)
            out[os.path.basename(f)] = sorted({n['name'] for n in gone})
    return out


def kind(road):
    """The order roads are built in: the roads, the branches off them, the run-offs along them."""
    return 2 if 'along' in road else 1 if road.get('joins') else 0


def main():
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union, nearest_points
    import restyle_paddock as rp
    roads = rd.load()
    if not roads:
        print('no roads described')
        return
    # (the roads with no heights yet, as one made or changed in the road editor (top-race-editor): theirs recorded on
    # the ground the drape lays the roads on, kept by their shape (roads.py's HEIGHTS), the roads they
    # join first)
    need = [roads[x] for x in rd.priority(roads) if not roads[x].get('heights')]     # (from the top: each on those above it)
    if need:
        import json
        import describe_roads as dr
        real = dr.ground()
        os.makedirs(rd.HEIGHTS, exist_ok=True)
        for r in need:
            r['heights'] = dr.heights_for(r, roads, real)
            json.dump(r['heights'], open(os.path.join(rd.HEIGHTS, rd.shape_key(r, roads) + '.json'), 'w'))
        print('heights recorded: ' + ', '.join(r['name'] for r in need))
    for f, names in retire().items():
        print('retired from %s: %s' % (f, ', '.join(names)))
    track = gltf_source.load_kn5(bp.TRACK)
    runoffs = gltf_source.load_kn5(RUNOFFS) if os.path.exists(RUNOFFS) else None
    built, tops, built_mesh = [], {}, {}
    # (a road's mesh in pieces, where it was too big for one: built into the first, the others emptied)
    pieces = {}
    # (a road takes the place of its name's mesh, or of road['mesh'], the one it had before it was renamed:
    # that mesh named as the road now)
    of = {r.get('mesh', r['name']): r['name'] for r in roads.values()}
    for n, m in bp.meshes(track):
        if n['name'] in of and len(n['faces']) and np.allclose(m, np.eye(4)):
            pieces.setdefault(of[n['name']], []).append(n)
            n['name'] = of[n['name']]
    # (a road the track has no mesh of yet, as one drawn in the road editor (top-race-editor): a mesh of its own, as
    # the track's roads in its material)
    for name in roads:
        if name in pieces:
            continue
        mat = roads[name].get('material', 'Tarmac Dark')
        tpl = next((x for x, m in bp.meshes(track) if x['name'].startswith('1ROAD_') and len(x['faces']) and np.allclose(m, np.eye(4))
                    and track['materials'][x['material']]['name'] == mat), None) or             next(x for x, m in bp.meshes(track) if x['name'].startswith('1ROAD_') and len(x['faces']) and np.allclose(m, np.eye(4)))
        n = dict(tpl, name=name, children=[], verts=tpl['verts'][:3].copy(), faces=np.array([[0, 1, 2]], '<u2'))
        track['root']['children'].append(n)
        pieces[name] = [n]
    order = [x for x in rd.priority(roads) if x in pieces]
    for name in order:
        n, rest = pieces[name][0], pieces[name][1:]
        road = roads[name]
        if 'along' in road:
            road = rd.along_road(road, roads)                   # (a run-off, along another's edge)
        old = np.concatenate([bp.triangles(x, np.eye(4)) for x in pieces[name]])
        patches = [x for x, _ in bp.meshes(runoffs)] if runoffs is not None else []
        mine = [x for x in patches if x['name'].startswith('1ROAD_PATCH_') and x['name'].endswith('_' + name) and len(x['faces'])]
        S = bp.Surface(np.concatenate([old] + [bp.triangles(x, np.eye(4)) for x in mine]))
        near = cKDTree(old.reshape(-1, 3)[:, [0, 2]])
        Vo = old.reshape(-1, 3)

        def height(x, z):
            y = S.ys(np.c_[x, z]) - rd.ROAD
            gone = np.isnan(y)
            if gone.any():
                y[gone] = Vo[near.query(np.c_[x, z][gone])[1], 1] - rd.ROAD
            return y
        P, U, F = rd.mesh(road, height)                        # (as high as it was: the drape lays it as described)
        if 'along' in roads[name]:
            # (a run-off's inner edge on its road's edge as built, point for point where they meet)
            Pp, Fp = built_mesh[roads[name]['along']['road']]
            outline = unary_union([Polygon(Pp[f][:, [0, 2]]).buffer(1e-4) for f in Fp if Polygon(Pp[f][:, [0, 2]]).area > 1e-6]).boundary
            ptop = bp.Surface(Pp[Fp])
            for i_ in range(len(P)):
                q = Point(P[i_, 0], P[i_, 2])
                if outline.distance(q) < SNAP:
                    r = nearest_points(outline, q)[0]
                    P[i_, 0], P[i_, 2] = r.x, r.y
                    y_ = ptop.y(r.x, r.y)
                    if np.isfinite(y_):
                        P[i_, 1] = y_
        built_mesh[name] = (P, F)
        top, pts = bp.Surface(P[F]), cKDTree(P[:, [0, 2]])

        def on_top(x, z, top=top, pts=pts, P=P):
            # (its lines on it as it lies; just past its edge, as high as its nearest point)
            y = top.ys(np.c_[x, z])
            gone = np.isnan(y)
            if gone.any():
                y[gone] = P[pts.query(np.c_[x, z][gone])[1], 1]
            return y - rd.ROAD
        tops[name] = on_top
        V = np.zeros((len(P), n['verts'].shape[1]))
        V[:, :3] = P
        # (facing up, as the ground under it: the drape lays its points on it)
        fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
        N = np.zeros((len(P), 3))
        for i in range(3):
            np.add.at(N, F[:, i], fn)
        V[:, 3:6] = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        V[:, 6:8] = U
        if V.shape[1] > 8:
            V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
        n['verts'], n['faces'] = V.astype(n['verts'].dtype), idx(F, len(V))
        # (in its material, as the road says, where the track has one of that name)
        mats = [x['name'] for x in track['materials']]
        if roads[name].get('material') in mats:
            n['material'] = mats.index(roads[name]['material'])
        cut = ''
        ups = [x for x in rd.above(roads[name], roads) if x in built_mesh]
        if ups:
            # (out where it lies under the roads above it (a branch under those it joins too), its edge on
            # theirs there; its edge points within SNAP_NEAR of theirs put on them, no slit between)
            import shapely
            own = rd.outline(roads[name], roads)
            over = unary_union([rd.outline(roads[x], roads) for x in ups]).intersection(own.buffer(1.))   # (only near it: quick)
            before = len(n['faces'])
            if over.intersection(own).area > 1e-3:
                rp.clip_in(n, over.buffer(1e-5))
            B = over.boundary
            V2 = n['verts'][:, [0, 2]].astype(float)
            pts = shapely.points(V2)
            d = shapely.distance(B, pts)
            near = (d > 1e-7) & (d < SNAP_NEAR) & ~shapely.contains_xy(over, V2[:, 0], V2[:, 1])
            if near.any():
                ends = np.array([np.asarray(g.coords)[1] for g in shapely.shortest_line(pts[near], B)])
                n['verts'][near, 0], n['verts'][near, 2] = ends[:, 0], ends[:, 1]
            cut = ', under %s: %d faces to %d, %d edge points onto theirs' % ('/'.join(ups), before, len(n['faces']), int(near.sum()))
        bp.sphere(n)
        for x in rest:
            x['faces'] = x['faces'][:0]
        for x in mine:
            x['faces'] = x['faces'][:0]
        built.append('%s (%d faces%s%s)' % (name, len(n['faces']), cut, ', %d patches of it out' % len(mine) if mine else ''))
    bp.save(track, bp.TRACK)
    if runoffs is not None:
        bp.save(runoffs, RUNOFFS)
    painted = paint_lines(roads, tops)
    if painted:
        built.append('their edge lines (%d)' % painted)
    print('roads built from their descriptions: ' + ', '.join(built))


if __name__ == '__main__':
    main()
