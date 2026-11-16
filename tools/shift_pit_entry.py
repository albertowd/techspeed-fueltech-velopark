"""The vp_1500's pit entry moved west where it goes in under the pit building's roof, so its tyre wall
lines up with the one along the building's columns (as the photos from it show: the tyres along its
left run on into the columns' line), the floor between it and the workshop's wall there.

    python tools/shift_pit_entry.py

Its road (ROAD), the tyre wall along its west side (TYRES) and, in its CORRIDOR (x, z), the
vp_1500's pit boxes (the layouts' AC_PIT_ points), their paint and every line on it, and the layouts'
pit lane AI points move SHIFT metres west (x) from z FROM on, easing in over EASE_IN from FROM and
out over EASE_OUT to TO, where the road meets the columns' tyre wall already. Marked done by its
road's west edge by z 100 (DONE_X).
"""
import glob, os, struct

import numpy as np

import build_paths as bp
import gltf_source

ROAD, TYRES = '1ROAD_25', ('TYRES_56', 'TYRES_56F', 'TYRES_107', 'TYRES_107F', '1WALL_TYRE_56', '1WALL_TYRE_107')
CORRIDOR = ((-43., -31.), (58., 112.))
SHIFT, FROM, EASE_IN, TO, EASE_OUT = 2.2, 78., 14., 110., 6.
# and its east leg a little west from the loop's bottom up, its curve the tighter (it ran under the
# building east of it): LEG metres from LEG_FROM, over LEG_EASE; east of x LEG_X only, easing in over
# LEG_X_EASE (the loop's bottom as it is)
LEG, LEG_FROM, LEG_EASE, LEG_X, LEG_X_EASE = .8, 62., 12., 31., 3.
DONE_X = -33.8        # (its west edge by z 100 past this, x: moved already)
# its tyre walls (laid again after it moved): round the grass it loops round (ISLAND_AT in it, up to
# ISLAND_TOP, z), and along its outer edge, as drawn (EAST_WALL, x, z); WALL_OFF from the roads' edges,
# a stack (TEMPLATE's) every WALL_STEP, their walls to hit WALL_H high, all in WALLS_BOX (x, z)
ISLAND_AT, ISLAND_TOP = (-25., 80.), 112.
ISLAND_PITS = (-40., 95., -28., 125.)     # (x0, z0, x1, z1: by the pits, whose tyre wall runs on along the columns)
EAST_WALL = ((-33.6, 58.3), (-33.7, 61.6), (-35.5, 64.), (-37.3, 65.7), (-38.8, 68.1), (-39.7, 71.1), (-40.3, 73.5),
             (-40.6, 76.4), (-40.4, 79.4), (-40., 81.8), (-40., 84.1))
WALL_OFF, WALL_STEP, WALL_H, TEMPLATE = .45, .62, .75, ('TYRES_56', 'TYRES_56F')
WALLS_BOX = ((-46., -8.), (50., 120.))
LAYOUTS = ('vp_1500', 'vp_1500_amateur', 'vp_1500_short', 'vp_1500_short_amateur', 'vp_1500_hotlap', 'vp_2500')


def dx(z, x=None):
    """How far west a point at z (and x, east of the loop's bottom: its east leg's) moves."""
    z = np.asarray(z, float)
    s = lambda t: (lambda c: c * c * (3 - 2 * c))(np.clip(t, 0., 1.))
    roof = SHIFT * s((z - FROM) / EASE_IN) * (1 - s((z - (TO - EASE_OUT)) / EASE_OUT))
    leg = LEG * s((z - LEG_FROM) / LEG_EASE) * (1 - s((z - (TO - EASE_OUT)) / EASE_OUT))
    out = np.maximum(roof, leg)
    if x is not None:
        out = out * s((-np.asarray(x, float) - LEG_X) / LEG_X_EASE)
    return out


def in_corridor(x, z):
    (x0, x1), (z0, z1) = CORRIDOR
    return (x > x0) & (x < x1) & (z > z0) & (z < z1)


def move_mesh(n, m, keep=None):
    """A mesh's points moved (those keep marks, all by default): world x += dx(world z)."""
    V = n['verts'].astype(float)
    W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
    sel = np.ones(len(W), bool) if keep is None else keep(W[:, 0], W[:, 2])
    W[sel, 0] += dx(W[sel, 2], W[sel, 0])
    V[:, :3] = (np.c_[W, np.ones(len(W))] @ np.linalg.inv(m))[:, :3]
    n['verts'] = V.astype(n['verts'].dtype)
    bp.sphere(n)
    return int(sel.sum())


def vacated(track):
    """Where the road was and is no more (x, z): its faces moved back east by as much, less it as it is."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    T = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track) if n['name'] == ROAD and len(n['faces'])])
    now = unary_union([Polygon(t[:, [0, 2]]).buffer(.01) for t in T if Polygon(t[:, [0, 2]]).area > 1e-6])
    B = T.copy()
    B[:, :, 0] -= dx(B[:, :, 2], B[:, :, 0])
    was = unary_union([Polygon(t[:, [0, 2]]).buffer(.01) for t in B if Polygon(t[:, [0, 2]]).area > 1e-6])
    return was.difference(now).buffer(0)                      # (thin as it may be: no slit left)


def roads_near(track):
    """The roads and kerbs round the pit entry (WALLS_BOX), as one shape, x, z."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    (x0, x1), (z0, z1) = WALLS_BOX
    tris = []
    for k in (track, gltf_source.load_kn5(bp.PADDOCK), gltf_source.load_kn5(bp.KERBS)):
        for n, m in bp.meshes(k):
            if n['name'].startswith(('1ROAD', '1KERB')) and len(n['faces']):
                T = bp.triangles(n, m)[:, :, [0, 2]]
                c = T.mean(1)
                T = T[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > z0) & (c[:, 1] < z1)]
                tris += [Polygon(t).buffer(.01) for t in T if Polygon(t).area > 1e-4]
    return unary_union(tris).buffer(-.01)


def island_path(roads):
    """Round the grass the entry loops round (ISLAND_AT in it), WALL_OFF in from its edge, from the
    north (ISLAND_TOP) down its west side, round its foot and up its east side: (n, 2) x, z."""
    from shapely.geometry import Point, box
    (x0, x1), (z0, z1) = WALLS_BOX
    grass = box(x0, z0, x1, z1).difference(roads)
    isl = next(g for g in getattr(grass, 'geoms', [grass]) if g.contains(Point(*ISLAND_AT)))
    inner = isl.buffer(-WALL_OFF)
    inner = max(getattr(inner, 'geoms', [inner]), key=lambda g: g.area)
    from shapely.ops import linemerge
    cut = inner.exterior.intersection(box(x0, z0, x1, ISLAND_TOP).difference(box(*ISLAND_PITS)))     # (not along the pits' own wall)
    cut = linemerge(cut) if cut.geom_type == 'MultiLineString' else cut      # (its ring's start no break in it)
    return np.array(max(getattr(cut, 'geoms', [cut]), key=lambda g: g.length).coords)


def east_path(roads):
    """Along the entry's outer edge from by the side road to the race control, as drawn (EAST_WALL: the
    ground there is paved, no edge of grass to follow), smoothed: (n, 2) x, z."""
    P = np.array(EAST_WALL, float)
    for _ in range(4):                                          # (Chaikin's corners cut, its ends kept)
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [a * .75 + b * .25, a * .25 + b * .75]
        Q.append(P[-1])
        P = np.array(Q)
    return P


def lay_wall(k, name, path, top):
    """A tyre wall along path (x, z) laid in k: TEMPLATE's first stack every WALL_STEP along it on top
    (a surface), as TYRES_<name> and TYRES_<name>F, its wall to hit (1WALL_TYRE_<name>) along it.
    How many stacks."""
    import restyle_paddock as rp
    from shapely.geometry import LineString
    line = LineString(path)
    P = np.array([line.interpolate(d).coords[0] for d in np.arange(0., line.length + 1e-6, WALL_STEP)])
    drawn = [n for n, _ in bp.meshes(k) if n['name'] in TEMPLATE]
    st, mid = rp.tyre_stacks(drawn)
    t = 0
    ys = top.ys(P)
    ys = np.where(np.isnan(ys), np.nanmedian(ys), ys)
    off = 0
    for n in drawn:
        Vn = n['verts'].astype(float)
        Fn = n['faces'].astype(int)
        mine = st[off:off + len(Vn)] == t
        off += len(Vn)
        fsel = mine[Fn].all(1)
        used = np.unique(Fn[fsel])
        remap = np.full(len(Vn), -1)
        remap[used] = np.arange(len(used))
        Vt, Ft = Vn[used], remap[Fn[fsel]]
        base = Vt[:, 1].min()
        rows, faces = [], []
        for i, (q, y) in enumerate(zip(P, ys)):
            v = Vt.copy()
            v[:, 0] += q[0] - mid[t][0]
            v[:, 2] += q[1] - mid[t][1]
            v[:, 1] += y - base
            rows.append(v)
            faces.append(Ft + i * len(Vt))
        V = np.concatenate(rows)
        node = dict(n, name=('TYRES_%s' % name) + ('F' if n['name'].endswith('F') else ''), children=[],
                    verts=V.astype(n['verts'].dtype), faces=np.concatenate(faces).astype('<u2'))
        bp.sphere(node)
        k['root']['children'].append(node)
    wall = next(n for n, _ in bp.meshes(k) if n['name'].startswith('1WALL_TYRE') and len(n['faces']))
    Vw, Fw = [], []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        y = min(ys[i], ys[i + 1])
        q = [(a[0], y - .05, a[1]), (b[0], y - .05, b[1]), (b[0], y + WALL_H, b[1]), (a[0], y + WALL_H, a[1])]
        nrm = np.cross(np.subtract(q[1], q[0]), np.subtract(q[3], q[0]))
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        b0 = len(Vw)
        for p_ in q:
            v = np.zeros(wall['verts'].shape[1])
            v[:3], v[3:6] = p_, nrm
            Vw.append(v)
        Fw += [(b0, b0 + 1, b0 + 2), (b0, b0 + 2, b0 + 3)]
    node = dict(wall, name='1WALL_TYRE_%s' % name, children=[], verts=np.array(Vw).astype(wall['verts'].dtype),
                faces=np.array(Fw).astype('<u2'))
    bp.sphere(node)
    k['root']['children'].append(node)
    return len(P)


def clear_near(k, paths, r=.6):
    """The tyre stacks (and their walls to hit) within r of paths (x, z) out of k, but those laid along
    them (TYRES_ENTRY_1500_...). How many stacks."""
    import restyle_paddock as rp
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString
    pts = []
    for p in paths:
        ln = LineString(p)
        pts.append(np.array([ln.interpolate(d).coords[0] for d in np.arange(0., ln.length, .2)]))
    tree = cKDTree(np.concatenate(pts))
    import restyle_paddock as rp_
    keep = {x for names in rp_.PIT_LINE_TYRES for x in names}          # (the pits' own walls, laid again by restyle)
    gone = 0
    groups = {}
    for n, _ in bp.meshes(k):
        if n['name'].startswith('TYRES_') and not n['name'].startswith('TYRES_ENTRY_1500') and len(n['faces'])                 and n['name'] not in keep:
            groups.setdefault(n['name'].rstrip('F'), []).append(n)
    for base, drawn in groups.items():
        st, mid = rp.tyre_stacks(drawn)
        near = tree.query(mid)[0] < r
        if not near.any():
            continue
        gone += int(near.sum())
        off = 0
        for n in drawn:
            F = n['faces'].astype(int)
            s_ = st[off:off + len(n['verts'])]
            off += len(n['verts'])
            n['faces'] = F[~near[s_][F].any(1)].astype('<u2')
            bp.sphere(n)
    for n, _ in bp.meshes(k):
        if n['name'].startswith('1WALL_TYRE') and not n['name'].startswith('1WALL_TYRE_ENTRY_1500') and len(n['faces']):
            V = n['verts'][:, [0, 2]].astype(float)
            F = n['faces'].astype(int)
            c = V[F].mean(1)
            n['faces'] = F[tree.query(c)[0] >= r].astype('<u2')
    return gone


def walls(track):
    """The tyre walls round the pit entry as the photos show them: one all round the grass it loops
    round, and one along its outer edge from by the side road to the race control. How many stacks."""
    roads = roads_near(track)
    paths = [island_path(roads), east_path(roads)]
    k = gltf_source.load_kn5(bp.TYRES)
    if any(n['name'].startswith('TYRES_ENTRY_1500') for n, _ in bp.meshes(k)):
        return 0
    out = clear_near(k, paths)
    top = bp.Surface(np.concatenate([bp.triangles(n, m) for k_ in (track, gltf_source.load_kn5(bp.PADDOCK)) for n, m in bp.meshes(k_)
                                     if n['name'].startswith(('1ROAD', '1GRASS')) and len(n['faces'])]))
    made = lay_wall(k, 'ENTRY_1500_ISLAND', paths[0], top) + lay_wall(k, 'ENTRY_1500_EAST', paths[1], top)
    bp.save(k, bp.TYRES)
    print('  %d tyre stacks round the pit entry out, %d laid again in two walls' % (out, made))
    return made


def main():
    track = gltf_source.load_kn5(bp.TRACK)
    road = [(n, m) for n, m in bp.meshes(track) if n['name'] == ROAD]
    W = np.concatenate([bp.triangles(n, m).reshape(-1, 3) for n, m in road])
    near = W[(W[:, 2] > 98.) & (W[:, 2] < 102.)]
    if len(near) and near[:, 0].max() > DONE_X:
        print('the pit entry is moved already')
        return
    for n, m in road:
        move_mesh(n, m)
    bp.save(track, bp.TRACK)
    tyres = gltf_source.load_kn5(bp.TYRES)
    t = sum(move_mesh(n, m) for n, m in bp.meshes(tyres) if n['name'] in TYRES)
    bp.save(tyres, bp.TYRES)
    path = os.path.join(bp.MODELS, 'velopark_lines.glb')
    lines = gltf_source.load_kn5(path)
    li = sum(move_mesh(n, m, in_corridor) for n, m in bp.meshes(lines) if len(n['verts']))
    bp.save(lines, path)
    boxes = 0
    for lay in LAYOUTS:
        f = os.path.join(bp.MODELS, lay + '.glb')
        if not os.path.exists(f):
            continue
        k = gltf_source.load_kn5(f)
        moved = False
        import kn5tree
        for n, m in kn5tree.walk(k['root']):
            if n['kind'] == 1 and n['name'].startswith('AC_PIT_'):
                M4 = np.array(n['matrix'], float).reshape(4, 4)
                o = M4[3, :3]
                if in_corridor(o[0], o[2]):
                    M4[3, 0] += float(dx(o[2], o[0]))
                    n['matrix'] = list(M4.ravel())
                    boxes += 1
                    moved = True
        if moved:
            bp.save(k, f)
    ai = 0
    for f in glob.glob(os.path.join(bp.ROOT, 'source', 'track', '*', 'ai', 'pit_lane.ai')):
        b = bytearray(open(f, 'rb').read())
        _, n_, _, _ = struct.unpack_from('<4i', b, 0)
        P = np.frombuffer(bytes(b[16:16 + n_ * 20]), '<f4').reshape(-1, 5).copy()
        sel = in_corridor(P[:, 0], P[:, 2])
        if sel.any():
            P[sel, 0] += dx(P[sel, 2], P[sel, 0]).astype('<f4')
            b[16:16 + n_ * 20] = P.astype('<f4').tobytes()
            open(f, 'wb').write(bytes(b))
            ai += int(sel.sum())
    walls(gltf_source.load_kn5(bp.TRACK))
    print('the vp_1500\'s pit entry %.1f m west under the pit building\'s roof: its road, %d points of its tyre wall, '
          '%d of the lines on it, %d pit boxes, %d pit lane AI points' % (SHIFT, t, li, boxes, ai))


if __name__ == '__main__':
    main()
