"""Everything laid on the ground made one surface (tools/surface.py), every merge proved where it is made.

    python tools/merge_floor.py [--near X0 Z0 X1 Z1] [--dry-run] [--keep]

After the drape and the pavers, the floors of the models (MODELS) are taken as they lie: their tops, the
faces facing up (their sides and skirts are not needed once they meet edge to edge). They go in this
order, each over those after it:

1. the rumbles described in roads.json (tools/build_kerbs.py's geometry), the old kerbs along their roads out;
2. the old kerbs left (their tops);
3. the roads described, in their order (roads.json's, the editor's), each made again from its description,
   its rows and columns, its points as high as the drape laid them;
4. the other roads and the paving (the paddock, the kids' track, the paths, the beds, the patches);
5. the grass laid on the ground (beds, islands, run-offs);
6. the near ground (1GRASS_0), where it lies within NEAR of any of them. The rest of it, and its edge
   there, stay as they are: the merge meets that edge point for point.

The drape's ramps (1GRASS_EDGE_RAMP) and the folds go: the ground eases up to every edge itself, over
surface.RAMP. Each region is proved as soon as it is filled, then the whole (surface.check), and the edge
with the ground left as it was. Any problem stops the build, with where it is. The meshes are written
back to their models, each with its material and its mapping, one normal at each point across the seams;
the paint under the rumbles goes.

--near X0 Z0 X1 Z1: only that part of the map, built and proved but not written (a quick look at one place);
--all: all of it merged again (else only round what changed since the last merge, the rest kept as it made
it: tools/merge_state.py); --dry-run: all of it built and proved, not written; --models DIR: the floors read from there (never written); --keep: the regions saved (build/cache), for
tools/surface.py's merge to be tried on them again by itself (python tools/merge_floor.py --again).
"""
import json
import os
import re
import sys
import time

import numpy as np

import build_kerbs as bk
import build_paths as bp
import gltf_source
import roads as rd
import surface as sf
sf.VERBOSE = True
sf.WORKERS = max(1, (os.cpu_count() or 2) - 2)

MODELS = ['velopark.glb', 'velopark_kerbs.glb', 'velopark_kids.glb', 'velopark_paddock.glb', 'velopark_runoffs.glb']
FLOOR = ('1ROAD', '1KERB', '1GRASS', 'GRASS', 'KERB')
FAR = ('GRASS_1', 'FAR_GROUND')                               # (the far ground: past the near ground's edge)
DROP = ('1GRASS_EDGE_RAMP', 'GRASS_FOLDS', 'SKIRT_', 'KERB_BED_SKIRT_', '1ROAD_KERB_E_')
SIDE = re.compile(r'^KERB_.*_SIDE$')                          # (an old kerb's side: under the surface now)
NEAR = 6.            # metres round the floors the near ground is merged with them
UP = .3              # a face is a top when its normal rises at least this much of its length
CRACK = 1.           # square metres: a gap in the ground's mesh smaller than this is a crack in it, not a hole
BEFORE = os.path.join(bp.ROOT, 'build', 'cache', 'floors_before_merge')   # (the floors' models as the merge found them)
MERGED = os.path.join(bp.MODELS, 'merged_floors.json')   # (the meshes it made, by model: proved by it)
ISSUES = os.path.join(bp.ROOT, 'build', 'cache', 'merge_floor_problems.json')
KEPT = os.path.join(bp.ROOT, 'build', 'cache', 'merge_floor_regions.pkl')   # (--keep: the regions, to try the merge on again)
EDGE_TOL = 1e-6      # metres: the merge's points on the ground left as it was, as high as its own there


def load(where=None):
    """The floor models: {file: kn5} (from where: another folder of them, its textures not read)."""
    if where:
        def read(p):
            doc, binary = gltf_source.read_glb(p)
            return gltf_source.from_gltf(doc, binary, lambda uri: b'', lambda name: b'')
        return {f: read(os.path.join(where, f)) for f in MODELS if os.path.exists(os.path.join(where, f))}
    return {f: gltf_source.load_kn5(os.path.join(bp.MODELS, f)) for f in MODELS if os.path.exists(os.path.join(bp.MODELS, f))}


def tops(n, m):
    """A floor mesh's tops: its triangles (x, y, z) facing up, their mapping (u, v), and which of its faces
    they are. A face facing down is taken (turned up) only where none of its tops lies over it."""
    import shapely
    f = n['faces'].astype(int)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    idx = np.nonzero(ok)[0]
    p = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
    T, UV = p[f[idx]], n['verts'][:, 6:8].astype(float)[f[idx]]
    nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(nn, axis=1)
    steep = np.abs(nn[:, 1]) < UP * np.maximum(ln, 1e-12)
    up, down = (nn[:, 1] > 0) & ~steep, (nn[:, 1] < 0) & ~steep
    if down.any() and up.any():
        P = shapely.polygons(np.concatenate([T[:, :, [0, 2]], T[:, :1, [0, 2]]], axis=1))
        tree = shapely.STRtree(P[up])
        a, b = tree.query(P[down], predicate='intersects')
        if len(a):
            hit = shapely.area(shapely.intersection(P[down][a], P[up][b])) > 1e-8
            under = np.zeros(int(down.sum()), bool)
            under[a[hit]] = True
            d = np.nonzero(down)[0]
            down[d[under]] = False
    keep = up | down
    return T[keep], UV[keep], idx[keep]


def kind(name, described):
    """What a floor mesh is in the merge: 'road' (made again from its description), 'drop', 'ground',
    'kerb' (an old kerb's top), 'paved', 'grass'; None: not a floor."""
    u = name.upper()
    if not u.startswith(FLOOR):
        return None
    if u.startswith(FAR):
        return 'far'
    if name in described:
        return 'road'
    if name.startswith(DROP) or SIDE.match(name):
        return 'drop'
    if name == bp.GROUND or re.fullmatch(re.escape(bp.GROUND) + r'_\d+', name) or name == bp.FILLER:
        return 'ground'
    if name.startswith('1ROAD_KERB_') and not name.startswith('1ROAD_KERB_BED_'):
        return 'kerb'
    if u.startswith(('1ROAD', '1KERB', 'KERB')):
        return 'paved'
    return 'grass'


def standing(models):
    """What stands on the floor (a building, a stand, a tyre wall: anything not a floor), seen from above:
    its triangles' shapes, in a tree."""
    import shapely
    polys = []
    for f, k in models.items():
        for n, m in bp.meshes(k):
            if not len(n['faces']) or n['name'].upper().startswith(FLOOR) or n['name'].startswith('AC_'):
                continue
            X = bp.triangles(n, m)[:, :, [0, 2]]
            a = np.abs((X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1]) - (X[:, 2, 0] - X[:, 0, 0]) * (X[:, 1, 1] - X[:, 0, 1]))
            X = X[a > 1e-6]
            if len(X):
                polys.append(shapely.polygons(np.concatenate([X, X[:, :1]], axis=1)))
    if not polys:
        return None
    polys = np.concatenate(polys)
    return polys, shapely.STRtree(polys)


def walled(ring, stand, share=.5):
    """Whether a gap (one no bigger than a crack never is) lies mostly under what stands there: inside a
    building, not a gap to fill."""
    import shapely
    from shapely.geometry import Polygon
    h = Polygon(ring)
    if stand is None or h.area <= CRACK:
        return False
    polys, tree = stand
    hit = tree.query(h, predicate='intersects')
    if not len(hit):
        return False
    return shapely.union_all(polys[hit]).intersection(h).area >= share * h.area


def order_key(name):
    """The paved and the grass among themselves: a kerb's bed first, the paving, then by name."""
    return (0 if 'KERB_BED' in name or 'KERB_FILL' in name else 1 if name.startswith('1KERB') else 2, name)


def described_regions(roads, models, window):
    """The rumbles' and the described roads' regions, the first over all after: [(Region, kind, model,
    template node, material)], their roads' meshes as the drape laid them read for their heights."""
    of = {r['name']: r['name'] for r in roads.values()}
    of.update({r['mesh']: r['name'] for r in roads.values() if r.get('mesh') and r['mesh'] not in roads})
    laid = {}
    for f, k in models.items():
        for n, m in bp.meshes(k):
            if n['name'] in of and len(n['faces']):
                laid.setdefault(of[n['name']], []).append((f, n, m))
    out = []
    meshes = {}
    for name in rd.priority(roads):
        r = roads[name]
        rr = rd.along_road(r, roads) if 'along' in r else r
        got = laid.get(name)
        if not got:
            raise SystemExit('%s: described but not laid (run tools/build_roads.py and the drape first)' % name)
        # (as tools/build_roads.py laid it, in the track's model; an old mesh of that name in another model, a
        # road it took the place of, not)
        got = [g for g in got if g[0] == 'velopark.glb'] or got
        T = np.concatenate([bp.triangles(n, m) for _, n, m in got])
        S = bp.Surface(T)
        from scipy.spatial import cKDTree
        V = T.reshape(-1, 3)
        vt = cKDTree(V[:, [0, 2]])

        kv = sf.key(V[:, [0, 2]])
        o = np.argsort(kv)
        ks, ky = kv[o], V[o, 1]

        def height(x, z, S=S, V=V, vt=vt, ks=ks, ky=ky):
            # (its own points by where they are, all at once: the drape laid each on its description's point;
            # any other where its mesh lies there, or its nearest point)
            q = np.c_[np.atleast_1d(x), np.atleast_1d(z)]
            k_ = sf.key(q)
            i = np.clip(np.searchsorted(ks, k_), 0, len(ks) - 1)
            y = np.where(ks[i] == k_, ky[i], np.nan)
            gone = np.isnan(y)
            if gone.any():
                y[gone] = S.ys(q[gone])
                still = np.isnan(y)
                if still.any():
                    y[still] = V[vt.query(q[still])[1], 1]
            return y - rd.ROAD
        meshes[name] = rd.mesh(rr, height)            # (its points as high as the drape laid them)
    allT = np.concatenate([P[F] for P, U, F in meshes.values()])
    top_s = bp.Surface(allT)
    av = allT.reshape(-1, 3)
    from scipy.spatial import cKDTree
    at = cKDTree(av[:, [0, 2]])

    def top(p):
        y = top_s.ys(p)
        gone = np.isnan(y)
        if gone.any():
            y[gone] = av[at.query(p[gone])[1], 1]
        return y
    kerbs = models['velopark_kerbs.glb']
    tpl = {}
    for n, _ in bp.meshes(kerbs):
        mat = kerbs['materials'][n['material']]['name']
        if n['name'].startswith('1ROAD_KERB_') and mat not in tpl and n['verts'].shape[0]:
            tpl[mat] = n
    for name in rd.priority(roads):
        r = roads[name]
        num = re.sub(r'^1(ROAD|KERB)_', '', name)
        for i, k in enumerate(r.get('kerbs') or []):
            for P, U, F, mat, suffix in bk.geometry(r, k, roads, top):
                T, TU = crop(P[F], U[F], window)
                if len(T):
                    out.append((region(bk.NAME % (num, i, suffix), T, TU), 'rumble', 'velopark_kerbs.glb',
                                tpl.get(mat) or next(iter(tpl.values())), mat))
    for name in rd.priority(roads):
        P, U, F = meshes[name]
        T, TU = crop(P[F], U[F], window)
        f, n, m = laid[name][0]
        if len(T):
            out.append((region(name, T, TU), 'road', f, n, roads[name].get('material') or models[f]['materials'][n['material']]['name']))
    return out, laid


def region(name, T, UV, ground=False):
    """A region of the merge, its setup (outline, folds) as worked out before for the same triangles, if it
    was (merge_state's), else worked out and kept."""
    import merge_state as ms
    fp = ms.fingerprint(T, UV)
    k = ms.known(fp)
    r = sf.Region(name, tris=T, uvs=UV, ground=ground, known=k)
    if k is None:
        ms.remember(fp, r)
    r.fp = fp
    return r


def crop(T, TU, window, margin=2.):
    if window is None:
        return T, TU
    x0, z0, x1, z1 = window
    c = T[:, :, [0, 2]]
    sel = (c[:, :, 0].max(1) >= x0 - margin) & (c[:, :, 0].min(1) <= x1 + margin) & (c[:, :, 1].max(1) >= z0 - margin) & (c[:, :, 1].min(1) <= z1 + margin)
    return T[sel], TU[sel]


def main():
    import shapely
    from shapely.geometry import box
    args = sys.argv[1:]
    window = tuple(float(a) for a in args[args.index('--near') + 1:args.index('--near') + 5]) if '--near' in args else None
    t0 = time.time()
    if '--again' in args:
        return again()
    roads = rd.load()
    where = args[args.index('--models') + 1] if '--models' in args else None
    models = load(where)
    described = {r.get('mesh', r['name']) for r in roads.values()} | set(roads)
    # (the old kerbs along the roads with rumbles described, in any of the models (the kids' in theirs): out, every part of them)
    gone = set().union(*(bk.old_kerbs(k, roads) for k in models.values())) if any(r.get('kerbs') for r in roads.values()) else set()
    regs, laid = described_regions(roads, models, window)
    # the floors as they lie, each by its kind
    found = {}                     # (model, name) -> [(node, matrix)]
    dropped, nodes_of = [], {}
    for f, k in models.items():
        for n, m in bp.meshes(k):
            if not len(n['faces']):
                continue
            kd = kind(n['name'], described)
            if kd is None:
                continue
            if kd in ('road', 'drop') or any(bk.parts_of(b, n['name']) for b in gone):
                dropped.append((f, n))
                continue
            found.setdefault((kd, f, n['name']), []).append((n, m))
    ground_parts = []              # (node, its faces taken as the ground's, their triangles, uv)
    groups = {'kerb': [], 'paved': [], 'grass': [], 'far': []}
    for (kd, f, name), parts in found.items():
        if kd == 'ground':
            for n, m in parts:
                T, UV, idx = tops(n, m)
                ground_parts.append((f, n, m, T, UV, idx))
            continue
        T = np.concatenate([tops(n, m)[0] for n, m in parts])
        UV = np.concatenate([tops(n, m)[1] for n, m in parts])
        T, UV = crop(T, UV, window)
        if len(T):
            groups[kd].append((name, f, parts[0][0], T, UV))
    names = {r.name for r, *_ in regs}
    for kd in ('kerb', 'paved', 'grass', 'far'):
        for name, f, n, T, UV in sorted(groups[kd], key=lambda g: order_key(g[0])):
            nm = name if name not in names else '%s@%s' % (name, f)
            names.add(nm)
            regs.append((region(nm, T, UV), kd, f, n, models[f]['materials'][n['material']]['name']))
    print('%d regions: %d rumbles, %d roads, %d old kerbs, %d paved, %d grass, %d far ground (%.0f s)' % (
        len(regs), sum(1 for x in regs if x[1] == 'rumble'), sum(1 for x in regs if x[1] == 'road'), len(groups['kerb']),
        len(groups['paved']), len(groups['grass']), len(groups['far']), time.time() - t0))
    # the near ground, where it lies within NEAR of them
    t = time.time()
    outl = [r.outline_ for r, *_ in regs]
    GT = np.concatenate([g[3] for g in ground_parts])
    GU = np.concatenate([g[4] for g in ground_parts])
    owner = np.concatenate([np.full(len(g[3]), i) for i, g in enumerate(ground_parts)])
    sel = np.zeros(len(GT), bool)
    if window is not None:
        GP = shapely.polygons(np.concatenate([GT[:, :, [0, 2]], GT[:, :1, [0, 2]]], axis=1))
        tree = shapely.STRtree(GP)
        cover = shapely.union_all(outl).buffer(NEAR, quad_segs=2).intersection(box(*window))
        shapely.prepare(cover)
        sel[tree.query(cover, predicate='intersects')] = True
        sel &= shapely.within(GP, box(*window))
    else:
        sel[:] = True                                  # (all of it: the whole map, one surface)
    ground = region(bp.GROUND, GT[sel], GU[sel], ground=True)
    if window is None:
        # (the world: the whole map, every floor's outline and the ground's; its gaps, where nothing lies,
        # ground too (the cracks a T in a mesh leaves, the corners nothing covered), but those a building or
        # anything else standing there meets all round (its walls on their edge: inside it))
        from shapely.geometry import Polygon, MultiPolygon
        import merge_state as ms
        fps = [r.fp for r, *_ in regs] + [ground.fp]
        W = ms.world_known(fps)
        if W is None:
            U_ = shapely.union_all(outl + [ground.outline_])
            stand = standing(models)
            parts = []
            for p in shapely.get_parts(U_):
                if p.area <= CRACK:
                    continue
                parts.append(Polygon(p.exterior, [h for h in p.interiors if walled(h, stand)]))
            W = shapely.set_precision(MultiPolygon(parts) if len(parts) > 1 else parts[0], sf.GRID)
            ms.world_remember(fps, W)
    else:
        W = sf.outline(GT[sel])
    W = shapely.set_precision(W, sf.GRID)
    holes = sum(len(p.interiors) for p in shapely.get_parts(W))
    regs.append((ground, 'ground', 'velopark.glb', next(g[1] for g in ground_parts if g[1]['name'] == bp.GROUND), None))
    print('the near ground: %d of its %d faces merged; the world %.0f m2 (%d pieces, %d holes: inside what stands there) (%.0f s)' % (
        sel.sum(), len(GT), W.area, len(shapely.get_parts(W)), holes, time.time() - t))
    if '--keep' in args:
        import pickle
        pickle.dump(([(r.name, r.tris, r.uvs, r.ground) for r, *_ in regs], W, GT[~sel].reshape(-1, 3)), open(KEPT, 'wb'))
    # the merge, each region proved as it is filled: only where something changed since the last one
    # (tools/merge_state.py), unless there is none kept, or the world changed, or --all
    t = time.time()
    import merge_state as ms
    kinds = [kd for _, kd, *_ in regs]
    plan_ = ms.plan([r for r, *_ in regs], kinds, W) if window is None and '--all' not in args and not (~sel).any() else None
    edge_issues = []
    try:
        if plan_ is None:
            fixed = edge_heights(W, GT[~sel].reshape(-1, 3)) if window is None and (~sel).any() else None
            result, faces, own = sf.build([r for r, *_ in regs], W, fixed=fixed)
            print('merged all of it:', end=' ')
        elif not plan_['changed']:
            result = plan_['kept']
            print('nothing changed since the last merge: it kept, all of it;', end=' ')
        else:
            zone = plan_['zone']
            local = [x for x in (ms.crop(r, zone) for r, *_ in regs) if x is not None]
            print('changed since the last merge: %s; made again: %.0f m2 round them (%d regions there), the rest kept' % (
                ', '.join(plan_['changed'][:12]) + (' and %d more' % (len(plan_['changed']) - 12) if len(plan_['changed']) > 12 else ''),
                zone.area, len(local)))
            made, faces, own = sf.build(local, zone, fixed=plan_['fixed'])
            kept, edge_issues, welded = zone_edge(made, zone, plan_['kept'], plan_['fixed'])
            result = ms.join(kept, made)
            print('    its edge on what was kept: %d points welded into the kept faces' % welded)
            print('merged:', end=' ')
    except sf.MergeError as e:
        print('MERGE FAILED at %s:' % e.name)
        for what, x, z, how in e.issues[:20]:
            print('    %s at (%.2f, %.2f) %.4g' % (what, x, z, how))
        return 1
    n_faces = sum(len(F) for P, U, F in result.values())
    print('%d faces in %.0f s' % (n_faces, time.time() - t))
    # the whole proved, and its edge on the ground left as it was
    t = time.time()
    issues = sf.check(result, W) + edge_issues
    welds = np.zeros((0, 3))
    if window is None and (~sel).any():
        more, welds = edge_check(result, W, GT[~sel].reshape(-1, 3))
        issues += more
        print('    its edge on the ground left as it was: %d points on the edges of its triangles, to weld there' % len(welds))
    print('proved in %.0f s: %s' % (time.time() - t, 'no problems' if not issues else '%d PROBLEMS' % len(issues)))
    if issues:
        report(issues)
        return 1
    if window is None:
        ms.save([r for r, *_ in regs], kinds, result, W)          # (for the next: only what changes made again)
    if window is not None or '--dry-run' in args or where:
        print('(not written)')
        return 0
    # (the floors' models as they were before it, kept: the road editor reads a road from its mesh there,
    # not from the merged one, cut where others lie over it)
    import shutil
    os.makedirs(BEFORE, exist_ok=True)
    for f in models:
        shutil.copy2(os.path.join(bp.MODELS, f), os.path.join(BEFORE, f))
    write(models, regs, result, ground_parts, sel, owner, dropped, W, welds)
    print('written in %.0f s all told: %d regions; %d meshes out (%s)' % (
        time.time() - t0, len(result), len(dropped), ', '.join(sorted({n['name'] for _, n in dropped}))[:600]))
    return 0


def again():
    """The merge tried again on the regions --keep saved, by itself: built and proved, nothing written."""
    import pickle
    t = time.time()
    saved, W, rest = pickle.load(open(KEPT, 'rb'))
    regs = [sf.Region(n, tris=T, uvs=U, ground=g) for n, T, U, g in saved]
    print('%d regions read (%.0f s)' % (len(regs), time.time() - t))
    t = time.time()
    try:
        result, faces, own = sf.build(regs, W, fixed=edge_heights(W, rest))
    except sf.MergeError as e:
        print('MERGE FAILED at %s (%.0f s):' % (e.name, time.time() - t))
        for what, x, z, how in e.issues[:20]:
            print('    %s at (%.2f, %.2f) %.4g' % (what, x, z, how))
        return 1
    print('merged: %d faces in %.0f s' % (sum(len(F) for P, U, F in result.values()), time.time() - t))
    more, welds = edge_check(result, W, rest)
    print('%d points to weld' % len(welds))
    issues = sf.check(result, W) + more
    print('no problems' if not issues else '%d PROBLEMS' % len(issues))
    report(issues)
    if issues:
        # (the faces round the first problems, and the merge's faces there, kept to look at)
        import pickle
        pts = np.array([[x, z] for _, x, z, _ in sorted(issues, key=lambda i: -i[3])[:40]])
        near = {}
        for n, (P, U, F) in result.items():
            c = P[F][:, :, [0, 2]].mean(1)
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pts).query(c)
            sel = d < 3.
            if sel.any():
                near[n] = P[F[sel]]
        fz = [(f, o) for f, o in zip(faces, own) if f.distance(__import__('shapely').Point(*pts[0])) < 1.]
        pickle.dump(dict(issues=issues, near=near, faces=fz), open(os.path.join(bp.ROOT, 'build', 'cache', 'merge_floor_problem_area.pkl'), 'wb'))
    return 1 if issues else 0


def zone_edge(made, zone, kept, fixed):
    """The part made again on the part kept (merge_state's): every point of its edge one of the kept faces'
    (as high), or on one of their edges (welded into it there). The kept faces (welded), the problems, how
    many welded."""
    import shapely
    import merge_state as ms
    B = zone.boundary
    shapely.prepare(B)
    issues, todo = [], []
    for n, (P, U, F) in made.items():
        on = shapely.distance(B, shapely.points(P[:, [0, 2]])) < sf.GRID * 2
        for p in P[on]:
            k_ = int(sf.key(p[[0, 2]]))
            if k_ in fixed:
                if abs(fixed[k_] - p[1]) > EDGE_TOL:
                    issues.append(('edge point moved (%s)' % n, p[0], p[2], p[1] - fixed[k_]))
            else:
                todo.append(p)
    kept = dict(kept)
    welded = 0
    if todo:
        todo = np.array(todo)
        lo, hi = todo[:, [0, 2]].min(0) - 1., todo[:, [0, 2]].max(0) + 1.
        left = np.ones(len(todo), bool)
        for n, (P, U, F) in list(kept.items()):
            if not len(F) or not left.any():
                continue
            c = P[F][:, :, [0, 2]]
            if (c.max((0, 1)) < lo).any() or (c.min((0, 1)) > hi).any():
                continue
            mine = np.array([left[i] and on_edge(P[F], q) is not None for i, q in enumerate(todo)])
            if mine.any():
                P, U, F, d = ms.weld(P, U, F, todo[mine])
                kept[n] = (P, U, F)
                welded += d
                left &= ~mine
        for p in todo[left]:
            issues.append(("edge point on nothing kept", p[0], p[2], 0.))
    return kept, issues, welded


def edge_heights(W, rest):
    """The heights the merge keeps on its edge (the world's): the ground left as it was, its points there
    (where it has two at one place, a step in it, the higher)."""
    import shapely
    on = shapely.distance(W.boundary, shapely.points(rest[:, [0, 2]])) < sf.GRID * 2
    out = {}
    for k_, y in zip(sf.key(rest[on][:, [0, 2]]).tolist(), rest[on][:, 1].tolist()):
        out[k_] = max(y, out.get(k_, -np.inf))
    return out


def edge_check(result, W, rest):
    """The merge's edge on the ground left as it was: every point of it there one of that ground's, as high;
    or on one of its triangles' edges (a T it had, from the drape's cuts: that triangle to be split there,
    welded). Its problems, and the points to weld (x, y, z)."""
    import shapely
    fixed = edge_heights(W, rest)
    T = rest.reshape(-1, 3, 3)
    L = sf.outline(T).boundary
    shapely.prepare(L)
    issues, welds = [], []
    for n, (P, U, F) in result.items():
        on = shapely.distance(W.boundary, shapely.points(P[:, [0, 2]])) < sf.GRID * 2
        on[on] = shapely.distance(L, shapely.points(P[on][:, [0, 2]])) < sf.GRID * 2      # (where it meets that ground)
        for p in P[on]:
            k_ = int(sf.key(p[[0, 2]]))
            if k_ in fixed:
                if abs(fixed[k_] - p[1]) > EDGE_TOL:
                    issues.append(('edge point moved (%s)' % n, p[0], p[2], p[1] - fixed[k_]))
            elif on_edge(T, p) is not None:
                welds.append(p)
            else:
                issues.append(("edge point not the ground's (%s)" % n, p[0], p[2], 0.))
    return issues, np.array(welds).reshape(-1, 3)


def on_edge(T, p, tol=sf.GRID * 2):
    """The triangle (of T, (n, 3, 3)) and which of its edges (0: a-b, 1: b-c, 2: c-a) has p (x, y, z) on
    it, between its ends; None if none."""
    X = T[:, :, [0, 2]]
    q = np.asarray(p, float)[[0, 2]]
    near = np.nonzero((X.min(1) <= q + tol).all(1) & (X.max(1) >= q - tol).all(1))[0]
    for i in near:
        for j in range(3):
            a, b = X[i, j], X[i, (j + 1) % 3]
            ab = b - a
            L2 = ab @ ab
            if L2 < 1e-12:
                continue
            t = ((q - a) @ ab) / L2
            off = abs((q[0] - a[0]) * ab[1] - (q[1] - a[1]) * ab[0]) / np.sqrt(L2)
            if 1e-6 < t < 1 - 1e-6 and off < tol:
                return i, j
    return None


def weld(n, points):
    """A mesh's triangles split at points on their edges (x, y, z): each such triangle in two, the point a
    corner of both, as high as it is given, its mapping and normal between the edge's ends."""
    V = n['verts'].astype(float)
    F = n['faces'].astype(int)
    done = 0
    for p in points:
        T = V[F][:, :, :3]
        hit = on_edge(T, p)
        if hit is None:
            continue
        i, j = hit
        a, b, c = F[i, j], F[i, (j + 1) % 3], F[i, (j + 2) % 3]
        A, B = V[a, [0, 2]], V[b, [0, 2]]
        t = ((p[[0, 2]] - A) @ (B - A)) / ((B - A) @ (B - A))
        v = V[a] + (V[b] - V[a]) * t
        v[:3] = p
        V = np.r_[V, v[None]]
        m = len(V) - 1
        F = np.r_[np.delete(F, i, 0), [[a, m, c], [m, b, c]]]
        done += 1
    n['verts'] = V.astype(n['verts'].dtype)
    n['faces'] = F.astype('<u2' if len(V) < 65536 else '<u4')
    return done


def report(issues):
    """Problems printed (by kind, the first of each) and all of them kept in build/cache."""
    import json
    from collections import Counter
    if not issues:
        return
    print('   ', dict(Counter(w for w, *_ in issues)))
    seen = Counter()
    for what, x, z, how in issues:
        seen[what] += 1
        if seen[what] <= 5:
            print('    %s at (%.2f, %.2f) %.4g' % (what, x, z, how))
    json.dump([(w, float(x), float(z), float(h)) for w, x, z, h in issues], open(ISSUES, 'w'), indent=0)
    print('    (all of them: %s)' % os.path.relpath(ISSUES, bp.ROOT))


def write(models, regs, result, ground_parts, sel, owner, dropped, W, welds):
    """The merged meshes into their models (see the module's note), the meshes they replace emptied."""
    import restyle_paddock as rp
    # (one normal at each point, of every face round it, across the seams and the ground left as it was)
    Ps, Fs, base = [], [], 0
    for P, U, F in result.values():
        Ps.append(P)
        Fs.append(F + base)
        base += len(P)
    welded = 0
    import shapely
    for i, (f, n, m, T, UV, idx) in enumerate(ground_parts):
        mine = owner == i
        merged = idx[sel[mine]]
        allf = np.ones(len(n['faces']), bool)
        allf[merged] = False
        # (and its faces that are not tops (steep, or under others of its own) where the merge is: out)
        Pn = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
        other = np.ones(len(n['faces']), bool)
        other[idx] = False
        if other.any():
            c = Pn[n['faces'].astype(int)[other]].mean(1)
            o = np.nonzero(other)[0]
            allf[o[shapely.contains_xy(W, c[:, 0], c[:, 2])]] = False
        n['faces'] = n['faces'][allf].astype(n['faces'].dtype)
        if len(welds) and np.allclose(m, np.eye(4)) and len(n['faces']):
            welded += weld(n, welds)
        Pn = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, :3]
        Ps.append(Pn)
        Fs.append(n['faces'].astype(int) + base)
        base += len(Pn)
    if welded != len(welds):
        raise SystemExit('%d of the %d points to weld the ground left as it was at not found on its edges' % (len(welds) - welded, len(welds)))
    P = np.concatenate(Ps)
    F = np.concatenate(Fs)
    k = sf.key(P[:, [0, 2]])
    uk, inv = np.unique(k, return_inverse=True)
    inv = inv.ravel()
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    fn[fn[:, 1] < 0] *= -1
    N = np.zeros((len(uk), 3))
    for c in range(3):
        np.add.at(N, inv[F[:, c]], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    normal = dict(zip(uk.tolist(), map(tuple, N)))

    def normals_of(Q):
        return np.array([normal.get(x, (0., 1., 0.)) for x in sf.key(Q[:, [0, 2]]).tolist()])
    for f, n in dropped:
        n['faces'] = n['faces'][:0]
    # (the ground left as it was: its merged faces out of it, its points' normals as the merge's at its edge)
    for f, n, m, T, UV, idx in ground_parts:
        if np.allclose(m, np.eye(4)) and len(n['faces']):
            used = np.unique(n['faces'].astype(int))
            Q = n['verts'][used, :3].astype(float)
            Nn = normals_of(Q)
            n['verts'][used, 3:6] = Nn.astype(n['verts'].dtype)
        bp.sphere(n)
    # (each region's mesh: in place of the meshes it was, in its material)
    taken = set()
    for r, kd, f, tpl, mat in regs:
        k = models[f]
        if kd in ('kerb', 'paved', 'grass', 'far'):
            name = r.name.split('@')[0]
            for n, _ in bp.meshes(k):
                if n['name'] == name:
                    n['faces'] = n['faces'][:0]      # (even one wholly under others: out)
        else:
            name = r.name
        if r.name not in result:
            continue
        P, U, F = result[r.name]
        V = np.zeros((len(P), tpl['verts'].shape[1]))
        V[:, :3], V[:, 3:6], V[:, 6:8] = P, normals_of(P), U
        if V.shape[1] > 8:
            V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
        mats = [x['name'] for x in k['materials']]
        if mat in mats:
            mi = mats.index(mat)
        elif tpl in [x for x, _ in bp.meshes(k)]:
            mi = tpl['material']
        else:
            raise SystemExit('%s: no material %s in %s' % (r.name, mat, f))
        node = dict(tpl, name=name, children=[], material=mi, verts=V.astype('<f4'),
                    faces=F.astype('<u2' if len(V) < 65536 else '<u4'))
        bp.sphere(node)
        k['root']['children'].append(node)
        taken.add((f, name))
    for f, k in models.items():
        bp.save(k, os.path.join(bp.MODELS, f))
    # (the paint: out under the rumbles (an old line along an old kerb, say), and cut to the roads as they lie
    # now, none of it over the grass: tools/paint_roads.py, after the floor is written, which it reads)
    lines_path = os.path.join(bp.MODELS, 'velopark_lines.glb')
    if os.path.exists(lines_path):
        import shapely
        import paint_roads as pr
        L = gltf_source.load_kn5(lines_path)
        # (only where a rumble rises over its road: a tapering end come down onto another road keeps that road's
        # line over it)
        flat = lambda p: np.zeros(len(p))
        roads = rd.load()
        raised = [sf.outline(P[F][P[F][:, :, 1].max(1) > bk.RAISED]) for r in roads.values() if 'along' not in r
                  for k in r.get('kerbs') or [] for P, U, F, *_ in bk.geometry(r, k, roads, flat)
                  if (P[F][:, :, 1].max(1) > bk.RAISED).any()]
        if raised:
            under = shapely.union_all(raised)
            for n, m in bp.meshes(L):
                if len(n['faces']) and np.allclose(m, np.eye(4)) and n['name'].startswith(('LINE_', 'PAINT')):
                    rp.clip_in(n, under.buffer(.01))
        pr._ROADS.clear()
        print('    %d lines of paint cut back to the roads under them' % pr.paint_on_roads(L))
        pr._FLOOR.clear()
        print('    the paint laid on the roads as they lie: %d points moved, by %.3f m at most' % pr.lay_on_roads(L))
        bp.save(L, lines_path)
    # (the meshes it made, proved here: tools/check_holes.py and tools/seal_holes.py look at the others' edges)
    out = {}
    for f, name in sorted(taken):
        out.setdefault(f, []).append(name)
    json.dump(out, open(MERGED, 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
