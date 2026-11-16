"""The merge kept between builds (tools/merge_floor.py), so the next one makes again only where something changed:
the area round the triangles that changed (in any region: added, gone, moved, mapped otherwise) and those of the
regions that came or went or changed places in the order, MARGIN wide; everything else as the last merge made it,
face for face, and the edge between the two held point for point, at its heights (surface.build's fixed), a point
the new side puts on an old edge welded into it.

    save(regions, result, world)        after a merge: its regions' inputs (their triangles, mapping, kind, order)
                                        and what it made of them, in STATE
    plan(regions, world)                before one: None (no merge kept, or the world changed: all of it made
                                        again), or what to make again and what to keep
"""
import json
import os

import numpy as np

import build_paths as bp
import surface as sf

STATE = os.path.join(bp.ROOT, 'build', 'cache', 'merge_state')
MARGIN = sf.RAMP + 1.          # metres round what changed made again (the ground eases over RAMP: all of it)
VERSION = 1


def tri_keys(tris, uvs):
    """Each triangle as one number: its corners (on the grid, in order round it from the least), their heights
    and mapping (to 0.01 mm, 1e-6): two the same only if they are the same triangle, as high, mapped the same."""
    X = sf.key(sf.snap(tris[:, :, [0, 2]]))
    first = np.argmin(X, axis=1)
    idx = (first[:, None] + np.arange(3)) % 3
    rows = np.arange(len(tris))[:, None]
    k = X[rows, idx]
    y = np.round(tris[:, :, 1][rows, idx] * 1e5).astype(np.int64)
    u = np.round(uvs[rows, idx] * 1e6).astype(np.int64).reshape(len(tris), -1) if uvs is not None else np.zeros((len(tris), 6), np.int64)
    h = np.zeros(len(tris), np.uint64)
    for col in (k[:, 0], k[:, 1], k[:, 2], y[:, 0], y[:, 1], y[:, 2]) + tuple(u.T):
        h = (h * np.uint64(1000003)) ^ col.astype(np.uint64)
    return h


def save(regions, kinds, result, world):
    """A merge kept (see the module's note): regions [Region], kinds [str], result {name: (P, UV, F)}."""
    import shapely
    os.makedirs(STATE, exist_ok=True)
    arrays, index = {}, []
    for i, (r, kd) in enumerate(zip(regions, kinds)):
        index.append(dict(name=r.name, kind=kd, ground=bool(r.ground)))
        arrays['t%d' % i] = r.tris
        arrays['u%d' % i] = r.uvs if r.uvs is not None else np.zeros((len(r.tris), 3, 2))
        if r.name in result:
            P, U, F = result[r.name]
            arrays['P%d' % i], arrays['U%d' % i], arrays['F%d' % i] = P, U, F
    np.savez(os.path.join(STATE, 'merge.npz'), **arrays)
    json.dump(dict(version=VERSION, regions=index, world=shapely.to_wkt(world, rounding_precision=6)),
              open(os.path.join(STATE, 'merge.json'), 'w'))


def load():
    """The merge kept: (index, arrays, world) or None."""
    import shapely
    j, n = os.path.join(STATE, 'merge.json'), os.path.join(STATE, 'merge.npz')
    if not (os.path.exists(j) and os.path.exists(n)):
        return None
    meta = json.load(open(j))
    if meta.get('version') != VERSION:
        return None
    return meta['regions'], np.load(n), shapely.from_wkt(meta['world'])


def plan(regions, kinds, world):
    """What to make again: None (all of it), or dict(zone (the area made again, shapely, its edge the kept
    faces' own), kept {name: (P, UV, F)} the faces kept, fixed {key: height} the points on the zone's edge,
    changed [names], area)."""
    import shapely
    from shapely.geometry import Polygon
    old = load()
    if old is None:
        return None
    index, A, W0 = old
    if abs(W0.area - world.area) > 1e-3 or W0.symmetric_difference(world).area > 1e-3:
        return None                                        # (the world itself changed: all of it again)
    was = {d['name']: i for i, d in enumerate(index)}
    now = [r.name for r in regions]
    # (the order of those that were and are: one that changed places changed with the ones it passed)
    moved = set()
    common = [n for n in now if n in was]
    prev_order = sorted(common, key=lambda n: was[n])
    if prev_order != common:
        for a, b in zip(prev_order, common):
            if a != b:
                moved |= {a, b}
    pieces, changed = [], []

    def polys(T):
        X = T[:, :, [0, 2]]
        return shapely.polygons(np.concatenate([X, X[:, :1]], axis=1))
    for r, kd in zip(regions, kinds):
        if r.name not in was:
            changed.append(r.name)
            pieces.append(polys(r.tris))
            continue
        i = was[r.name]
        T0, U0 = A['t%d' % i], A['u%d' % i]
        if r.name in moved or index[i]['kind'] != kd:
            changed.append(r.name)
            pieces += [polys(T0), polys(r.tris)]
            continue
        k0, k1 = tri_keys(T0, U0), tri_keys(r.tris, r.uvs if r.uvs is not None else np.zeros((len(r.tris), 3, 2)))
        gone, new = ~np.isin(k0, k1), ~np.isin(k1, k0)
        if gone.any() or new.any():
            changed.append(r.name)
            pieces += [polys(T0[gone]), polys(r.tris[new])]
    for n, i in was.items():
        if n not in set(now):
            changed.append(n)
            pieces.append(polys(A['t%d' % i]))
    if not changed:
        return dict(zone=Polygon(), kept={d['name']: (A['P%d' % i], A['U%d' % i], A['F%d' % i]) for i, d in enumerate(index)
                                          if 'P%d' % i in A.files}, fixed={}, changed=[], area=0.)
    allp = np.concatenate([p for p in pieces if len(p)])
    allp = allp[shapely.area(allp) > 1e-12]
    D = shapely.union_all(allp).buffer(MARGIN, quad_segs=4)
    shapely.prepare(D)
    # (the last merge's faces there, whole, made again; their outline the zone, its edge theirs)
    kept, redo = {}, []
    for i, d in enumerate(index):
        if 'P%d' % i not in A.files:
            continue
        P, U, F = A['P%d' % i], A['U%d' % i], A['F%d' % i]
        if d['name'] not in set(now):
            continue                                       # (gone: all of it is in the zone)
        T = P[F][:, :, [0, 2]]
        hit = shapely.intersects(D, polys(P[F]))
        if hit.any():
            redo.append(P[F[hit]])
        kept[d['name']] = (P, U, F[~hit])
    for n in set(was) - set(now):
        i = was[n]
        if 'P%d' % i in A.files:
            P, F = A['P%d' % i], A['F%d' % i]
            redo.append(P[F])
    zone = sf.outline(np.concatenate(redo)) if redo else Polygon()
    zone = shapely.set_precision(zone.intersection(world), sf.GRID)
    # (the heights its edge is held at: the kept faces' points on it)
    fixed = {}
    B = zone.boundary
    shapely.prepare(B)
    for P, U, F in kept.values():
        if not len(F):
            continue
        used = np.unique(F)
        Q = P[used]
        on = shapely.distance(B, shapely.points(Q[:, [0, 2]])) < sf.GRID * 2
        for k_, y in zip(sf.key(Q[on][:, [0, 2]]).tolist(), Q[on][:, 1].tolist()):
            fixed.setdefault(k_, y)
    return dict(zone=zone, kept=kept, fixed=fixed, changed=sorted(changed), area=zone.area)


def crop(region, zone, margin=2.):
    """A region's triangles near the zone (within margin), as a region of its own: the part the merge makes
    again. None if none."""
    import shapely
    X = region.tris[:, :, [0, 2]]
    polys = shapely.polygons(np.concatenate([X, X[:, :1]], axis=1))
    near = shapely.intersects(zone.buffer(margin, quad_segs=2), polys)
    if not near.any():
        return None
    return sf.Region(region.name, tris=region.tris[near], uvs=region.uvs[near] if region.uvs is not None else None,
                     height=region._height, uv=region._uv, spacing=region.spacing, ground=region.ground)


def weld(P, U, F, points):
    """A mesh's triangles (P, UV, F) split at points on their edges (x, y, z): each in two, the point a corner of
    both, at its height, its mapping between the edge's ends. (P, U, F) and how many were welded."""
    import merge_floor as mf
    done = 0
    for p in points:
        hit = mf.on_edge(P[F], p)
        if hit is None:
            continue
        i, j = hit
        a, b, c = F[i, j], F[i, (j + 1) % 3], F[i, (j + 2) % 3]
        A_, B_ = P[a, [0, 2]], P[b, [0, 2]]
        t = ((p[[0, 2]] - A_) @ (B_ - A_)) / ((B_ - A_) @ (B_ - A_))
        P = np.r_[P, [p]]
        U = np.r_[U, [U[a] + (U[b] - U[a]) * t]]
        m = len(P) - 1
        F = np.r_[np.delete(F, i, 0), [[a, m, c], [m, b, c]]]
        done += 1
    return P, U, F, done


def join(kept, made):
    """The kept faces and those made again, each region's one mesh, its points one a place (by key)."""
    out = {}
    for n in set(kept) | set(made):
        parts = [x for x in (kept.get(n), made.get(n)) if x is not None and len(x[2])]
        if not parts:
            continue
        P = np.concatenate([p for p, u, f in parts])
        U = np.concatenate([u for p, u, f in parts])
        F, base = [], 0
        for p, u, f in parts:
            F.append(f + base)
            base += len(p)
        F = np.concatenate(F)
        used = np.unique(F)
        k = sf.key(P[used][:, [0, 2]])
        _, first, inv = np.unique(k, return_index=True, return_inverse=True)
        remap = np.zeros(len(P), int)
        remap[used] = inv.ravel()
        out[n] = (P[used][first], U[used][first], remap[F])
    return out


# ---- what a region's setup worked out (its outline, its folds), kept by its triangles' fingerprint, and the
# world made of them all: worked out again only for triangles not seen before
KNOWN = os.path.join(STATE, 'known')


def fingerprint(tris, uvs):
    """A region's triangles (and mapping) as one name: the same only for the same triangles."""
    import hashlib
    k = np.sort(tri_keys(np.asarray(tris, float), None if uvs is None else np.asarray(uvs, float)))
    return hashlib.sha1(k.tobytes() + str(len(k)).encode()).hexdigest()


def known(fp):
    """(outline, folded) worked out before for a region of that fingerprint, or None."""
    import shapely
    p = os.path.join(KNOWN, fp + '.npz')
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p)
        return shapely.from_wkb(z['outline'].tobytes()), z['folded'].astype(bool)
    except Exception:
        return None


def remember(fp, region):
    import shapely
    os.makedirs(KNOWN, exist_ok=True)
    if region.outline_ is None:
        return
    np.savez(os.path.join(KNOWN, fp + '.npz'), outline=np.frombuffer(shapely.to_wkb(region.outline_), np.uint8),
             folded=region.folded.astype(np.uint8))


def world_known(fps):
    """The world made of regions of these fingerprints (in order) before, or None."""
    import hashlib
    import shapely
    p = os.path.join(KNOWN, 'world_' + hashlib.sha1('|'.join(fps).encode()).hexdigest() + '.wkb')
    return shapely.from_wkb(open(p, 'rb').read()) if os.path.exists(p) else None


def world_remember(fps, world):
    import hashlib
    import shapely
    os.makedirs(KNOWN, exist_ok=True)
    open(os.path.join(KNOWN, 'world_' + hashlib.sha1('|'.join(fps).encode()).hexdigest() + '.wkb'), 'wb').write(shapely.to_wkb(world))
