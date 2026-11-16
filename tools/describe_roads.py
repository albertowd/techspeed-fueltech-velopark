"""The track's roads described (see ROADS.md), read from their meshes as the committed models have them,
into tools/reference/roads.json: each its middle and its width each side, a point every KEEP rows.

    python tools/describe_roads.py [--model MODEL.glb] [--strip] NAME [NAME ...]     (--strip: read from its outline)
    python tools/describe_roads.py --heights [--lines PAINT.glb] NAME [NAME ...]     (to roads described already)

The track's own roads (as Race Track Builder made them) are mapped across from u -1 (one edge) to 0
(the other) and along by v, a row of points at each v: each row's ends are its edges, its middle
between them. Their texture's scale along and across is kept with them (map: along, u and v a metre).
"""
import json, os, sys

import numpy as np

import build_paths as bp
import gltf_source
import roads as rd

KEEP = 2              # every so many of a road's rows a point of its description
EDGE_NEAR, GAP = 1., 2.     # metres: an edge line's way in from the edge at most; a break in one
MOUTH_OUT = 1.        # metres past a road's edge another road there makes a gap in its line a mouth
STUB = 3.             # metres: a piece of old line shorter than this is no edge line
# no mouths: a road's edge line goes on past these (kerbs, patches, run-offs only lying beside it)
NOT_MOUTHS = ('1ROAD_KERB', '1ROAD_PATCH', '1ROAD_RUNOFF', '1KERB')
RUNOFFS = ('1ROAD_41', '1ROAD_30_ASPHALT')


def describe(n, m, material):
    """A road mesh (of the track's, mapped across u -1 to 0, along v) described: a dict as roads.json."""
    V = n['verts'].astype(float)
    W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
    u, v = np.round(V[:, 6], 4), np.round(V[:, 7], 4)
    rows = []
    for vv in np.unique(v):
        r = np.nonzero(v == vv)[0]
        a, b = W[r[np.argmin(u[r])], [0, 2]], W[r[np.argmax(u[r])], [0, 2]]
        rows.append((vv, a, b))
    rows.sort(key=lambda x: x[0])
    A = np.array([r[1] for r in rows])
    B = np.array([r[2] for r in rows])
    mid = (A + B) / 2
    t = np.gradient(mid, axis=0)
    left = np.c_[-t[:, 1], t[:, 0]]
    a_left = ((A - mid) * left).sum(1).mean() > 0
    wl = np.linalg.norm((A if a_left else B) - mid, axis=1)
    wr = np.linalg.norm((B if a_left else A) - mid, axis=1)
    keep = sorted(set(range(0, len(mid), KEEP)) | {len(mid) - 1})
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(mid, axis=0), axis=1))]
    vr = np.array([r[0] for r in rows])
    return {'name': n['name'], 'material': material, 'map': 'along',
            'u_per_m': float(1. / np.median(wl + wr)), 'v_per_m': float((vr[-1] - vr[0]) / max(s[-1], 1e-9)),
            'points': [[round(float(mid[i, 0]), 3), round(float(mid[i, 1]), 3), round(float(wl[i]), 3), round(float(wr[i]), 3)]
                       for i in keep]}


def describe_strip(n, m, material):
    """A road mesh described from its outline as a strip (tools/road_grid.py): its rows' ends its edges,
    their middles its middle; its end rows as they are. Mapped flat from above."""
    import road_grid as rg
    g, why = rg.grid(n, m, sharpest=True)
    if g is None:
        raise SystemExit('%s: %s' % (n['name'], why))
    rows, _ = g
    A, B = rows[:, 0, [0, 2]], rows[:, 1, [0, 2]]
    mid = (A + B) / 2
    t = np.gradient(mid, axis=0)
    left = np.c_[-t[:, 1], t[:, 0]]
    a_left = ((A - mid) * left).sum(1).mean() > 0
    L_, R_ = (A, B) if a_left else (B, A)
    wl, wr = np.linalg.norm(L_ - mid, axis=1), np.linalg.norm(R_ - mid, axis=1)
    road = {'name': n['name'], 'material': material, 'map': 'flat',
            'points': [[round(float(x), 3), round(float(z), 3), round(float(a_), 3), round(float(b_), 3)]
                       for (x, z), a_, b_ in zip(mid, wl, wr)]}
    for key, i in (('start_left', 0), ('end_left', -1)):
        d = (L_[i] - R_[i]) / max(np.linalg.norm(L_[i] - R_[i]), 1e-9)
        road[key] = [round(float(d[0]), 5), round(float(d[1]), 5)]
    return road


def end_rows(n, m, road):
    """A road's end rows as its mesh has them (a road ends on the row the next one begins on, square to
    neither, maybe): at its start and its end, the way from its middle to its left edge (unit, x, z)."""
    V = n['verts'].astype(float)
    W = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
    u, v = np.round(V[:, 6], 4), np.round(V[:, 7], 4)
    vs = np.unique(v)
    mid, left, wl, wr, at = rd.fine(road)
    out = {}
    for key, vv, i in (('start_left', vs[0], 0), ('end_left', vs[-1], -1)):
        r = np.nonzero(v == vv)[0]
        a, b = W[r[np.argmin(u[r])], [0, 2]], W[r[np.argmax(u[r])], [0, 2]]
        d = (a - b) / max(np.linalg.norm(a - b), 1e-9)
        if d @ left[i] < 0:
            d = -d
        out[key] = [round(float(d[0]), 5), round(float(d[1]), 5)]
    return out


def lines_of(road, paint):
    """The edge lines along a road, read from the paint's model (its LINE_ meshes): each a stretch of
    one, along one side, near the edge (EDGE_NEAR in from it at most, a little out), from where it
    begins along the road to where it ends (broken where it stops for GAP or more): its side, the
    mesh's name, from, to, its middle's way in from the edge (pad)."""
    from scipy.spatial import cKDTree
    mid, left, wl, wr, at = rd.fine(road)
    tree = cKDTree(mid)
    out = []
    for n, m in bp.meshes(paint):
        if not n['name'].startswith('LINE_') or not len(n['faces']):
            continue
        V = (np.c_[n['verts'][:, :3].astype(float), np.ones(len(n['verts']))] @ m)[:, [0, 2]]
        d, i = tree.query(V)
        t = np.c_[left[i, 1], -left[i, 0]]
        off = ((V - mid[i]) * left[i]).sum(1)
        along_ok = np.abs(((V - mid[i]) * t).sum(1)) < rd.FINE
        for side, sgn, w in (('left', 1., wl), ('right', -1., wr)):
            pad = w[i] - sgn * off
            ok = along_ok & (sgn * off > 0) & (pad > -.3) & (pad < EDGE_NEAR)
            if ok.sum() < 4:
                continue
            sv = np.sort(at[i[ok]])
            breaks = np.nonzero(np.diff(sv) > GAP)[0]
            for a0, a1 in zip(np.r_[0, breaks + 1], np.r_[breaks, len(sv) - 1]):
                if sv[a1] - sv[a0] < 1.:
                    continue
                inseg = ok & (at[i] >= sv[a0]) & (at[i] <= sv[a1])
                out.append({'side': side, 'name': n['name'], 'from': round(float(sv[a0]), 3), 'to': round(float(sv[a1]), 3),
                            'pad': round(float(np.median(pad[inseg])), 3)})
    return merge_lines(road, out)


def merge_lines(road, lines):
    """A road's edge lines one along each side, from where the first begins to where the last ends: two
    of them joined across the gap between them, but where another road leaves or joins it there (a
    mouth: road just past its edge, MOUTH_OUT out, along most of the gap). Each keeps its first's name."""
    import shapely
    k = gltf_source.load_kn5(bp.TRACK)
    others = []
    for f in (bp.TRACK, bp.PADDOCK, bp.KERBS):
        for n, m in bp.meshes(gltf_source.load_kn5(f)):
            if n['name'].startswith('1ROAD') and n['name'] != road['name'] and len(n['faces'])                     and not n['name'].startswith(NOT_MOUTHS) and n['name'] not in RUNOFFS:
                others.append(bp.triangles(n, m))
    S = bp.Surface(np.concatenate(others))
    out = []
    for side, sgn in (('left', 1.), ('right', -1.)):
        # (stubs out: a piece of old line left where roads met, not an edge line)
        mine = sorted([l for l in lines if l['side'] == side and l['to'] - l['from'] >= STUB], key=lambda l: l['from'])
        merged = []
        for l in mine:
            if merged and l['from'] <= merged[-1]['to'] + .01:
                merged[-1]['to'] = max(merged[-1]['to'], l['to'])
                continue
            if merged:
                a0, a1 = merged[-1]['to'], l['from']
                at = np.arange(a0, a1, .5) if a1 - a0 > .5 else np.array([(a0 + a1) / 2])
                mid, left = rd.fine_at(road, at)
                mid_, left_, wl, wr, at_ = rd.fine(road)
                w = np.interp(at, at_, wl if side == 'left' else wr)
                P = mid + left * (sgn * (w + MOUTH_OUT))[:, None]
                mouth = np.isfinite(S.ys(P)).mean() > .5
                if not mouth:
                    merged[-1]['to'] = l['to']
                    continue
            merged.append(dict(l))
        out += merged
    mid_, left_, wl_, wr_, at_ = rd.fine(road)
    for l in out:
        l['pad'] = rd.LINE_PAD                                  # (one way in from the edge, every line: ROADS.md)
        # (on to the road's very ends, where the next road's line goes on from)
        if l['from'] < STUB:
            l['from'] = 0.
        if l['to'] > at_[-1] - STUB:
            l['to'] = round(float(at_[-1]), 3)
    return out


def ground():
    """The ground the roads lie on (tools/terrain.py's, its edits on it): height(x, z)."""
    import terrain
    return terrain.ground(3)


def heights_for(road, roads, real):
    """A road's heights (see roads.py's record_heights) on the ground (real): a run-off's as
    along_heights; a branch's (road['joins']) as the roads it joins where it lies over them, the ground
    beyond; any other's the ground's."""
    if 'along' in road:
        return rd.along_heights(road, roads, real)
    surfaces = []
    for x in set([j for j in (road.get('joins') or ()) if j] + rd.above(road, roads)):
        if x and x in roads and roads[x].get('heights'):                # (those above it: their heights first)
            r2 = rd.along_road(roads[x], roads) if 'along' in roads[x] else roads[x]
            P, _, F = rd.mesh(r2)
            surfaces.append(bp.Surface(P[F]))

    def h(x, z):
        y = np.asarray(real(x, z), float) + rd.ROAD
        for S in surfaces:
            ys = S.ys(np.c_[x, z])
            y = np.where(np.isfinite(ys), ys, y)
        return y
    return rd.record_heights(road, h)


def main():
    args = sys.argv[1:]
    path = bp.TRACK
    if '--model' in args:
        i = args.index('--model')
        path = args[i + 1]
        args = args[:i] + args[i + 2:]
    if '--ends' in args:
        # (the roads' end rows, as the model given has them)
        names = [a_ for a_ in args if a_ != '--ends']
        k = gltf_source.load_kn5(path)
        have = rd.load()
        for n, m in bp.meshes(k):
            if n['name'] in names:
                have[n['name']].update(end_rows(n, m, have[n['name']]))
                print('%s: its end rows' % n['name'])
        json.dump({'roads': list(have.values())}, open(rd.ROADS, 'w'), indent=1)
        return
    heights = '--heights' in args
    paint = None
    if '--lines' in args:
        i = args.index('--lines')
        paint = gltf_source.load_kn5(args[i + 1])
        args = args[:i] + args[i + 2:]
    names = [a_ for a_ in args if a_ != '--heights']
    have = rd.load()
    if heights or paint is not None:
        # (the heights, as the roads lie in the track's model now; the lines, from the paint's model given)
        k = gltf_source.load_kn5(path)
        if heights:
            # (the ground the roads lie on: tools/terrain.py's)
            real = ground()
        for name in names:
            road = have[name]
            if heights and 'along' in road:
                road['heights'] = rd.along_heights(road, have, real)
                print('%s: a run-off, heights every %.0f m, %d rows' % (name, road['heights']['step'], len(road['heights']['rows'])))
                continue
            if heights:
                road['heights'] = rd.record_heights(road, lambda x, z: real(x, z) + rd.ROAD)
                print('%s: heights every %.0f m, %d rows' % (name, road['heights']['step'], len(road['heights']['rows'])))
            if paint is not None:
                road['lines'] = lines_of(road, paint)
                print('%s: %d edge lines: %s' % (name, len(road['lines']), ', '.join('%s %s %.0f-%.0f' % (l['name'], l['side'][0], l['from'], l['to']) for l in road['lines'])))
        json.dump({'roads': list(have.values())}, open(rd.ROADS, 'w'), indent=1)
        return
    k = gltf_source.load_kn5(path)
    strip = '--strip' in args
    names = [a_ for a_ in names if a_ != '--strip']
    for n, m in bp.meshes(k):
        if n['name'] in names:
            d = (describe_strip if strip else describe)(n, m, k['materials'][n['material']]['name'])
            have[n['name']] = d
            print('%s: %d points, %.1f to %.1f m wide' % (n['name'], len(d['points']),
                  min(p[2] + p[3] for p in d['points']), max(p[2] + p[3] for p in d['points'])))
    os.makedirs(os.path.dirname(rd.ROADS), exist_ok=True)
    json.dump({'roads': list(have.values())}, open(rd.ROADS, 'w'), indent=1)


if __name__ == '__main__':
    main()
