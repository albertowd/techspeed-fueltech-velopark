"""The paint kept on the roads: every line of paint cut to the roads and kerbs under it, none of it over the
grass beside them (tools/check_holes.py's paint check, measured the same way), and laid on them as they lie
(the merge moves the roads' points where they meet: the paint, laid before it, would float or sink). Run by
tools/merge_floor.py after it writes the floor, on velopark_lines.glb.
"""
import os

import numpy as np

import build_paths as bp
import gltf_source
from restyle_paddock import clip_in

CRACK = .005               # metres: a gap narrower than twice this between roads' faces is road (to paint on; under tools/check_holes.py's PAINT_NEAR)
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


LAY_NEAR = .5             # metres: the road a point of paint is laid on is the one under it nearest its height, within this


def lay_on_roads(L):
    """Every point of L's paint put on the road or kerb under it (1ROAD, 1KERB, every model), roads.LINE_LIFT
    over it: of those under it, the one nearest its height (a floor over the road, as a shed's over the pits'
    grid, not taken); a point with none within LAY_NEAR left as it is. How many points moved, and the most."""
    import glob
    import roads as rd
    if not _FLOOR:
        T = [bp.triangles(n, m) for f in glob.glob(os.path.join(bp.MODELS, 'velopark*.glb'))
             for n, m in bp.meshes(gltf_source.load_kn5(f)) if n['name'].startswith(('1ROAD', '1KERB')) and len(n['faces'])]
        _FLOOR.append(bp.Surface(np.concatenate(T)))
    S = _FLOOR[0]
    moved, most = 0, 0.
    for n, m in bp.meshes(L):
        if not len(n['faces']) or not np.allclose(m, np.eye(4)):
            continue
        V = n['verts']
        y = S.nearest_y(V[:, [0, 2]].astype(float), V[:, 1].astype(float) - rd.LINE_LIFT, LAY_NEAR) + rd.LINE_LIFT
        on = ~np.isnan(y)
        d = np.abs(y[on] - V[on, 1])
        moved += int((d > 1e-3).sum())
        most = max(most, float(d.max()) if len(d) else 0.)
        V[on, 1] = y[on]
        bp.sphere(n)
    return moved, most


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
