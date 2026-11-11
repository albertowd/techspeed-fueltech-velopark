"""The ground beside paths A and D graded into them, in the track's sources.

    python tools/grade_paths.py

build_paths.py lays the paths across the infield eased along and across, which the ground they
cross is not: over path A's hump it stands up to 0.3 m over a path's side within a metre or two,
as a bank the path is cut through, and elsewhere it falls away from it, the path standing on it
as on a wall, its skirts showing. This grades the ground of source/models/velopark.glb (1GRASS_0
and the old strips in it, 1GRASS_PATHS) within BAND of each path's sides anew: as a surface
bending as little as it can between the path's side and the ground as it is past that band,
meeting each in height and in slope, as a bank graded beside a road would; the ground under a
path just under it, unseen, so faces reaching under its side do not dip there; and the dents
laying the paths left in the ground near them (corners of its long faces pulled down under a
path, running off across the grass as trenches) filled the same way. Where something else
lies on the ground there (a road, a kerb, the paddock), the ground under it stays as it is. The
ground's big faces there are cut by a grid of a metre first, so it has points to bend at; the
tyre stacks standing on it go up or down with it, each as a whole, and the paths' skirts down
under it. The paths themselves are left as they are. Run again, it finds the ground graded and
changes nothing.
"""
import os, sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, median_filter

import build_paths as bp
import gltf_source
from join_paddock import Grid, R, chunks, fill, refine

ROOT = bp.ROOT
KIDS = os.path.join(bp.MODELS, 'velopark_kids.glb')
BAND = 4.            # metres from a path's side over which the ground is graded into it
BELOW = .025         # how far under a path's side the ground meets it: past the depth build_paths
                     # takes for the ground showing through a path, half its UNDER
DONE = .02           # metres; grading most of the band less than this, the ground is graded
SMALL = .002         # metres; a point graded less than this stays
CUT = .01            # metres; the ground's faces are cut where it is graded more than this
STACK = 1.5          # metres; a piece of a tyre mesh smaller than this across is one stack
DENT = .05           # metres the ground lies under the ground round it for a dent
DENT_WIDTH = 3.      # metres across which the ground round a point is taken
DENT_REACH = 15.     # metres from the paths dents are looked for


def pieces(F, n):
    """Which piece of a mesh each of its n points is in, pieces sharing no point."""
    label = np.arange(n)
    while True:
        low = label[F].min(1)
        before = label.copy()
        for k in range(3):
            np.minimum.at(label, F[:, k], low)
        label = label[label]
        if (label == before).all():
            return label


def main():
    track = gltf_source.load_kn5(bp.TRACK)
    pad = gltf_source.load_kn5(bp.PADDOCK)
    tyres = gltf_source.load_kn5(bp.TYRES)
    paths = [n for n, m in bp.meshes(pad) if n['name'].startswith('1ROAD_PATH_')]
    if not paths:
        sys.exit('no paths in %s: run build_paths.py first' % os.path.relpath(bp.PADDOCK, ROOT))
    grounds = [(n, m) for n, m in bp.meshes(track) if n['name'] in (bp.GROUND, bp.FILLER)]
    ids = {id(p) for p in paths}
    for n, m in grounds + [(p, m) for p, m in bp.meshes(pad) if id(p) in ids]:
        if not np.allclose(m, np.eye(4)):
            sys.exit('%s is not at the model\'s origin' % n['name'])
    grounds = [n for n, _ in grounds]
    tops = np.concatenate([bp.top_of(p) for p in paths])
    lo = tops[:, :, [0, 2]].min((0, 1)) - BAND - 3
    hi = tops[:, :, [0, 2]].max((0, 1)) + BAND + 3

    def near(t):
        c = t[:, :, [0, 2]]
        return t[(c.max(1) >= lo).all(1) & (c.min(1) <= hi).all(1)]
    g = Grid(lo, hi)
    path = g.top(tops)
    ground = g.top(np.concatenate([near(bp.triangles(n, np.eye(4))) for n in grounds]))
    # everything else lying on the ground: roads, kerbs and their grass beds, the paddock
    others = []
    for k in (track, gltf_source.load_kn5(bp.KERBS), gltf_source.load_kn5(KIDS), pad):
        for n, m in bp.meshes(k):
            nm = n['name']
            if nm[:1].isdigit() and 'WALL' not in nm and id(n) not in ids and all(n is not x for x in grounds):
                t = near(bp.triangles(n, m))
                if len(t):
                    others.append(t)
    others = np.concatenate(others)
    covered = ~np.isnan(g.top(others))
    on_top = bp.Surface(others)

    # the band beside the paths, and the ground graded in it: known at the paths' sides (just
    # under them) and past the band, and under whatever else lies on the ground
    open_path = ~np.isnan(path) & ~covered
    # (from where a path shows: where it runs on under a road, the ground past that road is not
    # beside it)
    dist = distance_transform_edt(~open_path) * R
    # and not across a road from it either
    reach = binary_dilation(open_path, structure=np.ones((3, 3), bool), iterations=int(round(BAND / R)), mask=~covered)
    band = np.isnan(path) & (dist <= BAND) & ~covered & reach
    # and the dents near them: lowering the ground under the paths pulled down corners of its
    # long faces reaching under them, leaving narrow trenches running off across the grass
    _, (jn, i_n) = distance_transform_edt(np.isnan(ground), return_indices=True)
    level = median_filter(ground[jn, i_n], size=int(round(DENT_WIDTH / R)) | 1)
    # (not at the edge of what lies on the ground, which the ground under it, kept as it is, holds)
    dent = binary_dilation((ground < level - DENT) & (dist <= DENT_REACH), iterations=2)
    band |= dent & np.isnan(path) & ~binary_dilation(covered, iterations=int(round(1 / R))) & ~np.isnan(ground)
    known = np.where(open_path, path - BELOW, ground)
    known[band] = np.nan
    base = np.where(np.isnan(ground), known, ground)
    _, (jn, i_n) = distance_transform_edt(np.isnan(base), return_indices=True)
    graded = fill(band, known, base[jn, i_n])
    # and under the paths, unseen, just under them: where a path stands over the ground it was
    # laid on, the faces reaching under its side from the ground graded beside it would dip
    # down to that ground at its side
    goal = np.where(band, graded, np.where(open_path, path - bp.UNDER - SMALL * 2, ground))
    graded_area = (band | open_path) & ~np.isnan(ground)
    up = np.where(graded_area, goal - ground, 0.)
    up[np.abs(up) < SMALL] = 0
    most = np.percentile(np.abs(up[graded_area]), 90)
    # (the ground at a road's edge hangs from its points under the road, which stay, so it may
    # not reach the grade there)
    free = ~binary_dilation(covered, iterations=int(round(1 / R)))
    worst = np.abs(up[free]).max()
    if most < DONE and worst < DENT:
        print('the ground beside the paths is graded already: nine in ten of its points would move less than %.3f m, '
              'none more than %.3f m' % (most, worst))
        return
    print('the ground graded beside the paths over %.0f m2: up to %.3f m up and %.3f m down'
          % (band.sum() * R * R, up.max(), -up.min()))
    # the goal carried on past where the ground is, so a point at its edge, found between
    # cells, goes where the cells round it do
    _, (jn, i_n) = distance_transform_edt(np.isnan(goal), return_indices=True)
    goal = goal[jn, i_n]
    gx, gz = np.gradient(goal, R, axis=(1, 0))
    moving = binary_dilation(np.abs(up) > CUT, iterations=int(round(1.5 / R)))
    graded_cells = binary_dilation(up != 0, iterations=1)

    def cell(F, P):
        i = np.clip(np.round((P[:, 0] - g.x0) / R).astype(int), 0, g.nx - 1)
        j = np.clip(np.round((P[:, -1] - g.z0) / R).astype(int), 0, g.nz - 1)
        return F[j, i]

    root = track['root']
    for n in grounds:
        V, F, new, cut, Fc = refine(n['verts'].astype(float), n['faces'].astype(int), lambda P: cell(moving, P))
        xz = V[:, [0, 2]]
        # each point graded but where something else lies over it, found exactly: the cells
        # beside a road's edge are cells of it too
        go = cell(graded_cells, xz)
        go[go] = np.isnan(on_top.ys(xz[go]))
        du = np.zeros(len(V))
        du[go] = g.at(goal, xz[go]) - V[go, 1]
        du[np.abs(du) < SMALL] = 0
        V[:, 1] += du
        moved = du != 0
        flat = moved & (V[:, 4] > .3)
        nn = np.c_[-g.at(gx, xz[flat]), np.ones(flat.sum()), -g.at(gz, xz[flat])]
        V[flat, 3:6] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
        redo = new | moved
        V[redo, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[redo]
        parts = chunks(V, F)
        i = next(k for k, c in enumerate(root['children']) if c is n)
        nodes = []
        for k, (v, f) in enumerate(parts):
            node = dict(n)
            node.update(name=n['name'] if k == 0 else '%s_%d' % (n['name'], k + 1), children=[],
                        verts=v.astype('<f4'), faces=f.astype('<u2'))
            bp.sphere(node)
            nodes.append(node)
        root['children'][i:i + 1] = nodes
        print('%s: %d faces cut into %d; %d points, %d of them graded' % (n['name'], cut.sum(), len(Fc), len(V), moved.sum()))
    # the ground under the paths' tops still, as build_paths lays it
    grounds = [n for n, _ in bp.meshes(track) if n['name'].startswith((bp.GROUND, bp.FILLER))]
    bp.sink(grounds, paths)
    after = g.top(np.concatenate([near(bp.triangles(n, np.eye(4))) for n in grounds]))
    _, (jn, i_n) = distance_transform_edt(np.isnan(after), return_indices=True)
    after = after[jn, i_n]

    # the paths' skirts down under the ground beside them
    for p in paths:
        V = p['verts']
        rows = V[:, :3].astype(float).reshape(-1, bp.COLUMNS + 2, 3)
        for skirt, edge in ((0, 1), (bp.COLUMNS + 1, bp.COLUMNS)):
            y = np.minimum(g.at(after, rows[:, skirt][:, [0, 2]]) - bp.DROP, rows[:, edge, 1])
            idx = np.arange(len(rows)) * (bp.COLUMNS + 2) + skirt
            V[idx, 1] = y.astype(V.dtype)
        bp.sphere(p)

    # the tyre stacks on the ground, each up or down as a whole as the ground under its middle
    # went; the long walls round them point by point
    shift = np.where(~np.isnan(ground), after - np.where(np.isnan(ground), 0, ground), 0.)
    shift[np.abs(shift) < SMALL] = 0
    stacks = points = 0
    for n, m in bp.meshes(tyres):
        if not np.allclose(m, np.eye(4)):
            continue
        V = n['verts']
        P = V[:, :3].astype(float)
        inside = (P[:, [0, 2]] >= lo).all(1) & (P[:, [0, 2]] <= hi).all(1)
        if not inside.any():
            continue
        label = pieces(n['faces'].astype(int), len(V))
        dy = np.zeros(len(V))
        for piece in np.unique(label[inside]):
            k = label == piece
            q = P[k]
            if np.ptp(q[:, [0, 2]], 0).max() < STACK:
                d = g.at(shift, ((q.min(0) + q.max(0)) / 2)[None, [0, 2]])[0]
                dy[k] = d
                stacks += d != 0
            else:
                dy[k] = g.at(shift, q[:, [0, 2]])
                points += (dy[k] != 0).sum()
        if (dy != 0).any():
            V[:, 1] += dy.astype(V.dtype)
            bp.sphere(n)
    print('%d tyre stacks and %d points of the tyre walls with the ground' % (stacks, points))
    bp.save(track, bp.TRACK)
    bp.save(pad, bp.PADDOCK)
    bp.save(tyres, bp.TYRES)
    print('the ground graded beside the paths in %s' % os.path.relpath(bp.TRACK, ROOT))


if __name__ == '__main__':
    main()
