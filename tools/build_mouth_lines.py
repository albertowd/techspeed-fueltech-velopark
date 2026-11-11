"""Lines painted across the road where roads part, in source/models/velopark_paint.glb.

    python tools/build_mouth_lines.py

Before the vp_1500's last corner, T12, its lap turns right off the road it came up, which carries
on north; the real track paints a line across the whole road where the two begin to part, just
before the southernmost of the cones standing between them. Each of MOUTHS is drawn so: through
its point, square to the road as it runs before the junction (the way its edge line runs there), from the edge line on one
side to the edge line on the other, WIDTH wide, as the other lines are, OVER over the asphalt and
mapped as they are. Run again, it finds the lines painted.
"""
import os

import numpy as np

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
PAINT = os.path.join(bp.MODELS, 'velopark_paint.glb')
# name: (a point the line runs through, a few metres before the southernmost cone, AC_POBJECT_7 of
# vp_1500, where the road's west edge still runs straight; a point of the road's edge line before the junction, the way the road runs there)
MOUTHS = {'LINE_MOUTH_T12': ((59.4, 184.1), (62.5, 174.5))}
WIDTH = .12              # metres, as the other lines
STEP = .25               # metres between its points
OVER = .0025             # metres over the asphalt, as the other lines lie
UV = (10., 20.)          # metres a texture, along x and z, as the other lines are mapped


def direction(points, at, toward):
    """Which way the line through points runs at at, pointing toward toward."""
    near = points[np.linalg.norm(points - at, axis=1) < 1.5]
    d = np.linalg.svd(near - near.mean(0))[2][0]
    return d if d @ (toward - at) > 0 else -d


def main():
    k = gltf_source.load_kn5(PAINT)
    names = {n['name'] for n, m in bp.meshes(k)}
    todo = {n: v for n, v in MOUTHS.items() if n not in names}
    if not todo:
        print('the mouths are painted already')
        return
    lines = [(n, n['verts'][:, [0, 2]].astype(float)) for n, m in bp.meshes(k)
             if k['materials'][n['material']]['name'] == 'LinePaint' and len(n['faces'])]
    template, _ = lines[0]
    paint = next(i for i, m in enumerate(k['materials']) if m['name'] == 'LinePaint')
    track = gltf_source.load_kn5(bp.TRACK)
    asphalt = []
    for model in (track, gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_kerbs.glb'))):
        for n, m in bp.meshes(model):
            if n['name'].startswith('1ROAD') and len(n['faces']) and 'Kerb' not in model['materials'][n['material']]['name']:
                t = bp.triangles(n, m)
                if len(t):
                    asphalt.append(t)
    top = bp.Surface(np.concatenate(asphalt))
    for name, (at, before) in todo.items():
        at, before = np.array(at), np.array(before)
        P = np.concatenate([Q for _, Q in lines])
        # square to the road as it runs before the junction: the way its edge line runs there
        _, near = min(lines, key=lambda x: np.linalg.norm(x[1] - before, axis=1).min())
        along = direction(near, near[np.linalg.norm(near - before, axis=1).argmin()], at)
        across = np.array([along[1], -along[0]])
        # out to the inner edge of the edge line each side: where the line across meets the
        # edges of the lines' faces
        E = np.concatenate([np.stack([n['verts'][f][:, :, [0, 2]].astype(float) for f in
                                      (n['faces'][:, [0, 1]], n['faces'][:, [1, 2]], n['faces'][:, [2, 0]])]).reshape(-1, 2, 2)
                            for n, _ in lines])
        a0, a1 = (E[:, 0] - at) @ along, (E[:, 1] - at) @ along
        cross = (a0 * a1 < 0)
        t = a0[cross] / (a0[cross] - a1[cross])
        hits = (E[cross, 0] + t[:, None] * (E[cross, 1] - E[cross, 0]) - at) @ across
        ends = []
        for sgn in (-1, 1):
            o = sgn * hits
            o = o[o > .3]
            if not len(o):
                raise SystemExit('%s finds no edge line on one side' % name)
            ends.append(at + sgn * across * o.min())
        p0, p1 = ends
        L = np.linalg.norm(p1 - p0)
        C = p0 + np.linspace(0, 1, max(2, int(L / STEP)))[:, None] * (p1 - p0)
        d = np.gradient(C, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        side = np.c_[-d[:, 1], d[:, 0]] * WIDTH / 2
        xz = np.stack([C - side, C + side], 1).reshape(-1, 2)
        y = top.ys(xz)
        # its ends on the edge lines may lie just over the asphalt's edge: as the points beside them
        ok = ~np.isnan(y)
        if ok.sum() < len(y) * .8:
            raise SystemExit("%s runs off the asphalt: %s to %s" % (name, xz[0], xz[-1]))
        y[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], y[ok])
        P = np.c_[xz[:, 0], y + OVER, xz[:, 1]]
        N = np.tile([0., 1., 0.], (len(P), 1))
        U = xz / UV
        F = []
        for i in range(len(C) - 1):
            a0, a1, b0, b1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
            F += [(a0, b0, b1), (a0, b1, a1)]
        F = np.array(F)
        up = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 1] < 0
        F[up] = F[up][:, ::-1]
        T = gltf_source.tangents(P, N, U, F)
        node = dict(template)
        node.update(name=name, children=[], material=paint, verts=np.c_[P, N, U, T].astype('<f4'), faces=F.astype('<u2'))
        bp.sphere(node)
        k['root']['children'].append(node)
        print('  %s, %.1f m from %s to %s' % (name, L, p0.round(2), p1.round(2)))
    bp.save(k, PAINT)


if __name__ == '__main__':
    main()
