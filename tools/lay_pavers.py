"""The strips of paving tools/build_pavers.py lays outside a run-off (STRIPS, as R41's beyond
STRAIGHT5) laid again on the ground as tools/drape_track.py left it.

    python tools/lay_pavers.py

The drape moves the paving with the roads round it, by their change carried out to it, which leaves
it a few centimetres up and down where the asphalt beside it and the grass under it lie smooth, and
a step along the run-off's edge. Here each of its rows is laid as build_pavers.py lays it: its inner
end on the run-off's asphalt (a little over it, OVER, as it laps onto it), rising smoothly to LIFT
over the ground at its outer end, never less than half that over the ground; its skirt SKIRT under
that. The grass ramp along its outer edge (1GRASS_EDGE_RAMP) moves up or down with it there.
"""
import os

import numpy as np

import build_paths as bp
import build_pavers as pv
import gltf_source

OVER = .003              # metres the paving's inner end lies over the asphalt it laps onto
ROAD = .02               # metres the asphalt lies over the ground (tools/drape_track.py's ROAD)
COLS = 7                 # points a row: six across it, and its skirt's
RUNOFFS = os.path.join(bp.MODELS, 'velopark_runoffs.glb')
KERBS = os.path.join(bp.MODELS, 'velopark_kerbs.glb')


def lay():
    """Every strip laid again; how many rows, and the points of the ramp moved with them."""
    track = gltf_source.load_kn5(bp.TRACK)
    models = {bp.TRACK: track, RUNOFFS: gltf_source.load_kn5(RUNOFFS), KERBS: gltf_source.load_kn5(KERBS)}
    names = {pv.KERB + s[len('1ROAD_'):] for s in pv.STRIPS}
    asphalt, strips = [], []
    for f, k in models.items():
        for n, m in bp.meshes(k):
            if not len(n['faces']):
                continue
            if n['name'] in names:
                strips.append((f, n, m))
            elif n['name'].startswith('1ROAD') and n.get('visible', 1):
                asphalt.append(bp.triangles(n, m))
    top = bp.Surface(np.concatenate(asphalt))
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track)
                                        if n['name'].startswith(bp.GROUND) and len(n['faces'])]))
    rows, moved = 0, {}
    for f, n, m in strips:
        if not np.allclose(m, np.eye(4)) or len(n['verts']) % COLS:
            raise SystemExit('%s: not the grid build_pavers.py lays' % n['name'])
        V = n['verts'].astype(float)
        G = V[:, :3].reshape(-1, COLS, 3)
        across = G[:, COLS - 2, [0, 2]] - G[:, 0, [0, 2]]
        wide = np.linalg.norm(across, axis=1)
        out = across / np.maximum(wide, 1e-9)[:, None]
        old = G[:, COLS - 2].copy()
        for r in range(len(G)):
            p0 = G[r, 0, [0, 2]]
            y0 = top.y(*p0)
            if np.isnan(y0):
                y0 = top.y(*(p0 - out[r] * .06))
            if np.isnan(y0):
                continue
            # (the ground under it is out: the grass just past its outer edge, and the asphalt's
            # own height over the ground at its inner end, between them)
            p1 = G[r, COLS - 2, [0, 2]]
            g1 = next((y for y in (ground.y(*(p1 + out[r] * s)) for s in (.3, .6, 1., 1.5)) if not np.isnan(y)), np.nan)
            y0 += OVER
            for c in range(COLS - 1):
                f_ = np.linalg.norm(G[r, c, [0, 2]] - p0) / max(wide[r], 1e-9)
                g = (y0 - OVER - ROAD) * (1 - f_) + g1 * f_
                if c == 0 or np.isnan(g):
                    G[r, c, 1] = y0
                    continue
                w = f_ * f_ * (3 - 2 * f_)
                G[r, c, 1] = max(y0 * (1 - w) + (g + pv.LIFT) * w, g + pv.LIFT * .5)
            G[r, COLS - 1, 1] = G[r, COLS - 2, 1] - pv.SKIRT
            rows += 1
        V[:, :3] = G.reshape(-1, 3)
        F = n['faces'].astype(int)
        fn = np.cross(V[F[:, 1], :3] - V[F[:, 0], :3], V[F[:, 2], :3] - V[F[:, 0], :3])
        N = np.zeros((len(V), 3))
        for i in range(3):
            np.add.at(N, F[:, i], fn)
        V[:, 3:6] = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        V[:, 8:11] = gltf_source.tangents(V[:, :3], V[:, 3:6], V[:, 6:8], F)[:, :3]
        n['verts'] = V.astype(n['verts'].dtype)
        bp.sphere(n)
        for a, b in zip(old, G[:, COLS - 2]):
            moved[(round(a[0], 2), round(a[2], 2))] = (a[1], b[1])
    # the ramp along their outer edges with them
    ramped = 0
    if moved:
        from scipy.spatial import cKDTree
        keys = np.array(list(moved))
        tree = cKDTree(keys)
        for n, m in bp.meshes(track):
            if not n['name'].startswith('1GRASS_EDGE_RAMP') or not len(n['verts']):
                continue
            V = n['verts'].astype(float)
            d, i = tree.query(V[:, [0, 2]])
            for j in np.where(d < .01)[0]:
                was, now = moved[tuple(keys[i[j]])]
                if abs(V[j, 1] - was) < .02:
                    V[j, 1] = now
                    ramped += 1
            n['verts'] = V.astype(n['verts'].dtype)
            bp.sphere(n)
    if rows:
        for f in {f for f, _, _ in strips} | ({bp.TRACK} if ramped else set()):
            bp.save(models[f], f)
    return rows, ramped


def main():
    rows, ramped = lay()
    print('%d rows of paving laid again on the ground, %d points of the grass ramp along it with them' % (rows, ramped))


if __name__ == '__main__':
    main()
