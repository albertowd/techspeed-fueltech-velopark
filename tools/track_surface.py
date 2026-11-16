"""A track's surface as tools/surface.py builds it, from its roads.json (see ROADS.md): its rumbles over
everything, its roads in their order (the run-offs with them), the ground under all. For the road editor's
preview of a part of the map (window) and for the build.

    python tools/track_surface.py X0 Z0 X1 Z1       the surface there built and checked
"""
import sys
import time

import numpy as np

import build_kerbs as bk
import build_paths as bp
import roads as rd
import surface as sf

_GROUND = []
_MESHES = {}         # (a road's description) -> its mesh, kept while it is the same


def ground():
    """The ground the roads lie on (tools/drape_track.py's): height(x, z). Read once."""
    if not _GROUND:
        import describe_roads as dr
        _GROUND.append(dr.ground())
    return _GROUND[0]


def crop(P, U, F, window, margin=2.):
    """A mesh's faces in a window (x0, z0, x1, z1) and margin metres round it, as triangles and their
    texture places."""
    T, TU = P[F], U[F]
    if window is None:
        return T, TU
    x0, z0, x1, z1 = window
    c = T[:, :, [0, 2]]
    sel = (c[:, :, 0].max(1) >= x0 - margin) & (c[:, :, 0].min(1) <= x1 + margin) & (c[:, :, 1].max(1) >= z0 - margin) & (c[:, :, 1].min(1) <= z1 + margin)
    return T[sel], TU[sel]


def regions(roads, window=None, flat=False):
    """The surface's regions, the first over all after it: [Region], and each one's kind (rumble, road,
    ground). roads: as rd.load() gives them (their order the build's); window: the part of the map, or all
    of it; flat: the ground level (no heights read: quicker, for a look at the seams alone)."""
    order = rd.priority(roads)
    import json
    meshes = {}
    for n in order:
        r = roads[n]
        rr = rd.along_road(r, roads) if 'along' in r else r
        if not r.get('heights') or flat:
            rr = dict(rr, heights=None)
        key = json.dumps({k: v for k, v in rr.items() if k not in ('lines', 'kerbs')}, sort_keys=True, default=str)
        if key not in _MESHES:
            if len(_MESHES) > 60:
                _MESHES.clear()
            _MESHES[key] = rd.mesh(rr) if rr.get('heights') else rd.mesh(rr, lambda x, z: np.zeros(len(np.atleast_1d(x))))
        meshes[n] = _MESHES[key]
    # (a rumble level with the roads' top where it starts)
    if flat:
        def top(p):
            return np.zeros(len(p))
    else:
        allT = np.concatenate([P[F] for P, U, F in meshes.values()]) if meshes else np.zeros((0, 3, 3))
        top_s = bp.Surface(allT)

        def top(p):
            y = top_s.ys(p)
            return np.where(np.isfinite(y), y, 0.)
    out, kinds = [], []
    for n in order:
        r = roads[n]
        for i, k in enumerate(r.get('kerbs') or []):
            for P, U, F, mat, suffix in bk.geometry(r, k, roads, top):
                T, TU = crop(P, U, F, window)
                if len(T):
                    out.append(sf.Region(bk.NAME % (n.replace('1ROAD_', '').replace('1KERB_', ''), i, suffix), tris=T, uvs=TU))
                    kinds.append('rumble')
    for n in order:
        T, TU = crop(*meshes[n], window)
        if len(T):
            out.append(sf.Region(n, tris=T, uvs=TU))
            kinds.append('road')
    g = (lambda x, z: np.zeros(len(np.atleast_1d(x)))) if flat else (lambda x, z: np.asarray(ground()(x, z), float))
    out.append(sf.Region('ground', ground=True, height=g, uv=lambda x, z: np.c_[np.asarray(x) / 50., np.asarray(z) / 50.], spacing=2.))
    kinds.append('ground')
    return out, kinds


def build(roads, window, flat=False):
    """The surface in a window: {name: (P, UV, F)}, each region's kind, its problems (surface.check)."""
    regs, kinds = regions(roads, window, flat)
    meshes, faces, owner = sf.build(regs, window)
    return meshes, dict(zip([r.name for r in regs], kinds)), sf.check(meshes, window)


def main():
    window = tuple(float(a) for a in sys.argv[1:5]) if len(sys.argv) >= 5 else None
    t = time.time()
    roads = rd.load()
    meshes, kinds, issues = build(roads, window, flat='--flat' in sys.argv)
    print('%d regions in %.1fs: %s' % (len(meshes), time.time() - t, ', '.join('%s %d' % (n, len(F)) for n, (P, U, F) in meshes.items())))
    print('no seam problems' if not issues else '%d problems, e.g. %s' % (len(issues), [(w, round(x, 2), round(z, 2)) for w, x, z, _ in issues[:8]]))
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
