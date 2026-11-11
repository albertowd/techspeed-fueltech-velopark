"""The track's narrow roads' asphalt mapped as the rest of it is, not stretched along them.

    python tools/map_roads.py [--dry-run]

The 0.5.0 track maps its asphalt along each road, the texture's width across it; so on the pit
lanes and the other narrow roads, 3.6 m wide, the texture is squeezed across them and stretched
along them, about six times as much as on the track and the paddock, which shows where they
meet the paddock's asphalt and the paths. This maps every road whose asphalt is more than OFF
off the paddock's scale, in every model of source/models, flat from above as the paddock's is,
at its scale: the asphalt being the same all over, the seams between them do not show. Run
again, it finds the roads mapped and changes nothing.
"""
import glob, os, re, sys

import numpy as np

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
OFF = .3             # how far a road's metres per texture may be off the paddock's, as a fraction
ROAD = re.compile(r'\d+ROAD_')


def scale(P, U, F):
    """Metres the median face goes along each of u and v."""
    e1, e2 = P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]
    d1, d2 = U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    ok = np.abs(det) > 1e-9
    if not ok.any():
        return None
    du = (e1[ok] * d2[ok, 1:2] - e2[ok] * d1[ok, 1:2]) / det[ok, None]
    dv = (e2[ok] * d1[ok, 0:1] - e1[ok] * d2[ok, 0:1]) / det[ok, None]
    return np.median(np.c_[np.linalg.norm(du, axis=1), np.linalg.norm(dv, axis=1)], axis=0)


def main():
    dry = '--dry-run' in sys.argv
    pad = gltf_source.load_kn5(bp.PADDOCK)
    asphalt = next(n for n, m in bp.meshes(pad) if n['name'] == bp.ASPHALT)
    uv_of = bp.flat_uv(asphalt)
    v = asphalt['verts'].astype(float)
    want = scale(v[:, :3], v[:, 6:8], asphalt['faces'].astype(int))
    total = mapped = 0
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        k = gltf_source.load_kn5(f)
        changed = False
        for n, m in bp.meshes(k):
            mat = k['materials'][n['material']]
            tex = next((t['name'] for t in mat['textures'] if t['sampler'] == 'txDiffuse'), '')
            if not (ROAD.match(n['name']) and n['visible'] and n['renderable'] and len(n['faces'])
                    and os.path.basename(tex).lower().startswith('tarmac')):
                continue
            V, F = n['verts'], n['faces'].astype(int)
            P = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ m)[:, :3]
            have = scale(P, V[:, 6:8].astype(float), F)
            if have is None:
                # faces with no area, which the build drops
                continue
            total += 1
            if (np.abs(have / want - 1) <= OFF).all():
                continue
            print('  %-24s %-16s %5.1f x %5.1f m per texture' % (os.path.basename(f), n['name'], *have))
            mapped += 1
            if dry:
                continue
            U = uv_of(P)
            N = V[:, 3:6].astype(float)
            T = gltf_source.tangents(V[:, :3].astype(float), N, U, F)
            n['verts'] = np.c_[V[:, :6], U, T].astype(V.dtype)
            changed = True
        if changed:
            bp.save(k, f)
            print('roads mapped in %s' % os.path.relpath(f, ROOT))
    if mapped:
        print('%d of %d roads %s as the paddock is, %.1f x %.1f m per texture' % (mapped, total, 'to map' if dry else 'mapped', *want))
    else:
        print('the roads are mapped already: %d roads' % total)


if __name__ == '__main__':
    main()
