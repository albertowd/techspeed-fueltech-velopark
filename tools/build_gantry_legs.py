"""The start gantries' legs meeting at the board, in source/models/velopark_gantry_*.glb.

    python tools/build_gantry_legs.py

The gantry stands at each end of its board on two tubes, a V seen from the side: their feet
apart along the track, both leaning out. In the 0.5.0 track they stop beside the board's end,
one at its front and one at its back, near its top. On the real gantry they meet each other in
one point, halfway up the board's end, and hold the board there. This makes each tube run
from its foot, where it stands, to that point, as thick as it was. The tyres round the feet
and the walls cars hit stay as they are. Run again, it finds the legs met and changes nothing.
"""
import glob, os

import numpy as np

import build_paths as bp
import gltf_source
from plant_trees import pieces

ROOT = bp.ROOT
SEG = 16             # sides of a tube


def tube(a, b, r):
    """A tube from a to b, its ends open, as kn5 vertices (position, normal, uv, tangent)."""
    ax = (b - a) / np.linalg.norm(b - a)
    u = np.cross(ax, [0, 0, 1] if abs(ax[2]) < .9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    ang = np.linspace(0, 2 * np.pi, SEG + 1)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
    P = np.r_[a + r * ring, b + r * ring]
    N = np.r_[ring, ring]
    L = np.linalg.norm(b - a)
    U = np.c_[np.r_[ang, ang] / (2 * np.pi) * 2, np.r_[np.zeros(SEG + 1), np.full(SEG + 1, L / 2)]]
    T = np.tile(ax, (len(P), 1))
    F = [(i, i + 1, SEG + 2 + i) for i in range(SEG)] + [(i, SEG + 2 + i, SEG + 1 + i) for i in range(SEG)]
    return np.c_[P, N, U, T], np.array(F)


def legs_of(node):
    """The tubes of GANTRY_STEEL, each as (its faces, top, foot, radius)."""
    V, F = node['verts'][:, :3].astype(float), node['faces'].astype(int)
    lab = pieces(F, len(V))
    out = []
    for t in np.unique(lab[F[:, 0]]):
        sel = lab == t
        q = V[sel]
        if np.ptp(q[:, 1]) < 2:
            continue
        # its axis, and its ends where its rings cross it
        c = q.mean(0)
        ax = np.linalg.svd(q - c, full_matrices=False)[2][0]
        ax = ax if ax[1] > 0 else -ax
        t = (q - c) @ ax
        top, foot = c + ax * t.max(), c + ax * t.min()
        r = np.median(np.linalg.norm(np.cross(q - foot, ax), axis=1))
        out.append((sel[F].all(1), top, foot, r))
    return out


def main():
    for f in sorted(glob.glob(os.path.join(bp.MODELS, 'velopark_gantry_*.glb'))):
        k = gltf_source.load_kn5(f)
        if any(not np.allclose(m, np.eye(4)) for n, m in bp.meshes(k)):
            print('%s: a mesh is moved, left as it is' % os.path.basename(f))
            continue
        by = {n['name']: n for n, m in bp.meshes(k)}
        steel, board = by['GANTRY_STEEL'], by['GANTRY_BOARD']
        legs = legs_of(steel)
        B = board['verts'][:, :3].astype(float)
        mid = B.mean(0)
        along = B[np.argmax(B[:, 0])] - B[np.argmin(B[:, 0])]
        along[1] = 0
        along /= np.linalg.norm(along)
        proj = (B - mid) @ along
        ends = {}
        for l in legs:
            ends.setdefault(1 if (l[1] - mid) @ along > 0 else -1, []).append(l)
        if all(len(ls) == 2 and np.linalg.norm(ls[0][1] - ls[1][1]) < .02 for ls in ends.values()):
            print('%s: the legs meet already' % os.path.basename(f))
            continue
        gone = np.zeros(len(steel['faces']), bool)
        for l in legs:
            gone |= l[0]
        bp.split(steel, gone)
        new_v, new_f, n = [steel['verts'].astype(float)], [steel['faces'].astype(int)], len(steel['verts'])
        for s, ls in ends.items():
            # the board's end: its corners furthest out that way, halfway up
            far = B[proj > proj.max() - .05] if s > 0 else B[proj < proj.min() + .05]
            apex = far.mean(0)
            apex[1] = (far[:, 1].min() + far[:, 1].max()) / 2
            for l in ls:
                V, F = tube(l[2], apex, l[3])
                new_v.append(V)
                new_f.append(F + n)
                n += len(V)
        steel['verts'] = np.concatenate(new_v).astype('<f4')
        steel['faces'] = np.concatenate(new_f).astype('<u2')
        bp.sphere(steel)
        bp.save(k, f)
        print('%s: the legs meet halfway up the board\'s ends' % os.path.relpath(f, ROOT))


if __name__ == '__main__':
    main()
