"""Faces drawn the wrong way round, in every model of source/models.

    python tools/check_normals.py [--fix] [model.glb ...]

A face shows from the side its corners turn counter-clockwise, and is lit by its points' normals:
the two must agree, and a closed piece (a post, a beam, a box) must face out, or a game draws it
inside out, lit from behind or not at all. This lists, model by model, each mesh with faces whose
turn is against their normals, and closed pieces turned inwards. A face made to show from both
sides is two faces, each right. A see-through card (leaves, wire mesh: a material cut out by its
texture's alpha) is drawn from both sides as made, a face for each, and is left out.
Exits 1 when it finds any.

--fix turns each face against its normals (whose points' normals are right, as they are on the
track's ground, roads and paint) to face their way, in every model but FIX_NOT; a closed piece
inside out is made wrong where it is made (tools/shapes.py), and is only reported.
"""
import os, sys

import numpy as np

import build_paths as bp
import gltf_source
from grade_paths import pieces

FIX_NOT = ('velopark_karts.glb',)     # (the kart project's model, as it is)
BIG = 20000          # faces: a mesh with more has its closed pieces not looked at, one by one
WELD = .001          # metres: points this close are one, for a piece's closed surface


def check(n):
    """(faces whose turn is against their normals, closed pieces turned inwards, closed pieces)."""
    V = n['verts']
    P = V[:, :3].astype(float)
    N = V[:, 3:6].astype(float)
    F = n['faces'].astype(int)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    if not len(F):
        return 0, 0, 0
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    g = np.cross(b - a, c - a)                            # (each face's way, by its turn)
    area = np.linalg.norm(g, axis=1)
    vn = N[F].sum(1)
    ok = (area > 1e-9) & (np.linalg.norm(vn, axis=1) > 1e-6)
    against = int((np.einsum('ij,ij->i', g[ok], vn[ok]) < 0).sum())
    if len(F) > BIG:
        return against, 0, 0                              # (a big mesh: tyres, trees, ground, made of pieces checked as made)
    # the closed pieces (every edge, its ends welded, in two faces): a convex one (a post, a beam, a
    # box: every point of it behind each face's plane) inside out when a face of it faces in
    _, weld = np.unique(np.round(P / WELD).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.ravel()
    W = weld[F]
    lab = pieces(W, weld.max() + 1)[W[:, 0]]
    inward = closed = 0
    for p in np.unique(lab):
        sel = np.nonzero(lab == p)[0]
        if len(sel) > 400:
            continue                                      # (big pieces: not posts and boxes)
        f = W[sel]
        e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        if not (cnt == 2).all():
            continue
        closed += 1
        q = P[np.unique(F[sel])]
        cen = q.mean(0)
        gs, ms = g[sel], (a[sel] + b[sel] + c[sel]) / 3
        nn = gs / np.maximum(np.linalg.norm(gs, axis=1, keepdims=True), 1e-12)
        side = np.einsum('fpk,fk->fp', q[None] - ms[:, None], nn)
        convex = ((side <= 1e-4).all(1) | (side >= -1e-4).all(1)).all()
        if convex and (np.einsum('ij,ij->i', nn, ms - cen) < -1e-6).any():
            inward += 1
    return against, inward, closed


def card(k, n):
    """Whether n is of a see-through material, drawn from both sides."""
    return bool(k['materials'][n['material']].get('alpha_tested'))


def fix(n):
    """The faces of n against their normals turned round. How many."""
    V, F = n['verts'], n['faces'].astype(int)
    P, N = V[:, :3].astype(float), V[:, 3:6].astype(float)
    g = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = np.einsum('ij,ij->i', g, N[F].sum(1)) < 0
    if flip.any():
        F[flip] = F[flip][:, [0, 2, 1]]
        n['faces'] = F.astype(n['faces'].dtype)
    return int(flip.sum())


def main():
    args = sys.argv[1:]
    files = [a for a in args if a.endswith('.glb')] or sorted(f for f in os.listdir(bp.MODELS) if f.endswith('.glb'))
    if '--fix' in args:
        for f in files:
            path = os.path.join(bp.MODELS, os.path.basename(f))
            if os.path.basename(f) in FIX_NOT:
                continue
            k = gltf_source.load_kn5(path)
            turned = sum(fix(n) for n, _ in bp.meshes(k) if len(n['faces']) and not n['name'].startswith(('1WALL', 'AC_')) and not card(k, n))
            if turned:
                bp.save(k, path)
                print('%s: %d faces turned to their normals' % (os.path.basename(f), turned))
    bad = 0
    for f in files:
        k = gltf_source.load_kn5(os.path.join(bp.MODELS, os.path.basename(f)))
        rows = []
        for n, _ in bp.meshes(k):
            if not len(n['faces']) or n['name'].startswith(('1WALL', 'AC_')) or card(k, n):
                continue                                  # (walls to hit and dummies are not drawn; cards both ways)
            against, inward, closed = check(n)
            if against or inward:
                rows.append('  %-40s %6d faces against their normals, %4d of %4d closed pieces inside out'
                            % (n['name'], against, inward, closed))
        print('%s: %s' % (os.path.basename(f), 'all faces the right way' if not rows else '%d meshes wrong' % len(rows)))
        for r in rows:
            print(r)
        bad += len(rows)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
