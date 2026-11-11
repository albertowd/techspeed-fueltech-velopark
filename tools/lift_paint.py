"""The track's paint back over the asphalt wherever the asphalt came up over it, in the track's sources.

    python tools/lift_paint.py

The paint (lines, the grid's boxes and numbers) lies a few millimetres over the asphalt it is
painted on. Where the asphalt was laid again after it, as where tools/join_paddock.py eased the
paddock into the roads, the asphalt can come up over the paint of another model than
velopark_paint.glb, which that moves with it: the edge line of the straight out of T2, in
velopark_kerbs.glb, went up to 4 mm under the paddock beside it, and showed only in pieces. This
lifts every point of paint lying on the asphalt (within NEAR of it) less than LOW over the
highest asphalt under it to OVER over it, and the points of a face whose middle, or an edge's,
is less than LOW over it as much as it is short, in every model. Run again, it finds the paint over the asphalt.
"""
import glob, os

import numpy as np

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
PAINT = ('LinePaint', 'GridNumbers')     # the materials of paint
NOT_ASPHALT = ('KerbRamp', 'KerbConcrete', 'KerbFloor', 'KerbRound', 'Rumble1')
LOW = .002           # metres over the asphalt under which paint is lifted
OVER = .0025         # to this, as the rest of the paint lies
NEAR = .05           # metres from the asphalt paint lies on it


def main():
    models = {f: gltf_source.load_kn5(f) for f in sorted(glob.glob(os.path.join(bp.MODELS, 'velopark*.glb')))}
    asphalt = []
    for k in models.values():
        for n, m in bp.meshes(k):
            if n['name'].startswith('1ROAD') and len(n['faces']) and k['materials'][n['material']]['name'] not in NOT_ASPHALT:
                t = bp.triangles(n, m)
                if len(t):
                    asphalt.append(t)
    top = bp.Surface(np.concatenate(asphalt))
    total = 0
    for f, k in models.items():
        lifted = 0
        for n, m in bp.meshes(k):
            if k['materials'][n['material']]['name'] not in PAINT or not len(n['faces']):
                continue
            if not np.allclose(m, np.eye(4)):
                print('%s %s is moved: left as it is' % (os.path.basename(f), n['name']))
                continue
            V = n['verts']
            y = top.ys(V[:, [0, 2]].astype(float))
            low = ~np.isnan(y) & (V[:, 1] < y + LOW) & (V[:, 1] > y - NEAR)
            if low.any():
                V[low, 1] = (y[low] + OVER).astype(V.dtype)
                lifted += low.sum()
            # and between its points, where the asphalt bends up under a face of it: the face's
            # points lifted as much as its middle or the middle of an edge is short
            F = n['faces'].astype(int)
            for _ in range(3):
                t = V[F, :3].astype(float)
                mids = np.stack([t.mean(1), (t[:, 0] + t[:, 1]) / 2, (t[:, 1] + t[:, 2]) / 2, (t[:, 2] + t[:, 0]) / 2], 1)
                ym = top.ys(mids[:, :, [0, 2]].reshape(-1, 2)).reshape(-1, 4)
                short = np.where(np.isnan(ym) | (mids[:, :, 1] < ym - NEAR), 0, ym + LOW - mids[:, :, 1]).max(1)
                if not (short > 0).any():
                    break
                up = np.zeros(len(V))
                for c in range(3):
                    np.maximum.at(up, F[:, c], np.where(short > 0, short + OVER - LOW, 0))
                V[:, 1] += up.astype(V.dtype)
                lifted += (up > 0).sum()
            if lifted:
                bp.sphere(n)
        if lifted:
            bp.save(k, f)
            print('  %-24s %5d points of paint lifted' % (os.path.basename(f), lifted))
        total += lifted
    print('%d points of paint lifted over the asphalt' % total if total else 'the paint is over the asphalt')


if __name__ == '__main__':
    main()
