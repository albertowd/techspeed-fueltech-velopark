"""Parked rental karts round the pits, in source/models/velopark_karts.glb.

    python tools/place_karts.py

The Techspeed rental karts of the kart project (d:\\projects\\kart-techspeed, or KART_PROJECT),
its third level of detail (the farthest has no rims), as it is built for Assetto Corsa (its
build/ac): parked in the pit building's workshop and under the white tent north of it, the spare
rentals (screenshots in tools/reference/site). None stands within CLEAR of a layout's pit box or start, where a player's
kart may stand. Each is drawn from LOD_OUT metres in, and has nothing to hit. Their meshes are
merged by material, their textures in source/textures/velopark_karts/. Run tools/build_outside.py
first (it lays the workshop's floor they stand on). Run again, it makes the model again.
"""
import configparser, glob, math, os, sys

import numpy as np

import build_outside as bo
import build_paths as bp
import gltf_source
import kn5io
import kn5tree

ROOT = bp.ROOT
KART_PROJECT = os.environ.get('KART_PROJECT', r'D:\projects\kart-techspeed')
KART = os.path.join(KART_PROJECT, 'build', 'ac', 'content', 'cars', 'techspeed_rental_11hp', 'techspeed_rental_11hp_LOD_C.kn5')
OUT = os.path.join(bp.MODELS, 'velopark_karts.glb')
TEX = os.path.join(ROOT, 'source', 'textures', 'velopark_karts')
LAYOUTS = os.path.join(ROOT, 'source', 'track')
BUILD = os.path.join(ROOT, 'build', 'ac', 'content', 'tracks', 'techspeed_fueltech_velopark')
CLEAR = 2.5              # metres from a pit box or a start a kart may not stand
LOD_OUT = 60.            # metres from which a kart is drawn


def spots():
    """Where each kart stands: (x, z, heading: the way its nose points, radians from +z to +x)."""
    c = np.array(bo.PIT_BUILDING[0], float)
    ez = np.array(bo.PIT_BUILDING[1], float)
    ez /= np.linalg.norm(ez)
    ex = np.array([ez[1], -ez[0]])
    along = math.atan2(ez[0], ez[1])
    out = []
    # the workshop: two rows down its middle, between the doorways and either side of them
    for x in (-1.7, 1.7):
        for z in (-10., -2.2, 2.2, 10.):
            p = c + ex * x + ez * z
            out.append((p[0], p[1], along + (math.pi if x < 0 else 0.), 'workshop'))
    # the tent: two rows of four, noses south towards the pits
    tc = np.array(bo.tent_on_floor()[0], float)
    tz = np.array(bo.tent_on_floor()[2], float)
    tz /= np.linalg.norm(tz)
    tx = np.array([tz[1], -tz[0]])
    for x in (-3.6, -1.2, 1.2, 3.6):
        for z in (-1.9, 1.9):
            p = tc + tx * x + tz * z
            out.append((p[0], p[1], math.atan2(-tz[0], -tz[1]), 'tent'))
    return out


def taken():
    """Every layout's pit boxes and starts, (x, z)."""
    P = []
    for f in glob.glob(os.path.join(BUILD, 'models_*.ini')):
        c = configparser.ConfigParser()
        c.read(f)
        for s in c.sections():
            path = os.path.join(BUILD, c[s]['FILE'])
            if not os.path.exists(path):
                continue        # (this model, not made yet)
            m = kn5io.Model(path)
            P += [(w[3, 0], w[3, 2]) for k, w in m.dummies.items() if k.startswith(('AC_PIT', 'AC_START', 'AC_HOTLAP'))]
    return np.array(P, float)


def floor_at():
    """The top where a kart stands: the ground, the roads, the workshop's floor."""
    tris = []
    for f in ('velopark.glb', 'velopark_kerbs.glb', 'velopark_runoffs.glb', 'velopark_paddock.glb', 'velopark_outside.glb'):
        for n, m in bp.meshes(gltf_source.load_kn5(os.path.join(bp.MODELS, f))):
            if len(n['faces']) and (n['name'].startswith(('1GRASS', '1ROAD', '1KERB')) or n['name'].endswith('_FLOOR')):
                tris.append(bp.triangles(n, m))
    return bp.Surface(np.concatenate(tris))


def main():
    if not os.path.exists(KART):
        sys.exit('no kart at %s: build the kart project for Assetto Corsa, or set KART_PROJECT' % KART)
    kart = kn5tree.load(KART)
    # the kart's meshes in its own space, by material
    by_mat = {}
    for n, m in kn5tree.walk(kart['root']):
        if n['kind'] == 2 and len(n['faces']) and n['visible']:
            V = n['verts'].astype(float).copy()
            V[:, :3] = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
            V[:, 3:6] = V[:, 3:6] @ m[:3, :3]
            V[:, 8:11] = V[:, 8:11] @ m[:3, :3]
            by_mat.setdefault(n['material'], []).append((V, n['faces'].astype(int)))
    floor = floor_at()
    busy = taken()
    placed = []
    for x, z, heading, where in spots():
        if len(busy) and np.min(np.hypot(busy[:, 0] - x, busy[:, 1] - z)) < CLEAR:
            continue
        # its four wheels' ground, the highest of them
        c, s = math.cos(heading), math.sin(heading)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # its +z to (sin h, cos h)
        wheels = np.array([(.58, .65), (-.58, .65), (.58, -.47), (-.58, -.47)])
        W = wheels @ R[[0, 2]][:, [0, 2]].T + (x, z)
        y = np.nanmax(floor.ys(W))
        placed.append((np.array([x, y, z]), R, where))
    print('%d karts parked (%s)' % (len(placed), ', '.join('%d in the %s' % (sum(p[2] == w for p in placed), w)
                                                           for w in sorted({p[2] for p in placed}))))
    children = []
    for mi, parts in sorted(by_mat.items()):
        allv, allf, nv = [], [], 0
        runs = []
        for at, R, _ in placed:
            for V, F in parts:
                W = V.copy()
                W[:, :3] = V[:, :3] @ R.T + at
                W[:, 3:6] = V[:, 3:6] @ R.T
                W[:, 8:11] = V[:, 8:11] @ R.T
                if nv + len(W) > 60000 and allv:
                    runs.append((np.concatenate(allv), np.concatenate(allf)))
                    allv, allf, nv = [], [], 0
                allv.append(W)
                allf.append(F + nv)
                nv += len(W)
        if allv:
            runs.append((np.concatenate(allv), np.concatenate(allf)))
        name = kart['materials'][mi]['name'].upper()
        for i, (V, F) in enumerate(runs):
            node = dict(kind=2, name='KART_%s%s' % (name, '' if i == 0 else '_%d' % i), active=1, children=[], material=mi,
                        cast_shadows=1, visible=1, transparent=0, layer=0, lod_in=0., lod_out=LOD_OUT, renderable=1,
                        verts=V.astype('<f4'), faces=F.astype('<u2'))
            bp.sphere(node)
            children.append(node)
    k = dict(version=kart.get('version', 6), extra=kart.get('extra', 0), textures=kart['textures'], materials=kart['materials'],
             root=dict(kind=1, name='velopark_karts', active=1, matrix=list(np.eye(4).ravel()), children=children))
    os.makedirs(TEX, exist_ok=True)
    for t in kart['textures']:
        open(os.path.join(TEX, t['name']), 'wb').write(t['data'])
    doc, binary = gltf_source.to_gltf(k, lambda t: '../textures/velopark_karts/' + t['name'])
    gltf_source.write_glb(OUT, doc, binary)
    print('%d meshes, %d triangles, in %s' % (len(children), sum(len(c['faces']) for c in children), os.path.relpath(OUT, ROOT)))
    added = 0
    for f in sorted(glob.glob(os.path.join(LAYOUTS, 'models_*.ini'))):
        text = open(f, newline='').read()
        if 'velopark_karts.kn5' in text:
            continue
        nl = '\r\n' if '\r\n' in text else '\n'
        n = text.count('[MODEL_')
        text = text.rstrip() + nl + nl + '[MODEL_%d]%sFILE=velopark_karts.kn5%sPOSITION=0,0,0%sROTATION=0,0,0%s' % (n, nl, nl, nl, nl)
        open(f, 'w', newline='').write(text)
        added += 1
    print('%d layouts given the model' % added)


if __name__ == '__main__':
    main()
