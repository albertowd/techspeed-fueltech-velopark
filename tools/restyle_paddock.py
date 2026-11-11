"""The paddock's buildings as they are: grey block walls, salmon red pillars, in
source/models/velopark_paddock.glb.

    python tools/restyle_paddock.py

The paddock's buildings were all white, walls and posts alike. The real ones (the screenshots
of the vp_1500's start and the south) have walls of grey painted block and pillars and posts
painted salmon red. This takes every pillar and post out of the walls' mesh (a piece of it
narrower than PILLAR and taller than TALL) into a mesh of its own, BLD_PILLARS, of a material of
its own; and paints the walls grey (BuildingWallGrey.dds, made here). It takes out too the
buildings tools/build_outside.py makes again (REMADE: the race control tower, the fuel station),
each piece of the buildings' meshes whose middle stands in one, and their walls to hit. Run
again, it finds the paddock restyled.
"""
import os

import numpy as np
from PIL import Image

import build_paths as bp
import gltf_source
import kn5io
import kn5tree
from plant_trees import pieces

TEX = os.path.join(bp.ROOT, 'source', 'textures')
PILLAR, TALL = .8, 2.         # metres: a piece narrower than this and taller than that is a pillar
# the buildings made again by tools/build_outside.py: (x0, x1, z0, z1), and their walls to hit
REMADE = {'tower': ((-52., -42., 75., 86.), ('1WALL_BLD_C3',)),
          'fuel station': ((178., 214., -23., -5.), ('1WALL_BLD_W1', '1WALL_BLD_W2'))}
BUILDING_MESHES = ('BLD_WALLS', 'BLD_ROOFS', 'BLD_DARK', 'BLD_DOORS', 'BLD_SOLAR')


def grey_wall(n=512, seed=3):
    """Painted concrete block: courses of blocks, a darker plinth, a little grime."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    g = np.asarray(Image.fromarray((rng.random((32, 32)) * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float) / 255
    course = (y * 10) % 1
    off = (np.floor(y * 10) % 2) * .5
    joint = ((course < .04) | (((x * 5 + off) % 1) < .02)).astype(float)
    base = 120 + 14 * (g - .5) - 28 * joint
    rgb = base[..., None] * np.array([.96, .98, 1.02])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def pillar_paint(n=256, seed=4):
    rng = np.random.default_rng(seed)
    g = np.asarray(Image.fromarray((rng.random((16, 16)) * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float) / 255
    rgb = (np.array([196., 104., 92.]) * (.9 + .15 * g[..., None]))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def inside(c, box, pad=1.):
    x0, x1, z0, z1 = box
    return (c[:, 0] > x0 - pad) & (c[:, 0] < x1 + pad) & (c[:, 1] > z0 - pad) & (c[:, 1] < z1 + pad)


def main():
    k = gltf_source.load_kn5(bp.PADDOCK)
    if any(n['name'] == 'BLD_PILLARS' for n, m in bp.meshes(k)):
        print('the paddock is restyled already')
        return
    nodes = {n['name']: (n, m) for n, m in bp.meshes(k)}
    # the remade buildings out
    gone_walls = {w for _, walls in REMADE.values() for w in walls}
    taken = 0
    for name in BUILDING_MESHES:
        if name not in nodes:
            continue
        n, m = nodes[name]
        V = n['verts'][:, :3].astype(float) + m[3, :3]
        F = n['faces'].astype(int)
        lab = pieces(F, len(V))
        flab = lab[F[:, 0]]
        ids = np.unique(flab)
        c = np.array([V[lab == t][:, [0, 2]].mean(0) for t in ids])
        out = np.zeros(len(ids), bool)
        for box, _ in REMADE.values():
            out |= inside(c, box)
        if out.any():
            bp.split(n, np.isin(flab, ids[out]))
            taken += out.sum()
    for parent, _ in kn5tree.walk(k['root']):
        parent['children'] = [ch for ch in parent['children'] if ch['name'] not in gone_walls]
    # the pillars out of the walls, into their own mesh
    n, m = nodes['BLD_WALLS']
    V = n['verts'][:, :3].astype(float) + m[3, :3]
    F = n['faces'].astype(int)
    lab = pieces(F, len(V))
    flab = lab[F[:, 0]]
    ids = np.unique(flab)
    pillar = []
    for t in ids:
        p = V[lab == t]
        span = np.ptp(p[:, [0, 2]], axis=0)
        if span.max() < PILLAR and np.ptp(p[:, 1]) > TALL:
            pillar.append(t)
    sel = np.isin(flab, pillar)
    verts, faces = bp.split(n, sel)
    # the new materials and textures
    for name, img in (('BuildingWallGrey.dds', grey_wall()), ('BuildingPillar.dds', pillar_paint())):
        data = kn5io.dds(img, 'DXT1')
        open(os.path.join(TEX, name), 'wb').write(data)
        k['textures'] = [t for t in k['textures'] if t['name'] != name] + [dict(name=name, active=1, data=data)]
    wall_mat = next(i for i, mt in enumerate(k['materials']) if mt['name'] == 'BuildingWall')
    k['materials'][wall_mat]['textures'] = [dict(sampler='txDiffuse', slot=0, name='BuildingWallGrey.dds')]
    base = k['materials'][wall_mat]
    k['materials'].append(dict(base, name='BuildingPillar', props=[dict(p) for p in base['props']],
                               textures=[dict(sampler='txDiffuse', slot=0, name='BuildingPillar.dds')]))
    pnode = dict(n, name='BLD_PILLARS', children=[], material=len(k['materials']) - 1, verts=verts, faces=faces.astype('<u2'))
    bp.sphere(pnode)
    parent = next(p for p, _ in kn5tree.walk(k['root']) if any(ch is n for ch in p.get('children', [])))
    parent['children'].append(pnode)
    # the texture no mesh uses any more out
    used = {t['name'] for mt in k['materials'] for t in mt['textures']}
    k['textures'] = [t for t in k['textures'] if t['name'] in used]
    doc, _ = gltf_source.read_glb(bp.PADDOCK)
    import urllib.parse
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(bp.PADDOCK, doc, binary)
    print('%d pieces of the remade buildings out, %d of their walls; %d pillars and posts painted' % (taken, len(gone_walls), len(pillar)))


if __name__ == '__main__':
    main()
