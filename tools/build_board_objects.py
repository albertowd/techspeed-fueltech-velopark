"""The layouts' distance boards made loose, standing on the ground, in each layout's own model.

    python tools/build_board_objects.py

The boards counting down to the corners (MarkerBoards) were walls of the 0.5.0 track, fixed where
they stood, in models of their own (source/models/vp_*_boards.glb, and velopark_board50.glb, a
50 m board), a few of them half in the ground where it falls away, their numbers hard to see.
Each becomes a loose object as the cones are, that a kart knocks over, in the model of the layout
it belongs to (source/models/vp_<layout>.glb), beside its cones: AC takes loose objects from a
track as AC_POBJECT_0, AC_POBJECT_1 and on, so each is numbered on from the layout's last cone, a
dummy AC_POBJECT_<n> where it stands, holding the box AC moves it by, unseen, and the board drawn,
BOARD_<its board>. It stands on the ground under it, its foot on the plane through the ground at
its four corners, LIFT over it, so none of it is under the ground and it does not topple as it
settles. The layouts load the boards' own models no more (source/track/models_<layout>.ini), and
they go. The BeamNG.drive export makes the boards loose too (tools/export_beamng.py). Run again,
it finds the boards loose.
"""
import glob, os, re, urllib.parse

import numpy as np

import build_paths as bp
import gltf_source
import kn5tree

ROOT = bp.ROOT
LIFT = .01               # metres over the ground a board's foot is put, to settle on it
GROUND_MODELS = ('velopark', 'velopark_kerbs', 'velopark_paddock', 'velopark_kids')


def ground_surface():
    tris = []
    for f in GROUND_MODELS:
        k = gltf_source.load_kn5(os.path.join(bp.MODELS, f + '.glb'))
        for n, m in bp.meshes(k):
            name = n['name']
            if name[:1].isdigit() and 'WALL' not in name and len(n['faces']):
                t = bp.triangles(n, m)
                if len(t):
                    tris.append(t)
    return bp.Surface(np.concatenate(tris))


def node(template, **kw):
    out = dict(template)
    out.update(children=[], **kw)
    return out


def layouts():
    """Each layout: its ini, the model of its own and the models of boards it loads."""
    out = []
    for ini in sorted(glob.glob(os.path.join(ROOT, 'source', 'track', 'models_*.ini'))):
        files = re.findall(r'(?m)^FILE=(\S+)\.kn5\s*$', open(ini, encoding='utf-8').read())
        own = 'vp_' + os.path.basename(ini)[len('models_vp_'):-len('.ini')]
        boards = [f for f in files if f.endswith('_boards') or f == 'velopark_board50']
        out.append((ini, own, boards))
    return out


def drop_models(ini, gone):
    """The models gone out of a layout's ini, its MODEL_ sections numbered again."""
    raw = open(ini, 'rb').read().decode('utf-8')
    nl = '\r\n' if '\r\n' in raw else '\n'
    blocks = [b for b in re.split(r'(?m)^(?=\[MODEL_\d+\])', raw.replace('\r\n', '\n')) if b.strip()]
    keep = [b for b in blocks if not any('FILE=%s.kn5' % g in b for g in gone)]
    keep = [re.sub(r'^\[MODEL_\d+\]', '[MODEL_%d]' % i, b) for i, b in enumerate(keep)]
    text = '\n'.join(b.strip('\n') + '\n' for b in keep)
    open(ini, 'wb').write(text.replace('\n', nl).encode('utf-8'))


def save(k, path):
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)


def loose(n, m, mi, number, ground, dummy, box_t):
    """A board's mesh as a loose object standing on the ground, AC_POBJECT_<number>, or None."""
    V = n['verts'].astype(float)
    P = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
    # its foot: the corners of its bottom, and the ground's plane through them
    bottom = P[:, 1].min()
    foot = P[np.abs(P[:, 1] - bottom) < 1e-3][:, [0, 2]]
    g = ground.ys(foot)
    if np.isnan(g).all():
        return None
    g = np.where(np.isnan(g), np.nanmax(g), g)
    plane = np.linalg.lstsq(np.c_[foot, np.ones(len(foot))], g, rcond=None)[0]
    mid = foot.mean(0)
    at = np.array([mid[0], plane @ [mid[0], mid[1], 1] + LIFT, mid[1]])
    # each point as high over the plane as it was over its foot
    Q = P.copy()
    Q[:, 1] = P[:, 1] - bottom + np.c_[P[:, [0, 2]], np.ones(len(P))] @ plane + LIFT
    Q -= at
    # it stands on the ground at its highest under it, nowhere in it
    under = ground.ys(P[:, [0, 2]])
    dip = np.nanmax(np.where(np.isnan(under), -np.inf, under - (Q[:, 1] + at[1])))
    if dip > 0:
        at[1] += dip
    name = n['name'][len('1WALL_BOARD_'):]
    drawn = dict(n)
    drawn.update(name='BOARD_' + name, children=[], material=mi, verts=np.c_[Q, V[:, 3:]].astype('<f4'))
    bp.sphere(drawn)
    lo, hi = Q.min(0), Q.max(0)
    # the box AC moves it by: the board's own corners, its faces as the cone's box's
    corners = Q[np.unique(np.round(Q, 4), axis=0, return_index=True)[1]]
    if len(corners) != 8:
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    B = np.zeros((8, 11))
    B[:, :3] = order_box(corners)
    box = dict(box_t)
    box.update(name='AC_POBJECT_%d' % number, children=[], material=mi, verts=B.astype('<f4'), faces=box_t['faces'].copy())
    bp.sphere(box)
    holder = node(dummy, name='BOARD_' + name, matrix=list(np.eye(4).ravel()))
    holder['children'] = [drawn]
    obj = node(dummy, name='AC_POBJECT_%d' % number, matrix=list(np.r_[np.eye(4)[:3].ravel(), at, 1.]))
    obj['children'] = [box, holder]
    return obj


def main():
    todo = [(ini, own, boards) for ini, own, boards in layouts() if boards]
    if not todo:
        print('the boards are loose already')
        return
    ground = ground_surface()
    cone = gltf_source.load_kn5(os.path.join(bp.MODELS, 'vp_1500.glb'))
    for n, m in kn5tree.walk(cone['root']):
        if n['kind'] == 1 and n['name'].startswith('AC_POBJECT'):
            dummy = n
            box_t = next(c for c in n['children'] if c['kind'] == 2)
            break
    used = set()
    for ini, own, board_models in todo:
        path = os.path.join(bp.MODELS, own + '.glb')
        k = gltf_source.load_kn5(path)
        objects = [n for n, m in kn5tree.walk(k['root']) if n['kind'] == 1 and re.match(r'AC_POBJECT_\d+$', n['name'])]
        parent = next(p for p, _ in kn5tree.walk(k['root']) if any(c is objects[0] for c in p['children']))
        number = max(int(n['name'][len('AC_POBJECT_'):]) for n in objects) + 1
        for bm in board_models:
            b = gltf_source.load_kn5(os.path.join(bp.MODELS, bm + '.glb'))
            for n, m in bp.meshes(b):
                if not n['name'].startswith('1WALL_BOARD_'):
                    continue
                # its material and texture, as the boards' model has them
                mat = b['materials'][n['material']]
                mi = next((i for i, x in enumerate(k['materials']) if x['name'] == mat['name']), None)
                if mi is None:
                    k['materials'].append(mat)
                    mi = len(k['materials']) - 1
                    for t in mat['textures']:
                        if not any(x['name'] == t['name'] for x in k['textures']):
                            k['textures'].append(next(x for x in b['textures'] if x['name'] == t['name']))
                obj = loose(n, m, mi, number, ground, dummy, box_t)
                if obj is None:
                    print('  %s %s: no ground under it, left out' % (own, n['name']))
                    continue
                parent['children'].append(obj)
                print('  %-22s %-16s AC_POBJECT_%d' % (own, n['name'][len('1WALL_BOARD_'):], number))
                number += 1
        save(k, path)
        drop_models(ini, board_models)
        used.update(board_models)
    for bm in sorted(used):
        os.remove(os.path.join(bp.MODELS, bm + '.glb'))
    print("the boards loose in each layout's model; %s gone" % ', '.join(sorted(used)))


def order_box(C):
    """Eight corners in the order of the cone's box: top then bottom round one way, as its faces
    take them: 0 1 2 3 a side (top, top, bottom, bottom), 4 7 6 5 the other."""
    c = C.mean(0)
    top = C[C[:, 1] > c[1]]
    bot = C[C[:, 1] <= c[1]]
    ang = lambda p: np.arctan2(p[:, 2] - c[2], p[:, 0] - c[0])
    top = top[np.argsort(ang(top))]
    bot = bot[np.argsort(ang(bot))]
    # as the cone's: 0 top a, 1 top b, 2 bottom b, 3 bottom a, 4 top d, 5 bottom d, 6 bottom c, 7 top c
    a, b, cc, d = range(4)
    return np.array([top[a], top[b], bot[b], bot[a], top[d], bot[d], bot[cc], top[cc]])


if __name__ == '__main__':
    main()
