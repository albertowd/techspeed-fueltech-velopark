"""Each thing of the track in the model it belongs to, in source/models.

    python tools/organize_models.py

The models grew as the track did, and things went where the tool that made them wrote: the kerbs'
model held tyre walls, painted lines and run-offs, the paddock's the asphalt round the vp_1000's
corners. This puts them in order:

- velopark.glb: the old invisible walls the track came with sunk deep under the ground, where
  nothing reaches them, out.
- velopark_kerbs.glb keeps the kerbs (and their beds); its tyre walls go to velopark_tyres.glb,
  its painted lines to velopark_lines.glb, the asphalt, pavers and grass laid round the track's
  edges (run-offs, patches, islands) to velopark_runoffs.glb.
- velopark_paint.glb, the grid marks, pit boxes, parking bays and lines, goes into
  velopark_lines.glb: every line painted on the track, in one model.
- vp_<layout>_lines.glb, the lines across the mouths of the roads a layout does not take, goes
  into the layout's own model, vp_<layout>.glb.
- velopark_paddock.glb's asphalt round the vp_1000's corners, west and north of it, goes to
  velopark_runoffs.glb; the paddock keeps the pits, the car park and the middle's.
- velopark_extra.glb, the trees, is velopark_trees.glb (its forest for BeamNG velopark_trees.json);
  the kids track's pole's wall in it goes to velopark_poles.glb.
- The layouts' models (track/ac/models_*.ini) follow.

A mesh keeps its name, material and textures. Run again, it finds them in order.
"""
import copy, glob, os, re, sys, urllib.parse

import numpy as np

import build_paths as bp
import gltf_source
import kn5tree
from grade_paths import pieces

M = bp.MODELS
LINES = os.path.join(M, 'velopark_lines.glb')
RUNOFFS = os.path.join(M, 'velopark_runoffs.glb')
TREES = os.path.join(M, 'velopark_trees.glb')
OLD_TREES = os.path.join(M, 'velopark_extra.glb')
FOREST, OLD_FOREST = os.path.join(M, 'velopark_trees.json'), os.path.join(M, 'velopark_extra_trees.json')
PAINT = os.path.join(M, 'velopark_paint.glb')
POLES = os.path.join(M, 'velopark_poles.glb')
SUNK = -20.             # metres: a wall all under this is one of the sunk ones
PAINTS = ('LinePaint', 'GridNumbers')
# the kerbs' model keeps these; its tyre walls and lines go, and the rest are run-offs
KERB = ('1ROAD_KERB', 'KERB_', '1GRASS_KERB_')
TYRE = ('TYRES_', '1WALL_TYRE_')
# the paddock's asphalt round the vp_1000's corners: its pieces with their middle west of x or
# north of z
RUNOFF_WEST, RUNOFF_NORTH = -100., 185.


class Models:
    """The models as they are loaded, and the texture paths their files give."""

    def __init__(self):
        self.k, self.uris, self.changed = {}, {}, set()

    def get(self, path, new_root=None):
        if path not in self.k:
            if os.path.exists(path):
                self.k[path] = gltf_source.load_kn5(path)
                doc, _ = gltf_source.read_glb(path)
                self.uris.update({i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])})
            elif new_root:
                self.k[path] = dict(version=6, extra=0, textures=[], materials=[],
                                    root=dict(kind=1, name=new_root, active=1, matrix=list(np.eye(4).ravel()), children=[]))
            else:
                return None
        return self.k[path]

    def save(self):
        for path in sorted(self.changed):
            k = self.k[path]
            # only the materials its meshes have, and their textures
            nodes = [n for n, _ in bp.meshes(k)]
            keep = sorted({n['material'] for n in nodes})
            remap = {o: i for i, o in enumerate(keep)}
            k['materials'] = [k['materials'][o] for o in keep]
            for n in nodes:
                n['material'] = remap[n['material']]
            used = {t['name'] for m in k['materials'] for t in m['textures']}
            k['textures'] = [t for t in k['textures'] if t['name'] in used]
            doc, binary = gltf_source.to_gltf(k, lambda t: self.uris.get(t['name'], '../textures/' + t['name']))
            gltf_source.write_glb(path, doc, binary)
            print('  saved %s' % os.path.relpath(path, bp.ROOT))


def detach(k, keep):
    """The mesh nodes of k for which keep(node) is false, taken out of it (none under a transform)."""
    placed = {id(n) for n, m in bp.meshes(k) if not np.allclose(m, np.eye(4))}
    parents = [p for p, _ in kn5tree.walk(k['root']) if p['kind'] == 1]
    out = []
    for parent in parents:
        stay = []
        for c in parent['children']:
            if c['kind'] == 2 and not keep(c):
                if id(c) in placed:
                    sys.exit('%s stands under a transform: not moved' % c['name'])
                out.append(c)
            else:
                stay.append(c)
        parent['children'] = stay
    return out


def attach(src, dst, nodes):
    """The nodes, of src, into dst: their materials as dst has them (by name), or brought along with
    their textures. A name dst has already gets a number."""
    names = {n['name'] for n, _ in bp.meshes(dst)}
    for n in nodes:
        mat = src['materials'][n['material']]
        i = next((i for i, m in enumerate(dst['materials']) if m['name'] == mat['name']), None)
        if i is None:
            dst['materials'].append(copy.deepcopy(mat))
            i = len(dst['materials']) - 1
            have = {t['name'] for t in dst['textures']}
            for t in mat['textures']:
                held = next((x for x in src['textures'] if x['name'] == t['name']), None)
                # (one the source does not hold: a library material's texture, which the step that makes it adds)
                if t['name'] not in have and held is not None:
                    dst['textures'].append(held)
                    have.add(t['name'])
        n['material'] = i
        name, j = n['name'], 1
        while n['name'] in names:
            n['name'] = '%s_%d' % (name, j)
            j += 1
        names.add(n['name'])
        dst['root']['children'].append(n)


def move(ms, src_path, dst_path, keep, root=None):
    src = ms.get(src_path)
    if src is None:
        return 0
    nodes = detach(src, keep)
    if nodes:
        attach(src, ms.get(dst_path, root), nodes)
        ms.changed.update((src_path, dst_path))
    return len(nodes)


def paddock_runoffs(ms):
    """The pieces of the paddock's asphalt round the vp_1000's corners, as one mesh of the run-offs."""
    pad = ms.get(bp.PADDOCK)
    taken = []
    for n, m in list(bp.meshes(pad)):
        if not n['name'].startswith(bp.ASPHALT) or not len(n['faces']):
            continue
        V, F = n['verts'][:, :3].astype(float), n['faces'].astype(int)
        _, weld = np.unique(np.round(V[:, [0, 2]] / .01).astype(np.int64), axis=0, return_inverse=True)
        weld = weld.ravel()
        comp = pieces(weld[F], weld.max() + 1)[weld[F[:, 0]]]
        out = np.zeros(len(F), bool)
        for c in np.unique(comp):
            mid = V[F[comp == c]].reshape(-1, 3).mean(0)
            if mid[0] < RUNOFF_WEST or mid[2] > RUNOFF_NORTH:
                out |= comp == c
        if out.any():
            v, f = bp.split(n, out)
            node = dict(n, verts=v, faces=f.astype('<u2'), children=[], name='1ROAD_RUNOFF_PADDOCK')
            bp.sphere(node)
            taken.append(node)
    if taken:
        attach(pad, ms.get(RUNOFFS, 'velopark_runoffs'), taken)
        ms.changed.update((bp.PADDOCK, RUNOFFS))
    return sum(len(n['faces']) for n in taken)


def layouts():
    """Each layout's models in its models_*.ini (track/ac), as the models now are."""
    have = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(M, '*.glb'))}
    rename = {'velopark_extra': 'velopark_trees', 'velopark_paint': 'velopark_lines'}
    done = 0
    for path in sorted(glob.glob(os.path.join(bp.AC_MADE, 'models_*.ini'))):
        text = open(path, newline='').read()
        nl = '\r\n' if '\r\n' in text else '\n'
        files = [rename.get(f, f) for f in re.findall(r'(?m)^FILE=(.*?)\.kn5\s*$', text)]
        files = [f for i, f in enumerate(files) if f in have and f not in files[:i]]
        if 'velopark_runoffs' in have and 'velopark_runoffs' not in files:
            files.insert(files.index('velopark_kerbs') + 1 if 'velopark_kerbs' in files else len(files), 'velopark_runoffs')
        if 'velopark_lines' in have and 'velopark_lines' not in files:
            files.append('velopark_lines')
        new = nl.join('[MODEL_%d]%sFILE=%s.kn5%sPOSITION=0,0,0%sROTATION=0,0,0%s' % (i, nl, f, nl, nl, nl)
                      for i, f in enumerate(files))
        if new != text:
            open(path, 'w', newline='').write(new)
            done += 1
    return done


def main():
    ms = Models()
    # the walls sunk under the ground
    track = ms.get(bp.TRACK)
    sunk = detach(track, lambda n: not (n['name'].startswith('1WALL') and len(n['verts'])
                                        and n['verts'][:, 1].astype(float).max() < SUNK))
    if sunk:
        ms.changed.add(bp.TRACK)
    print('%d sunk walls out of velopark.glb' % len(sunk))
    # the kerbs' model: its tyre walls, its lines, its run-offs
    kerbs = ms.get(bp.KERBS)
    mat = lambda n: kerbs['materials'][n['material']]['name']
    print('%d tyre walls from the kerbs to velopark_tyres.glb'
          % move(ms, bp.KERBS, bp.TYRES, lambda n: not n['name'].startswith(TYRE)))
    print('%d lines from the kerbs to velopark_lines.glb'
          % move(ms, bp.KERBS, LINES, lambda n: mat(n) not in PAINTS, 'velopark_lines'))
    print('%d run-offs from the kerbs to velopark_runoffs.glb'
          % move(ms, bp.KERBS, RUNOFFS, lambda n: n['name'].startswith(KERB), 'velopark_runoffs'))
    # the paint's model, and the layouts' own lines
    gone = []
    if os.path.exists(PAINT):
        print('%d lines from velopark_paint.glb to velopark_lines.glb' % move(ms, PAINT, LINES, lambda n: False, 'velopark_lines'))
        gone.append(PAINT)
    for f in sorted(glob.glob(os.path.join(M, 'vp_*_lines.glb'))):
        own = f[:-len('_lines.glb')] + '.glb'
        print('%d lines from %s to %s' % (move(ms, f, own, lambda n: False), os.path.basename(f), os.path.basename(own)))
        gone.append(f)
    # the paddock's asphalt round the vp_1000
    print('%d faces of the paddock\'s asphalt round the vp_1000 to velopark_runoffs.glb' % paddock_runoffs(ms))
    # the trees, and the kids pole's wall among them
    trees = TREES if os.path.exists(TREES) else OLD_TREES
    print('%d walls from the trees to velopark_poles.glb' % move(ms, trees, POLES, lambda n: 'POLE' not in n['name']))
    ms.save()
    for f in gone:
        os.remove(f)
        print('  %s gone' % os.path.relpath(f, bp.ROOT))
    if os.path.exists(OLD_TREES):
        os.replace(OLD_TREES, TREES)
        print('  velopark_extra.glb is velopark_trees.glb')
    if os.path.exists(OLD_FOREST):
        os.replace(OLD_FOREST, FOREST)
        print('  velopark_extra_trees.json is velopark_trees.json')
    print('%d layouts\' models listed again' % layouts())


if __name__ == '__main__':
    main()
