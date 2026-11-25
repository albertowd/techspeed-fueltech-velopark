"""The track's trees planted where the source places them (TRACK_FORMAT.md, "Moving there", 7: `trees`), into
track/models/velopark_trees.glb, and BeamNG's forest of them beside it.

    python tools/place_trees.py

Each tree of the placements (track.json's trees: [{model, x, z, heading, scale}]) is its catalogue model (a template
of kind "tree": a tree of the plantation, or a mass of its understory), sized by its scale, turned to its heading
(radians from +z toward +x, as the objects') and standing on the floor where it is. Its parts go into the meshes of
the TILE metre square it stands in, one for each part of each tile, as few points as a mesh may hold apart
(EUCA_<PART>_<i>_<j>_<n>); its hull, if it has one (its trunk's wall), into the tile's unseen wall that cars hit
(1WALL_EUCA_TRUNKS_<i>_<j>_<n>). Each is drawn as its catalogue's draw says. BeamNG's forest of them
(velopark_trees.json: each tree as [model, x, y, z, yaw, scale], yaw build_trees' way, -heading) is written beside
it. Run before the objects step, which clears the trees where a solid object stands.
"""
import json
import os

import numpy as np

import build_paths as bp
import build_trees as bt
import gltf_source
import templates
import track_data

TREES = os.path.join(bp.MODELS, 'velopark_trees.glb')
FOREST = os.path.join(bp.MODELS, 'velopark_trees.json')
FLOOR = ('velopark.glb', 'velopark_kerbs.glb', 'velopark_runoffs.glb', 'velopark_paddock.glb', 'velopark_kids.glb')


def floor():
    """The top of the floor: its ground, roads and kerbs."""
    tris = []
    for f in FLOOR:
        p = os.path.join(bp.MODELS, f)
        if not os.path.exists(p):
            continue
        for n, m in bp.meshes(gltf_source.load_kn5(p)):
            if len(n['faces']) and n['name'].startswith(('1GRASS', '1ROAD', '1KERB', 'GRASS')):
                tris.append(bp.triangles(n, m))
    return bp.Surface(np.concatenate(tris))


def material_of(k, name):
    """The index of a library material in model k, added if it has none of it, its textures with it."""
    names = [m['name'] for m in k['materials']]
    if name in names:
        return names.index(name)
    m = gltf_source._material(name, {})
    have = {t['name'] for t in k['textures']}
    import materials
    for t in m['textures']:
        if t['name'] not in have:
            k['textures'].append(dict(name=t['name'], active=1,
                                      data=materials.read_texture(os.path.join(materials.TEXTURES, t['name']))))
            have.add(t['name'])
    k['materials'].append(m)
    return len(k['materials']) - 1


def laid(V, x, y, z, heading, scale):
    """A template's points (n, 11) sized, turned and stood at x, y, z."""
    W = np.asarray(V, float).copy()
    W[:, :3] *= scale
    return templates.world(W, np.array([x, z]), y, heading)


def main():
    data, _ = track_data.part('trees')
    stands = (data or {}).get('trees', [])
    cat = templates.catalogue()
    if os.path.exists(TREES):
        k = gltf_source.load_kn5(TREES)
        k['root']['children'] = []
    else:
        k = dict(version=6, extra=0, textures=[], materials=[],
                 root=dict(kind=1, name='velopark_trees', active=1, matrix=list(np.eye(4).ravel()), children=[]))
    ys = floor().ys(np.array([[t['x'], t['z']] for t in stands], float)) if stands else np.zeros(0)
    parts_of, tiles, walls, forest = {}, {}, {}, []
    for t, y in zip(stands, ys):
        if np.isnan(y):
            continue
        if t['model'] not in parts_of:
            parts_of[t['model']] = templates.load(t['model'])
        key = tuple(np.floor(np.array([t['x'], t['z']]) / bt.TILE).astype(int))
        draw = tuple(float(d or 0.) for d in (cat[t['model']].get('draw') or (0., 0.)))
        for part, V, F, mat in parts_of[t['model']]:
            if part == 'hull':
                walls.setdefault(key, []).append((laid(V, t['x'], y, t['z'], t['heading'], t['scale']), F))
            else:
                tiles.setdefault((key, part, mat, draw), []).append((laid(V, t['x'], y, t['z'], t['heading'], t['scale']), F))
        forest.append([t['model'], float(t['x']), float(y), float(t['z']), -float(t['heading']), float(t['scale'])])
    template = dict(kind=2, name='', active=1, children=[], material=0, cast_shadows=1, visible=1, transparent=0, layer=0,
                    lod_in=0., lod_out=0., renderable=1)

    def nodes(name, runs, mat, flags):
        out = []
        for i, run in enumerate(bt.chunks([(V[:, :3], V[:, 3:6], V[:, 6:8], F) for V, F in runs])):
            out.append(bt.node(template, '%s_%d' % (name, i), run, material_of(k, mat), flags))
        return out
    root = k['root']['children']
    tris = 0
    for (key, part, mat, draw), runs in sorted(tiles.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for nd in nodes('%s%s_%d_%d' % (bt.PREFIX, part.upper(), key[0], key[1]), runs, mat, dict(lod_in=draw[0], lod_out=draw[1])):
            root.append(nd)
            tris += len(nd['faces'])
    for key in sorted(walls):
        root += nodes('1WALL_%sTRUNKS_%d_%d' % (bt.PREFIX, key[0], key[1]), walls[key], 'EucaBark', dict(renderable=0, cast_shadows=0))
    used = {t['name'] for m in k['materials'] for t in m['textures']}
    k['textures'] = [t for t in k['textures'] if t['name'] in used]
    bt.save(k, TREES) if os.path.exists(TREES) else gltf_source.write_glb(TREES, *gltf_source.to_gltf(k, lambda t: '../textures/' + t['name']))
    json.dump(dict(trees=forest), open(FOREST, 'w'), indent=0)
    print('%d trees of %d kinds planted, %d triangles; BeamNG\'s forest in %s' % (
        len(forest), len(parts_of), tris, os.path.relpath(FOREST, bp.ROOT)))


if __name__ == '__main__':
    main()
