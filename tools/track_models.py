"""The track's models as every game's build reads them (TRACK_FORMAT.md, "Models"): each mesh of a model (.glb) with
what it is (extras.track: its role, its ground, whether it is seen, how far it is drawn, its shadows, whether grass
grows on it), its material's name in the library (source/materials.json) and its corners and triangles where it
stands, in the track's coordinates (y up). Nothing here is one game's: AC's names, flags and shaders are made by its
own build (tools/gltf_source.py through tools/materials.py); its round trip's data (extras.kn5) is not read.
"""
import os

import numpy as np

import gltf_source
import materials

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'track', 'models')
TEXTURES = os.path.join(ROOT, 'source', 'textures')


def path(stem):
    """A model's file in track/models, by its name."""
    return os.path.join(MODELS, stem + '.glb')


def texture(png, model=None):
    """A texture's file (a PNG the library names): the model's own (source/textures/<model>/, as tools/gltf_source.py
    looks first) where it holds one, else the track's."""
    own = os.path.join(TEXTURES, model, png) if model else None
    return own if own and os.path.exists(own) else os.path.join(TEXTURES, png)


def ground(mesh):
    """The kind of ground a mesh is driven as (a key of surfaces.json), or None if nothing drives on it: a wall's
    `wall`; a surface's its material's `surface` (a rumble's paint, a road's tarmac), or its own where its material
    has none."""
    if mesh['role'] == 'wall':
        return 'wall'
    if mesh['role'] != 'surface':
        return None
    return materials.library()['materials'].get(mesh['material'], {}).get('surface') or mesh['surface']


class Model:
    """A model's meshes, [{name, role, surface, visible, draw (near, far or None), shadows, grass, material,
    v (n, 8): position, normal, uv; f (m, 3): its triangles}], in the order the model has them. A mesh over the
    65536 corners a 16-bit index reaches is cut in pieces of its name, as games take them."""

    def __init__(s, glb):
        s.name = os.path.splitext(os.path.basename(glb))[0]
        doc, binary = gltf_source.read_glb(glb)
        names = []
        for g in doc.get('materials', []):
            x = (g.get('extras') or {}).get('track') or {}
            names.append(x.get('material') or gltf_source.BLENDER_COPY.sub('', g.get('name', 'material')))
        s.meshes = []

        def node(i, parent):
            g = doc['nodes'][i]
            m = np.array(gltf_source._matrix(g), float).reshape(4, 4) @ parent       # (rows taking row points)
            if 'mesh' in g:
                s._mesh(doc, binary, g, m, names)
            for c in g.get('children', []):
                node(c, m)
        for i in doc['scenes'][doc.get('scene', 0)]['nodes']:
            node(i, np.eye(4))

    def _mesh(s, doc, binary, g, m, names):
        track = (g.get('extras') or {}).get('track') or {}
        name = gltf_source.BLENDER_COPY.sub('', g.get('name', 'mesh'))
        draw = track.get('draw') or [0., None]
        for prim in doc['meshes'][g['mesh']]['primitives']:
            if prim.get('mode', 4) != 4:
                continue
            a = prim['attributes']
            pos = gltf_source._accessor(doc, binary, a['POSITION']).astype(float)
            faces = (gltf_source._accessor(doc, binary, prim['indices']).reshape(-1, 3) if 'indices' in prim
                     else np.arange(len(pos)).reshape(-1, 3)).astype(np.int64)
            uv = gltf_source._accessor(doc, binary, a['TEXCOORD_0']).astype(float) if 'TEXCOORD_0' in a else np.zeros((len(pos), 2))
            if 'NORMAL' in a:
                nor = gltf_source._accessor(doc, binary, a['NORMAL']).astype(float) @ np.linalg.inv(m[:3, :3]).T
            else:
                fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
                nor = np.zeros_like(pos)
                for c in range(3):
                    np.add.at(nor, faces[:, c], fn)
            nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
            pos = pos @ m[:3, :3] + m[3, :3]
            verts = np.c_[pos, nor, uv].astype('<f4')
            for v, f in gltf_source._chunks(verts, faces):
                s.meshes.append(dict(name=name, role=track.get('role', 'visual'), surface=track.get('surface'),
                                     visible=track.get('visible', True) is not False,
                                     draw=(float(draw[0] or 0.), None if draw[1] is None else float(draw[1])),
                                     shadows=track.get('shadows', True) is not False, grass=bool(track.get('grass')),
                                     material=names[prim['material']] if 'material' in prim and names else None,
                                     v=v, f=f.astype(np.int64)))

    def mesh(s, name):
        return next(x for x in s.meshes if x['name'] == name)
