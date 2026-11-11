"""The track's source models: kn5 models as glTF binaries (.glb), and back.

A model is source/models/<name>.glb; its textures are files of their own in source/textures,
which the .glb points at, so models share them and they can be edited as images. Everything a
kn5 holds that glTF has no place for is kept in the glTF "extras" under "kn5", so
kn5 -> glb -> kn5 gives the same bytes:

    the file         version, extra, and the textures in order: [{name, active, uri}]
    a material       shader, blend, alpha_tested, depth_mode, props [[name, value]], and the
                     texture slots [[sampler, slot, texture name]]; the props' 36 trailing
                     bytes when not all zero, as hex
    a node           active; a mesh's cast_shadows, visible, transparent, layer, lod_in,
                     lod_out, renderable and bounding sphere, with a hash of the vertices it
                     was kept for: some kn5 spheres are stale, and are kept only while the
                     mesh is as it was

A model edited in Blender (File > Import / Export > glTF 2.0, with "Custom Properties" on, as
"glTF Binary", textures left where they are) comes back as well: a mesh moved or turned has its
move baked into its vertices; a mesh of several materials becomes one kn5 mesh per material,
all of its name; Blender's ".001" endings on repeated names are dropped; missing tangents are
worked out; a material without kn5 data becomes a plain ksPerPixel one with its base colour
texture; the sphere of a mesh that changed is measured again.

Positions are AC's own: y up, as glTF. UVs too: both have v running down the texture.
"""
import ast, hashlib, json, os, re, struct, urllib.parse

import numpy as np

import kn5tree

GLB, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
FLOAT, USHORT, UINT, UBYTE = 5126, 5123, 5125, 5121
IDENTITY = [1., 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
MESH_KEYS = ('cast_shadows', 'visible', 'transparent', 'layer', 'lod_in', 'lod_out', 'renderable')
# what Blender adds to a name it has seen before
BLENDER_COPY = re.compile(r'\.\d{3,}$')
DEFAULT_PROPS = (('ksAmbient', .5), ('ksDiffuse', .5), ('ksSpecular', 0.), ('ksSpecularEXP', 1.),
                 ('ksEmissive', 0.), ('ksAlphaRef', 0.))


# ---- kn5 -> glb ------------------------------------------------------------------------
class _Bin:
    def __init__(s):
        s.parts, s.size, s.views, s.accessors = [], 0, [], []

    def add(s, arr, ctype, kind, target=None, minmax=False):
        data = np.ascontiguousarray(arr).tobytes()
        view = dict(buffer=0, byteOffset=s.size, byteLength=len(data))
        if target:
            view['target'] = target
        s.parts.append(data)
        s.size += len(data)
        pad = -s.size % 4
        s.parts.append(b'\0' * pad)
        s.size += pad
        s.views.append(view)
        acc = dict(bufferView=len(s.views) - 1, componentType=ctype, count=int(len(arr) if kind != 'SCALAR' else arr.size), type=kind)
        if minmax:
            acc['min'] = [float(x) for x in arr.min(0)]
            acc['max'] = [float(x) for x in arr.max(0)]
        s.accessors.append(acc)
        return len(s.accessors) - 1


def to_gltf(k, tex_uri):
    """A parsed kn5 (kn5tree) as (glTF json, binary chunk). tex_uri(texture) gives the uri the
    glb points at for each of the kn5's textures."""
    b = _Bin()
    images = [dict(name=t['name'], uri=urllib.parse.quote(tex_uri(t))) for t in k['textures']]
    by_name = {t['name']: i for i, t in enumerate(k['textures'])}
    textures = [dict(source=i) for i in range(len(images))]
    materials = []
    for m in k['materials']:
        x = dict(shader=m['shader'], blend=m['blend'], alpha_tested=m['alpha_tested'], depth_mode=m['depth_mode'],
                 props=[[p['name'], p['value']] for p in m['props']],
                 textures=[[t['sampler'], t['slot'], t['name']] for t in m['textures']])
        rest = {p['name']: p['rest'].hex() for p in m['props'] if p['rest'] != bytes(36)}
        if rest:
            x['rest'] = rest
        g = dict(name=m['name'], extras=dict(kn5=x), pbrMetallicRoughness=dict(metallicFactor=0., roughnessFactor=1.))
        slots = {t['sampler']: t['name'] for t in m['textures']}
        if slots.get('txDiffuse') in by_name:
            g['pbrMetallicRoughness']['baseColorTexture'] = dict(index=by_name[slots['txDiffuse']])
        if slots.get('txNormal') in by_name:
            g['normalTexture'] = dict(index=by_name[slots['txNormal']])
        if m['alpha_tested']:
            g['alphaMode'] = 'MASK'
        elif m['blend'] == 1:
            g['alphaMode'] = 'BLEND'
        materials.append(g)
    nodes, meshes = [], []

    def node(n):
        i = len(nodes)
        g = dict(name=n['name'])
        nodes.append(g)
        x = dict(active=n['active'])
        if n['kind'] == 1:
            if n['matrix'] != IDENTITY:
                g['matrix'] = n['matrix']
        else:
            v, f = n['verts'], n['faces']
            x.update({key: n[key] for key in MESH_KEYS}, sphere=n['sphere'], sphere_of=_hash(v, f))
            if len(v) and len(f):
                v, f, copies = _split_repeats(v, f)
                if copies:
                    # undone on the way back while the mesh stays as it is here
                    x.update(copies=copies, copies_of=_hash(v, f))
                attrs = dict(POSITION=b.add(v[:, 0:3], FLOAT, 'VEC3', 34962, True),
                             NORMAL=b.add(v[:, 3:6], FLOAT, 'VEC3', 34962),
                             TEXCOORD_0=b.add(v[:, 6:8], FLOAT, 'VEC2', 34962),
                             TANGENT=b.add(np.c_[v[:, 8:11], np.ones(len(v), '<f4')].astype('<f4'), FLOAT, 'VEC4', 34962))
                idx = b.add(f.ravel().astype('<u2' if len(v) < 65536 else '<u4'), USHORT if len(v) < 65536 else UINT, 'SCALAR', 34963)
                meshes.append(dict(name=n['name'], primitives=[dict(attributes=attrs, indices=idx, material=n['material'], mode=4)]))
                g['mesh'] = len(meshes) - 1
            else:
                # glTF has no empty mesh: kept as a node, with what it would hold
                x.update(empty=True, material=n['material'], verts=len(v))
        g['extras'] = dict(kn5=x)
        kids = [node(c) for c in n['children']]
        if kids:
            g['children'] = kids
        return i

    root = node(k['root'])
    # used: by a mesh with a face on three corners, as Blender drops the rest
    used = {n['material'] for n, _ in kn5tree.walk(k['root']) if n['kind'] == 2 and len(n['faces'])
            and ((n['faces'][:, 0] != n['faces'][:, 1]) & (n['faces'][:, 1] != n['faces'][:, 2]) & (n['faces'][:, 0] != n['faces'][:, 2])).any()}
    top = dict(version=k['version'], extra=k['extra'],
               textures=[[t['name'], t['active'], images[i]['uri']] for i, t in enumerate(k['textures'])],
               # as Blender drops a material no mesh uses
               unused_materials=[[g['name'], g['extras']['kn5']] for i, g in enumerate(materials) if i not in used])
    # on the root node too: Blender drops the file's own extras, but keeps a node's
    nodes[root]['extras']['kn5']['file'] = top
    doc = dict(asset=dict(version='2.0', generator='techspeed-fueltech-velopark gltf_source'), scene=0, scenes=[dict(nodes=[root])],
               nodes=nodes, meshes=meshes, materials=materials, textures=textures, images=images,
               buffers=[dict(byteLength=b.size)], bufferViews=b.views, accessors=b.accessors, extras=dict(kn5=top))
    for key in ('meshes', 'materials', 'textures', 'images', 'bufferViews', 'accessors'):
        if not doc[key]:
            del doc[key]
    return doc, b''.join(b.parts)


def write_glb(path, doc, binary):
    js = json.dumps(doc, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    binary += b'\0' * (-len(binary) % 4)
    out = struct.pack('<III', GLB, 2, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack('<II', len(js), JSON_CHUNK) + js + struct.pack('<II', len(binary), BIN_CHUNK) + binary
    open(path, 'wb').write(out)


def read_glb(path):
    b = open(path, 'rb').read()
    magic, _, _ = struct.unpack_from('<III', b, 0)
    assert magic == GLB, '%s is not a glTF binary' % path
    p, doc, binary = 12, None, b''
    while p < len(b):
        n, kind = struct.unpack_from('<II', b, p)
        chunk = b[p + 8:p + 8 + n]
        if kind == JSON_CHUNK:
            doc = json.loads(chunk)
        elif kind == BIN_CHUNK:
            binary = chunk
        p += 8 + n
    return doc, binary


# ---- glb -> kn5 ------------------------------------------------------------------------
COMPONENTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
DTYPES = {FLOAT: '<f4', USHORT: '<u2', UINT: '<u4', UBYTE: 'u1', 5120: 'i1', 5122: '<i2'}


def _accessor(doc, binary, i):
    a = doc['accessors'][i]
    v = doc['bufferViews'][a['bufferView']]
    dt = np.dtype(DTYPES[a['componentType']])
    n = COMPONENTS[a['type']]
    start = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride', 0) or dt.itemsize * n
    if stride == dt.itemsize * n:
        out = np.frombuffer(binary, dt, count=a['count'] * n, offset=start).reshape(a['count'], n)
    else:
        out = np.stack([np.frombuffer(binary, dt, count=n, offset=start + j * stride) for j in range(a['count'])])
    if a.get('normalized') and dt.kind in 'ui':
        out = out / float(np.iinfo(dt).max)
    return out.astype('<f4') if dt.kind == 'f' or a.get('normalized') else out


def _matrix(g):
    """A glTF node's local matrix, as kn5 stores one: 16 floats, rows taking row vectors."""
    if 'matrix' in g:
        return [float(x) for x in g['matrix']]
    t = g.get('translation', [0, 0, 0])
    x, y, z, w = g.get('rotation', [0, 0, 0, 1])
    s = g.get('scale', [1, 1, 1])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]) * s
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return [float(v) for v in m.T.ravel()]


def tangents(pos, nor, uv, faces):
    """Per-vertex tangents from the uv layout, square to the normals."""
    tan = np.zeros_like(pos)
    p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    u0, u1, u2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, u1 - u0, u2 - u0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > 1e-12
    t = np.where(ok[:, None], (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / np.where(ok, det, 1)[:, None], e1)
    for c in range(3):
        np.add.at(tan, faces[:, c], t)
    tan -= nor * (tan * nor).sum(1, keepdims=True)
    ln = np.linalg.norm(tan, axis=1, keepdims=True)
    return np.where(ln > 1e-9, tan / np.maximum(ln, 1e-9), np.array([1., 0, 0])).astype('<f4')


def _hash(verts, faces):
    h = hashlib.sha1(np.ascontiguousarray(verts, '<f4').tobytes())
    h.update(np.ascontiguousarray(faces, '<u4').tobytes())
    return h.hexdigest()[:16]


def _split_repeats(verts, faces):
    """A face on the same three corners as one before it (the back of a two-sided face) given
    corners of its own, copies of those: Blender keeps one face of a set of corners only.
    Returns the corners, the faces and, for each corner added, the one it copies."""
    key = np.sort(faces, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    again = np.ones(len(faces), bool)
    again[first] = False
    # nor those on fewer than three corners, which Blender drops anyway
    again &= (key[:, 0] != key[:, 1]) & (key[:, 1] != key[:, 2])
    if not again.any():
        return verts, faces, []
    rows = np.nonzero(again)[0]
    copies = faces[rows].ravel()
    f = faces.astype(np.int64)
    f[rows] = len(verts) + np.arange(len(copies)).reshape(-1, 3)
    return np.vstack([verts, verts[copies]]), f, [int(c) for c in copies]


def _sphere(verts, faces, x, moved):
    """The kn5's own sphere while the mesh is as it was, else one measured round it."""
    if not moved and 'sphere' in x and x.get('sphere_of') == _hash(verts, faces):
        return [float(v) for v in x['sphere']]
    pos = verts[:, :3]
    if not len(pos):
        return [0., 0., 0., 0.]
    c = (pos.min(0) + pos.max(0)) / 2
    return [float(x) for x in c] + [float(np.linalg.norm(pos - c, axis=1).max())]


def _chunks(verts, faces):
    """A mesh split into pieces each within the 65536 corners a kn5 mesh can index."""
    if len(verts) < 65536:
        yield verts, faces.astype('<u2')
        return
    start = 0
    while start < len(faces):
        used, end = {}, start
        while end < len(faces) and len(used) + 3 <= 65535:
            for c in faces[end]:
                used.setdefault(int(c), len(used))
            end += 1
        keys = np.array(list(used), int)
        remap = np.zeros(len(verts), int)
        remap[keys] = np.arange(len(keys))
        yield verts[keys], remap[faces[start:end]].astype('<u2')
        start = end


def _kn5(g):
    """The kn5 data in a glTF object's extras. Blender writes back a custom property it cannot
    hold as such, a dict of lists say, as the Python text of it."""
    def fix(v):
        if isinstance(v, str) and v[:1] in '[{':
            try:
                return fix(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                return v
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v
    x = fix((g.get('extras') or {}).get('kn5'))
    return x if isinstance(x, dict) else {}


def _material(name, x):
    rest = x.get('rest', {})
    return dict(name=name, shader=x['shader'], blend=x['blend'], alpha_tested=x['alpha_tested'], depth_mode=x['depth_mode'],
                props=[dict(name=n, value=float(v), rest=bytes.fromhex(rest[n]) if n in rest else bytes(36)) for n, v in x['props']],
                textures=[dict(sampler=s, slot=i, name=t) for s, i, t in x['textures']])


def from_gltf(doc, binary, read_texture, find_texture=None):
    """A glTF document as a kn5 (kn5tree data). read_texture(uri) gives a texture file's bytes;
    find_texture(name), if given, those of a texture a material names that the document does
    not point at, as after Blender, which can leave the textures out."""
    top = _kn5(doc) or next((_kn5(n)['file'] for n in doc.get('nodes', []) if 'file' in _kn5(n)), {})
    images = doc.get('images', [])
    texs = []

    def image_data(img):
        if 'uri' in img:
            return read_texture(urllib.parse.unquote(img['uri']))
        v = doc['bufferViews'][img['bufferView']]
        return binary[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]

    def image_name(img):
        if img.get('name'):
            return img['name']
        return os.path.basename(urllib.parse.unquote(img.get('uri', ''))) or 'texture%d.png' % images.index(img)

    # the kn5's own textures in their order, then any the model gained
    names = []
    for name, active, uri in top.get('textures', []):
        texs.append(dict(name=name, active=active, data=read_texture(urllib.parse.unquote(uri))))
        names.append(name)
    for img in images:
        if image_name(img) not in names:
            texs.append(dict(name=image_name(img), active=1, data=image_data(img)))
            names.append(image_name(img))

    def tex_name(info):
        if not info:
            return None
        return image_name(images[doc['textures'][info['index']]['source']])

    mats = []
    for g in doc.get('materials', []):
        x = _kn5(g)
        name = BLENDER_COPY.sub('', g.get('name', 'material'))
        if x:
            mats.append(_material(name, x))
        else:
            diffuse = tex_name(g.get('pbrMetallicRoughness', {}).get('baseColorTexture'))
            normal = tex_name(g.get('normalTexture'))
            alpha = g.get('alphaMode') == 'MASK'
            shader = 'ksPerPixel' + ('AT' if alpha else '') + ('_NM' if normal else '')
            slots = [('txDiffuse', diffuse)] + ([('txNormal', normal)] if normal else [])
            mats.append(dict(name=name, shader=shader, blend=1 if g.get('alphaMode') == 'BLEND' else 0,
                             alpha_tested=1 if alpha else 0, depth_mode=0,
                             props=[dict(name=n, value=v, rest=bytes(36)) for n, v in DEFAULT_PROPS],
                             textures=[dict(sampler=s, slot=i, name=t) for i, (s, t) in enumerate(slots) if t]))
    for name, x in top.get('unused_materials', []):
        if name not in {m['name'] for m in mats}:
            mats.append(_material(name, x))
    if not mats:
        mats.append(dict(name='default', shader='ksPerPixel', blend=0, alpha_tested=0, depth_mode=0,
                         props=[dict(name=n, value=v, rest=bytes(36)) for n, v in DEFAULT_PROPS], textures=[]))
    # the textures the materials name, when the document lost its list of them, as after
    # Blender. Not otherwise: a kn5 may use textures only another model of the track holds
    for m in mats if not top.get('textures') else ():
        for t in m['textures']:
            if t['name'] not in names:
                if find_texture is None:
                    raise ValueError('material %s names texture %s, which the model lacks' % (m['name'], t['name']))
                texs.append(dict(name=t['name'], active=1, data=find_texture(t['name'])))
                names.append(t['name'])

    def mesh_nodes(g, name, x, local):
        """The kn5 meshes of a glTF node with a mesh, its local matrix baked in."""
        out = []
        m = np.array(local, float).reshape(4, 4)
        moved = local != IDENTITY
        for prim in doc['meshes'][g['mesh']]['primitives']:
            if prim.get('mode', 4) != 4:
                raise ValueError('%s: only triangles can go into a kn5' % name)
            a = prim['attributes']
            pos = _accessor(doc, binary, a['POSITION']).astype('<f4')
            faces = (_accessor(doc, binary, prim['indices']).reshape(-1, 3) if 'indices' in prim
                     else np.arange(len(pos)).reshape(-1, 3)).astype(np.int64)
            nor = _accessor(doc, binary, a['NORMAL']) if 'NORMAL' in a else None
            uv = _accessor(doc, binary, a['TEXCOORD_0']) if 'TEXCOORD_0' in a else np.zeros((len(pos), 2), '<f4')
            tan = _accessor(doc, binary, a['TANGENT'])[:, :3] if 'TANGENT' in a else None
            if moved:
                pos = (pos @ m[:3, :3] + m[3, :3]).astype('<f4')
                inv = np.linalg.inv(m[:3, :3]).T
                if nor is not None:
                    nor = nor @ inv
                    nor = (nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)).astype('<f4')
                if tan is not None:
                    tan = (tan @ m[:3, :3]).astype('<f4')
            if nor is None:
                fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
                nor = np.zeros_like(pos)
                for c in range(3):
                    np.add.at(nor, faces[:, c], fn)
                nor = (nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)).astype('<f4')
            if tan is None:
                tan = tangents(pos, nor, uv, faces)
            verts = np.hstack([pos, nor, uv, tan]).astype('<f4')
            copies = x.get('copies')
            if copies and not moved and len(doc['meshes'][g['mesh']]['primitives']) == 1 and x.get('copies_of') == _hash(verts, faces):
                # the corners added for the backs of two-sided faces, taken out again
                n0 = len(verts) - len(copies)
                back = np.r_[np.arange(n0), np.array(copies, np.int64)]
                verts, faces = verts[:n0], back[faces]
            for v, f in _chunks(verts, faces):
                n = dict(kind=2, name=name, active=x.get('active', 1), children=[], verts=v, faces=f,
                         material=prim.get('material', 0))
                for key, default in zip(MESH_KEYS, (1, 1, 0, 0, 0., 0., 1)):
                    n[key] = x.get(key, default)
                n['sphere'] = _sphere(v, f, x, moved)
                out.append(n)
        return out

    def node(i, parent_moves=False):
        g = doc['nodes'][i]
        name = BLENDER_COPY.sub('', g.get('name', 'node%d' % i))
        x = _kn5(g)
        local = _matrix(g)
        kids = []
        for c in g.get('children', []):
            kids += node(c)
        if 'mesh' in g:
            own = mesh_nodes(g, name, x, local)
            if kids:
                # a mesh with nodes under it: they stay where they were, under a dummy that moves them
                own.append(dict(kind=1, name=name, active=1, matrix=local, children=kids))
            return own
        if x.get('empty'):
            n = dict(kind=2, name=name, active=x.get('active', 1), children=kids, verts=np.zeros((0, 11), '<f4'),
                     faces=np.zeros((0, 3), '<u2'), material=x.get('material', 0), sphere=x.get('sphere', [0., 0., 0., 0.]))
            for key, default in zip(MESH_KEYS, (1, 1, 0, 0, 0., 0., 1)):
                n[key] = x.get(key, default)
            return [n]
        return [dict(kind=1, name=name, active=x.get('active', 1), matrix=local, children=kids)]

    roots = []
    for i in doc['scenes'][doc.get('scene', 0)]['nodes']:
        roots += node(i)
    if len(roots) == 1 and roots[0]['kind'] == 1:
        root = roots[0]
    else:
        root = dict(kind=1, name='root', active=1, matrix=list(IDENTITY), children=roots)
    return dict(version=top.get('version', 6), extra=top.get('extra', 0), textures=texs, materials=mats, root=root)


# ---- files -------------------------------------------------------------------------------
def load_kn5(glb_path):
    """A source model as kn5 data, its textures read from where it points."""
    doc, binary = read_glb(glb_path)
    base = os.path.dirname(os.path.abspath(glb_path))
    stem = os.path.splitext(os.path.basename(glb_path))[0]

    def find(name):
        # one this model holds a version of its own of first, as the extraction left them
        for p in (os.path.join(base, '..', 'textures', stem, name), os.path.join(base, '..', 'textures', name)):
            if os.path.exists(p):
                return open(p, 'rb').read()
        raise FileNotFoundError('texture %s of %s is not in source/textures' % (name, stem))
    return from_gltf(doc, binary, lambda uri: open(os.path.join(base, uri), 'rb').read(), find)


def texture_files(glb_path):
    """The texture files a source model points at."""
    doc, _ = read_glb(glb_path)
    base = os.path.dirname(os.path.abspath(glb_path))
    stem = os.path.splitext(os.path.basename(glb_path))[0]
    top = _kn5(doc) or next((_kn5(n)['file'] for n in doc.get('nodes', []) if 'file' in _kn5(n)), {})
    files = {os.path.normpath(os.path.join(base, urllib.parse.unquote(u))) for _, _, u in top.get('textures', [])}
    files |= {os.path.normpath(os.path.join(base, urllib.parse.unquote(i['uri']))) for i in doc.get('images', []) if 'uri' in i}
    held = {os.path.basename(f) for f in files}
    # those the materials name, only when the list of them is lost, as from_gltf
    for g in doc.get('materials', []) if not top.get('textures') else ():
        for _, _, name in _kn5(g).get('textures', []):
            if name not in held:
                held.add(name)
                own = os.path.normpath(os.path.join(base, '..', 'textures', stem, name))
                files.add(own if os.path.exists(own) else os.path.normpath(os.path.join(base, '..', 'textures', name)))
    return sorted(files)
