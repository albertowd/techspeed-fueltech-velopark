"""Read and write Assetto Corsa kn5 models (version 6), keeping raw blocks for copying."""
import io, struct

import numpy as np
from PIL import Image


class Reader:
    def __init__(s, b):
        s.b, s.p = b, 0

    def i(s):
        s.p += 4
        return struct.unpack_from('<i', s.b, s.p - 4)[0]

    def f(s):
        s.p += 4
        return struct.unpack_from('<f', s.b, s.p - 4)[0]

    def st(s):
        n = s.i()
        s.p += n
        return s.b[s.p - n:s.p].decode('latin1')


class Model:
    """Textures and materials as raw byte blocks, meshes as arrays with their byte offsets."""

    def __init__(s, path):
        """From a file, or from a kn5's bytes."""
        b = bytes(path) if isinstance(path, (bytes, bytearray)) else open(path, 'rb').read()
        s.bytes = b
        r = Reader(b)
        assert b[:6] == b'sc6969'
        r.p = 6
        s.version = r.i()
        if s.version > 5:
            r.i()
        s.textures = {}
        for _ in range(r.i()):
            start = r.p
            r.i()
            name = r.st()
            size = r.i()
            r.p += size
            s.textures[name] = b[start:r.p]
        s.materials = []
        for _ in range(r.i()):
            start = r.p
            name = r.st()
            shader = r.st()
            blend, alpha_tested = b[r.p], b[r.p + 1]
            r.p += 2 + (4 if s.version > 4 else 0)
            for _ in range(r.i()):
                r.st()
                r.p += 40
            tex = {}
            for _ in range(r.i()):
                sampler = r.st()
                r.i()
                tex[sampler] = r.st()
            s.materials.append(dict(name=name, shader=shader, tex=tex, raw=b[start:r.p], blend=blend, alpha_tested=bool(alpha_tested)))
        s.meshes = []
        s.dummies = {}
        s.dummy_at = {}         # where each dummy's own matrix is in the file, and its parent's
        s._node(r, np.eye(4))

    def _node(s, r, parent):
        kind = r.i()
        name = r.st()
        kids = r.i()
        r.p += 1
        m = parent
        if kind == 1:
            s.dummy_at[name] = (r.p, parent)
            local = np.array(struct.unpack_from('<16f', s.bytes, r.p)).reshape(4, 4)
            r.p += 64
            m = local @ parent
            s.dummies[name] = m
        elif kind == 2:
            visible = bool(s.bytes[r.p + 1])
            r.p += 3
            vc = r.i()
            vp = r.p
            r.p += vc * 44
            ic = r.i()
            ip = r.p
            r.p += ic * 2
            mat = r.i()
            # the distances it is drawn from and to (0 to 0: always)
            lod_in, lod_out = struct.unpack_from('<ff', s.bytes, r.p + 4)
            renderable = bool(s.bytes[r.p + 28])
            r.p += 12 + 16 + 1
            d = np.frombuffer(s.bytes, dtype='<f4', count=vc * 11, offset=vp).reshape(-1, 11).copy()
            # rows of the kn5 matrix: v' = v @ m
            d[:, :3] = d[:, :3] @ m[:3, :3] + m[3, :3]
            idx = np.frombuffer(s.bytes, dtype='<u2', count=ic, offset=ip).reshape(-1, 3).copy()
            s.meshes.append(dict(name=name, mat=mat, v=d, f=idx, vp=vp, ip=ip, ic=ic, visible=visible, renderable=renderable,
                                 lod_in=lod_in, lod_out=lod_out))
        else:
            raise ValueError('node type %d' % kind)
        for _ in range(kids):
            s._node(r, m)

    def mesh(s, name):
        return next(x for x in s.meshes if x['name'] == name)

    def material_name(s, mesh):
        return s.materials[mesh['mat']]['name']


def blank_faces(data, mesh, faces=None):
    """Collapse triangles to nothing, in place, in a bytearray of the whole kn5."""
    sel = np.arange(len(mesh['f'])) if faces is None else np.asarray(faces)
    for k in sel:
        a = int(mesh['f'][k, 0])
        struct.pack_into('<3H', data, mesh['ip'] + k * 6, a, a, a)


# ---- Writing -----------------------------------------------------------------
def _s(t):
    b = t.encode('latin1')
    return struct.pack('<i', len(b)) + b


def texture_block(name, payload):
    return struct.pack('<i', 1) + _s(name) + struct.pack('<i', len(payload)) + payload


def material_block(name, shader, props, textures, alpha_tested=False):
    out = _s(name) + _s(shader) + struct.pack('<BBi', 0, 1 if alpha_tested else 0, 0)
    out += struct.pack('<i', len(props))
    for k, v in props.items():
        out += _s(k) + struct.pack('<f', v) + b'\0' * 36
    out += struct.pack('<i', len(textures))
    for slot, (sampler, tex) in enumerate(textures.items()):
        out += _s(sampler) + struct.pack('<i', slot) + _s(tex)
    return out


def dds(img, fmt='DXT1'):
    """DDS with a mip chain down to 4 pixels. Sides must be powers of two, so every
    level is exactly half the one above, as the game expects."""
    assert all(x & (x - 1) == 0 for x in img.size), img.size
    img = img.convert('RGBA' if fmt != 'DXT1' else 'RGB')
    levels = []
    w, h = img.size
    cur = img
    while True:
        buf = io.BytesIO()
        cur.save(buf, format='DDS', pixel_format=fmt)
        levels.append(buf.getvalue())
        if min(w, h) <= 4:
            break
        w, h = w // 2, h // 2
        cur = cur.resize((w, h), Image.LANCZOS)
    head = bytearray(levels[0][:128])
    flags = struct.unpack_from('<I', head, 8)[0] | 0x20000
    struct.pack_into('<I', head, 8, flags)
    struct.pack_into('<I', head, 28, len(levels))
    caps = struct.unpack_from('<I', head, 108)[0] | 0x8 | 0x400000
    struct.pack_into('<I', head, 108, caps)
    return bytes(head) + b''.join(l[128:] for l in levels)


def smooth_normals(v, faces, angle):
    """Per-corner normals, smoothed across faces closer than angle degrees.
    v: (n, 3) positions, faces: list of index lists (tris or quads). Returns tris and corner data."""
    tris = []
    for q in faces:
        for k in range(1, len(q) - 1):
            tris.append((q[0], q[k], q[k + 1]))
    tris = np.array(tris)
    p = v[tris]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-10
    tris, fn, area = tris[keep], fn[keep], area[keep]
    unit = fn / area[:, None]
    cos = np.cos(np.radians(angle))
    key = np.round(v, 4)
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.ravel()
    corner_g = group[tris]
    members = {}
    for t in range(len(tris)):
        for c in range(3):
            members.setdefault(corner_g[t, c], []).append(t)
    normals = np.zeros((len(tris), 3, 3))
    for t in range(len(tris)):
        for c in range(3):
            ts = members[corner_g[t, c]]
            near = [u for u in ts if unit[u] @ unit[t] > cos]
            n = fn[near].sum(0)
            normals[t, c] = n / np.linalg.norm(n)
    return tris, normals


def mesh_block(name, pos, normals, uvs, material, cast_shadows=False, lod_out=0.0, renderable=True):
    """pos, normals: (k, 3) corners, uvs: (k, 2). Corners come in threes. Welds equal corners."""
    tan = np.zeros_like(pos)
    for t in range(0, len(pos), 3):
        p0, p1, p2 = pos[t:t + 3]
        u0, u1, u2 = uvs[t:t + 3]
        e1, e2 = p1 - p0, p2 - p0
        d1, d2 = u1 - u0, u2 - u0
        det = d1[0] * d2[1] - d2[0] * d1[1]
        tv = (e1 * d2[1] - e2 * d1[1]) / det if abs(det) > 1e-12 else e1
        tan[t:t + 3] = tv
    tan -= normals * (tan * normals).sum(1, keepdims=True)
    ln = np.linalg.norm(tan, axis=1, keepdims=True)
    tan = np.where(ln > 1e-9, tan / np.maximum(ln, 1e-9), np.array([1., 0, 0]))
    rec = np.hstack([pos, normals, uvs, tan]).astype('<f4')
    key = np.round(rec[:, :8], 5)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = rec[first]
    idx = inv.ravel().astype('<u2')
    assert len(verts) < 65536, name
    centre = (pos.min(0) + pos.max(0)) / 2
    radius = float(np.linalg.norm(pos - centre, axis=1).max())
    out = struct.pack('<i', 2) + _s(name) + struct.pack('<iB', 0, 1)
    out += struct.pack('<BBB', 1 if cast_shadows else 0, 1, 0)
    out += struct.pack('<i', len(verts)) + verts.tobytes()
    out += struct.pack('<i', len(idx)) + idx.tobytes()
    out += struct.pack('<iiff', material, 0, 0.0, lod_out)
    out += struct.pack('<4f', *centre, radius) + struct.pack('<B', 1 if renderable else 0)
    return out, len(idx) // 3


def write(path, textures, materials, meshes, root='root'):
    """textures: raw texture blocks; materials: raw material blocks; meshes: raw mesh blocks."""
    out = b'sc6969' + struct.pack('<ii', 6, 0)
    out += struct.pack('<i', len(textures)) + b''.join(textures)
    out += struct.pack('<i', len(materials)) + b''.join(materials)
    out += struct.pack('<i', 1) + _s(root) + struct.pack('<iB', len(meshes), 1)
    out += np.eye(4, dtype='<f4').tobytes()
    out += b''.join(meshes)
    open(path, 'wb').write(out)
