"""The whole of a kn5 model (version 5 and 6) as plain data, and back to the same bytes.

    k = kn5tree.load(path)        # or kn5tree.parse(bytes)
    kn5tree.save(k, path)         # or kn5tree.dump(k) -> bytes

The data:
    version, extra           the header's version and, past version 5, the int after it
    textures                 [{name, active, data}], data the file's bytes (DDS, PNG or JPEG)
    materials                [{name, shader, blend, alpha_tested, depth_mode, props, textures}]
        props                [{name, value, rest}], rest the 36 bytes after the value
        textures             [{sampler, slot, name}]
    root                     the node tree

Each node is {kind, name, active, children} and then, for a dummy (kind 1), matrix: 16 floats
as stored, rows of a matrix that takes row vectors, v' = v @ m; for a mesh (kind 2):
    cast_shadows, visible, transparent    the three flags
    verts                    (n, 11) float32: position, normal, uv, tangent, in its parent's space
    faces                    (k, 3) uint16 corners
    material, layer, lod_in, lod_out, sphere (4 floats: middle and radius), renderable
"""
import struct

import numpy as np

MAGIC = b'sc6969'


class _R:
    def __init__(s, b):
        s.b, s.p = b, 0

    def take(s, fmt):
        v = struct.unpack_from(fmt, s.b, s.p)
        s.p += struct.calcsize(fmt)
        return v if len(v) > 1 else v[0]

    def raw(s, n):
        s.p += n
        return bytes(s.b[s.p - n:s.p])

    def st(s):
        return s.raw(s.take('<i')).decode('latin1')


def parse(b):
    r = _R(b)
    assert r.raw(6) == MAGIC, 'not a kn5'
    k = dict(version=r.take('<i'))
    k['extra'] = r.take('<i') if k['version'] > 5 else None
    k['textures'] = []
    for _ in range(r.take('<i')):
        active = r.take('<i')
        name = r.st()
        k['textures'].append(dict(name=name, active=active, data=r.raw(r.take('<i'))))
    k['materials'] = []
    for _ in range(r.take('<i')):
        m = dict(name=r.st(), shader=r.st())
        m['blend'], m['alpha_tested'] = r.take('<BB')
        m['depth_mode'] = r.take('<i') if k['version'] > 4 else None
        m['props'] = []
        for _ in range(r.take('<i')):
            name = r.st()
            m['props'].append(dict(name=name, value=r.take('<f'), rest=r.raw(36)))
        m['textures'] = []
        for _ in range(r.take('<i')):
            sampler = r.st()
            m['textures'].append(dict(sampler=sampler, slot=r.take('<i'), name=r.st()))
        k['materials'].append(m)
    k['root'] = _node(r)
    assert r.p == len(b), 'kn5 has %d bytes past its node tree' % (len(b) - r.p)
    return k


def _node(r):
    n = dict(kind=r.take('<i'), name=r.st())
    kids = r.take('<i')
    n['active'] = r.take('<B')
    if n['kind'] == 1:
        n['matrix'] = list(r.take('<16f'))
    elif n['kind'] == 2:
        n['cast_shadows'], n['visible'], n['transparent'] = r.take('<BBB')
        vc = r.take('<i')
        n['verts'] = np.frombuffer(r.raw(vc * 44), '<f4').reshape(-1, 11).copy()
        ic = r.take('<i')
        n['faces'] = np.frombuffer(r.raw(ic * 2), '<u2').reshape(-1, 3).copy()
        n['material'], n['layer'], n['lod_in'], n['lod_out'] = r.take('<iiff')
        n['sphere'] = list(r.take('<4f'))
        n['renderable'] = r.take('<B')
    else:
        raise ValueError('kn5 node %r is of kind %d, which is not read' % (n['name'], n['kind']))
    n['children'] = [_node(r) for _ in range(kids)]
    return n


def load(path):
    return parse(open(path, 'rb').read())


def _s(t):
    b = t.encode('latin1')
    return struct.pack('<i', len(b)) + b


def dump(k):
    out = [MAGIC, struct.pack('<i', k['version'])]
    if k['version'] > 5:
        out.append(struct.pack('<i', k['extra'] or 0))
    out.append(struct.pack('<i', len(k['textures'])))
    for t in k['textures']:
        out += [struct.pack('<i', t['active']), _s(t['name']), struct.pack('<i', len(t['data'])), t['data']]
    out.append(struct.pack('<i', len(k['materials'])))
    for m in k['materials']:
        out += [_s(m['name']), _s(m['shader']), struct.pack('<BB', m['blend'], m['alpha_tested'])]
        if k['version'] > 4:
            out.append(struct.pack('<i', m['depth_mode'] or 0))
        out.append(struct.pack('<i', len(m['props'])))
        for p in m['props']:
            out += [_s(p['name']), struct.pack('<f', p['value']), p['rest']]
        out.append(struct.pack('<i', len(m['textures'])))
        for t in m['textures']:
            out += [_s(t['sampler']), struct.pack('<i', t['slot']), _s(t['name'])]
    _dump_node(k['root'], out)
    return b''.join(out)


def _dump_node(n, out):
    out += [struct.pack('<i', n['kind']), _s(n['name']), struct.pack('<iB', len(n['children']), n['active'])]
    if n['kind'] == 1:
        out.append(struct.pack('<16f', *n['matrix']))
    else:
        v = np.ascontiguousarray(n['verts'], '<f4')
        f = np.ascontiguousarray(n['faces'], '<u2')
        assert v.ndim == 2 and v.shape[1] == 11 and len(v) < 65536, n['name']
        out += [struct.pack('<BBB', n['cast_shadows'], n['visible'], n['transparent']),
                struct.pack('<i', len(v)), v.tobytes(), struct.pack('<i', f.size), f.tobytes(),
                struct.pack('<iiff', n['material'], n['layer'], n['lod_in'], n['lod_out']),
                struct.pack('<4f', *n['sphere']), struct.pack('<B', n['renderable'])]
    for c in n['children']:
        _dump_node(c, out)


def save(k, path):
    data = dump(k)
    open(path, 'wb').write(data)
    return data


def walk(n, parent=None):
    """Every node with the matrix taking its vertices to the model's space (row vectors)."""
    m = np.eye(4) if parent is None else parent
    if n['kind'] == 1:
        m = np.array(n['matrix'], float).reshape(4, 4) @ m
    yield n, m
    for c in n['children']:
        yield from walk(c, m)
