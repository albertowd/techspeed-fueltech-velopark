"""Read textures, materials and meshes of a kn5 file (read only)."""
import struct, sys, os, re

class R:
    def __init__(s, b): s.b, s.p = b, 0
    def i(s):
        v = struct.unpack_from('<i', s.b, s.p)[0]; s.p += 4; return v
    def f(s):
        v = struct.unpack_from('<f', s.b, s.p)[0]; s.p += 4; return v
    def by(s):
        v = s.b[s.p]; s.p += 1; return v
    def st(s):
        n = s.i(); v = s.b[s.p:s.p+n].decode('latin1'); s.p += n; return v
    def fs(s, n):
        v = struct.unpack_from('<%df' % n, s.b, s.p); s.p += 4*n; return v

def mul(a, b):
    return tuple(sum(a[r*4+k]*b[k*4+c] for k in range(4)) for r in range(4) for c in range(4))

IDENT = (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)

def read(path):
    r = R(open(path, 'rb').read())
    assert r.b[:6] == b'sc6969', path
    r.p = 6
    ver = r.i()
    if ver > 5: r.i()
    tex = []
    for _ in range(r.i()):
        r.i(); name = r.st(); size = r.i()
        tex.append((name, size, r.p)); r.p += size
    mats = []
    for _ in range(r.i()):
        name = r.st(); shader = r.st(); r.p += 2
        if ver > 4: r.p += 4
        props = {}
        for _ in range(r.i()):
            pn = r.st(); pv = r.f(); r.p += 36; props[pn] = pv
        samp = {}
        for _ in range(r.i()):
            sn = r.st(); r.i(); samp[sn] = r.st()
        mats.append(dict(name=name, shader=shader, props=props, tex=samp))
    nodes = []
    def node(pm, path):
        t = r.i(); name = r.st(); kids = r.i(); r.by()
        m = pm
        if t == 1:
            m = mul(r.fs(16), pm)
        elif t == 2:
            r.p += 3
            vc = r.i(); start = r.p; r.p += vc*44
            ic = r.i(); ip = r.p; r.p += ic*2
            mid = r.i(); r.i(); r.f(); r.f(); r.p += 16 + 1
            nodes.append(dict(name=name, path=path, vc=vc, ic=ic, mat=mid, vp=start, ip=ip, m=pm))
        else:
            raise ValueError('node type %d' % t)
        for _ in range(kids): node(m, path + '/' + name)
    node(IDENT, '')
    return r.b, ver, tex, mats, nodes

def bbox(b, n):
    m = n['m']; lo = [1e9]*3; hi = [-1e9]*3
    for i in range(n['vc']):
        x, y, z = struct.unpack_from('<3f', b, n['vp'] + i*44)
        p = (x*m[0]+y*m[4]+z*m[8]+m[12], x*m[1]+y*m[5]+z*m[9]+m[13], x*m[2]+y*m[6]+z*m[10]+m[14])
        for k in range(3):
            lo[k] = min(lo[k], p[k]); hi[k] = max(hi[k], p[k])
    return lo, hi

if __name__ == '__main__':
    d = sys.argv[1]
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.kn5') or re.match(r'vp_.*(boards)', fn): continue
        try:
            b, ver, tex, mats, nodes = read(os.path.join(d, fn))
        except Exception as e:
            print(fn, 'ERR', repr(e)); continue
        print('==', fn, 'v', ver, 'tex', len(tex), 'mats', len(mats), 'meshes', len(nodes))
        print('  tex :', ', '.join('%s(%dk)' % (t[0], t[1]//1024) for t in tex)[:900])
        print('  mats:', ', '.join(m['name'] for m in mats)[:900])
        names = {}
        for n in nodes:
            k = re.sub(r'[\d_.]+$', '', n['name'])
            a = names.setdefault(k, [0, 0, set()]); a[0] += 1; a[1] += n['ic']//3; a[2].add(mats[n['mat']]['name'])
        for k, a in sorted(names.items(), key=lambda x: -x[1][1])[:24]:
            print('   %-30s n=%-5d tris=%-8d %s' % (k, a[0], a[1], sorted(a[2])[:4]))
