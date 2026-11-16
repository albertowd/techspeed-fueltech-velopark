"""The track's material library and its textures, as TRACK_FORMAT.md has them: source/materials.json, and the
textures in source/textures as PNG.

    python tools/materials.py convert     (the models' materials into the library; the DDS textures into PNG)

Each material by its name: what every game reads (its textures, how see-through it is, the ground it is to
drive on) and, under targets.ac, Assetto Corsa's own (its shader, flags, the shader's values and its texture
slots). The models (source/models) keep only each material's name: tools/gltf_source.py reads the rest from here
(LIBRARY, set by tools/build_paths.py), and writes here what a tool changed.

A texture is named in the models as Assetto Corsa names it (X.dds) and kept as PNG (X.png, beside where the DDS
was): read as PNG by every tool, made DDS only by Assetto Corsa's build (ac_dds), in the format it had (the
library's textures: DXT1, DXT5, or as it was, uncompressed), a normal map's green turned back down as AC reads it
(kept up, as glTF has it).
"""
import hashlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, 'source', 'materials.json')
SURFACES_FILE = os.path.join(ROOT, 'source', 'surfaces.json')
TEXTURES = os.path.join(ROOT, 'source', 'textures')
CACHE = os.path.join(ROOT, 'build', 'cache', 'dds')
VERSION = 1
# (the ground a mesh is driven as, by AC's name for it: the first that fits)
SURFACES = (('1ROAD_KERB', 'rumble'), ('1KERB', 'kerb'), ('1ROAD', 'asphalt'), ('1GRASS', 'grass'), ('1WALL', 'wall'), ('1SAND', 'sand'))

_LIB = []
_DIRTY = {'materials': set(), 'textures': set()}   # (what this run changed: only that is written back, over the file as it is)


def library():
    """The library, read once: {'version', 'materials': {name: entry}, 'textures': {png: {'ac': format, 'normal'}}}."""
    if not _LIB:
        lib = json.load(open(LIBRARY, encoding='utf-8')) if os.path.exists(LIBRARY) else {}
        lib.setdefault('version', VERSION)
        lib.setdefault('materials', {})
        lib.setdefault('textures', {})
        _LIB.append(lib)
    return _LIB[0]


def save():
    """What this run changed written over the library as it is now (another may have written it since)."""
    if _LIB:
        lib = _LIB[0]
        disk = json.load(open(LIBRARY, encoding='utf-8')) if os.path.exists(LIBRARY) else {}
        for part in ('materials', 'textures'):
            merged = dict(disk.get(part, {}))
            for name in _DIRTY[part]:
                if name in lib[part]:
                    merged[name] = lib[part][name]
            lib[part] = {**merged, **{k: v for k, v in lib[part].items() if k not in merged}} if not disk else merged
        out = dict(version=lib['version'], materials=dict(sorted(lib['materials'].items())), textures=dict(sorted(lib['textures'].items())))
        text = json.dumps(out, indent=1, ensure_ascii=False)
        if not os.path.exists(LIBRARY) or open(LIBRARY, encoding='utf-8').read() != text:
            open(LIBRARY, 'w', encoding='utf-8').write(text)


def png_of(name):
    """A texture's file name as kept (X.dds: X.png)."""
    stem, ext = os.path.splitext(name)
    return stem + '.png' if ext.lower() == '.dds' else name


def entry_of(name, x):
    """A material's library entry from Assetto Corsa's (x: as gltf_source keeps it: shader, blend, alpha_tested,
    depth_mode, props [[name, value]], textures [[sampler, slot, texture]], rest)."""
    slots = {s: t for s, _, t in x.get('textures', [])}
    e = {}
    if slots.get('txDiffuse'):
        e['colour'] = png_of(slots['txDiffuse'])
    if slots.get('txNormal'):
        e['normal'] = png_of(slots['txNormal'])
    e['alpha'] = 'mask' if x.get('alpha_tested') else 'blend' if x.get('blend') == 1 else 'opaque'
    e['targets'] = {'ac': {k: x[k] for k in ('shader', 'blend', 'alpha_tested', 'depth_mode', 'props', 'textures', 'rest') if k in x}}
    return e


class Hook:
    """What tools/gltf_source.py asks of the library: a material's Assetto Corsa data by its name, and a model's
    materials to keep (those its drawn meshes use, so a marker's left-over material does not stand for one)."""

    def get(self, name):
        m = library()['materials'].get(name)
        return m['targets']['ac'] if m and 'ac' in m.get('targets', {}) else None

    def put(self, name, x, drawn=True):
        mats = library()['materials']
        old = mats.get(name)
        if old is not None and not drawn:
            return
        new = entry_of(name, x)
        for k in ('metres', 'surface'):                   # (what the library holds that the model does not)
            if old and k in old:
                new[k] = old[k]
        if old != new:
            mats[name] = new
            _DIRTY['materials'].add(name)
            save()


def read_texture(path):
    """A texture's bytes by its path as AC names it (X.dds): its PNG (X.png) where it is kept so."""
    if not os.path.exists(path) and path.lower().endswith('.dds') and os.path.exists(path[:-4] + '.png'):
        path = path[:-4] + '.png'
    return open(path, 'rb').read()


def texture_format(data):
    """A DDS's format: 'DXT1', 'DXT5', 'DXT3' or 'RGBA' (uncompressed)."""
    if data[:4] != b'DDS ':
        return None
    four = data[84:88]
    return four.decode('ascii', 'replace') if four.strip(b'\x00') else 'RGBA'


def ac_dds(name, data):
    """A texture's bytes as Assetto Corsa's build needs them: a PNG made DDS (its format as the library has it,
    DXT1 unless it says; a normal map's green turned down), kept in build/cache/dds; anything else as it is."""
    if data[:8] != b'\x89PNG\r\n\x1a\n' or os.path.splitext(name)[1].lower() != '.dds':
        return data
    t = library()['textures'].get(png_of(os.path.basename(name)), {})
    fmt, normal = t.get('ac', 'DXT1'), t.get('normal', False)
    key = hashlib.sha1(data + fmt.encode() + bytes([normal])).hexdigest()
    path = os.path.join(CACHE, key + '.dds')
    if os.path.exists(path):
        return open(path, 'rb').read()
    import numpy as np
    from PIL import Image
    import kn5io
    img = Image.open(io.BytesIO(data))
    if normal:
        a = np.array(img.convert('RGBA'))
        a[..., 1] = 255 - a[..., 1]
        img = Image.fromarray(a, 'RGBA')
    if fmt in ('DXT1', 'DXT5', 'DXT3'):
        sq = all(v & (v - 1) == 0 for v in img.size)
        out = kn5io.dds(img, fmt) if sq else _dds_plain(img, fmt)
    else:
        out = _dds_plain(img, 'RGBA')
    os.makedirs(CACHE, exist_ok=True)
    open(path, 'wb').write(out)
    return out


def _dds_plain(img, fmt):
    """A DDS of one level (sides not powers of two, or uncompressed)."""
    buf = io.BytesIO()
    if fmt == 'RGBA':
        img.convert('RGBA').save(buf, format='DDS')
    else:
        img.convert('RGBA' if fmt != 'DXT1' else 'RGB').save(buf, format='DDS', pixel_format=fmt)
    return buf.getvalue()


def save_texture(name, img, fmt='DXT1', normal=False, folder=TEXTURES):
    """A texture a tool made (img, PIL; a normal map's green down, as AC's are made): kept as PNG (its green up),
    its format for AC in the library. Its bytes, as the models hold it."""
    import numpy as np
    from PIL import Image
    if normal:
        a = np.array(img.convert('RGBA'))
        a[..., 1] = 255 - a[..., 1]
        img = Image.fromarray(a, 'RGBA')
    png = png_of(name)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()
    path = os.path.join(folder, png)
    if not os.path.exists(path) or open(path, 'rb').read() != data:
        open(path, 'wb').write(data)
    tex = library()['textures']
    if tex.get(png) != {'ac': fmt, 'normal': normal}:
        tex[png] = {'ac': fmt, 'normal': normal}
        _DIRTY['textures'].add(png)
        save()
    return data


def convert():
    """The models' materials into the library (with the ground each is driven as, where its meshes are), and every
    DDS texture of source/textures into PNG, its format kept in the library; the DDS out."""
    import glob
    import numpy as np
    from PIL import Image
    import build_paths as bp
    import gltf_source
    lib = library()
    normals, uses = set(), {}
    for f in sorted(glob.glob(os.path.join(bp.MODELS, '*.glb'))):
        doc, _ = gltf_source.read_glb(f)
        drawn = set()
        for n in doc.get('nodes', []):
            if 'mesh' in n and not n.get('name', '').startswith('AC_'):
                for p in doc['meshes'][n['mesh']]['primitives']:
                    if 'material' in p:
                        drawn.add(p['material'])
                        key = next((v for k, v in SURFACES if n.get('name', '').startswith(k)), None)
                        if key:
                            uses.setdefault(doc['materials'][p['material']]['name'], []).append(key)
        for i, g in enumerate(doc.get('materials', [])):
            x = (g.get('extras') or {}).get('kn5')
            if not x:
                continue
            for s, _, t in x.get('textures', []):
                if s == 'txNormal':
                    normals.add(png_of(t))
            if g['name'] not in lib['materials'] or i in drawn:
                Hook().put(g['name'], x, drawn=i in drawn)
    for name, keys in uses.items():
        if name in lib['materials']:
            lib['materials'][name]['surface'] = max(set(keys), key=keys.count)
    if not os.path.exists(SURFACES_FILE):
        json.dump(dict(version=VERSION, surfaces={v: dict(targets=dict(ac=k.lstrip('1').split('_')[0])) for k, v in SURFACES}),
                  open(SURFACES_FILE, 'w', encoding='utf-8'), indent=1)
    scans = os.path.join(ROOT, 'tools', 'reference', 'materials', 'state')
    for f in glob.glob(os.path.join(scans, '*.json')):
        s = json.load(open(f))
        base = os.path.splitext(os.path.basename(f))[0]
        for name, e in lib['materials'].items():
            if e.get('colour') == base + '.png' and s.get('metres'):
                e['metres'] = float(s['metres'])
    converted = 0
    for dds in sorted(glob.glob(os.path.join(TEXTURES, '**', '*.dds'), recursive=True)):
        data = open(dds, 'rb').read()
        fmt = texture_format(data) or 'RGBA'
        img = Image.open(io.BytesIO(data))
        img.load()
        png = png_of(os.path.basename(dds))
        normal = png in normals
        if normal:
            a = np.array(img.convert('RGBA'))
            a[..., 1] = 255 - a[..., 1]
            img = Image.fromarray(a, 'RGBA')
        elif img.mode == 'RGBA' and np.asarray(img)[..., 3].min() == 255 and fmt == 'DXT1':
            img = img.convert('RGB')
        img.save(os.path.join(os.path.dirname(dds), png), format='PNG')
        lib['textures'][png] = {'ac': fmt, 'normal': normal}
        os.remove(dds)
        converted += 1
    _DIRTY['materials'] |= set(lib['materials'])
    _DIRTY['textures'] |= set(lib['textures'])
    save()
    return len(lib['materials']), converted


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1:] == ['convert']:
        print('%d materials in the library, %d textures made PNG' % convert())
    else:
        print(__doc__)
