"""The track's surfaces in scanned materials: the asphalt, the concrete pavers and the light poles' steel.

    python tools/scan_materials.py [--dry-run]

Each surface in PLAN takes a material scanned by Poly Haven (polyhaven.com, CC0: free to use and
pass on, in any game): its colour, normal, roughness and ambient occlusion maps, fetched once
into tools/reference/materials/<scan>/, kept out of the repository and fetched again when missing.

For Assetto Corsa each becomes three textures in source/textures, scan_<material>*.dds:
- the colour, darkened where the ambient occlusion is, and, for the asphalt, tinted to the average
  colour of the texture it replaces, so the track keeps its look and the Custom Shaders Patch's
  night settings for it still fit;
- the normal map, as Assetto Corsa reads them, DirectX's (green down), as Kunos' tracks' are;
- txMaps, how each point shines, from its roughness: red the specular, green the gloss, blue the
  reflection.
The material takes ksPerPixelMultiMap, which reads all three, and keeps its name, so the Custom
Shaders Patch config (rain, grass, night) still finds it. Each of its meshes, in every model of
source/models, gets its uv scaled so the texture repeats at the scan's real size, all of them by
the same factor so they meet as they did. Run again, it finds the meshes at that size and the
textures made and changes nothing; a mesh some tool made again at the old size is scaled again.

tools/export_beamng.py carries the same textures to BeamNG: the normal map with its green turned
up (BeamNG reads OpenGL's), and the gloss as BeamNG's roughness map.
"""
import glob, json, os, sys, urllib.request

import numpy as np
from PIL import Image

import build_paths as bp
import gltf_source
import kn5tree
from map_roads import scale

ROOT = bp.ROOT
MODELS = os.path.join(ROOT, 'source', 'models')
TEXTURES = os.path.join(ROOT, 'source', 'textures')
CACHE = os.path.join(ROOT, 'tools', 'reference', 'materials')
SIZE = 2048             # texture side, pixels
OFF = .3                # how far a mesh's metres per texture may be off the scan's, as a fraction
AO = .5                 # how much of the ambient occlusion darkens the colour
JOINTS = .15            # the share of a paving's texture its joints take

# material: (scan, tint to the average of the texture it had, ksSpecular, ksSpecularEXP, fresnelMaxLevel,
# options: colour, the one colour of all a paving's blocks; clean, straight lines and bands across
# the scan taken out; res, the scan's maps fetched (2k, 4k); size and nm_size, the sides of the
# colour and of the normal and shine textures)
# The asphalt is aerial_asphalt_01, a 30 m scan, the user's pick of twelve tried on 2026-11-12:
# grey, coarse enough to read from a kart, worn, with curved tyre marks, and hardly repeating; its
# one crack straight across and its bands taken out, as they crossed the road every 30 m. (Before:
# asphalt_02, 3 m, whose cracks repeated too often; finer scans blurred to flat grey from a kart.)
SHADE = .8              # the asphalt's colour, of the old texture's average: the Velopark's is darker
PLAN = {
    'Tarmac Dark': ('aerial_asphalt_01', True, .4, 25., .08, dict(clean=True, res='4k', size=4096, nm_size=2048, shade=SHADE)),
    'Tarmac Light': ('aerial_asphalt_01', True, .4, 25., .08, dict(clean=True, res='4k', size=4096, nm_size=2048, shade=SHADE)),
    # the real pavers are all light grey, where the scan mixes light and dark ones: each block the
    # light grey the track's paving had
    'Pavers': ('brick_pavement_03', False, .3, 20., .06, dict(colour=(.6, .595, .57))),
    # the light poles: their masts cast concrete (a cast concrete wall's scan, grey, its mottling and pores kept), their
    # frames galvanised steel (a painted plate's scan, its colour taken out and calmed: its scratches blurred away),
    # in the grey the mast had and the dark the frame had; mapped in metres by tools/templates.py
    'PoleMast': ('concrete_wall_006', True, .2, 12., .04, dict(grey=True)),
    'PoleFrame': ('green_metal_rust', True, .35, 25., .12, dict(grey=True, calm=.35)),
}
STATE = os.path.join(CACHE, 'state')    # what each material was and has, by its textures' name


def fetch(scan, res='2k'):
    """The scan's maps, from Poly Haven, and its size in metres."""
    folder = os.path.join(CACHE, scan)
    os.makedirs(folder, exist_ok=True)
    info_path = os.path.join(folder, 'info.json')
    keys = ('Diffuse', 'nor_dx', 'Rough', 'AO', 'Displacement')
    # (2k ones as key.png, as the first fetched)
    file = lambda k: os.path.join(folder, k + ('' if res == '2k' else '_' + res) + '.png')
    if not os.path.exists(info_path) or not all(os.path.exists(file(k)) for k in keys):
        get = lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'velopark-tools'})).read()
        info = json.loads(get('https://api.polyhaven.com/info/' + scan))
        files = json.loads(get('https://api.polyhaven.com/files/' + scan))
        for key in keys:
            out = file(key)
            if not os.path.exists(out):
                print('  fetching %s %s %s' % (scan, key, res))
                open(out, 'wb').write(get(files[key][res]['png']['url']))
        json.dump(dict(name=info['name'], authors=info['authors'], dimensions_mm=info['dimensions'],
                       url='https://polyhaven.com/a/' + scan, license='CC0'), open(info_path, 'w'), indent=1)
    info = json.load(open(info_path))
    maps = {k: Image.open(file(k)) for k in keys}
    return maps, info['dimensions_mm'][0] / 1000.


def channels(img, size=None):
    """An image as floats 0 to 1, (h, w, c), 8 or 16 bits a channel, resized to size x size."""
    a = np.asarray(img).astype(np.float32)
    a = a / (65535. if a.max() > 255 else 255.)
    if a.ndim == 2:
        a = a[..., None]
    a = a[..., :3]
    if size and a.shape[:2] != (size, size):
        a = np.stack([np.asarray(Image.fromarray(a[..., c], 'F').resize((size, size), Image.LANCZOS)) for c in range(a.shape[2])], -1)
    return a


def rgb(img):
    return np.asarray(img.convert('RGB'), float) / 255.


def even(col, height, colour):
    """Each block of a paving the one colour, its own grain kept: the blocks found as what stands
    over the joints in the height map, each across the texture's edges as it repeats, and each
    scaled to colour, the joints by the mean of the blocks' scales, softly between."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label
    n = len(height)
    blocks = height > np.percentile(height, JOINTS * 100)
    # (shrunk first, so two blocks a thin bridge joins come apart, then each grown back over its own)
    tiled = np.tile(blocks, (3, 3))
    lab, count = label(binary_erosion(tiled, iterations=4))
    _, (jj, ii) = distance_transform_edt(lab == 0, return_indices=True)
    lab = np.where(tiled, lab[jj, ii], 0)[n:2 * n, n:2 * n]
    # a block sunk below the rest reads as joint: what of the joints lies apart from their
    # network, the largest piece of them, is a block too
    low, _ = label(np.tile(~blocks, (3, 3)))
    low = low[n:2 * n, n:2 * n]
    sizes = np.bincount(low.ravel())
    sizes[0] = 0
    apart = (low != 0) & (low != sizes.argmax())
    lab = np.where(apart, low + lab.max() + 1, lab)
    ids, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(lab.shape)
    sums = np.stack([np.bincount(inv.ravel(), col[..., c].ravel()) for c in range(3)], 1)
    area = np.bincount(inv.ravel())
    mean = sums / np.maximum(area, 1)[:, None]
    scale = colour / np.maximum(mean, 1e-3)
    big = (ids != 0) & (area > 200)
    joint = scale[big].mean(0)
    scale[~big] = joint
    field = scale[inv]
    field = np.stack([gaussian_filter(field[..., c], 2, mode='wrap') for c in range(3)], -1)
    return col * field


def clean(col, others=()):
    """Straight lines across a scan (a crack across a road, repeating with it) filled from beside
    them, in it and in the others (its normal map, its roughness) alike; and the bands across it,
    its rows' and columns' brightness, evened out."""
    from scipy.ndimage import binary_dilation, gaussian_filter, uniform_filter1d
    n = len(col)
    s = n / 4096.
    lum = col.mean(2)
    dark = gaussian_filter(lum, 10 * s, mode='wrap') - gaussian_filter(lum, s, mode='wrap')
    out = [col] + list(others)
    for axis in (0, 1):
        # a line along this axis: dark over a long run of it
        run = uniform_filter1d(np.maximum(dark, 0), size=n // 12, axis=1 - axis, mode='wrap')
        line = binary_dilation(run > np.percentile(run, 99.3), iterations=max(1, int(6 * s)))
        w = gaussian_filter(line.astype(np.float32), 3 * s, mode='wrap')[..., None]
        k = int(40 * s)
        out = [a * (1 - w) + (np.roll(a, k, axis) + np.roll(a, -k, axis)) / 2 * w for a in out]
    col = out[0]
    for axis in (1, 0):
        lum = col.mean(2)
        f = lum.mean() / uniform_filter1d(lum.mean(axis), int(81 * s) | 1, mode='wrap')
        col = col * (f[None, :, None] if axis == 0 else f[:, None, None])
    return [col] + out[1:]


def textures(scan, base, tint_to, opts):
    """The scan's three textures, by name: (image, AC's format, whether a normal map: its green down, as AC's)."""
    maps, metres = fetch(scan, opts.get('res', '2k'))
    size, nm_size = opts.get('size', SIZE), opts.get('nm_size', SIZE)
    col = channels(maps['Diffuse'], size)
    col *= 1 - AO + AO * channels(maps['AO'], size)[..., :1]
    nrm = channels(maps['nor_dx'], nm_size)
    gloss = 1 - channels(maps['Rough'], nm_size)
    if opts.get('clean'):
        col = clean(col)[0]
        small = channels(maps['Diffuse'], nm_size)
        _, nrm, gloss = clean(small, (nrm, gloss))
    gloss = gloss[..., 0]
    if opts.get('colour') is not None:
        col = even(col, channels(maps['Displacement'], size)[..., 0], np.array(opts['colour']))
    if opts.get('grey'):
        col = np.repeat(col @ np.array([.299, .587, .114])[:, None], 3, -1)       # (its colour out, its light and dark kept)
    if opts.get('calm') is not None:
        # (its marks out: its colour blurred, its light and dark that much of what they were round its mean, its
        # bumps that much of theirs; only its soft mottling left, as galvanised steel's)
        from scipy.ndimage import gaussian_filter
        c = float(opts['calm'])
        col = gaussian_filter(col, (len(col) / 96, len(col) / 96, 0), mode='wrap')
        mean = col.reshape(-1, 3).mean(0)
        col = mean + (col - mean) * c
        flat = np.array([.5, .5, 1.])
        nrm = flat + (gaussian_filter(nrm, (len(nrm) / 256, len(nrm) / 256, 0), mode='wrap') - flat) * c
        gloss = gloss.mean() + (gaussian_filter(gloss, len(gloss) / 96, mode="wrap") - gloss.mean()) * c
    if tint_to is not None:
        col *= tint_to / col.reshape(-1, 3).mean(0)
    col *= opts.get('shade', 1.)                                # (darker or lighter than that, as the real one is)
    tmaps = np.stack([gloss, gloss, gloss * .5], -1)
    out = {}
    for suffix, arr, fmt in (('', col, 'DXT1'), ('_nm', nrm, 'DXT5'), ('_maps', tmaps, 'DXT1')):
        name = '%s%s.dds' % (base, suffix)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255 + .5).astype(np.uint8), 'RGB')
        out[name] = (img, fmt, suffix == '_nm')
    return out, metres


def average(k, name):
    """The average colour of a model's texture, 0 to 1."""
    t = next(t for t in k['textures'] if t['name'] == name)
    import io
    return rgb(Image.open(io.BytesIO(t['data']))).reshape(-1, 3).mean(0)


def material(m, base, spec, spec_exp, fresnel):
    old = {p['name']: p['value'] for p in m['props']}
    props = [('ksAmbient', old.get('ksAmbient', .2)), ('ksDiffuse', old.get('ksDiffuse', .2)), ('ksSpecular', spec),
             ('ksSpecularEXP', spec_exp), ('ksEmissive', 0.), ('ksAlphaRef', old.get('ksAlphaRef', 1.)),
             ('fresnelC', 0.), ('fresnelEXP', 5.), ('fresnelMaxLevel', fresnel), ('nmObjectSpace', 0.), ('isAdditive', 0.),
             ('useDetail', 0.), ('detailUVMultiplier', 1.), ('shadowBiasMult', 0.)]
    m['shader'] = 'ksPerPixelMultiMap'
    m['props'] = [dict(name=n, value=float(v), rest=bytes(36)) for n, v in props]
    # txDetail is the colour again: the shader reads it only with useDetail, as Kunos' own leave it
    m['textures'] = [dict(sampler=s, slot=i, name=t) for i, (s, t) in enumerate(
        (('txDiffuse', base + '.dds'), ('txNormal', base + '_nm.dds'), ('txMaps', base + '_maps.dds'), ('txDetail', base + '.dds')))]


def shown(n):
    """A mesh the game draws: the layouts' markers (AC_START...) are meshes the game hides."""
    return n['visible'] and n['renderable'] and len(n['faces']) and not n['name'].startswith('AC_')


def metres_of(n, mx):
    """Metres a texture of the mesh covers, the mean of its u and v; None where its uv are
    too thin to tell (strips along an edge, mapped as the road beside them)."""
    V, F = n['verts'], n['faces'].astype(int)
    P = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ mx)[:, :3]
    have = scale(P, V[:, 6:8].astype(float), F)
    return None if have is None else float(have.mean())


def main():
    dry = '--dry-run' in sys.argv
    models = {f: gltf_source.load_kn5(f) for f in sorted(glob.glob(os.path.join(MODELS, '*.glb')))}
    changed = set()
    for mat_name, (scan, tint, spec, spec_exp, fresnel, opts) in PLAN.items():
        users = [(f, k, mi) for f, k in models.items() for mi, m in enumerate(k['materials']) if m['name'] == mat_name
                 and any(n['material'] == mi and shown(n) for n, _ in bp.meshes(k))]
        if not users:
            print('%-14s no model shows it' % mat_name)
            continue
        # named after the material: two may take one scan, each in its own colour
        base = 'scan_' + mat_name.lower().replace(' ', '_')
        is_scan = lambda k, mi: k['materials'][mi]['textures'][0]['name'] == base + '.dds'
        # what it was, kept from the first run: the average colour of its texture, and the metres
        # that texture covered, which the uv of a mesh made again by some tool still have; and the
        # scan it has, and the metres a texture of it covers
        state_path = os.path.join(STATE, base + '.json')
        state = json.load(open(state_path)) if os.path.exists(state_path) else {}
        if not state:
            # (kept beside the scan, before)
            for old in glob.glob(os.path.join(CACHE, '*', base + '.json')):
                state = json.load(open(old))
                state.setdefault('scan', os.path.basename(os.path.dirname(old)))
        fresh = [(f, k, mi) for f, k, mi in users if not is_scan(k, mi)]
        if fresh and 'from_metres' not in state:
            f, k, mi = fresh[0]
            was = k['materials'][mi]['textures'][0]['name']
            state['tint'] = [float(x) for x in average(k, was)]
            sizes = [metres_of(n, mx) for f, k, mi in fresh for n, mx in bp.meshes(k) if n['material'] == mi and shown(n)]
            state['from_metres'] = float(np.median([x for x in sizes if x is not None]))
        if 'from_metres' not in state:
            sys.exit('%s: no state in %s and no model with its old texture to make it from' % (mat_name, state_path))
        made, metres = textures(scan, base, np.array(state['tint']) if tint else None, opts)
        factor = state['from_metres'] / metres
        # a model that has another scan already has its meshes at that scan's size
        was_scan = state.get('scan', scan)
        if was_scan != scan:
            was_metres = state.get('metres') or json.load(open(os.path.join(CACHE, was_scan, 'info.json')))['dimensions_mm'][0] / 1000.
            swap = was_metres / metres
        state.update(scan=scan, metres=metres)
        # a model taking the scan has every mesh of the material scaled; one that has it, only
        # those back at the old size
        todo = []
        for f, k, mi in users:
            first = not is_scan(k, mi)
            for n, mx in bp.meshes(k):
                if n['material'] != mi or not len(n['faces']):
                    continue
                h = metres_of(n, mx)
                # (back at the old size: near it, and nearer it than the scan's)
                if first or (h is not None and abs(np.log(h / state['from_metres'])) < min(np.log(1 + OFF), abs(np.log(h / metres)))):
                    todo.append((f, n, np.array([factor, factor])))
                elif was_scan != scan:
                    todo.append((f, n, np.array([swap, swap])))
                else:
                    todo.append((f, n, np.ones(2)))
        # and each mapped square, the scan's size along both: the old textures were not square
        # (TarmacDark 2048 by 4096), and the roads were mapped for them, along each road
        squared = 0
        for j, (f, n, by) in enumerate(todo):
            V, F = n['verts'], n['faces'].astype(int)
            mx = next(m_ for n_, m_ in bp.meshes(models[f]) if n_ is n)
            P3 = (np.c_[V[:, :3].astype(float), np.ones(len(V))] @ mx)[:, :3]
            have = scale(P3, V[:, 6:8].astype(float) * by, F)
            if have is not None:
                fix = have / metres
                if np.abs(np.log(fix)).max() > np.log(1.05):
                    todo[j] = (f, n, by * fix)
                    squared += 1
        todo = [(f, n, by) for f, n, by in todo if np.abs(by - 1).max() > 1e-6]
        print('%-14s %d meshes mapped square' % ('', squared))
        print('%-14s %-18s %.2f m a texture, was %.1f m%s: uv scaled on %d meshes in %d models'
              % (mat_name, scan, metres, state['from_metres'], '' if was_scan == scan else ', then %s' % was_scan,
                 len(todo), len({f for f, _, _ in todo})))
        if dry:
            continue
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        json.dump(state, open(state_path, 'w'), indent=1)
        import materials
        made = {name: materials.save_texture(name, img, fmt, normal) for name, (img, fmt, normal) in made.items()}   # (kept as PNG)
        for f, n, by in todo:
            V = n['verts']
            n['verts'] = np.c_[V[:, :6], V[:, 6:8].astype(float) * by, V[:, 8:]].astype(V.dtype)
            changed.add(f)
        for f, k, mi in users:
            if not is_scan(k, mi):
                material(k['materials'][mi], base, spec, spec_exp, fresnel)
                changed.add(f)
            have = {t['name'] for t in k['textures']}
            for name, data in made.items():
                if name not in have:
                    k['textures'].append(dict(name=name, active=1, data=data))
                    changed.add(f)
    for f in sorted(changed):
        k = models[f]
        # the textures no material names any more go
        used = {t['name'] for m in k['materials'] for t in m['textures']}
        k['textures'] = [t for t in k['textures'] if t['name'] in used]
        save(k, f)
        print('  saved %s' % os.path.relpath(f, ROOT))
    if not changed and not dry:
        print('the surfaces have their scans already')


def save(k, path):
    """As build_paths.save, with the textures this adds pointed at in source/textures."""
    import urllib.parse
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)


if __name__ == '__main__':
    main()
