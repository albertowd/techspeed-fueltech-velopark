"""The track's concrete paving blocks, in source/models/velopark.glb and velopark_kerbs.glb.

    python tools/build_pavers.py

The real track is paved with concrete blocks, BLOCK long and half as wide, laid in rows a half
block apart with dark joints between them, at R29, the light concrete beside the L1500's top
hairpin, and on the run-offs beyond the floor kerbs STRAIGHT4 (R30), STRAIGHT5 (R41) and
LAST_OUT (R43, R44). The 0.5.0 track has them as light tarmac, and 0.5.1 patched them with
pieces of the roads beside them, light and dark: none of which a layout drives on (PAVED). This
gives them all a material of their own, Pavers, as Tarmac Light's but for its texture, made here
(source/textures/pavers.dds): grey blocks, each a little lighter or darker than the next, their
edges worn darker, and mapped flat from above, TILE metres a texture; so each run-off is one
paving. Beyond STRAIGHT4 and STRAIGHT5 (BANDS) the run-off is first a strip of asphalt along the
track, as wide all along as WIDTHS says, from the track's own road (its dark asphalt, and the
pieces of it 0.5.1 patched): past STRAIGHT4 a little more than its kerb, past STRAIGHT5 half the
run-off. The faces are cut along its edge, the asphalt side made the track's dark asphalt (a
mesh of its own, _ASPHALT, mapped as the paddock's is) and the rest paved. Run
again, it finds them paved.

The paving is driven as kerb, a kart rumbling over it as over the kerbs: its meshes are named
1KERB_ (as 1KERB_29) in place of 1ROAD_, AC's kerb surface, which the BeamNG.drive export makes
BeamNG's rumble strip.
"""
import os, urllib.parse

import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

import numpy as np
from PIL import Image, ImageFilter

import build_paths as bp
import gltf_source
import kn5io
import kn5tree

ROOT = bp.ROOT
PAVED = {'velopark.glb': ('1ROAD_29', '1ROAD_30', '1ROAD_43', '1ROAD_44'),
         'velopark_kerbs.glb': ('1ROAD_PATCH_STRAIGHT4_1ROAD_30', '1ROAD_PATCH_LAST_OUT_1ROAD_43', '1ROAD_PATCH_LAST_OUT_1ROAD_44')}
# the paved roads with a strip of asphalt along the track first, and the kerb beside it
BANDS = {'1ROAD_30': '1ROAD_KERB_STRAIGHT4', '1ROAD_PATCH_STRAIGHT4_1ROAD_30': '1ROAD_KERB_STRAIGHT4'}
# how wide the strip is, from the track's road: ('kerb', m), m past the kerb's outer edge; or
# ('half', None), half as wide as the run-off
WIDTHS = {'1ROAD_KERB_STRAIGHT4': ('kerb', .8)}
# paving laid anew outside a run-off of asphalt, along the straight of its kerb: the mesh, the
# kerb, from and to where along x the kerb runs straight, how wide, in velopark_kerbs.glb
STRIPS = {'1ROAD_PAVERS_STRAIGHT5': ('1ROAD_KERB_STRAIGHT5', (119., 147.), 2.5)}
STRIP_STEP = .5          # metres between its rows
STRIP_TAPER = 5.         # metres over which a strip grows out of its run-off's edge, at each end
LIFT = .015              # metres over the ground its outside lies
SKIRT = .3               # metres its outer edge goes down into the ground
ASPHALT = 'Tarmac Dark'
MATERIAL = 'Pavers'
TEXTURE = 'pavers.dds'
BLOCK = .2               # metres, a block's length; its width half that
JOINT = .004             # metres, the joints between them
TILE = 6.4               # metres a texture covers, each way: so wide its blocks' shades do not
                         # make a pattern seen from far
PX = 2048                # pixels a texture, each way


def wrapped(rng, n, spread):
    """A smooth noise over the texture, n knots each way, as it repeats."""
    k = rng.normal(128, spread, (n, n)).clip(0, 255)
    k = np.tile(k, (3, 3))
    big = np.asarray(Image.fromarray(k.astype(np.uint8)).resize((PX * 3, PX * 3), Image.BICUBIC), float)
    return big[PX:2 * PX, PX:2 * PX] - 128


def texture(seed=3):
    """Rows of blocks along u, each row half a block on from the last, a whole number of them
    across the texture both ways, so it repeats."""
    rng = np.random.default_rng(seed)
    m = PX / TILE                                  # pixels a metre
    y, x = np.mgrid[0:PX, 0:PX] / m
    row = np.floor(y / (BLOCK / 2)).astype(int)
    shift = np.where(row % 2, BLOCK / 2, 0)
    col = np.floor((x + shift) / BLOCK).astype(int) % int(round(TILE / BLOCK))
    # how far into its block each pixel is, from its nearest edge
    fx = ((x + shift) % BLOCK)
    fy = y % (BLOCK / 2)
    edge = np.minimum(np.minimum(fx, BLOCK - fx), np.minimum(fy, BLOCK / 2 - fy))
    # each block its own shade, and a fine grain over all
    n_rows, n_cols = int(round(TILE / (BLOCK / 2))), int(round(TILE / BLOCK))
    shade = rng.normal(0, 6, (n_rows, n_cols)) + np.asarray(Image.fromarray(rng.normal(128, 40, (6, 6)).clip(0, 255)
        .astype(np.uint8)).resize((n_cols, n_rows), Image.BICUBIC), float) / 40 * 4 - 12.8
    base = 152 + shade[row % n_rows, col]
    grain = np.asarray(Image.fromarray(rng.normal(128, 40, (PX, PX)).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.7)), float) - 128
    stain = wrapped(rng, 8, 50)
    dirt = wrapped(rng, 48, 60)
    lum = base + .3 * grain + .2 * stain - 18 * np.clip(dirt / 60 - .3, 0, 1)
    # worn edges a little darker, the joints dark with sand in them
    lum = lum - 14 * np.clip(1 - edge / .008, 0, 1)
    lum = np.where(edge < JOINT / 2, 96 + .3 * grain + .2 * stain, lum)
    rgb = np.stack([lum * 1.0, lum * .99, lum * .95], -1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def save(k, path):
    doc, _ = gltf_source.read_glb(path)
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(path, doc, binary)


def main():
    data = None
    paved = 0
    km = gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_kerbs.glb'))
    tm = gltf_source.load_kn5(bp.TRACK)
    flat = lambda t: [Polygon(x) for x in t if Polygon(x).area > 1e-6]
    paved_names = {x for v in PAVED.values() for x in v}
    # the track's own road: its dark asphalt, and the pieces of it 0.5.1 patched
    road, runoff = [], {}
    for model in (tm, km):
        for n, m in bp.meshes(model):
            if not n['name'].startswith('1ROAD') or not len(n['faces']):
                continue
            t = bp.triangles(n, m)
            if not len(t):
                continue
            if n['name'] in BANDS:
                runoff.setdefault(BANDS[n['name']], []).append(t[:, :, [0, 2]])
            elif n['name'] not in paved_names and model['materials'][n['material']]['name'] == ASPHALT:
                road.append(t[:, :, [0, 2]])
    road = np.concatenate(road)
    kerbs, notes = {}, []
    for n, m in bp.meshes(km):
        if n['name'] not in WIDTHS:
            continue
        kerb = unary_union(flat(bp.triangles(n, m)[:, :, [0, 2]]))
        near = unary_union(flat(road[shapely.distance(kerb, shapely.points(road.mean(1))) < 40]))
        out = np.percentile(shapely.distance(near, shapely.points(np.array(kerb.exterior.coords if kerb.geom_type == 'Polygon'
                                                                           else [c for g in kerb.geoms for c in g.exterior.coords]))), 95)
        rule, x = WIDTHS[n['name']]
        if rule == 'kerb':
            width = out + x
        else:
            pts = np.concatenate(runoff[n['name']]).reshape(-1, 2)
            width = max(np.percentile(shapely.distance(near, shapely.points(pts)), 90) / 2, out + .3)
        kerbs[n['name']] = (near, width)
        notes.append('  past %s, asphalt %.2f m from the track, the kerb out to %.2f m' % (n['name'], width, out))
    uv_paddock = bp.flat_uv(next(n for n, m in bp.meshes(gltf_source.load_kn5(bp.PADDOCK)) if n['name'] == bp.ASPHALT))
    for f, roads in PAVED.items():
        path = os.path.join(bp.MODELS, f)
        k = gltf_source.load_kn5(path)
        mi = next((i for i, m in enumerate(k['materials']) if m['name'] == MATERIAL), None)
        todo = [(n, m) for n, m in bp.meshes(k) if n['name'] in roads and n['material'] != mi]
        if not todo:
            kerbed = rumble(k)
            if kerbed:
                save(k, path)
                paved += kerbed
            continue
        if not todo:
            continue
        if data is None:
            data = kn5io.dds(texture(), 'DXT1')
            open(os.path.join(ROOT, 'source', 'textures', TEXTURE), 'wb').write(data)
        if mi is None:
            k['textures'].append(dict(name=TEXTURE, active=1, data=data))
            like = next(m for m in k['materials'] if m['name'] in ('Tarmac Light', 'Tarmac Dark'))
            k['materials'].append(dict(like, name=MATERIAL, props=[dict(p) for p in like['props']],
                                       textures=[dict(sampler='txDiffuse', slot=0, name=TEXTURE)]))
            mi = len(k['materials']) - 1
        pave(k, todo, mi, kerbs, uv_paddock)
        rumble(k)
        save(k, path)
        paved += len(todo)
    laid = strips(data)
    print('\n'.join(notes + ['%d roads paved' % paved] if paved else []) + ('\n' if paved and laid else '') +
          ('\n'.join(laid) if laid else '') if paved or laid else 'the paving is laid already')


KERB = '1KERB_'           # the paving's meshes' names begin so: driven as kerb


def rumble(k):
    """The paving's meshes named to be driven as kerb; how many were not."""
    mi = next((i for i, m in enumerate(k['materials']) if m['name'] == MATERIAL), None)
    n_ = 0
    for n, m in bp.meshes(k):
        if n['material'] == mi and n['name'].startswith('1ROAD_'):
            n['name'] = KERB + n['name'][len('1ROAD_'):]
            n_ += 1
    return n_


def strips(data):
    """The STRIPS laid, as each new mesh of paving in velopark_kerbs.glb: rows square to the
    kerb's straight every STRIP_STEP, from the outer edge of the asphalt there out for its width,
    lying on the run-off's edge at its inside and LIFT over the ground at its outside, its outer
    edge going down SKIRT into the ground; at its ends it grows out of the asphalt's edge over
    STRIP_TAPER, from nothing to its width."""
    path = os.path.join(bp.MODELS, 'velopark_kerbs.glb')
    k = gltf_source.load_kn5(path)
    names = {n['name'].replace(KERB, '1ROAD_', 1) for n, m in bp.meshes(k)}
    todo = {n: v for n, v in STRIPS.items() if n not in names}
    if not todo:
        return []
    tm = gltf_source.load_kn5(bp.TRACK)
    asphalt, ground = [], []
    for model in (tm, k):
        for n, m in bp.meshes(model):
            if not len(n['faces']):
                continue
            t = bp.triangles(n, m)
            if not len(t):
                continue
            if n['name'].startswith('1ROAD') and n['name'] not in STRIPS:
                asphalt.append(t)   # the kerbs too: the strip's rows start on them
            elif 'GRASS' in n['name'][:8]:
                ground.append(t)
    top = bp.Surface(np.concatenate(asphalt))
    under = bp.Surface(np.concatenate(ground))
    mi = next(i for i, m in enumerate(k['materials']) if m['name'] == MATERIAL)
    template = next(n for n, m in bp.meshes(k) if n['name'].startswith('1ROAD_PATCH'))
    out = []
    for name, (kerb_name, (x0, x1), width) in todo.items():
        kn = next(n for n, m in bp.meshes(k) if n['name'] == kerb_name)
        K = kn['verts'][:, [0, 2]].astype(float)
        K = K[(K[:, 0] >= x0) & (K[:, 0] <= x1)]
        c = K.mean(0)
        u = np.linalg.svd(K - c)[2][0]
        u = u if u[0] > 0 else -u
        n_ = np.array([u[1], -u[0]])
        # outward: away from the track, where there is no asphalt a few metres off
        if np.isnan(top.y(*(c + n_ * 6))) < np.isnan(top.y(*(c - n_ * 6))):
            n_ = -n_
        t = (K - c) @ u
        rows = []
        for s_ in np.arange(t.min(), t.max() + 1e-6, STRIP_STEP):
            # growing out of the run-off's edge at its ends, as smoothly as a road merging
            e = np.clip(min(s_ - t.min(), t.max() - s_) / STRIP_TAPER, 0, 1)
            wide = width * max(e * e * (3 - 2 * e), .02)
            p = c + u * s_
            # out to the asphalt's outer edge
            d = 0.
            while not np.isnan(top.y(*(p + n_ * (d + .05)))) and d < 15:
                d += .05
            inner = p + n_ * (d - .02)
            y_in = top.y(*(p + n_ * (d - .08)))
            if np.isnan(y_in) or d >= 15:
                raise SystemExit('%s: no asphalt out from the kerb at %s' % (name, p.round(1)))
            row = []
            for f in np.linspace(0, 1, 6):
                q = inner + n_ * wide * f
                g = under.y(*q)
                g = y_in if np.isnan(g) else g
                w = f * f * (3 - 2 * f)
                row.append([q[0], max(y_in * (1 - w) + (g + LIFT) * w, g + LIFT * .5) if f > 0 else y_in, q[1]])
            last = row[-1]
            row.append([last[0] + n_[0] * .05, last[1] - SKIRT, last[2] + n_[1] * .05])
            rows.append(row)
        G = np.array(rows)
        node = bp.mesh(template, name, G, lambda P: P[:, [0, 2]] / TILE)
        node['material'] = mi
        node['name'] = KERB + name[len('1ROAD_'):]
        k['root']['children'].append(node)
        out.append('  %s laid, %.1f m long and %.1f m wide' % (name, np.ptp(t), width))
    if data is not None and not any(x['name'] == TEXTURE for x in k['textures']):
        k['textures'].append(dict(name=TEXTURE, active=1, data=data))
    save(k, path)
    return out


def cut(V, F, f):
    """A mesh's faces cut along where f, a value at each vertex, is 0, as it goes between them:
    its vertices with those made on the edges it crosses, and its faces, each wholly on one side."""
    V, f = list(V), list(f)
    made = {}

    def at(a, b):
        key = (min(a, b), max(a, b))
        if key not in made:
            t = f[a] / (f[a] - f[b])
            V.append(V[a] + t * (V[b] - V[a]))
            f.append(0.)
            made[key] = len(V) - 1
        return made[key]
    out = []
    for tri in F:
        s = [f[i] < 0 for i in tri]
        if all(s) or not any(s):
            out.append(tuple(tri))
            continue
        # the one on its own side, and the other two, in the face's turn
        k = next(i for i in range(3) if s[i] != s[(i + 1) % 3] and s[i] != s[(i + 2) % 3])
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        ab, ac = at(a, b), at(a, c)
        out += [(a, ab, ac), (ab, b, c), (ab, c, ac)]
    return np.array(V), np.array(out), np.array(f)


def pave(k, todo, mi, kerbs, uv_paddock):
    for n, m in todo:
        V = n['verts'].astype(float)
        F = n['faces'].astype(int)
        P = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
        asphalt = None
        if n['name'] in BANDS:
            near, width = kerbs[BANDS[n['name']]]
            f = shapely.distance(near, shapely.points(P[:, [0, 2]])) - width
            V, F, f = cut(V, F, f)
            P = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
            near = f[F].mean(1) < 0
            if near.any():
                asphalt = dict(n)
                asphalt['verts'] = V.astype('<f4')
                asphalt['faces'] = F.astype('<u2')
                taken = bp.split(asphalt, ~near)
                Pa = (np.c_[asphalt['verts'][:, :3].astype(float), np.ones(len(asphalt['verts']))] @ m)[:, :3]
                Ua = uv_paddock(Pa)
                Ta = gltf_source.tangents(asphalt['verts'][:, :3].astype(float), asphalt['verts'][:, 3:6].astype(float), Ua,
                                          asphalt['faces'].astype(int))
                asphalt['verts'] = np.c_[asphalt['verts'][:, :6], Ua, Ta].astype('<f4')
                asphalt.update(name=n['name'] + '_ASPHALT', children=[],
                               material=next(i for i, x in enumerate(k['materials']) if x['name'] == ASPHALT))
                bp.sphere(asphalt)
                n['verts'], n['faces'] = V.astype('<f4'), F.astype('<u2')
                bp.split(n, near)
                V = n['verts'].astype(float)
                F = n['faces'].astype(int)
                P = (np.c_[V[:, :3], np.ones(len(V))] @ m)[:, :3]
        U = P[:, [0, 2]] / TILE
        T = gltf_source.tangents(V[:, :3], V[:, 3:6], U, F)
        n['verts'] = np.c_[V[:, :6], U, T].astype('<f4')
        n['faces'] = F.astype('<u2')
        n['material'] = mi
        bp.sphere(n)
        if asphalt is not None:
            for parent, _ in kn5tree.walk(k['root']):
                if any(c is n for c in parent['children']):
                    parent['children'].append(asphalt)
                    break
        print('  %s paved%s' % (n['name'], ', its asphalt along the kerb kept' if asphalt is not None else ''))


if __name__ == '__main__':
    main()
