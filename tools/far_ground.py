"""The far ground (GRASS_1 of source/models/velopark.glb) shown as the land round the track is.

    python tools/far_ground.py

The far ground, from the near ground's edge out to 1.5 km, took the near ground's texture, the
ground photo (GoogleMap.jpg, 1060 m round the origin), far outside it: the photo repeated over
it, mirrored, its smeared edges showing as streaks. This gives it a material of its own,
BackgroundFar (Background's values), with a texture of its own, FarGround.dds, made from
tools/reference/surroundings/surroundings.jpg (Google's satellite pictures round the track, from
tools/surroundings.py), its colours matched to the ground photo's where both show the same land
(round the near ground's edge), SIZE pixels a side; and maps the far ground onto it by where
each point lies. Run again, it finds the far ground mapped.
"""
import io, json, os

import numpy as np
from PIL import Image

import build_paths as bp
import gltf_source

ROOT = bp.ROOT
SURROUNDINGS = os.path.join(ROOT, 'tools', 'reference', 'surroundings')
TEX = os.path.join(ROOT, 'source', 'textures')
FAR = 'GRASS_1'
NAME, TEXTURE = 'BackgroundFar', 'FarGround.dds'
PHOTO, PHOTO_SIZE = 'GoogleMap.jpg', 1060.
MATCH = (330., 500.)        # metres from the origin (the larger of x and z): the ring both show, to match
SIZE = 4096


def main():
    k = gltf_source.load_kn5(bp.TRACK)
    if any(m['name'] == NAME for m in k['materials']):
        print('the far ground is mapped already')
        return
    info = json.load(open(os.path.join(SURROUNDINGS, 'surroundings.json')))
    (xa, xb), (za, zb) = info['area_x_z']
    pic = np.asarray(Image.open(os.path.join(SURROUNDINGS, info['image'])).convert('RGB'), float)
    H, W = pic.shape[:2]
    # the ground photo onto the same pixels, to match the colours where both show the land
    data = next(t['data'] for t in k['textures'] if t['name'] == PHOTO)
    n = int(PHOTO_SIZE * W / (xb - xa))
    photo = np.asarray(Image.open(io.BytesIO(data)).convert('RGB').resize((n, n), Image.LANCZOS), float)
    c0, r0 = int((xb - PHOTO_SIZE / 2) * W / (xb - xa)), int((zb - PHOTO_SIZE / 2) * H / (zb - za))
    rows, cols = np.mgrid[r0:r0 + n, c0:c0 + n]
    x = xb - (cols + .5) * (xb - xa) / W
    z = zb - (rows + .5) * (zb - za) / H
    ring = (np.maximum(np.abs(x), np.abs(z)) > MATCH[0]) & (np.maximum(np.abs(x), np.abs(z)) < MATCH[1])
    a, b = pic[r0:r0 + n, c0:c0 + n][ring], photo[ring]
    out = (pic - a.mean(0)) / a.std(0) * b.std(0) + b.mean(0)
    print('colours matched: the pictures %s +- %s, the ground photo %s +- %s' % (
        a.mean(0).round(), a.std(0).round(), b.mean(0).round(), b.std(0).round()))
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS)
    import materials
    tex = materials.save_texture(TEXTURE, img, 'DXT1')                    # (kept as PNG, AC's DDS made by its build)
    k['textures'].append(dict(name=TEXTURE, active=1, data=tex))
    base = next(m for m in k['materials'] if m['name'] == 'Background')
    k['materials'].append(dict(base, name=NAME, props=[dict(p) for p in base['props']],
                               textures=[dict(t, name=TEXTURE) if t['sampler'] == 'txDiffuse' else dict(t) for t in base['textures']]))
    node, m = next((n_, m_) for n_, m_ in bp.meshes(k) if n_['name'] == FAR)
    node['material'] = len(k['materials']) - 1
    V = node['verts'][:, :3].astype(float) + m[3, :3]
    node['verts'][:, 6] = (xb - V[:, 0]) / (xb - xa)
    node['verts'][:, 7] = (zb - V[:, 2]) / (zb - za)
    doc, _ = gltf_source.read_glb(bp.TRACK)
    import urllib.parse
    uris = {i['name']: urllib.parse.unquote(i['uri']) for i in doc.get('images', [])}
    doc, binary = gltf_source.to_gltf(k, lambda t: uris.get(t['name'], '../textures/' + t['name']))
    gltf_source.write_glb(bp.TRACK, doc, binary)
    u, v = node['verts'][:, 6], node['verts'][:, 7]
    print('%s mapped onto %s: u %.2f to %.2f, v %.2f to %.2f' % (FAR, TEXTURE, u.min(), u.max(), v.min(), v.max()))


if __name__ == '__main__':
    main()
