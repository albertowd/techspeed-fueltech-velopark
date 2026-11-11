"""The trees round the track as the ground photo and the screenshots show them, in
source/models/velopark_extra.glb (and BeamNG's forest, source/models/velopark_extra_trees.json).

    python tools/plant_outside.py [--prune | --strip]

After tools/build_outside.py, whose roads, fences and buildings nothing is planted on, nor
anything new within VERGE metres of the track's roads (the grass inside its fence):

- the plantation's trees out of the way of the dirt roads, the fences and the buildings (within
  CLEAR of them), and out of the strip between the north straight and the north road where the
  photo shows grass (crowns on less than OPEN of the ground round them): each tree with its
  leaves, the understory's clumps about it and the wall round its trunk;
- the old trees, boards of a broad leaf or an ash some 21 m high, out of every model;
- in the woods north of the north road, the town's, eucalyptus as the plantation's, WOODS_APART
  apart, under them the understory, where the photo shows crowns;
- everywhere else the photo shows crowns (on more than CROWNS of it within CROWN metres), broad
  leaf trees, NEAR_APART apart within NEAR metres of the track (BIG of them the size of the
  south east wood's) and FAR_APART further out, these with fewer clumps of leaves;
- palms (PALMS) and clumps of bamboo (BAMBOO) where the screenshots show them, round the south
  east's fence and the paddock's south.

--strip plants the strip between the vp_1000's north tyre wall and the north fence thick with
eucalyptus, as it is (strip()).

Nothing new stands inside the track's outline (its roads and tyre walls, with the infields they
enclose): --prune takes out any that does, from a planting before that rule.

Each kind is made in VARIANTS of its own, as tools/build_trees.py makes them, turned and sized
at random where it stands; their meshes cut in its TILE metre squares, named EUCA_<part>_..._n<i>
(so tools/export_beamng.py makes them BeamNG's forest). Run again, it finds them planted.
"""
import io, json, math, os, sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_opening, uniform_filter
from scipy.spatial import cKDTree

import build_paths as bp
import build_trees as bt
import gltf_source
import kn5io
import kn5tree
from plant_trees import pieces

ROOT = bp.ROOT
EXTRA = bt.EXTRA
OUTSIDE = os.path.join(bp.MODELS, 'velopark_outside.glb')
TEX = bt.TEX
CLEAR = 3.               # metres from a road, fence, building, kerb or tyre wall
VERGE = 11.              # and no new tree nearer the track's roads: its grass verge, inside its fence
OUTLINE = 3.             # metres round the roads and tyre walls the track's outline is, its infields in it
OPEN, STRIP = .15, (295., 337.)   # the north strip's z, and the share of crowns under which it is grass
NORTH_ROAD = 337.        # the woods north of the road, from here north
WOODS_APART = 7.5
NEAR, NEAR_APART, FAR_APART = 80., 6., 11.
CROWNS, CROWN = .45, 3.
BIG = .35                # of the broad leaf trees, the big ones
EXTENT = ((-500., 500.), (-430., 440.))  # where trees are planted
PALMS = [(-150., 25.), (-156., 30.), (-162., 36.), (-131., 19.), (-95., 12.), (-88., 9.), (-75., 8.),
         (-180., 262.), (-185., 256.), (-122., 16.)]
BAMBOO = [(-104., 18.), (-112., 21.), (-119., 23.), (-100., 27.)]
# the kinds made: kind, heights, how many variants, their first seed, and (broad leaf trees seen
# only from afar) how many clumps of leaves
VARIANTS = {'broad_n': ('broad', (9., 15.), 8, 400, None), 'broad_big_n': ('broad', (15., 21.), 6, 420, None),
            'broad_far_n': ('broad', (10., 19.), 6, 480, 22),
            'palm_n': ('palm', (6., 10.), 4, 440, None), 'bamboo_n': ('bamboo', (8., 11.), 3, 460, None)}
SEED = 11


# ---- the photo -----------------------------------------------------------------------------------
def crowns(track):
    """The share of crowns round each half metre of the ground photo (as an image, 0.5 m a pixel)."""
    data = next(t['data'] for t in track['textures'] if t['name'] == bt.PHOTO)
    n = 2120
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert('RGB').resize((n, n), Image.BOX), float)
    L = rgb.mean(2)
    G = rgb[..., 1] - (rgb[..., 0] + rgb[..., 2]) / 2
    mean = uniform_filter(L, 7)
    spread = np.sqrt(np.maximum(uniform_filter(L ** 2, 7) - mean ** 2, 0))
    crown = binary_opening((spread > 14) & (mean < 120) & (uniform_filter(G, 7) > 4), iterations=2)
    return uniform_filter(crown.astype(float), int(2 * CROWN / .5) + 1)


def at(img, P):
    P = np.atleast_2d(P)
    j = np.clip(((530 - P[:, 0]) * 2).astype(int), 0, img.shape[1] - 1)
    i = np.clip(((530 - P[:, 1]) * 2).astype(int), 0, img.shape[0] - 1)
    return img[i, j]


# ---- what nothing is planted near ----------------------------------------------------------------
def blocked(models, grid, outside_only=False):
    tris = []
    for path, k in models.items():
        if outside_only and path != OUTSIDE:
            continue
        for n, m in bp.meshes(k):
            if not len(n['faces']) or k['materials'][n['material']].get('shader') == 'ksTree':
                continue
            name = n['name']
            if 'GRASS' in name[:7] or name.startswith(('EUCA_', '1WALL_EUCA', 'TREE_')) or not (n['visible'] or name[:1].isdigit()):
                continue
            t = bp.triangles(n, m)
            if len(t) and t[:, :, 1].max() > -20:
                tris.append(t)
    return binary_dilation(grid.mark(np.concatenate(tris)), iterations=int(round(CLEAR / bt.R)))


NEW_KINDS = ('broad_n', 'broad_big_n', 'broad_far_n', 'palm_n', 'bamboo_n')


def track_outline(track):
    """The track's outline from above: its roads and tyre walls, OUTLINE metres round, with what
    they enclose (the infields) filled."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for k in (track, gltf_source.load_kn5(os.path.join(bp.MODELS, 'velopark_tyres.glb'))):
        for n, m in bp.meshes(k):
            if (n['name'].startswith('1ROAD_') or 'TYRE' in n['name']) and len(n['faces']):
                t = bp.triangles(n, m)
                if len(t) and np.abs(t[:, :, 1]).max() < 30:
                    polys += [Polygon(x[:, [0, 2]]).buffer(.05) for x in t if Polygon(x[:, [0, 2]]).area > .01]
    U = unary_union(polys).buffer(OUTLINE)
    return unary_union([Polygon(g.exterior) for g in getattr(U, 'geoms', [U])])


def prune():
    """The new trees (NEW_KINDS) standing inside the track's outline out: their stands, the
    pieces of their meshes nearest them and the walls round their trunks."""
    from shapely import contains_xy
    extra = gltf_source.load_kn5(EXTRA)
    forest = json.load(open(bt.FOREST))
    stands = forest['trees']
    S = np.array([[s[1], s[3]] for s in stands])
    new = np.array([s[0].startswith(NEW_KINDS) for s in stands])
    gone = new & contains_xy(track_outline(gltf_source.load_kn5(bp.TRACK)), S[:, 0], S[:, 1])
    if not gone.any():
        print('no new tree inside the track')
        return
    tree_of = cKDTree(S)
    emptied, pieces_out = [], 0
    for n, m in bp.meshes(extra):
        if not (n['name'].startswith((bt.PREFIX, '1WALL_' + bt.PREFIX)) and '_n' in n['name']) or not len(n['faces']):
            continue
        V = n['verts'][:, :3].astype(float) + m[3, :3]
        F = n['faces'].astype(int)
        lab = pieces(F, len(V))
        flab = lab[F[:, 0]]
        ids = np.unique(flab)
        c = np.array([V[lab == t][:, [0, 2]].mean(0) for t in ids])
        d, i = tree_of.query(c)
        out = (d < 12) & gone[i]
        if not out.any():
            continue
        sel = np.isin(flab, ids[out])
        pieces_out += out.sum()
        if sel.all():
            emptied.append(n)
        else:
            bp.split(n, sel)
    drop = {id(x) for x in emptied}
    for parent, _ in kn5tree.walk(extra['root']):
        parent['children'] = [c for c in parent['children'] if id(c) not in drop]
    forest['trees'] = [s for s, g in zip(stands, gone) if not g]
    bt.save(extra, EXTRA)
    json.dump(forest, open(bt.FOREST, 'w'), indent=0)
    print('%d new trees inside the track out (%d pieces)' % (gone.sum(), pieces_out))


STRIP_X = (-168., 62.)     # x the strip north of the vp_1000's north straight runs over
STRIP_APART = 3.5          # metres between its eucalyptus
STRIP_FROM_TRACK = 3.      # metres off the track's roads and tyre walls
STRIP_FROM_FENCE = 1.5     # metres short of the north fence


def strip():
    """Eucalyptus, young ones among them and the understory, thick in the strip between the
    vp_1000's north tyre wall and the north fence, as the real one is: STRIP_APART apart, from
    STRIP_FROM_TRACK off the track to STRIP_FROM_FENCE short of the fence, over STRIP_X. Their
    meshes named EUCA_<part>_..._s<i>: run again, it finds them planted."""
    from shapely import contains_xy
    from shapely.geometry import LineString
    import build_outside as bo
    extra = gltf_source.load_kn5(EXTRA)
    if any(n['name'].startswith(bt.PREFIX) and n['name'].split('_')[-1][:1] == 's' for n, m in bp.meshes(extra)):
        print('the strip north of the vp_1000 is planted already')
        return
    track = gltf_source.load_kn5(bp.TRACK)
    rng = np.random.default_rng(SEED + 7)
    # the north fence, as tools/build_outside.py lays it
    N = bo.osm_lines()[94747576]
    fence = LineString(bo.offset(N[(N[:, 0] > -300) & (N[:, 0] < 230)], -bo.FENCE_IN))
    F = np.array(fence.coords)
    F = F[np.argsort(F[:, 0])]
    fence_z = lambda x: np.interp(x, F[:, 0], F[:, 1])
    outline = track_outline(track)
    near_track = outline.buffer(STRIP_FROM_TRACK - OUTLINE)
    (x0, x1) = STRIP_X
    P = bt.jittered(np.array([x0, 270.]), np.array([x1, 335.]), STRIP_APART, rng)
    P = P[(P[:, 1] < fence_z(P[:, 0]) - STRIP_FROM_FENCE) & ~contains_xy(near_track, P[:, 0], P[:, 1])]
    forest = json.load(open(bt.FOREST))
    stands = forest['trees']
    S = np.array([[q[1], q[3]] for q in stands])
    P = P[cKDTree(S).query(P)[0] > STRIP_APART * .7]
    variants = forest['variants']
    euca = [n for n, v in variants.items() if v.get('kind') == 'euca']
    young = [n for n, v in variants.items() if v.get('kind') == 'young']
    shapes = {n: bt.shape_of(variants[n]) for n in euca + young}
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track) if 'GRASS' in n['name'][:7] and len(n['faces'])]))
    tiles, walls, new = {}, {}, []
    kinds = np.where(rng.random(len(P)) < .25, 'young', 'euca')
    for (x, z), kind in zip(P, kinds):
        y = ground.y(x, z)
        if np.isnan(y):
            continue
        names = young if kind == 'young' else euca
        name = names[rng.integers(len(names))]
        shape, r = shapes[name]
        scale, yaw = float(rng.uniform(.9, 1.1)), float(rng.uniform(0, 2 * np.pi))
        key = tuple(np.floor(np.array([x, z]) / bt.TILE).astype(int))
        for part in shape:
            tiles.setdefault(key, {}).setdefault(part, []).extend(bt.placed(shape[part], yaw, scale, np.array([x, y, z])))
        new.append([name, float(x), float(y), float(z), yaw, scale])
        a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ring = np.c_[np.cos(a), np.zeros(6), np.sin(a)] * r * scale * 1.1
        Vw = np.r_[ring + [x, y - .3, z], ring + [x, y + bt.TRUNK, z]]
        Fw = [(i, (i + 1) % 6, 6 + (i + 1) % 6) for i in range(6)] + [(i, 6 + (i + 1) % 6, 6 + i) for i in range(6)]
        walls.setdefault(key, []).append((Vw, np.r_[ring, ring] / np.linalg.norm(ring[0]), np.zeros((12, 2)), np.array(Fw)))
    # the understory under them
    C = P[rng.integers(len(P), size=len(P) * 4)] + rng.normal(0, 2., (len(P) * 4, 2))
    C = C[(C[:, 1] < fence_z(C[:, 0]) - STRIP_FROM_FENCE) & ~contains_xy(near_track, C[:, 0], C[:, 1])]
    for x, z in C:
        g = ground.y(x, z)
        if np.isnan(g):
            continue
        H = rng.uniform(3.5, 7.)
        yy = H * rng.uniform(0, 1) ** 1.2
        size = rng.uniform(1.6, 3.) * (1 + .3 * yy / H)
        c = np.array([x, g + yy, z])
        key = tuple(np.floor(np.array([x, z]) / bt.TILE).astype(int))
        tiles.setdefault(key, {}).setdefault('shrub', []).extend(
            bt.cards(c + [0, size * .25, 0], size, size, rng.uniform(0, np.pi), rng.normal(0, .3), c - [0, size * .3, 0], .8))
    mats = {m['name']: i for i, m in enumerate(extra['materials'])}
    part_mat = {'bark': mats['EucaBark'], 'leaves': mats['EucaLeaves'], 'shrub': mats['EucaShrub']}
    template = dict(kind=2, name='', active=1, children=[], material=0, cast_shadows=1, visible=1, transparent=0, layer=0,
                    lod_in=0., lod_out=0., renderable=1)
    root = extra['root']['children']
    tris = 0
    for key in sorted(tiles):
        for part, parts in tiles[key].items():
            for i, run in enumerate(bt.chunks(parts)):
                nd = bt.node(template, '%s%s_%d_%d_s%d' % (bt.PREFIX, part.upper(), key[0], key[1], i), run, part_mat[part])
                root.append(nd)
                tris += len(nd['faces'])
    for key in sorted(walls):
        for i, run in enumerate(bt.chunks(walls[key])):
            root.append(bt.node(template, '1WALL_%sTRUNKS_%d_%d_s%d' % (bt.PREFIX, key[0], key[1], i), run, mats['EucaBark'],
                                dict(renderable=0, cast_shadows=0)))
    bt.save(extra, EXTRA)
    forest['trees'] = stands + new
    json.dump(forest, open(bt.FOREST, 'w'), indent=0)
    print('%d eucalyptus (%d young) and %d clumps of understory in the strip north of the vp_1000: %d triangles' % (
        len(new), sum(1 for q in new if q[0] in young), len(C), tris))


def main():
    extra = gltf_source.load_kn5(EXTRA)
    if any(n['name'].startswith(bt.PREFIX) and '_n' in n['name'] for n, m in bp.meshes(extra)):
        print('the trees round the track are planted already')
        return
    if not os.path.exists(OUTSIDE):
        sys.exit('no %s: run tools/build_outside.py first' % os.path.relpath(OUTSIDE, ROOT))
    track = gltf_source.load_kn5(bp.TRACK)
    paths = [bp.TRACK, EXTRA, OUTSIDE] + [os.path.join(bp.MODELS, f + '.glb') for f in
                                          ('velopark_paddock', 'velopark_kerbs', 'velopark_tyres', 'velopark_paint', 'velopark_poles', 'velopark_kids')]
    models = {p: (track if p == bp.TRACK else extra if p == EXTRA else gltf_source.load_kn5(p)) for p in paths}
    (xa, xb), (za, zb) = EXTENT
    grid = bt.Grid(np.array([xa, za]), np.array([xb, zb]))
    near = blocked(models, grid)
    corridor = blocked(models, grid, outside_only=True)
    share = crowns(track)
    forest = json.load(open(bt.FOREST))
    stands = forest['trees']
    S = np.array([[s[1], s[3]] for s in stands])

    # the plantation's trees out of the corridors, and out of the north strip where it is grass
    cells = grid.cells(S)
    strip = (S[:, 1] > STRIP[0]) & (S[:, 1] < STRIP[1]) & (at(share, S) < OPEN)
    gone_stand = corridor[tuple(cells.T)] | strip
    tree_of = cKDTree(S)
    took_parts = 0
    emptied = []
    for n, m in bp.meshes(extra):
        name = n['name']
        if not name.startswith((bt.PREFIX, '1WALL_' + bt.PREFIX)) or not len(n['faces']):
            continue
        V = n['verts'][:, :3].astype(float) + m[3, :3]
        F = n['faces'].astype(int)
        lab = pieces(F, len(V))
        flab = lab[F[:, 0]]
        gone = np.zeros(len(F), bool)
        ids = np.unique(flab)
        c = np.array([V[lab == t][:, [0, 2]].mean(0) for t in ids])
        d, i = tree_of.query(c)
        if 'SHRUB' in name:
            out = corridor[tuple(grid.cells(c).T)] | ((d < 5) & gone_stand[i]) | \
                  ((c[:, 1] > STRIP[0]) & (c[:, 1] < STRIP[1]) & (at(share, c) < OPEN))
        else:
            out = (d < 9) & gone_stand[i]
        for t, o in zip(ids, out):
            if o:
                gone |= flab == t
        took_parts += out.sum()
        if gone.all():
            emptied.append(n)
        elif gone.any():
            bp.split(n, gone)
    # the old trees, boards, out of every model
    old = 0
    for k in (track, extra):
        for n, m in bp.meshes(k):
            if k['materials'][n['material']].get('shader') == 'ksTree':
                emptied.append(n)
                old += 1
    drop = {id(x) for x in emptied}
    for k in (track, extra):
        for parent, _ in kn5tree.walk(k['root']):
            parent['children'] = [c for c in parent['children'] if id(c) not in drop]
    kept = [s for s, g in zip(stands, gone_stand) if not g]
    print('  %d of the plantation\'s trees out (%d pieces), %d meshes of old trees out' % (gone_stand.sum(), took_parts, old))

    # where the new trees stand
    rng = np.random.default_rng(SEED)
    K = np.array([[s[1], s[3]] for s in kept]) if kept else np.zeros((0, 2))
    taken = cKDTree(K) if len(K) else None
    lo, hi = np.array([xa, za]), np.array([xb, zb])
    road_tris = np.concatenate([bp.triangles(n, m) for n, m in bp.meshes(track) if n['name'].startswith('1ROAD_') and len(n['faces'])])
    road_pts = cKDTree(road_tris.reshape(-1, 3)[:, [0, 2]])

    from shapely import contains_xy
    outline = track_outline(track)

    def free(P, apart):
        # not within the track's verge, its fence's side of it, nor in its infields: grass, the
        # tyre rings round the poles there are no trees
        ok = ~near[tuple(grid.cells(P).T)] & (road_pts.query(P)[0] > VERGE) & ~contains_xy(outline, P[:, 0], P[:, 1])
        if taken is not None:
            ok &= taken.query(P)[0] > apart * .7
        return ok
    woods = bt.jittered(np.array([xa, NORTH_ROAD]), hi, WOODS_APART, rng)
    woods = woods[free(woods, WOODS_APART) & (at(share, woods) > CROWNS)]
    P = np.r_[bt.jittered(lo, hi, NEAR_APART, rng)]
    dn = road_pts.query(P)[0]
    P = P[(dn < NEAR) | (rng.random(len(P)) < (NEAR_APART / FAR_APART) ** 2)]
    P = P[(P[:, 1] < NORTH_ROAD) & free(P, NEAR_APART) & (at(share, P) > CROWNS)]
    # the palms and bamboo out past the track's fence, where the screenshots show them
    def beyond(spots):
        out = []
        for p in np.array(spots, float):
            for _ in range(40):
                d, i = road_pts.query(p)
                if d > VERGE + 2.5:
                    break
                away = p - road_pts.data[i]
                p = p + away / max(np.linalg.norm(away), 1e-6) * .5
            out.append(p)
        return np.array(out)
    palms, bamboos = beyond(PALMS), beyond(BAMBOO)
    # none where a palm or bamboo goes
    special = np.r_[palms, bamboos]
    P = P[cKDTree(special).query(P)[0] > 4.]
    woods = woods[cKDTree(special).query(woods)[0] > 4.]

    # the trees made
    made = {}
    for prefix, (kind, hs, count, seed, clumps) in VARIANTS.items():
        for i in range(count):
            made['%s%d' % (prefix, i)] = dict(kind=kind, height=float(rng.uniform(*hs)), seed=seed + i, **({'clumps': clumps} if clumps else {}))
    variants = dict(forest['variants'])
    euca = [n for n, v in variants.items() if v.get('kind') == 'euca']
    young = [n for n, v in variants.items() if v.get('kind') == 'young']
    for n, v in made.items():
        variants[n] = v
    shapes = {n: bt.shape_of(v) for n, v in variants.items()}
    for n in made:
        made[n]['radius'] = shapes[n][1]
    ground = bp.Surface(np.concatenate([bp.triangles(n, m) for k in (track, models[os.path.join(bp.MODELS, 'velopark_kerbs.glb')])
                                        for n, m in bp.meshes(k) if 'GRASS' in n['name'][:7] and len(n['faces'])]))
    tiles, walls, new = {}, {}, []

    def plant(kinds, spots, sizes):
        count = 0
        for x, z in spots:
            g = ground.ys(np.array([[x, z], [x + .4, z], [x - .4, z], [x, z + .4], [x, z - .4]]))
            if np.isnan(g).all():
                continue
            y = float(np.nanmin(g))
            kind = kinds[rng.integers(len(kinds))]
            shape, r = shapes[kind]
            scale, yaw = float(rng.uniform(*sizes)), float(rng.uniform(0, 2 * np.pi))
            key = tuple(np.floor(np.array([x, z]) / bt.TILE).astype(int))
            for part in shape:
                tiles.setdefault(key, {}).setdefault(part, []).extend(bt.placed(shape[part], yaw, scale, np.array([x, y, z])))
            new.append([kind, float(x), y, float(z), yaw, scale])
            a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            ring = np.c_[np.cos(a), np.zeros(6), np.sin(a)] * r * scale * 1.1
            Vw = np.r_[ring + [x, y - .3, z], ring + [x, y + bt.TRUNK, z]]
            Fw = [(i, (i + 1) % 6, 6 + (i + 1) % 6) for i in range(6)] + [(i, 6 + (i + 1) % 6, 6 + i) for i in range(6)]
            walls.setdefault(key, []).append((Vw, np.r_[ring, ring] / np.linalg.norm(ring[0]), np.zeros((12, 2)), np.array(Fw)))
            count += 1
        return count

    n_woods = plant(euca, woods, (.9, 1.1))
    # young eucalyptus among them, a quarter as many
    Y = woods[rng.random(len(woods)) < .25]
    Y = Y + rng.normal(0, WOODS_APART / 2, Y.shape)
    n_young = plant(young, Y[~near[tuple(grid.cells(Y).T)]], (.85, 1.15))
    big = rng.random(len(P)) < BIG
    close = road_pts.query(P)[0] < NEAR
    n_broad = plant([n for n in made if n.startswith('broad_n')], P[~big & close], (.85, 1.15))
    n_big = plant([n for n in made if n.startswith('broad_big_n')], P[big & close], (.9, 1.1))
    n_broad += plant([n for n in made if n.startswith('broad_far_n')], P[~close], (.85, 1.2))
    n_palm = plant([n for n in made if n.startswith('palm_n')], palms, (.9, 1.1))
    n_bamboo = plant([n for n in made if n.startswith('bamboo_n')], bamboos, (.9, 1.1))
    # the understory under the new woods, as the plantation's: clumps of leaves on crossed cards
    clumps = 0
    if len(woods):
        C = woods[rng.integers(len(woods), size=len(woods) * 3)] + rng.normal(0, 2.5, (len(woods) * 3, 2))
        C = C[~near[tuple(grid.cells(C).T)]]
        for x, z in C:
            g = ground.y(x, z)
            if np.isnan(g):
                continue
            H = rng.uniform(3.5, 7.)
            yy = H * rng.uniform(0, 1) ** 1.2
            size = rng.uniform(1.6, 3.0) * (1 + .3 * yy / H)
            c = np.array([x, g + yy, z])
            key = tuple(np.floor(np.array([x, z]) / bt.TILE).astype(int))
            tiles.setdefault(key, {}).setdefault('shrub', []).extend(
                bt.cards(c + [0, size * .25, 0], size, size, rng.uniform(0, np.pi), rng.normal(0, .3), c - [0, size * .3, 0], .8))
            clumps += 1

    # the textures and materials the new kinds need
    have = {t['name'] for t in extra['textures']}
    for name, img, fmt in (('palm_fronds.dds', bt.palm_texture(), 'DXT5'), ('bamboo_leaves.dds', bt.bamboo_texture(), 'DXT5'),
                           ('bamboo_culm.dds', bt.culm_texture(), 'DXT1')):
        data = kn5io.dds(img, fmt)
        open(os.path.join(TEX, name), 'wb').write(data)
        if name not in have:
            extra['textures'].append(dict(name=name, active=1, data=data))
    mats = {m['name']: i for i, m in enumerate(extra['materials'])}
    like_at = extra['materials'][mats['EucaBroad']]
    like = extra['materials'][mats['EucaBark']]
    for part, tex, base in (('palm', 'palm_fronds.dds', like_at), ('bamboo', 'bamboo_leaves.dds', like_at), ('culm', 'bamboo_culm.dds', like)):
        name = 'Euca' + part.title()
        if name not in mats:
            extra['materials'].append(dict(base, name=name, props=[dict(p) for p in base['props']],
                                           textures=[dict(sampler='txDiffuse', slot=0, name=tex)]))
            mats[name] = len(extra['materials']) - 1
    part_mat = {'bark': mats['EucaBark'], 'leaves': mats['EucaLeaves'], 'shrub': mats['EucaShrub'], 'broad': mats['EucaBroad'],
                'palm': mats['EucaPalm'], 'bamboo': mats['EucaBamboo'], 'culm': mats['EucaCulm']}
    template = dict(kind=2, name='', active=1, children=[], material=0, cast_shadows=1, visible=1, transparent=0, layer=0,
                    lod_in=0., lod_out=0., renderable=1)
    root = extra['root']['children']
    meshes_n = tris = 0
    for key in sorted(tiles):
        for part, parts in tiles[key].items():
            for i, run in enumerate(bt.chunks(parts)):
                nd = bt.node(template, '%s%s_%d_%d_n%d' % (bt.PREFIX, part.upper(), key[0], key[1], i), run, part_mat[part])
                root.append(nd)
                meshes_n += 1
                tris += len(nd['faces'])
    for key in sorted(walls):
        for i, run in enumerate(bt.chunks(walls[key])):
            root.append(bt.node(template, '1WALL_%sTRUNKS_%d_%d_n%d' % (bt.PREFIX, key[0], key[1], i), run, mats['EucaBark'],
                                dict(renderable=0, cast_shadows=0)))
    for k, path in ((track, bp.TRACK), (extra, EXTRA)):
        bt.save(k, path)
    json.dump({'variants': {n: dict(v, radius=shapes[n][1]) for n, v in variants.items()}, 'trees': kept + new},
              open(bt.FOREST, 'w'), indent=0)
    print('%d eucalyptus (%d young) in the north woods, %d broad leaf, %d big, %d palms, %d bamboo, %d clumps: %d meshes, %d triangles' % (
        n_woods, n_young, n_broad, n_big, n_palm, n_bamboo, clumps, meshes_n, tris))


if __name__ == '__main__':
    prune() if '--prune' in sys.argv else strip() if '--strip' in sys.argv else main()
