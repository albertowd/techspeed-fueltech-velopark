"""The road editor: the roads described in tools/reference/roads.json (see ROADS.md) over the satellite
picture, to edit in the browser, and saved back.

    python tools/road_editor.py [--port 8765] [--no-browser] [--track tools/reference/track.json]

Then http://127.0.0.1:8765 (opened by itself). The page (tools/road_editor/) shows the satellite
picture (source/textures/GoogleMap.jpg), the track as its models have it now (source/models: roads,
kerbs, paint, tyre walls) and the roads described, each as its middle (a curve of Bezier pieces: nodes
and their handles), its edges, its edge lines and kerbs, and its mesh as tools/roads.py makes it (the
wireframe). A road described by its points is shown as a curve fitted to them (fit), and saved as that
curve only once changed. Saving writes roads.json (the one before it kept in build/cache/roads_backup/):
a road whose shape changed loses its heights, and the next build records them again
(tools/build_roads.py).
"""
import hashlib
import http.server
import io
import json
import os
import sys
import threading
import time
import webbrowser
from urllib.parse import urlparse, parse_qs

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_paths as bp
import gltf_source
import roads as rd

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'road_editor')
# (the track edited: tools/reference/track.json, or the file given with --track; its paths from the project's root)
TRACK_FILE = sys.argv[sys.argv.index('--track') + 1] if '--track' in sys.argv else os.path.join(bp.ROOT, 'tools', 'reference', 'track.json')
TRACK = json.load(open(TRACK_FILE)) if os.path.exists(TRACK_FILE) else {}
_at = lambda key, default: os.path.join(bp.ROOT, TRACK[key]) if key in TRACK else default
SATELLITE = _at('satellite', os.path.join(bp.ROOT, 'source', 'textures', 'GoogleMap.jpg'))
DRAWINGS = _at('drawings', os.path.join(bp.ROOT, 'tools', 'reference', 'road_drawings.json'))
rd.ROADS = _at('roads', rd.ROADS)
MODELS = _at('models', bp.MODELS)
# (where the roads not described yet are read from: the models as they were before their floors were merged
# into one surface (tools/merge_floor.py keeps them), each where it has one, else as they are)
MESHES_FROM = _at('meshes_from', os.path.join(bp.ROOT, 'build', 'cache', 'floors_before_merge'))


def shapes_of(path):
    """A model's meshes as kn5 data (gltf_source's), its textures not read: a copy kept away from them."""
    doc, binary = gltf_source.read_glb(path)
    return gltf_source.from_gltf(doc, binary, lambda uri: b'', lambda name: b'')


def mesh_model(f):
    p = os.path.join(MESHES_FROM, f)
    return p if os.path.exists(p) else os.path.join(MODELS, f)
CONTEXT = os.path.join(bp.ROOT, 'build', 'cache', 'editor_context.json')
BACKUPS = os.path.join(bp.ROOT, 'build', 'cache', 'roads_backup')
SPAN = float(TRACK.get('satellite_span', 1060.))   # metres the satellite picture spans (its pixels across: read from it)
PX = 8192
TILE = 512                      # pixels a tile
FIT = .05                       # metres a fitted curve keeps to the road's middle (and widths) within
GEOMETRY = ('curve', 'points', 'start_left', 'end_left')
SIMPLE = .02                    # metres the track's outlines (shown under the roads) are kept within
CONTEXT_MODELS = (('velopark.glb', ('1ROAD', '1KERB', '1WALL'), 'road'), ('velopark_runoffs.glb', ('1ROAD',), 'runoff'),
                  ('velopark_kerbs.glb', ('1ROAD_KERB', 'KERB', '1KERB'), 'kerb'), ('velopark_lines.glb', ('LINE_', 'GRID_'), 'line'),
                  ('velopark_tyres.glb', ('TYRES', '1WALL_TYRE'), 'tyre'), ('velopark_paddock.glb', ('1ROAD', 'BLD', '1WALL'), 'paddock'))


# ---------------------------------------------------------------- curves

def fit_curve(road):
    """A road's middle as a curve of Bezier pieces (see roads.py): nodes where needed so it keeps within
    FIT of the road's middle and widths; each node's handles along the road's way there, a third of the
    way to the next node each."""
    C, s, _ = rd.centre(road)
    keep = np.r_[True, np.diff(s) > 1e-9]
    C, s = C[keep], s[keep]
    T = np.gradient(C[:, :2], s, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)

    def nodes_of(idx):
        out = []
        for j, i in enumerate(idx):
            d_in = s[i] - s[idx[j - 1]] if j else 0.
            d_out = s[idx[j + 1]] - s[i] if j < len(idx) - 1 else 0.
            out.append({'p': [round(float(C[i, 0]), 4), round(float(C[i, 1]), 4)],
                        'in': [round(float(-T[i, 0] * d_in / 3), 4), round(float(-T[i, 1] * d_in / 3), 4)],
                        'out': [round(float(T[i, 0] * d_out / 3), 4), round(float(T[i, 1] * d_out / 3), 4)],
                        'left': round(float(C[i, 2]), 4), 'right': round(float(C[i, 3]), 4)})
        return out

    from scipy.spatial import cKDTree
    idx = [0, len(C) - 1]
    for _ in range(400):
        nodes = nodes_of(idx)
        B, at = rd.bezier(nodes, .05)
        tree = cKDTree(B[:, :2])
        d, j = tree.query(C[:, :2])
        err = np.maximum(d, np.maximum(np.abs(B[j, 2] - C[:, 2]), np.abs(B[j, 3] - C[:, 3])))
        worst, k = 0., None
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a < 3:
                continue
            m = a + 1 + int(np.argmax(err[a + 1:b]))
            if err[m] > FIT and err[m] > worst:
                worst, k = err[m], (a, b)
        if k is None:
            break
        a, b = k
        mid = int(np.searchsorted(s, (s[a] + s[b]) / 2))     # (split where it is worst: at the middle of that piece)
        idx = sorted(set(idx + [min(max(mid, a + 1), b - 1)]))
    return nodes_of(idx)


def road_meshes():
    """The roads in the track's models not described yet (in roads.json neither by name nor as a road's
    mesh): [{model, name, material, area, box}], the biggest first. Its road models: track.json's
    road_models, or every model with 1ROAD_ meshes in it."""
    import surface as sf
    have = set()
    for r in (json.load(open(rd.ROADS))['roads'] if os.path.exists(rd.ROADS) else []):
        have |= {r['name'], r.get('mesh')}
        have |= {r['name'][:-1]} if r['name'].endswith('B') else set()        # (a ring's second half: the ring)
    files = TRACK.get('road_models') or sorted(os.path.basename(f) for f in __import__('glob').glob(os.path.join(MODELS, '*.glb')))
    out = []
    for f in files:
        path = mesh_model(f)
        if not os.path.exists(path) or os.path.basename(f).startswith('vp_'):
            continue
        k = shapes_of(path)
        for n, m in bp.meshes(k):
            nm = n['name']
            if not nm.startswith('1ROAD_') or nm.startswith(('1ROAD_KERB', '1ROAD_PATCH')) or nm in have or not len(n['faces']):
                continue
            T = bp.triangles(n, m)
            c = T[:, :, [0, 2]]
            a = np.abs((c[:, 1, 0] - c[:, 0, 0]) * (c[:, 2, 1] - c[:, 0, 1]) - (c[:, 2, 0] - c[:, 0, 0]) * (c[:, 1, 1] - c[:, 0, 1])).sum() / 2
            if a < 1.:
                continue
            lo, hi = c.min((0, 1)), c.max((0, 1))
            out.append({'model': f, 'name': nm, 'material': k['materials'][n['material']]['name'], 'area': round(float(a), 1),
                        'box': [round(float(v), 2) for v in (lo[0], lo[1], hi[0], hi[1])]})
    return sorted(out, key=lambda x: -x['area'])


def bring_in(model, name):
    """A road mesh of the track's described (tools/read_road.py: by its rows, as a strip, or a ring in
    two halves joined end to end), each as a curve fitted to it: roads as the page has them, not saved."""
    import read_road as rr
    k = shapes_of(mesh_model(model))
    n, m = next((n, m) for n, m in bp.meshes(k) if n['name'] == name and len(n['faces']))
    roads, how = rr.describe_any(n, m, k['materials'][n['material']]['name'])
    out = []
    for r in roads:
        P = np.array(r['points'], float)
        # (a strip read from an outline flares where it meets another road, its mouth: as wide as it is along)
        med = np.median(P[:, 2] + P[:, 3])
        for c in (2, 3):
            P[:, c] = np.minimum(P[:, c], med * .65)
        r['points'] = P.round(3).tolist()
        q = {k_: v for k_, v in r.items() if k_ != 'points'}
        q['curve'] = fit_curve(r)
        q.update(lines=[], kerbs=[])
        out.append(q)
    if how == 'ring':
        a, b = out
        a['curve'][0]['joint'] = {'road': b['name'], 'node': 'end'}
        a['curve'][-1]['joint'] = {'road': b['name'], 'node': 'start'}
        b['curve'][0]['joint'] = {'road': a['name'], 'node': 'end'}
        b['curve'][-1]['joint'] = {'road': a['name'], 'node': 'start'}
        b['curve'][-1]['p'] = list(a['curve'][0]['p'])
        b['curve'][0]['p'] = list(a['curve'][-1]['p'])
    return {'roads': out, 'how': how}


def geometry_key(road):
    return hashlib.sha1(json.dumps({k: road.get(k) for k in GEOMETRY}, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- what the page asks for

def roads_for_page():
    """The roads, each with its curve (fitted where it has points: 'fitted'), no heights (kept here)."""
    out = []
    for r in (json.load(open(rd.ROADS))['roads'] if os.path.exists(rd.ROADS) else []):
        q = {k: v for k, v in r.items() if k not in ('heights', 'points')}
        if 'along' not in r and not r.get('curve'):
            q['curve'] = fit_curve(r)
            q['fitted'] = True
        q['shape'] = geometry_key(r)
        out.append(q)
    return out


def save(page_roads, deleted=()):
    """The page's roads written to roads.json: a road the page did not change in shape as it was (its
    points, its heights); one it did, its curve, no heights (the next build records them). A road in the
    file the page did not have (added since it opened) kept, unless the page deleted it."""
    old = {r['name']: r for r in (json.load(open(rd.ROADS))['roads'] if os.path.exists(rd.ROADS) else [])}
    os.makedirs(BACKUPS, exist_ok=True)
    if os.path.exists(rd.ROADS):
        with open(os.path.join(BACKUPS, time.strftime('roads-%Y%m%d-%H%M%S.json')), 'w') as f:
            f.write(open(rd.ROADS).read())
    out, changed = [], []
    for q in page_roads:
        q = dict(q)
        moved = q.pop('moved', False)
        q.pop('fitted', None)
        q.pop('shape', None)
        was = old.get(q.get('was') or q['name'])
        if was is not None and q['name'] != was['name']:
            q['mesh'] = was.get('mesh', was['name'])             # (renamed: the mesh it takes the place of kept)
        if was is not None and not moved:
            for k in GEOMETRY + ('heights',):
                q.pop(k, None)
                if k in was:
                    q[k] = was[k]
        else:
            q.pop('points', None)
            q.pop('heights', None)
            changed.append(q['name'])
        if 'along' in q:
            q.pop('curve', None)
        q.pop('was', None)
        out.append(q)
    have = {q.get('was') or q['name'] for q in page_roads} | {q['name'] for q in page_roads}
    out += [r for n, r in old.items() if n not in have and n not in set(deleted)]
    with open(rd.ROADS, 'w') as f:
        json.dump({'roads': out}, f, indent=1)
    return changed


def mesh_for_page(road, roads):
    """A road's mesh as tools/build_roads.py makes it: tools/roads.py's, cut where it lies under the roads
    above it (the page's order); its points (x, z) and faces."""
    from shapely.ops import unary_union
    road = dict(road)
    road.pop('heights', None)
    by = {r['name']: dict(r, heights=None) for r in roads}
    by[road['name']] = road
    P, U, F = rd.mesh(rd.along_road(road, by) if 'along' in road else road, lambda x, z: np.zeros(len(np.atleast_1d(x))))
    ups = rd.above(road, by) if road['name'] in by else []
    if ups:
        import restyle_paddock as rp
        mine = rd.outline(road, by)
        over = unary_union([rd.outline(by[x], by) for x in ups]).intersection(mine.buffer(.5))   # (only near it: quick)
        if over.intersection(mine).area > 1e-3:
            node = {'verts': np.c_[P, np.zeros((len(P), 5))].astype('<f4'), 'faces': F.astype('<u4')}
            rp.clip_in(node, over.buffer(1e-5))
            V, F = node['verts'].astype(float), node['faces'].astype(int)
            used = np.unique(F)
            remap = np.full(len(V), -1)
            remap[used] = np.arange(len(used))
            P, F = V[used, :3], remap[F]
    return {'P': np.round(P[:, [0, 2]], 3).ravel().tolist(), 'F': F.ravel().tolist(), 'under': ups}


def surface_for_page(page_roads, window):
    """The surface in a window of the map as the build makes it (tools/track_surface.py, the page's roads
    in its order), checked: each region's faces (x, z) and kind, its problems. Level (flat): the seams are
    what it is for."""
    import track_surface as ts
    t = time.time()
    by = {r['name']: dict(r, heights=None) for r in page_roads}
    x0, z0, x1, z1 = [float(v) for v in window]
    try:
        meshes, kinds, issues = ts.build(by, (x0, z0, x1, z1), flat=True)
    except Exception as e:
        return {'error': str(e)}
    out = []
    for n, (P, UV, F) in meshes.items():
        out.append({'name': n, 'kind': kinds.get(n, 'road'), 'P': np.round(P[:, [0, 2]], 4).ravel().tolist(), 'F': F.ravel().tolist()})
    return {'regions': out, 'issues': [[w, round(float(x), 3), round(float(z), 3), round(float(h), 4)] for w, x, z, h in issues],
            'window': [x0, z0, x1, z1], 'seconds': round(time.time() - t, 1)}


def context():
    """The track as its models have it now, for the page to show under the roads, each mesh as: its
    outline (the edges of its faces seen from above that no other of its faces shares: 'segs', x z x z
    each), or, for paint and kerbs, its faces seen from above ('tris', x z x z x z each), or, for tyre
    walls, a circle a stack ('rings', x z r each). Kept (CONTEXT) until the models change."""
    files = [(os.path.join(MODELS, f), tuple(pre), kind) for f, pre, kind in (TRACK.get('context_models') or CONTEXT_MODELS)]
    stamp = [os.path.getmtime(f) if os.path.exists(f) else 0 for f, _, _ in files]
    if os.path.exists(CONTEXT):
        have = json.load(open(CONTEXT))
        if have.get('stamp') == stamp and have.get('version') == 3:
            return have
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    items = []
    for f, pre, kind in files:
        if not os.path.exists(f):
            continue
        k = gltf_source.load_kn5(f)
        mats = [m['name'] for m in k['materials']]
        for n, m in bp.meshes(k):
            if not len(n['faces']) or not n['name'].startswith(pre):
                continue
            T = bp.triangles(n, m)[:, :, [0, 2]]
            a = np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])) / 2
            T = T[a > 1e-5]                                  # (seen from above: no wall faces)
            if not len(T):
                continue
            it = {'name': n['name'], 'kind': kind, 'material': mats[n['material']]}
            if kind == 'tyre':
                # (each stack: the points joined by faces, its middle and how far out)
                V = T.reshape(-1, 2)
                key = np.round(V / .005).astype(np.int64)
                _, idx = np.unique(key[:, 0] * 10 ** 7 + key[:, 1], return_inverse=True)
                F = idx.reshape(-1, 3)
                e = np.r_[F[:, [0, 1]], F[:, [1, 2]]]
                nv = idx.max() + 1
                _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
                P = np.zeros((nv, 2))
                P[idx] = V
                out = []
                for c in np.unique(lab):
                    Q = P[lab == c]
                    mid = (Q.min(0) + Q.max(0)) / 2
                    out += [round(float(mid[0]), 2), round(float(mid[1]), 2), round(float(np.linalg.norm(Q - mid, axis=1).max()), 2)]
                it['rings'] = out
            else:
                V = T.reshape(-1, 2)
                key = np.round(V / .005).astype(np.int64)
                _, idx = np.unique(key[:, 0] * 10 ** 7 + key[:, 1], return_inverse=True)
                F = idx.reshape(-1, 3)
                E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
                Eu, cnt = np.unique(E, axis=0, return_counts=True)
                P = np.zeros((idx.max() + 1, 2))
                P[idx] = V
                S_ = Eu[cnt == 1]
                if not len(S_):
                    continue
                # (its outline's edges joined into lines, each kept within SIMPLE of itself)
                from shapely.geometry import MultiLineString
                from shapely.ops import linemerge
                merged = linemerge(MultiLineString([[tuple(P[a]), tuple(P[b])] for a, b in S_]))
                it['lines'] = [np.round(np.asarray(l.simplify(SIMPLE).coords), 2).ravel().tolist()
                               for l in getattr(merged, 'geoms', [merged]) if l.length > .05]
            items.append(it)
        print('  the track now: %s' % os.path.basename(f), flush=True)
    out = {'stamp': stamp, 'version': 3, 'items': items}
    os.makedirs(os.path.dirname(CONTEXT), exist_ok=True)
    json.dump(out, open(CONTEXT, 'w'))
    return out


class Satellite:
    """The satellite picture in tiles of TILE pixels, at levels each half the one before (0 the picture's own)."""
    def __init__(self):
        self.levels, self.cache, self.lock = {}, {}, threading.Lock()

    def level(self, L):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with self.lock:
            if 0 not in self.levels:
                self.levels[0] = Image.open(SATELLITE).convert('RGB')
            if L not in self.levels:
                im = self.levels[0]
                self.levels[L] = im.resize((im.width >> L, im.height >> L), Image.LANCZOS)
            return self.levels[L]

    def tile(self, L, x, y):
        key = (L, x, y)
        if key not in self.cache:
            im = self.level(L)
            box = (x * TILE, y * TILE, min((x + 1) * TILE, im.width), min((y + 1) * TILE, im.height))
            if box[0] >= im.width or box[1] >= im.height or x < 0 or y < 0:
                return None
            b = io.BytesIO()
            im.crop(box).save(b, 'JPEG', quality=88)
            if len(self.cache) > 600:
                self.cache.clear()
            self.cache[key] = b.getvalue()
        return self.cache[key]


SAT = Satellite()
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8'}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, kind='application/json', code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store' if kind != 'image/jpeg' else 'max-age=86400')
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass                                        # (the page went away first: a tile no longer wanted)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                return self.send(open(os.path.join(HERE, 'index.html'), 'rb').read(), TYPES['.html'])
            if u.path.startswith('/static/'):
                f = os.path.join(HERE, os.path.basename(u.path))
                return self.send(open(f, 'rb').read(), TYPES.get(os.path.splitext(f)[1], 'application/octet-stream'))
            if u.path == '/api/roads':
                return self.send({'roads': roads_for_page(), 'span': SPAN, 'px': PX, 'tile': TILE,
                                  'materials': ['Tarmac Dark', 'Tarmac Light', 'Pavers', 'Concrete']})
            if u.path == '/api/meshes':
                return self.send({'meshes': road_meshes()})
            if u.path == '/api/context':
                return self.send(context())
            if u.path == '/api/drawings':
                return self.send(json.load(open(DRAWINGS)) if os.path.exists(DRAWINGS) else {})
            if u.path == '/api/tile':
                t = SAT.tile(int(q['l']), int(q['x']), int(q['y']))
                return self.send(t, 'image/jpeg') if t else self.send(b'', 'text/plain', 404)
            self.send({'error': 'not found'}, code=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send({'error': str(e)}, code=500)

    def do_POST(self):
        u = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        try:
            if u.path == '/api/save':
                changed = save(body['roads'], body.get('deleted', []))
                print('saved %d roads%s' % (len(body['roads']), ' (changed in shape: %s)' % ', '.join(changed) if changed else ''))
                return self.send({'ok': True, 'changed': changed, 'roads': roads_for_page()})
            if u.path == '/api/mesh':
                return self.send(mesh_for_page(body['road'], body.get('roads', [])))
            if u.path == '/api/surface':
                return self.send(surface_for_page(body['roads'], body['window']))
            if u.path == '/api/bring':
                return self.send(bring_in(body['model'], body['name']))
            if u.path == '/api/fit':
                return self.send({'curve': fit_curve(body['road'])})
            self.send({'error': 'not found'}, code=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send({'error': str(e)}, code=500)


def main():
    global PX
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        PX = Image.open(SATELLITE).size[0]
    except Exception:
        pass
    print('track: %s (%s)' % (TRACK.get('name', '?'), TRACK_FILE))
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8765
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = 'http://127.0.0.1:%d/' % port
    print('road editor at %s (Ctrl+C to stop)' % url)
    if '--no-browser' not in sys.argv:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
