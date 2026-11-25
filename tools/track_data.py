"""The track's own settings in tools/reference/track.json (TRACK_FORMAT.md), as every game's build reads them: each
layout's models (`layouts`: {layout: {models: [name, ...]}}, by their names in track/models, in the order a game
loads them), and the source's parts (roads, objects, terrain, trees).

The track is the one tools/build_paths.py points at (its ROOT, as the road editor sets it for a track of its own),
worked out at each call.
"""
import json
import os


def root():
    import build_paths
    return build_paths.ROOT


def track_path():
    return os.path.join(root(), 'tools', 'reference', 'track.json')


def track():
    return json.load(open(track_path(), encoding='utf-8'))


def layouts():
    """{layout: [model name, ...]}, as track.json lists them."""
    return {lay: list(v.get('models') or []) for lay, v in (track().get('layouts') or {}).items()}


def ac_models(models):
    """A layout's model list as AC reads it (models_<layout>.ini): each model a [MODEL_n] of its .kn5, where it is."""
    return '\r\n'.join('[MODEL_%d]\r\nFILE=%s.kn5\r\nPOSITION=0,0,0\r\nROTATION=0,0,0\r\n' % (i, m) for i, m in enumerate(models))


# ---- the source's parts (TRACK_FORMAT.md, "Moving there", 7): each in track.json, or its own file while it is not:
# track.json's key holds the part itself, or the path of its file (relative to the track's root, as its settings' paths
# are); a part's own paths are relative to the root when it is in track.json, to its file's folder when in a file
def parts():
    ref = os.path.dirname(track_path())
    return {'roads': os.path.join(ref, 'roads.json'), 'objects': os.path.join(ref, 'objects.json'),
            'terrain': os.path.join(ref, 'terrain', 'terrain.json'), 'trees': os.path.join(ref, 'trees.json')}


def _where(name):
    t = track() if os.path.exists(track_path()) else {}
    v = t.get(name)
    if isinstance(v, str):
        return None, os.path.join(root(), v)
    if v is not None:
        return v, None
    return None, parts()[name]


def part(name):
    """A part of the track's source and the folder its paths are relative to: (data, base), or (None, None) if the
    track has none. roads: {'roads': [...]}; objects: objects.json's body; terrain: terrain.json's description;
    trees: {'trees': [...]}."""
    data, path = _where(name)
    if data is not None:
        if name in ('roads', 'trees') and isinstance(data, list):
            data = {name: data}
        return data, root()
    if os.path.exists(path):
        return json.load(open(path, encoding='utf-8')), os.path.dirname(path)
    return None, None


def files(name):
    """The files a part is read from (for a build to know when it changed): track.json, and its own file."""
    _, path = _where(name)
    return [f for f in (track_path(), path) if f and os.path.exists(f)]
