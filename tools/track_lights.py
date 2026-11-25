"""The track's lights and screens, the same for every game (TRACK_FORMAT.md, "Lights and screens"): the light
series and the hand-placed lights and screens of source/lights.json, with those the build places: the catalogue
objects' lights where each object stands (places.json `lights`, from tools/place_objects.py) and the buildings'
lamps and screens (lights_outside.json, from tools/build_outside.py). Each game's build reads them through data().
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'source', 'lights.json')
MADE = os.path.join(ROOT, 'track', 'models')
PLACES = os.path.join(MADE, 'places.json')
OUTSIDE = os.path.join(MADE, 'lights_outside.json')


def _read(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}


def data():
    """{'series': {name: series}, 'lights': [{series, at, facing?}], 'screens': [screen]}: the source's, then the
    made ones', in that order (a series' lights in the order they come)."""
    src = _read(SOURCE)
    out = dict(series=dict(src.get('series', {})), lights=list(src.get('lights', [])), screens=list(src.get('screens', [])))
    for made in (_read(PLACES), _read(OUTSIDE)):
        out['lights'] += made.get('lights', [])
        out['screens'] += made.get('screens', [])
    return out


def write_outside(lights, screens):
    """The buildings' lamps and screens, as tools/build_outside.py places them, written where data() reads them.
    lights: [{series, at, facing?}]; screens: [screen]."""
    os.makedirs(MADE, exist_ok=True)
    json.dump(dict(version=1, lights=lights, screens=screens), open(OUTSIDE, 'w', encoding='utf-8'), indent=1)
