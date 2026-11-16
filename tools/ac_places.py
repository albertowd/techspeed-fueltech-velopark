"""Assetto Corsa's own nodes for a layout, made by its build from the layouts' places (source/models/places.json,
written by tools/place_objects.py from objects.json): its grid slots, pit boxes, hotlap start and timing gates
(AC_START_0 and so on: an empty node where each stands, turned to its heading: all AC reads of one), its physics
objects (AC_POBJECT_n: its model's hull, its model under it), and the number of its
pit boxes in its ui_track.json.
"""
import json
import os
import re

import numpy as np

import build_paths as bp

PLACES = os.path.join(bp.MODELS, 'places.json')


def places():
    return json.load(open(PLACES, encoding='utf-8'))['layouts'] if os.path.exists(PLACES) else {}


def matrix(x, y, z, heading):
    import place_objects as po
    M = np.eye(4)
    M[:3, :3] = po.turned(heading)
    M[3, :3] = x, y, z
    return [float(v) for v in M.ravel()]


def add(k, here):
    """A layout's places put into its model k (kn5 data): its spots and gates, its physics objects."""
    import place_objects as po
    import templates
    for s in here.get('spots', []):
        k['root']['children'].append(dict(kind=1, name='AC_' + s['name'], active=1, matrix=matrix(s['x'], s['y'], s['z'], s['heading']),
                                          children=[]))
    cat = templates.catalogue()
    for i, o in enumerate(here.get('objects', [])):
        parts = templates.load(o['model'])
        lods = cat[o['model']].get('lods') or {}
        top = dict(kind=1, name='AC_POBJECT_%d' % i, active=1, matrix=matrix(o['x'], o['y'], o['z'], o['heading']), children=[])
        hull = next(((V, F, m) for p, V, F, m in parts if p == 'hull'), None)
        if hull is not None:
            top['children'].append(po.mesh(k, top['name'], hull[0], hull[1], hull[2], renderable=False))
        vis = [(p, V, F, m) for p, V, F, m in parts if p != 'hull']
        node = dict(kind=1, name=o['name'], active=1, matrix=[float(v) for v in np.eye(4).ravel()], children=[])
        for p, V, F, mat in vis:
            node['children'].append(po.mesh(k, po.part_name(o['name'], p, len(vis)), V, F, mat, lod=lods.get(p, (0., 0.))))
        top['children'].append(node)
        k['root']['children'].append(top)


def pitboxes(track_folder, layout, n):
    """The layout's pit boxes in the built track's ui_track.json (ui/<layout>/ui_track.json), n."""
    p = os.path.join(track_folder, 'ui', layout, 'ui_track.json')
    if not os.path.exists(p):
        return False
    text = open(p, 'rb').read().decode('utf-8-sig')
    new = re.sub(r'("pitboxes"\s*:\s*")\d+(")', lambda m: '%s%d%s' % (m.group(1), n, m.group(2)), text)
    if new != text:
        open(p, 'w', encoding='utf-8', newline='').write(new)
    return True
