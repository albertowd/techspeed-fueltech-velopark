"""Parked rental karts round the pits, as objects of tools/reference/objects.json (the catalogue's kart).

    python tools/place_karts.py

The Techspeed rental karts (the catalogue's kart: the kart project's third level of detail, the farthest
with no rims): parked in the pit building's workshop and under the white tent north of it, the spare
rentals (screenshots in tools/reference/site). None stands within CLEAR of a layout's pit box or start, where a player's
kart may stand. Each is drawn within LOD_OUT metres (the catalogue's draw), and has nothing to hit. The objects
step lays them on the floor, the workshop's and the tent's. Run again, it writes them again.
"""
import math, os, sys

import numpy as np

import build_outside as bo
import build_paths as bp

ROOT = bp.ROOT
OBJECTS = os.path.join(ROOT, 'tools', 'reference', 'objects.json')
CLEAR = 2.5              # metres from a pit box or a start a kart may not stand
LOD_OUT = 60.            # metres from which a kart is drawn


def spots():
    """Where each kart stands: (x, z, heading: the way its nose points, radians from +z to +x)."""
    c = np.array(bo.PIT_BUILDING[0], float)
    ez = np.array(bo.PIT_BUILDING[1], float)
    ez /= np.linalg.norm(ez)
    ex = np.array([ez[1], -ez[0]])
    along = math.atan2(ez[0], ez[1])
    out = []
    # the workshop: two rows down its middle, between the doorways and either side of them
    for x in (-1.7, 1.7):
        for z in (-10., -2.2, 2.2, 10.):
            p = c + ex * x + ez * z
            out.append((p[0], p[1], along + (math.pi if x < 0 else 0.), 'workshop'))
    # the tent: two rows of four, noses south towards the pits
    tc = np.array(bo.tent_on_floor()[0], float)
    tz = np.array(bo.tent_on_floor()[2], float)
    tz /= np.linalg.norm(tz)
    tx = np.array([tz[1], -tz[0]])
    for x in (-3.6, -1.2, 1.2, 3.6):
        for z in (-1.9, 1.9):
            p = tc + tx * x + tz * z
            out.append((p[0], p[1], math.atan2(-tz[0], -tz[1]), 'tent'))
    return out


def main():
    """The karts into objects.json, through the road editor's own objects store (its format, a backup of the file
    before): each one KART_n, the catalogue's kart, no layouts named (so all have it, in the shared model), nothing
    to hit; those standing within CLEAR of a layout's pit box or start left out."""
    import json
    from pathlib import Path
    sys.path.insert(0, os.path.join(os.path.dirname(ROOT), 'top-race-editor', 'engine'))
    from top_race_engine.objects import ObjectsStore
    store = ObjectsStore(Path(OBJECTS), Path(ROOT, 'build', 'cache', 'roads_backup'), seed=lambda: {})
    data, _ = store.load()
    busy = np.array([(s['x'], s['z']) for L in data['layouts'].values()
                     for s in (L.get('grid') or []) + (L.get('pits') or []) + ([L['hotlap']] if L.get('hotlap') else [])], float)
    karts = []
    for x, z, heading, where in spots():
        if len(busy) and np.min(np.hypot(busy[:, 0] - x, busy[:, 1] - z)) < CLEAR:
            continue
        karts.append(dict(name='KART_%d' % len(karts), model='kart', curve=[{'p': [round(float(x), 3), round(float(z), 3)],
                                                                             'in': [0., 0.], 'out': [0., 0.]}],
                          heading=round(float(heading), 4), collision='none'))
    data['objects'] = [o for o in data['objects'] if not o['name'].startswith('KART_')] + karts
    store.save(data)
    print('%d karts parked as objects (%s)' % (len(karts), json.dumps(sorted({w for *_, w in spots()}))))


if __name__ == '__main__':
    main()
