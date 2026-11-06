"""Races for the BeamNG.drive level: a time trial mission for each layout, which BeamNG's
freeroam shows as a marker at the layout's start. Driving into it offers the race (with a list
to pick from where layouts share a start); starting it puts the car on the grid, counts down and
times the laps.

Each race is a closed path of checkpoints along the layout's racing line, as BeamNG's own race
tracks have: the start and finish where AC times the lap (its AC_TIME_0 gate), a split at each
of AC's other timing gates, and a checkpoint every CHECKPOINT_STEP metres between them, so a lap
cannot be cut short. A checkpoint spans the road there, wherever on it the car runs. Each has a
place to put a car back on the road ("recover"). The standing start is the grid's first slot (the
hotlap's own start for a hotlap); the rolling start, which the race's settings offer, is AC's
hotlap start, where a flying lap begins.
"""
import json, os

import numpy as np
from PIL import Image

CHECKPOINT_STEP = 40.    # metres along the racing line between checkpoints
HALF_ROAD = 6.           # metres either side of the racing line a checkpoint reaches, at most
# (AC's side distances run on over the run-offs)
LAPS = 3
# the lap times for one to five stars, from the lap's length: metres per second on average
STAR_SPEEDS = (11, 13, 15, 17, 19)


def quat(fwd, up=(0, 0, 1.)):
    """The rotation BeamNG's race start positions take, as x, y, z, w: its y facing fwd."""
    y = np.asarray(fwd, float)
    y = y / np.linalg.norm(y)
    x = np.cross(y, up)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    m = np.c_[x, y, z]
    w = np.sqrt(max(0., 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    if w > 1e-6:
        q = ((m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w)
    else:
        # turned half round: about the axis the matrix leaves where it is
        k = int(np.argmax(np.diag(m)))
        axis = np.zeros(3)
        axis[k] = 1
        q = (*axis, 0.)
    return [float(v) for v in q]


def _floats(p):
    return [float(v) for v in p]


def race_path(line, sides, gates, standing, rolling, on_road, lift):
    """BeamNG's race path (race.race.json) for a racing line (n, 3) and its sides (n, 2) or None,
    in BeamNG's axes. gates: [(left, right)] AC's timing gates, the first the start and finish.
    standing, rolling: (position, forward) of the starts. Returns the path and its lap length."""
    n = len(line)
    step = np.linalg.norm(np.roll(line, -1, 0) - line, axis=1)
    s = np.r_[0, np.cumsum(step)]
    length = float(s[-1])

    def along(i):
        d = line[(i + 2) % n] - line[(i - 2) % n]
        d[2] = 0
        return d / np.linalg.norm(d)

    def across(i):
        """The road's middle and half width at a point of the line."""
        f = along(i)
        left = np.array([-f[1], f[0], 0])
        sl, sr = (np.minimum(sides[i], HALF_ROAD) if sides is not None else (HALF_ROAD / 2,) * 2)
        mid = line[i] + left * (sl - sr) / 2
        return on_road(mid), (sl + sr) / 2

    # the gates where the racing line crosses them; the start's first, and the line from it on
    picks = []
    for g, (l, r) in enumerate(gates):
        mid = (l + r) / 2
        i = int(np.linalg.norm(line[:, :2] - mid[:2], axis=1).argmin())
        picks.append((i, g, float(np.linalg.norm(l[:2] - r[:2]) / 2)))
    first = picks[0][0]
    pos = lambda i: (s[i] - s[first]) % length
    nodes = sorted(picks, key=lambda x: pos(x[0]))
    # checkpoints between them, at most CHECKPOINT_STEP apart
    full = []
    for k, (i, g, half) in enumerate(nodes):
        full.append((i, g, half))
        a = pos(i)
        b = pos(nodes[k + 1][0]) if k + 1 < len(nodes) else length
        extra = int(np.ceil((b - a) / CHECKPOINT_STEP)) - 1
        for e in range(1, extra + 1):
            at = (s[first] + a + (b - a) * e / (extra + 1)) % length
            full.append((int(np.searchsorted(s, at)) % n, None, None))

    ids = iter(range(1, 10000))
    pathnodes, starts, segments = [], [], []

    def start_position(name, group, at, fwd):
        sid = next(ids)
        starts.append({'group': group, 'name': name, 'oldId': sid, 'pos': _floats(on_road(at) + (0, 0, lift)),
                       'rot': quat(fwd)})
        return sid

    for k, (i, g, half) in enumerate(full):
        mid, width = across(i)
        if g is not None:
            # a gate of AC's: as wide as AC draws it
            radius = max(width, min(half, 2 * HALF_ROAD)) + 1
        else:
            radius = width + 1
        nid = next(ids)
        name = 'Start' if k == 0 else ('Split %d' % g if g is not None else 'Checkpoint %d' % k)
        rec = start_position('%s Recovery Forward' % name, 'recovery', line[(i - 4) % n], along((i - 4) % n))
        pathnodes.append({'customFields': {'names': {}, 'tags': {}, 'types': {}, 'values': {}}, 'mode': 'manual',
                          'name': name, 'navRadiusScale': 1, 'normal': _floats(along(i)), 'oldId': nid,
                          'pos': _floats(mid + (0, 0, lift)), 'radius': float(radius), 'recovery': rec,
                          'reverseRecovery': -1, 'sidePadding': [1, 1], 'useAsSplit': g is not None, 'visible': True})
    for k, p in enumerate(pathnodes):
        segments.append({'capsules': {}, 'from': p['oldId'], 'mode': 'waypoint', 'name': 'Segment %d' % (k + 1),
                         'oldId': next(ids), 'to': pathnodes[(k + 1) % len(pathnodes)]['oldId']})
    standing_id = start_position('Start Position fwd', 'start', *standing)
    rolling_id = start_position('Start Position fwd roll', 'start', *rolling)
    # where the car stops after the finish: a little on from the line
    after = int(np.searchsorted(s, (s[first] + 30) % length)) % n
    start_position('finalPosFwd', 'start', across(after)[0], along(after))
    path = {'authors': '', 'classification': {'allowRollingStart': True, 'branching': False, 'closed': True,
                                              'reversible': False},
            'date': 0, 'defaultLaps': LAPS, 'defaultStartPosition': standing_id, 'description': '', 'difficulty': 24,
            'endNode': -1, 'forwardPrefabs': {}, 'hideMission': False, 'name': 'Path', 'pacenotes': {},
            'pathnodes': pathnodes, 'prefabs': {}, 'reversePrefabs': {}, 'reverseStartPosition': -1,
            'rollingReverseStartPosition': -1, 'rollingStartPosition': rolling_id, 'segments': segments,
            'simplifyAiPath': False, 'startNode': pathnodes[0]['oldId'], 'startPositions': starts}
    return path, length


def mission(level, folder, title, description, author, trigger, length):
    """A time trial's info.json. trigger: (position, forward) of its marker."""
    times = [round(length / v) for v in STAR_SPEEDS]
    data = {'allowFlip': True, 'allowRecover': True, 'allowRollingStart': True, 'canChangeLapAmount': True,
            'closed': True, 'defaultLaps': LAPS, 'reversible': False, 'mapPreviewMode': 'navgraph',
            'flipLimit': -1, 'flipPenalty': 5, 'recoverLimit': -1, 'recoverPenalty': 5, 'justFinishPenalty': 30,
            'startScreenText': ''}
    data.update({'lapTime%d' % (k + 1): t for k, t in enumerate(times)})
    # the older medals, by the whole race's time
    for medal, t in (('bronze', times[0]), ('silver', times[2]), ('gold', times[4])):
        data.update({medal + 'Time': t * LAPS, medal + 'TimeTotal': t * LAPS, medal + 'TimePenalty': 0})
    return {
        'author': author, 'date': 0, 'devMission': False, 'name': title, 'description': description,
        'missionType': 'timeTrial', 'missionTypeData': data,
        'additionalAttributes': {'difficulty': 'medium'}, 'customAdditionalAttributes': {},
        'careerSetup': {'showInFreeroam': True, 'skill': '(none)', 'starOutroTexts': {}, 'starRewards': {},
                        'defaultStarKeys': ['lapTime%d' % k for k in range(1, 6)],
                        'starsActive': dict({'justFinish': False}, **{'lapTime%d' % k: True for k in range(1, 6)})},
        'grouping': {'id': '', 'label': ''},
        'layers': [{'dir': '/gameplay/missions/%s/' % folder, 'fixed': True, 'isMissionFolderDir': True,
                    'isMissionTypeDir': False},
                   {'dir': '/gameplay/missionTypes/timeTrial/', 'fixed': True, 'isMissionFolderDir': False,
                    'isMissionTypeDir': True}],
        'retryBehaviour': 'infiniteRetries',
        'setupModules': {'environment': {'enabled': False}, 'traffic': {'enabled': False},
                         'vehicles': {'enabled': True, 'includePlayerVehicle': True, 'playerVehicleDiffs': {},
                                      'prioritizePlayerVehicle': False, 'vehicles': {}}},
        'startCondition': {'type': 'automatic'},
        'startTrigger': {'level': level, 'type': 'coordinates', 'pos': _floats(trigger[0]), 'radius': 3.,
                         'rot': quat(trigger[1])},
        'visibleCondition': {'type': 'always'},
    }


def write(out, level, races):
    """The missions in out/gameplay/missions/<level>/timeTrial, made again each time. races: a
    list of dicts with layout, title, description, author, path, length, trigger and preview (an
    image file)."""
    root = os.path.join(out, 'gameplay', 'missions', level, 'timeTrial')
    if os.path.exists(root):
        for base, dirs, files in os.walk(root, topdown=False):
            for f in files:
                os.remove(os.path.join(base, f))
            for d in dirs:
                os.rmdir(os.path.join(base, d))
    for k, r in enumerate(races):
        folder = '%s/timeTrial/%03d-%s' % (level, k + 1, r['layout'])
        d = os.path.join(out, 'gameplay', 'missions', *folder.split('/'))
        os.makedirs(d, exist_ok=True)
        json.dump(mission(level, folder, r['title'], r['description'], r['author'], r['trigger'], r['length']),
                  open(os.path.join(d, 'info.json'), 'w'), indent=1)
        json.dump(r['path'], open(os.path.join(d, 'race.race.json'), 'w'), indent=1)
        img = Image.open(r['preview']).convert('RGB')
        img.save(os.path.join(d, 'preview.jpg'), quality=92)
        img.resize((200, round(200 * img.height / img.width)), Image.LANCZOS).save(os.path.join(d, 'thumbnail.jpg'), quality=90)
    return root
