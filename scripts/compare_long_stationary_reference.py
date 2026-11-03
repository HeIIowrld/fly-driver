"""Compare fixed original and longer stationary references on two diagnostic histories."""
import hashlib
import json
from pathlib import Path

import numpy as np


BASE = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
CASES = (
    ('fast', '20261101-fast-body-fourview-probe', '20261101-fast-body-fourview-graph-sensory-neutral',
     '20261101-fast-body-fourview-sensory-long-reference'),
    ('reverse', '20261101-low-speed-fourview-prefix-reverse-probe',
     '20261101-low-speed-fourview-prefix-reverse-graph-evolving', '20261101-reverse-fourview-sensory-long-reference'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(probe, old, new):
    event_file = BASE/probe/'events.jsonl'
    rows = list(map(json.loads, event_file.read_text(encoding='utf8').splitlines()))
    events = [row for row in rows if row['kind'] == 'native']
    pre_throttle = 0
    for row in rows:
        if row['kind'] == 'request' and row['request']['requested_control']['throttle'] > 0:
            break
        pre_throttle += row['kind'] == 'native'
    paths = [BASE/old/'intents.npz', BASE/new/'intents.npz']
    receipts = [json.loads((p.parent/'result.json').read_text(encoding='utf8')) for p in paths]
    manifests = []
    for p, r in zip(paths, receipts, strict=True):
        if sha(p) != r['intents_sha256'] or r['original_start_index'] != 2:
            raise ValueError('Original arrays or episode start changed')
        manifest = json.loads((p.parent/'manifest.json').read_text(encoding='utf8'))['config']
        if manifest['source_events_sha256'] != sha(event_file):
            raise ValueError('Both replays must use the same original events')
        manifests.append(manifest)
    with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as c:
        if not np.array_equal(a['clocks'], c['clocks']) or len(events)-2 != len(a['clocks']):
            raise ValueError('Original decision clocks and complete source required')
        events = events[2:]
        native_yaw = np.array([e['native']['fields']['localAngularVel'][1] for e in events])
        prior = np.arange(len(events)) < pre_throttle-2
        common = a['ready'] & c['ready']
        unique = np.zeros(len(events), dtype=bool)
        seen = set()
        for i, event in enumerate(events):
            if not common[i] or event.get('camera') is None:
                continue
            capture = event['camera']['capture_id']
            if capture not in seen:
                seen.add(capture)
                unique[i] = True

        def stats(mask):
            x, y, body = a['geometry_yaw'][mask], c['geometry_yaw'][mask], native_yaw[mask]
            return dict(count=int(mask.sum()), original_mean_abs=float(np.abs(x).mean()) if len(x) else None,
                        long_mean_abs=float(np.abs(y).mean()) if len(y) else None,
                        original_matches=int((np.sign(x) == np.sign(body)).sum()),
                        long_matches=int((np.sign(y) == np.sign(body)).sum()))

        return dict(events_sha256=sha(event_file), original_intents_sha256=sha(paths[0]), long_intents_sha256=sha(paths[1]),
                    original_ready=int(a['ready'].sum()), long_ready=int(c['ready'].sum()),
                    original_first_ready=int(np.flatnonzero(a['ready'])[0])+2 if a['ready'].any() else None,
                    long_first_ready=int(np.flatnonzero(c['ready'])[0])+2 if c['ready'].any() else None,
                    common_prethrottle=stats(common & prior), unique_prethrottle=stats(unique & prior),
                    common_turn=stats(common & (np.abs(native_yaw) > .01)),
                    unique_turn=stats(unique & (np.abs(native_yaw) > .01)),
                    zero_control_equal=bool(np.array_equal(a['intents'][1], c['intents'][1])),
                    zero_control_interpretable=manifests[0]['command'] == manifests[1]['command'])


def main():
    result = dict(status='completed_reference_comparison', cases={name: compare(p, a, c) for name, p, a, c in CASES},
                  interpretation='Prethrottle and native yaw are diagnostic observations, not optimal steering labels; '
                  'two runs share one session; reverse baseline used a different speed command, so only geometry is comparable.',
                  training_performed=False, vehicle_control_sent=False, physiology_verified=False)
    output = BASE/'20261101-long-stationary-reference-comparison'
    output.mkdir(exist_ok=False)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
