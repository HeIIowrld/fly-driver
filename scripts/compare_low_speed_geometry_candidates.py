"""Compare two fixed geometry-port replays on the original native history."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'visual', 'mean11', 'last3'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    events = [row for row in map(json.loads, (args.probe / 'events.jsonl').read_text().splitlines())
              if row['kind'] == 'native']
    rows = json.loads((args.visual / 'port-rows.json').read_text())
    start = json.loads((args.mean11 / 'result.json').read_text())['original_start_index']
    with np.load(args.mean11 / 'intents.npz', allow_pickle=False) as mean, \
            np.load(args.last3 / 'intents.npz', allow_pickle=False) as last:
        if not np.array_equal(mean['clocks'], last['clocks']) or not np.array_equal(mean['ready'], last['ready']):
            raise ValueError('Original clocks and readiness must match')
        if len(events) != len(rows) or len(events) - start != len(mean['ready']):
            raise ValueError('Complete matching history required')
        if any(bool(row['ready']) != bool(ready) for row, ready in zip(rows[start:], mean['ready'], strict=True)):
            raise ValueError('Visual readiness differs from graph readiness')
        body = np.array([e['native']['fields']['localAngularVel'][1] for e in events[start:]])
        first = np.array([r['geometry_yaw'] if r['geometry_yaw'] is not None else np.nan for r in rows[start:]])
        second = last['geometry_yaw']
        mask = mean['ready'] & (np.abs(body) > .01) & np.isfinite(first) & np.isfinite(second)

        def score(values):
            return dict(match=int(np.sum(np.sign(values[mask]) == np.sign(body[mask]))),
                        count=int(mask.sum()), positive=int(np.sum(values[mask] > 0)),
                        negative=int(np.sum(values[mask] < 0)),
                        pearson=float(np.corrcoef(body[mask], values[mask])[0, 1]) if mask.sum() > 2 else None)

        result = dict(criterion='reference ready, abs(native localAngularVel[1]) > .01',
                      samples=int(mask.sum()), body_positive=int(np.sum(body[mask] > 0)),
                      body_negative=int(np.sum(body[mask] < 0)), mean11=score(first), last3=score(second),
                      zero_control_bitwise_equal=bool(np.array_equal(mean['intents'][1], last['intents'][1])),
                      first_intent_mean_abs_difference=float(np.abs(mean['intents'][0, :, 0] - last['intents'][0, :, 0]).mean()),
                      same_session_single_time_series=True, temporal_independence_not_assumed=True,
                      physical_timing_verified=False, actor_deployed=False, training_performed=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
