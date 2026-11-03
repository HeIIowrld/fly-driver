"""Check original event identity and saved intervention summary arithmetic."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','replay','output'):
        p.add_argument('--'+key,required=True)
    a = p.parse_args()
    root, replay = Path(a.candidate)/'recording', Path(a.replay)
    result, manifest = read(replay/'result.json'), read(replay/'manifest.json')['config']
    path = replay/'events.json'; original = root/'events.jsonl'
    if (result['status'] != 'completed_fixed_multisensory_interventions'
            or sha256(path) != result['events_sha256']
            or sha256(original) != manifest['events_sha256']):
        raise ValueError('Unchanged original and saved replay required')
    rows = read(path)
    events = [json.loads(line) for line in original.read_text(encoding='utf8').splitlines()]
    if len(rows) != len(events) or len(rows) != result['decisions']:
        raise ValueError('Complete original decisions required')
    for i,(row,event) in enumerate(zip(rows,events,strict=True)):
        if (row['index'] != i or event['index'] != i or row['packet'] != event['native']['packet_id']
                or row['time'] != event['decision_monotonic_seconds']):
            raise ValueError('Original decision identity differs')
    x = np.asarray([row['intents'] for row in rows])
    if x.shape != (len(rows),4,2) or not np.isfinite(x).all():
        raise ValueError('All intervention intents required')
    names = ((1,'zero_projection'),(2,'yaw_zero'),(3,'history_missing')) if result.get(
        'historical_context_weights_reused_without_retraining') else (
            (1,'tilt_zero'),(2,'load_zero'),(3,'scene_features_zero'))
    for k,name in names:
        delta = np.abs(x[:,k]-x[:,0])
        np.testing.assert_allclose(delta.mean(0),result['effects'][name]['mean_abs'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(delta.max(0),result['effects'][name]['max_abs'],rtol=0,atol=1e-14)
    run = begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(original),
        replay_sha256=sha256(path),result_sha256=sha256(replay/'result.json'),scope=__doc__))
    out = dict(status='completed',decisions=len(rows),original_event_identity_match=True,
        summary_arithmetic_matches=True,independent_model_execution=False,
        physiology_or_driving_benefit_verified=False,fit_allowed=False)
    write_json(run/'result.json',out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
