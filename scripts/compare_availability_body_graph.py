"""Compare two fixed graph replays differing only in the audited body proxy."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('front', 'availability', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    roots = [Path(a.front), Path(a.availability)]
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    results = [read(root/'result.json') for root in roots]
    manifests = [read(root/'manifest.json')['config'] for root in roots]
    for root, result in zip(roots, results, strict=True):
        if (result['status'] != 'completed_offline_signed_graph_intervention'
                or result['training_admissible'] is not False
                or sha256(root/'graph-intervention.npz') != result['arrays_sha256']):
            raise ValueError('Two exact diagnostic graph outputs required')
    for key in ('projection_sha256', 'replay_sha256', 'probe_sha256',
                'joint_checkpoint_sha256', 'temporal_checkpoint_sha256', 'task_goal', 'conditions'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Common graph input lineage differs: '+key)
    if (manifests[1]['body'] != 'group availability prior receipt'
            or results[1]['computation_dtype'] != 'float64'
            or results[0].get('computation_dtype', 'float64') != 'float64'):
        raise ValueError('Float64 front/availability diagnostic pair required')
    with np.load(roots[0]/'graph-intervention.npz') as first, np.load(roots[1]/'graph-intervention.npz') as second:
        np.testing.assert_array_equal(first['group_clock'], second['group_clock'])
        intent = second['intent']-first['intent']
        muscles = second['muscles']-first['muscles']
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'front_output_sha256': results[0]['arrays_sha256'],
        'availability_output_sha256': results[1]['arrays_sha256'],
        'training_admissible': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'differences.npz', intent=intent, muscles=muscles)
    out = {'status': 'completed_same_recording_body_proxy_comparison',
        'groups': len(intent), 'baseline_intent_max_abs': np.max(np.abs(intent[:, 0]), axis=0).tolist(),
        'baseline_intent_mean_abs': np.mean(np.abs(intent[:, 0]), axis=0).tolist(),
        'baseline_muscle_max_abs': float(np.max(np.abs(muscles[:, 0]))),
        'baseline_changed_groups': int(np.count_nonzero(np.any(intent[:, 0] != 0, axis=1))),
        'source_physical_time_verified': False, 'training_admissible': False,
        'causal_driving_improvement_verified': False, 'arrays_sha256': sha256(run/'differences.npz')}
    write_json(run/'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
