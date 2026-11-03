"""Offline common-observation comparison; action difference is not lap performance."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_sac_teacher import PrivilegedSACTeacher
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', required=True)
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    source = Path(a.events)
    rows = [json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
    rows = [x for x in rows if x['kind'] == 'teacher_request' and not x.get('automatic_shift_coast')]
    observations = np.array([x['observation'] for x in rows], dtype=np.float32)
    run = begin_run(a.output, dict(arguments=vars(a), events_sha256=sha256(source),
        teacher_only=True, student_optimizer_updates=0, seed=780101))
    predictions, summaries = {}, []
    for config_path in a.configs:
        c = json.loads(Path(config_path).read_text(encoding='utf8'))
        teacher = PrivilegedSACTeacher(c['checkpoint'], c['checkpoint_sha256'])
        action = teacher.predict(observations)['deterministic_policy_action']
        if action.shape != (len(rows), 3) or not np.isfinite(action).all():
            raise ValueError('Finite three-channel action required')
        key = Path(c['checkpoint']).stem
        predictions[key] = action
        baseline = next(iter(predictions.values()))
        summaries.append(dict(checkpoint=c['checkpoint'], sha256=c['checkpoint_sha256'],
            observations=len(rows), finite=True,
            delta_mean=action.mean(axis=0).tolist(),
            delta_std=action.std(axis=0).tolist(),
            mean_absolute_delta_difference_from_baseline=np.abs(action-baseline).mean(axis=0).tolist(),
            lap_performance_rank=None))
    np.savez_compressed(run/'predictions.npz', observations=observations, **predictions)
    write_json(run/'result.json', dict(status='completed_offline_only', models=summaries,
        source_events_sha256=sha256(source), predictions_sha256=sha256(run/'predictions.npz'),
        full_lap_verified=False, superiority_verified=False, student_optimizer_updates=0))
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
