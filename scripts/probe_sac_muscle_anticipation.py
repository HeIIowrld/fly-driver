"""Fixed TRAIN-only ridge decoding of future commands from baseline muscles.

Uses the original continuous neural trace, not an alternative vehicle actor.
Provides a readout-only comparator for the auxiliary-learning experiment.
"""
import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.anticipation_targets import future_command_targets, ridge_fit, ridge_predict
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.diagnose_sac_anticipation import load_verified, metric, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    trace = Path(a.trace)
    result = read(trace / 'result.json')
    config = read(trace / 'manifest.json')['config']
    if (result['status'] != 'completed_continuous_circuit_trace'
            or sha256(trace / 'trace.npz') != result['arrays_sha256']):
        raise ValueError('Exact original continuous trace required')
    run = begin_run(a.output, dict(seed=780311, program_sha256=sha256(__file__),
        trace_sha256=sha256(trace / 'trace.npz'), episodes=config['episodes'],
        normalization='Training episode only', ridge_regularization=.01,
        horizons_seconds=[.25, .5, 1., 2.], actor_deployed=False,
        scope='Fixed linear future decodability in original six flight muscles; consumed development'))
    data = [load_verified(e) for e in config['episodes']]
    with np.load(trace / 'trace.npz', allow_pickle=False) as arrays:
        muscles = [arrays[f'baseline_{i}_muscles'].copy() for i in range(3)]
    if any(m.shape != (len(d['labels']), 6) for m, d in zip(muscles, data)):
        raise ValueError('Original complete six-muscle events required')
    targets = [future_command_targets(d['decision_time'], d['labels'],
               d['label_valid'], d['fit_mask']) for d in data]
    reports, saved = [], {}
    for h, horizon in enumerate([.25, .5, 1., 2.]):
        train_x = np.concatenate([m[t[1][:, h]] for m, t in zip(muscles[:2], targets[:2])])
        train_y = np.concatenate([t[0][:, h][t[1][:, h]] for t in targets[:2]])
        fitted = ridge_fit(train_x, train_y, .01)
        valid = targets[2][1][:, h]
        prediction = ridge_predict(fitted, muscles[2][valid])
        reports.append(dict(horizon_seconds=horizon, **metric(prediction, targets[2][0][:, h][valid])))
        saved[f'h{h}_prediction'] = prediction
        saved[f'h{h}_indices'] = np.flatnonzero(valid)
        for key, value in fitted.items():
            saved[f'h{h}_{key}'] = value
    np.savez_compressed(run / 'probes.npz', **saved)
    write_json(run / 'result.json', dict(status='completed_baseline_muscle_future_probe',
        reports=reports, probe_fits=4, biological_optimizer_updates=0, vehicle_control_sent=False,
        arrays_sha256=sha256(run / 'probes.npz'),
        interpretation='Train-only fitted readout on pre-existing neural states; not a new sensory model or actor',
        termination_reason='Four fixed future horizons completed'))


if __name__ == '__main__':
    main()
