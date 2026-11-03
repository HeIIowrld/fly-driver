"""Audit ridge shrinkage and bounded speed response on saved synthetic states."""
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    source = base / '20261101-visual-corner-speed-fit-1'
    result = json.loads((source / 'result.json').read_text(encoding='utf8'))
    payload = source / 'evaluation.npz'
    if result['status'] != 'completed_synthetic_corner_speed_readout_fit' or sha256(payload) != result['output_sha256']:
        raise ValueError('Exact saved synthetic fit features required')
    with np.load(payload, allow_pickle=False) as data:
        features = data['fit_features'].copy()
        labels = data['fit_targets'].copy()
    if (features.shape != (60, 6) or labels.shape != (60,)
            or not np.isfinite(features).all()
            or not np.array_equal(labels.reshape(2, 3, 10)[:, :, 0], [[3.5, 2.5, 2.5]] * 2)):
        raise ValueError('Exact two-texture ordered paired samples required')
    raw = features.reshape(2, 3, 10, 6) / 1000.
    paired = np.stack((raw[:, 0] - raw[:, 1], raw[:, 0] - raw[:, 2]), axis=1)
    l1 = np.abs(paired).sum(axis=-1)
    max_delta, weight_cap = 10., 5000.
    # tanh is 1-Lipschitz. Fixed body speed and bias cancel in each pair.
    bound = max_delta * weight_cap * float(l1.mean())
    necessary = 1. / (max_delta * float(l1.mean()))
    design = np.column_stack((features, np.ones(len(features))))
    target_logits = np.arctanh((labels - 3.) / max_delta)
    ridge = []
    for strength in (.001, 1e-5, 1e-7, 1e-9):
        weights = np.linalg.solve(design.T @ design + np.diag([strength] * 6 + [0.]),
                                  design.T @ target_logits)
        prediction = 3. + max_delta * np.tanh(design @ weights)
        ridge.append(dict(lambda_value=strength,
                          mean_gap=float(prediction[labels == 3.5].mean() - prediction[labels == 2.5].mean()),
                          mse=float(np.mean((prediction - labels) ** 2)),
                          largest_raw_muscle_weight=float(np.abs(weights[:6]).max() * 1000.)))
    run = begin_run(base / '20261101-corner-speed-fixed-state-bounds',
                    dict(program_sha256=sha256(__file__), source_sha256=sha256(payload),
                         scope='post-hoc fixed synthetic states only, no candidate fitting or control'))
    report = dict(status='completed_corner_speed_fixed_state_bounds',
                  paired_count=int(l1.size), mean_paired_l1=float(l1.mean()),
                  mean_pairwise_difference_upper_bound_m_s=bound,
                  necessary_weight_cap_for_one_m_s=necessary,
                  declared_weight_cap=weight_cap, ridge_counterfactuals=ridge,
                  actor_qualified=False, optimizer_updates=0, vehicle_control_sent=False,
                  interpretation='Mathematical Lipschitz bound on fixed previously seen synthetic states; not a bound on changed recurrent trajectories or real flight/vehicle capability.')
    write_json(run / 'result.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
