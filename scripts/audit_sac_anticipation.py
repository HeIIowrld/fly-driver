"""Recompute R1 saved probe metrics/targets and paired R2 checkpoint invariants.

The audit does not fit, choose hyperparameters, actuate, or rerun neural episodes.
"""
import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.anticipation_targets import (diagnostic_features,
    future_command_targets, ridge_predict)
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.diagnose_sac_anticipation import load_verified, metric, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--information', required=True)
    parser.add_argument('--trace')
    parser.add_argument('--fit')
    parser.add_argument('--intervention', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.information)
    run = begin_run(args.output, dict(arguments=vars(args), seed=780305,
        program_sha256=sha256(__file__), information_result_sha256=sha256(source / 'result.json'),
        optimizer_updates=0, actor_deployed=False))
    result = dict(status='failed', optimizer_updates=0, vehicle_control_sent=False)
    try:
        report = read(source / 'result.json')
        cfg = read(source / 'manifest.json')['config']
        if (report['status'] != 'completed_information_probe_diagnostic'
                or sha256(source / 'probes-and-predictions.npz') != report['arrays_sha256']):
            raise ValueError('Completed information run/hash required')
        data = [load_verified(e) for e in cfg['episodes']]
        dev = data[2]
        features = diagnostic_features(dev)
        future, mask, _ = future_command_targets(dev['decision_time'], dev['labels'],
                                                 dev['label_valid'], dev['fit_mask'])
        verified = 0
        with np.load(source / 'probes-and-predictions.npz', allow_pickle=False) as arrays:
            for h, row in enumerate(report['reports']):
                indices = np.flatnonzero(dev['fit_mask'] if h == 0 else mask[:, h - 1])
                target = dev['labels'][indices] if h == 0 else future[indices, h - 1]
                np.testing.assert_array_equal(arrays[f'h{h}_dev_indices'], indices)
                np.testing.assert_array_equal(arrays[f'h{h}_dev_target'], target)
                for name, expected in row['probes'].items():
                    key = f'h{h}_{name}'
                    fitted = {field: arrays[key + '_' + field] for field in
                              ['mean', 'scale', 'weights', 'target_mean']}
                    prediction = ridge_predict(fitted, features[name][indices])
                    np.testing.assert_allclose(prediction, arrays[key + '_prediction'], atol=1e-12, rtol=1e-10)
                    actual = metric(prediction, target)
                    for field in actual:
                        np.testing.assert_allclose(actual[field], expected[field], atol=1e-12, rtol=1e-10)
                    verified += 1
        result['saved_probes_recomputed'] = verified
        if args.trace:
            trace = Path(args.trace)
            tr = read(trace / 'result.json')
            if (tr['status'] != 'completed_continuous_circuit_trace'
                    or sha256(trace / 'trace.npz') != tr['arrays_sha256']):
                raise ValueError('Completed original circuit trace/hash required')
            with np.load(trace / 'trace.npz', allow_pickle=False) as arrays:
                for i, d in enumerate(data):
                    actual = metric(arrays[f'baseline_{i}_prediction'][d['fit_mask']],
                                    d['labels'][d['fit_mask']])
                    for field in actual:
                        np.testing.assert_allclose(actual[field], tr['metrics'][i][field], rtol=1e-6, atol=1e-8)
                baseline = Path('runs/teacher-synchronized-20261102/student-fit-01')
                with np.load(baseline / 'heldout-predictions.npz', allow_pickle=False) as original:
                    delta = abs(arrays['baseline_2_prediction'] - original['full_0'])
                    result['cpu_original_replay_max_absolute_error'] = delta.max(0).tolist()
                    result['cpu_original_replay_matches_at_2e_4'] = bool(np.all(delta <= 2e-4))
        if args.fit:
            import torch
            from flybrain_drive.sac_unified_student import assemble_student
            torch.set_num_threads(1)
            fit = Path(args.fit)
            fr = read(fit / 'result.json')
            if (fr['status'] != 'completed_fixed_anticipation_learning'
                    or sha256(fit / 'predictions.npz') != fr['arrays_sha256']):
                raise ValueError('Completed paired fit/hash required')
            baseline = torch.load('runs/teacher-synchronized-20261102/student-fit-01/unified-sac-student.pt',
                                  map_location='cpu', weights_only=True)
            model, lineage = assemble_student()
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            fixed = [key for key in baseline['model'] if key not in trainable]
            with np.load(fit / 'predictions.npz', allow_pickle=False) as predictions:
                for arm in fr['arms']:
                    path = fit / f'{arm}.pt'
                    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                    if (sha256(path) != fr['arms'][arm]['checkpoint_sha256']
                            or checkpoint['manifest_sha256'] != sha256(fit / 'manifest.json')
                            or checkpoint['actor_qualified'] is not False
                            or checkpoint['lineage'] != lineage
                            or checkpoint['optimizer_updates'] != fr['arms'][arm]['updates']):
                        raise ValueError('Paired checkpoint metadata differs')
                    for key in fixed:
                        a, b = baseline['model'][key], checkpoint['model'][key]
                        if a.layout != torch.strided:
                            a, b = a.to_sparse_coo().coalesce(), b.to_sparse_coo().coalesce()
                            same = torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
                        else:
                            same = torch.equal(a, b)
                        if not same:
                            raise ValueError('Fixed circuit state changed: ' + key)
                    model.load_state_dict(checkpoint['model'], strict=True)
                    model.motor.backbone.assert_body_boundary_protected()
                    actual = metric(predictions[arm + '_dev_prediction'][dev['fit_mask']],
                                    dev['labels'][dev['fit_mask']])
                    for field in actual:
                        np.testing.assert_allclose(actual[field], fr['arms'][arm]['dev'][field], rtol=1e-6, atol=1e-8)
                    if arm == 'auxiliary':
                        plan = predictions[arm + '_dev_future']
                        if plan.shape != (len(dev['labels']), 4, 3) or not np.isfinite(plan).all():
                            raise ValueError('Finite saved future plans at all original events required')
                        for h in range(4):
                            actual_future = metric(plan[:, h][mask[:, h]], future[:, h][mask[:, h]])
                            for field in actual_future:
                                np.testing.assert_allclose(actual_future[field],
                                    fr['arms'][arm]['dev']['future'][h][field], rtol=1e-6, atol=1e-8)
                        head = checkpoint['auxiliary']
                        if (head['projection.weight'].shape != (12, 6)
                                or not all(torch.isfinite(v).all() for v in head.values())):
                            raise ValueError('Finite auxiliary head on existing six muscles required')
                        if args.trace:
                            with np.load(Path(args.trace) / 'trace.npz', allow_pickle=False) as calibration:
                                np.testing.assert_array_equal(head['reference'].numpy(), calibration['train_reference'])
                                np.testing.assert_array_equal(head['scale'].numpy(), np.maximum(calibration['train_std'], 1e-6))
                        result['future_horizon_metrics_recomputed'] = 4
            result['fixed_entries_verified_per_candidate'] = len(fixed)
            result['fit_updates'] = fr['optimizer_updates']
        intervention_metrics = {}
        for name in args.intervention:
            intervention_root = Path(name)
            ir = read(intervention_root / 'result.json')
            im = read(intervention_root / 'manifest.json')['config']
            fit_root = Path(im['arguments']['fit'])
            arm = im['arguments']['arm']
            fit_result = read(fit_root / 'result.json')
            if (ir['status'] != 'completed_continuous_information_intervention'
                    or sha256(intervention_root / 'predictions.npz') != ir['arrays_sha256']
                    or sha256(fit_root / 'result.json') != im['fit_result_sha256']
                    or sha256(fit_root / f'{arm}.pt') != im['checkpoint_sha256']):
                raise ValueError('Completed intervention/source hashes required')
            with np.load(intervention_root / 'predictions.npz', allow_pickle=False) as arrays:
                prediction = arrays['prediction']
                if prediction.shape != dev['labels'].shape or not np.isfinite(prediction).all():
                    raise ValueError('Original-event finite intervention commands required')
                actual = metric(prediction[dev['fit_mask']], dev['labels'][dev['fit_mask']])
                saved_future = arrays['future'].copy() if 'future' in arrays.files else None
            for field in actual:
                np.testing.assert_allclose(actual[field], ir['metrics'][field], rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(ir['weighted_mse_ratio_to_normal'],
                actual['weighted_mse']/fit_result['arms'][arm]['dev']['weighted_mse'], rtol=1e-6, atol=1e-8)
            if 'future' in ir:
                if saved_future is None or saved_future.shape != future.shape or not np.isfinite(saved_future).all():
                    raise ValueError('Finite future intervention plans at original events required')
                checked_future = []
                for h in range(4):
                    observed = metric(saved_future[:, h][mask[:, h]], future[:, h][mask[:, h]])
                    for field in observed:
                        np.testing.assert_allclose(observed[field], ir['future'][h][field], rtol=1e-6, atol=1e-8)
                    checked_future.append(observed)
                actual['future'] = checked_future
            intervention_metrics[name] = actual
        if intervention_metrics:
            result['intervention_saved_metrics_recomputed'] = intervention_metrics
        result.update(status='completed_anticipation_evidence_audit',
            scope='Saved-array/target/metric recomputation and fixed checkpoint checks; no independent neural rerun or new test/generalization claim',
            termination_reason='All requested audits completed')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit audit failure')
    write_json(run / 'result.json', result)
    print(result)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
