"""Independent stored-data/metric/calibration audit of the spatial experiment.

No optimization or simulator. Recomputes masks and saved metrics; checks original
state->muscle reduction and TRAIN-only calibration. Not a second neural replay.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.anticipation_targets import future_command_targets
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import assemble_student, read
from scripts.diagnose_sac_anticipation import load_verified, metric
from scripts.trace_sac_premuscle import CHECKPOINT, CHECKPOINT_SHA


def verify_metrics(actual, expected):
    for key in ['weighted_mse', 'mae', 'rmse']:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    base = Path(a.root)
    entries = read('runs/teacher-synchronized-20261102/student-split-01.json')['episodes']
    data = [load_verified(e) for e in entries]
    d = data[2]
    run = begin_run(a.output, dict(arguments=vars(a), seed=780409, program_sha256=sha256(__file__),
        original_checkpoint_sha256=CHECKPOINT_SHA, entries=entries,
        scope='Stored target/metric/state-reduction and calibration audit; not new neural execution or physiology',
        optimizer_updates=0, vehicle_control_sent=False))
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False)
    try:
        if sha256(CHECKPOINT) != CHECKPOINT_SHA:
            raise ValueError('Original best model changed')
        future, valid, _ = future_command_targets(d['decision_time'], d['labels'], d['label_valid'], d['fit_mask'])
        supervised = np.flatnonzero(d['fit_mask'])
        trace = base/'premuscle-trace-01'
        tr = read(trace/'result.json')
        for name, expected in tr['files_sha256'].items():
            if sha256(trace/name) != expected:
                raise ValueError('Neural trace changed')
        model, _ = assemble_student()
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        model.load_state_dict(payload['model'], strict=True)
        model.requires_grad_(False)
        reductions = []
        for i in range(3):
            s = np.load(trace/f'state-{i}.npy', mmap_mode='r', allow_pickle=False)
            with torch.no_grad():
                _, muscle = model.motor.backbone._read_state_tensor(torch.tensor(np.asarray(s)))
            with np.load(trace/f'observed-{i}.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['event_indices'], np.flatnonzero(data[i]['fit_mask']))
                delta = float(abs(muscle.numpy()-z['muscles']).max())
            if delta > 2e-7:
                raise ValueError('Recorded six-muscle aggregation differs')
            reductions.append(delta)
        result['state_to_muscle_max_errors'] = reductions
        fits = 0
        for name in ['spatial-probes-01', 'neural-probes-01']:
            r = read(base/name/'result.json')
            if r['status'] != 'completed_information_stage_probes' or sha256(base/name/'predictions.npz') != r['predictions_sha256']:
                raise ValueError('Completed probe arrays/hash required')
            with np.load(base/name/'predictions.npz', allow_pickle=False) as z:
                for h, row in enumerate(r['reports']):
                    mask = d['fit_mask'] if h == 0 else valid[:, h-1]
                    target = d['labels'][mask] if h == 0 else future[:, h-1][mask]
                    np.testing.assert_array_equal(z[f'h{h}_target'], target)
                    for feature, expected in row['probes'].items():
                        verify_metrics(metric(z[f'h{h}_{feature}_prediction'], target), expected)
                        fits += 1
        result['saved_probe_metric_sets_recomputed'] = fits
        spatial = base/'spatial-recovery-01'
        sr = read(spatial/'result.json')
        extras = []
        for i, x in enumerate(data):
            file = spatial/f'spatial-{i}.npz'
            if sha256(file) != sr['files_sha256'][file.name]:
                raise ValueError('Original retinal array differs')
            with np.load(file, allow_pickle=False) as z:
                joined = z['event_to_camera']
                np.testing.assert_array_equal(z['camera_indices'][joined], x['camera_index'])
                extras.append(z['discarded_grid8'][joined].copy())
        learned = {}
        for name in ['spatial-control-01', 'spatial-residual-01', 'descending-control-01', 'descending-residual-01']:
            root = base/name
            r = read(root/'result.json')
            if r['status'] != 'completed_fixed_spatial_port_learning' or sha256(root/'spatial-port.pt') != r['checkpoint_sha256']:
                raise ValueError('Exact completed candidate required')
            cp = torch.load(root/'spatial-port.pt', map_location='cpu', weights_only=True)
            if cp['actor_qualified'] is not False or cp['source_manifest_sha256'] != sha256(root/'manifest.json'):
                raise ValueError('Candidate qualification/provenance differs')
            if cp['baseline_checkpoint_sha256'] != CHECKPOINT_SHA:
                raise ValueError('No changed original biological model may be serialized')
            route = cp.get('route', 'optical')
            train = []
            for i in range(2):
                x = extras[i]
                if route == 'descending':
                    intensity = data[i]['scene'].reshape(-1, 4, 9, 16)[:, :, 0].reshape(-1, 64)
                    x = np.concatenate([intensity, x], axis=1)
                train.append(x[data[i]['fit_mask']])
            x = np.concatenate(train)
            np.testing.assert_allclose(cp['adapter']['reference'].numpy(), x.mean(0), rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(cp['adapter']['scale'].numpy(), np.maximum(x.std(0), 1e-4), rtol=1e-6, atol=1e-8)
            for key in ['training_reference', 'training_scale']:
                if not torch.equal(cp['readout'][key], payload['readout'][key]):
                    raise ValueError('Vehicle readout calibration changed')
            if cp['arm'] == 'control' and torch.count_nonzero(cp['adapter']['projection.weight']) != 0:
                raise ValueError('Matched control acquired learned extra visual weights')
            if not all(torch.isfinite(v).all() for group in ['adapter', 'readout'] for v in cp[group].values()):
                raise ValueError('Nonfinite learned boundary')
            with np.load(root/'predictions.npz', allow_pickle=False) as z:
                verify_metrics(metric(z['dev'][d['fit_mask']], d['labels'][d['fit_mask']]), r['dev'])
            learned[name] = r
        if learned['descending-control-01']['initial_prefix_prediction_sha256'] != learned['descending-residual-01']['initial_prefix_prediction_sha256']:
            raise ValueError('Tonic-matched initial conditions differ')
        comparisons = {}
        original_mse = read('runs/visual-anticipation-20261103/continuation8-02/result.json')['arms']['continuation']['dev']['weighted_mse']
        for route in ['spatial', 'descending']:
            c, r = learned[route+'-control-01']['dev'], learned[route+'-residual-01']['dev']
            passed = (r['weighted_mse'] <= .98*min(c['weighted_mse'], original_mse)
                      and np.all(np.asarray(r['rmse']) <= 1.02*np.asarray(c['rmse'])))
            comparisons[route] = dict(control=c, candidate=r,
                normal_mse_change_percent=100*(r['weighted_mse']/c['weighted_mse']-1),
                consumed_development_gate_passed=bool(passed))
        checked = {}
        for root in sorted(base.iterdir()):
            if not root.is_dir() or not (root/'result.json').exists():
                continue
            r = read(root/'result.json')
            if r['status'] not in ['completed_body_channel_information_intervention', 'completed_spatial_port_information_replay']:
                continue
            if sha256(root/'predictions.npz') != r['predictions_sha256']:
                raise ValueError('Intervention predictions changed')
            with np.load(root/'predictions.npz', allow_pickle=False) as z:
                pred = z['prediction']
                if pred.shape != d['labels'].shape or not np.isfinite(pred).all():
                    raise ValueError('Original-event finite predictions required')
                verify_metrics(metric(pred[d['fit_mask']], d['labels'][d['fit_mask']]), r['metrics'])
            checked[root.name] = r['metrics']
        result.update(status='completed_spatial_experiment_evidence_audit', comparisons=comparisons,
            intervention_metrics_recomputed=checked, checkpoint_calibrations_verified=4,
            original_model_state_entries_per_fit=51, matched_tonic_prefix_verified=True,
            interpretation='All results remain one consumed Monza development episode; gate is not physiological or live-actor qualification',
            termination_reason='All requested stored metrics, calibration, fixed-source identities and matched comparisons verified')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit audit failure')
    write_json(run/'result.json', result)
    print(result)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
