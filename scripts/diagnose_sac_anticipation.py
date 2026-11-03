"""R0/R1 fixed information probes on qualified episodes. No actor or control API.

Train first two historical train episodes; the already-consumed heldout is
REGRESSION/DEVELOPMENT only. Fixed hyperparameters, no test-based selection.
Linear information probes are NOT a replacement flight driver or evidence of
physiological function. Future teacher controls are supervision only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.anticipation_targets import (diagnostic_features,
    future_command_targets, ridge_fit, ridge_predict)
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_verified(entry):
    root = Path(entry['data'])
    result, contract = read(root / 'result.json'), read(root / 'contract.json')
    manifest = read(root / 'manifest.json')['config']
    cert = Path(manifest['arguments']['qualification']) / 'result.json'
    if (result['status'] != 'completed_qualified_actor_preprocessing'
            or contract['episode_id'] != entry['episode_id']
            or contract.get('unqualified_pixel_inputs') != 0
            or sha256(root / 'actor-only.npz') != entry['arrays_sha256']
            or result['arrays_sha256'] != entry['arrays_sha256']
            or sha256(root / 'contract.json') != entry['contract_sha256']
            or result['contract_sha256'] != entry['contract_sha256']
            or sha256(cert) != entry['qualification_sha256']
            or sha256(cert) != manifest['qualification_sha256']
            or sha256(cert) != contract['source_qualification_sha256']
            or not read(cert)['training_admissible']):
        raise ValueError('Qualified independent episode provenance changed')
    with np.load(root / 'actor-only.npz', allow_pickle=False) as arrays:
        data = {k: arrays[k].copy() for k in arrays.files}
    if (data['fit_mask'].dtype != np.bool_
            or data['label_valid'].dtype != np.bool_
            or data['body_valid'].dtype != np.bool_
            or not all(np.isfinite(x).all() for x in data.values())
            or np.any(data['fit_mask'] & ~data['label_valid'])
            or np.any(abs(data['labels'][:, 0]) > 1.00001)
            or np.any(data['labels'][:, 1:] < 0)
            or np.any(data['labels'][:, 1:] > 1)):
        raise ValueError('Original masks, finite inputs and command units required')
    return data


def metric(prediction, target):
    error = prediction - target
    return dict(weighted_mse=float(np.mean(error ** 2 * [4, 1, 1])),
                mae=np.mean(abs(error), 0).tolist(),
                rmse=np.sqrt(np.mean(error ** 2, 0)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    split_path = Path('runs/teacher-synchronized-20261102/student-split-01.json')
    entries = read(split_path)['episodes']
    if ([x['role'] for x in entries] != ['train', 'train', 'heldout']
            or len({x['episode_id'] for x in entries}) != 3):
        raise ValueError('Historical independent chronological split changed')
    run = begin_run(args.output, dict(seed=780303, program_sha256=sha256(__file__),
        split_sha256=sha256(split_path), episodes=entries,
        regression_consumed=True, heldout_role='Consumed historical development/regression only',
        horizons_seconds=[0, .25, .5, 1, 2], maximum_label_time_error_seconds=.025,
        probe='Fixed ridge alpha=.01 per training observation; train-only normalization',
        probe_roles=['body', 'vision', 'full', 'spatial_mean', 'intensity', 'motion', 'time_only'],
        primary_question='Do pooled visual inputs contain future command information beyond body/time?',
        actor_deployed=False, biological_model_training=False,
        physiological_validation=False, checkpoint='Original student checkpoint unchanged'))
    result = dict(status='failed', actor_deployed=False, vehicle_control_sent=False,
                  biological_optimizer_updates=0, probe_fits=0)
    try:
        data = [load_verified(entry) for entry in entries]
        features = [diagnostic_features(d) for d in data]
        targets = [future_command_targets(d['decision_time'], d['labels'],
                   d['label_valid'], d['fit_mask']) for d in data]
        baseline = Path('runs/teacher-synchronized-20261102/student-fit-01')
        checkpoint = baseline / 'unified-sac-student.pt'
        if sha256(checkpoint) != '24a8f4e03614636fe6eede71137faae161c8e82dcc1f212dc4bd4a75fb5203d4':
            raise ValueError('R0 original checkpoint differs')
        calibration = read(baseline / 'readout-calibration.json')
        std, scales = np.array(calibration['raw_training_std']), np.array(calibration['scale'])
        write_json(run / 'baseline.json', dict(checkpoint_sha256=sha256(checkpoint),
            calibration_sha256=sha256(baseline / 'readout-calibration.json'),
            raw_std=std.tolist(), scale=scales.tolist(),
            scale_over_std=(scales / std).tolist(),
            interpretation='Scale floor attenuates pre-fit muscle variations; not proof of causal bottleneck after fit'))
        arrays, reports = {}, []
        for horizon_index, horizon in enumerate([0, .25, .5, 1., 2.]):
            masks = [d['fit_mask'] if horizon == 0 else t[1][:, horizon_index - 1]
                     for d, t in zip(data, targets)]
            labels = [d['labels'] if horizon == 0 else t[0][:, horizon_index - 1]
                      for d, t in zip(data, targets)]
            train_y = np.concatenate([y[m] for y, m in zip(labels[:2], masks[:2])])
            dev_y = labels[2][masks[2]]
            constant = np.repeat(train_y.mean(0)[None], len(dev_y), axis=0)
            report = dict(horizon_seconds=horizon, counts=[int(m.sum()) for m in masks],
                          train_constant=metric(constant, dev_y), probes={})
            arrays[f'h{horizon_index}_dev_target'] = dev_y
            arrays[f'h{horizon_index}_dev_indices'] = np.flatnonzero(masks[2])
            for name in features[0]:
                x = np.concatenate([f[name][m] for f, m in zip(features[:2], masks[:2])])
                fitted = ridge_fit(x, train_y, regularization=.01)
                prediction = ridge_predict(fitted, features[2][name][masks[2]])
                report['probes'][name] = metric(prediction, dev_y)
                key = f'h{horizon_index}_{name}'
                arrays[key + '_prediction'] = prediction
                for field, value in fitted.items():
                    arrays[key + '_' + field] = value
                result['probe_fits'] += 1
                write_json(run / 'progress.json', dict(horizon=horizon, feature=name,
                                                       probe_fits=result['probe_fits']))
            reports.append(report)
        np.savez_compressed(run / 'probes-and-predictions.npz', **arrays)
        scene = data[0]['scene'][data[0]['fit_mask']]
        result.update(status='completed_information_probe_diagnostic', reports=reports,
            arrays_sha256=sha256(run / 'probes-and-predictions.npz'),
            train_scene_std_quantiles=np.quantile(scene.std(0), [0, .25, .5, .75, 1]).tolist(),
            conclusion_scope='Fixed linear decodability on one consumed Monza regression episode; no new-track, causal physiology or improved flight actor claim',
            termination_reason='Fixed 5 horizons x 7 information probes completed')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit failure')
    write_json(run / 'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
