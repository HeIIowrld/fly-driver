"""Enable reviewed yaw inputs with zero initial influence; preserve other model tensors."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes
from scripts.audit_native_student_replay import equal_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--config', default='configs/ac-csp-yaw-retina.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.source_run)
    report = json.loads((root / 'result.json').read_text(encoding='utf8'))
    file = root / 'checkpoint.pt'
    if report['status'] != 'completed' or sha256(file) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified source checkpoint required')
    source = torch.load(file, map_location='cpu', weights_only=True)
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    retina = CSPCameraRetina(config)
    if retina.yaw_spec is None:
        raise ValueError('Explicit reviewed yaw registration required')
    base = copy.deepcopy(config)
    del base['native_retina']['yaw_rotation_organs']
    constructor = copy.deepcopy(source['constructor'])
    if (constructor['sensory_calibration'] != CSPCameraRetina(base).calibration
            or not np.array_equal(constructor['directions'], retina.directions)
            or constructor.get('organ_validity_mode') != 'gate_only'):
        raise ValueError('Unchanged camera geometry and gate-only base registration required')
    _, teacher = load_episodes(args.teacher, retina.calibration)
    constructor['sensory_calibration'] = retina.calibration
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in source['graph'].items()}
    model = MultiViewCircuitController(graph, **constructor)
    model.load_state_dict(source['model'])
    reset = {}
    with torch.no_grad():
        for i, name in enumerate(model.adapter.group_names):
            features = getattr(model.adapter, f'group_features_{i}').tolist()
            columns = [j for j, feature in enumerate(features) if feature in (126, 127)]
            if not columns:
                continue
            columns += [j+len(features) for j in columns]
            model.adapter.organ_adapters[i].weight[:, columns] = 0
            reset[f'adapter.organ_adapters.{i}.weight'] = dict(group=name, columns=columns)
    if {r['group'] for r in reset.values()} != {'haltere_L', 'haltere_R'}:
        raise ValueError('Both existing haltere groups must receive only the declared yaw pair')
    after = model.state_dict()
    for key, original in source['model'].items():
        expected = original.clone() if key in reset else original
        if key in reset:
            expected[:, reset[key]['columns']] = 0
        if not equal_tensor(expected, after[key]):
            raise ValueError('Registration changed undeclared model entries: '+key)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862840,
        source_checkpoint_sha256=sha256(file), sensory_configuration=config,
        teacher_manifest_sha256=sha256(Path(args.teacher) / 'dataset.json')))
    payload = copy.deepcopy(source)
    payload.update(model=after, constructor=constructor, AC_ready=False,
        teacher_label_contract=teacher['teacher_label_contract'],
        native_yaw_registration=dict(configuration=retina.yaw_spec, reset_entries=reset,
            initial_yaw_influence='zero', measured_fly_tuning_verified=False,
            native_yaw_driving_verified=False),
        csp_input_registration=dict(configuration=config, camera_parameters_unchanged=True,
            source_checkpoint_sha256=sha256(file), physical_optics_calibrated=False),
        driving_trained='yaw input registered only; no yaw-conditioned training yet')
    torch.save(payload, run / 'checkpoint.pt')
    training = dict(source_checkpoint=str(run / 'checkpoint.pt'),
        source_checkpoint_sha256=sha256(run / 'checkpoint.pt'), seed=862841, epochs=6,
        chunk_length=16, learning_rate=.001, brain_ticks_per_decision=4,
        control_dt_seconds=.04, action_profile='normalized_ac_absolute',
        trainable_groups=['organs'], organ_group_subset=['haltere_L', 'haltere_R'],
        organ_validity_mode='gate_only', pedal_mode='independent', episodes_per_batch=2,
        readout_gain_parameterization='log', activation_gradient='leaky_surrogate',
        surrogate_negative_slope=.05, teacher_label_contract=teacher['teacher_label_contract'])
    write_json(run / 'training-config.json', training)
    write_json(run / 'result.json', dict(status='completed',
        checkpoint_sha256=sha256(run / 'checkpoint.pt'), reset_entries=reset,
        graph_and_undeclared_tensors_unchanged=True, camera_and_photometry_unchanged=True,
        initial_yaw_influence_zero=True, game_actions_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
