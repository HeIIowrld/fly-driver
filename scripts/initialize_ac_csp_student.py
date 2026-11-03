"""Register new engine camera rays and reset photometry; preserve the measured circuit."""

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
    parser.add_argument('--config', default='configs/ac-csp-retina.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source_root = Path(args.source_run)
    report = json.loads((source_root / 'result.json').read_text(encoding='utf8'))
    checkpoint = source_root / 'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified source checkpoint required')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    retina = CSPCameraRetina(config)
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    _, teacher_manifest = load_episodes(args.teacher, retina.calibration)
    if not teacher_manifest.get('expert_labels_available'):
        raise ValueError('Sealed actual expert labels required for the next training stage')
    constructor = copy.deepcopy(source['constructor'])
    constructor.update(directions=retina.directions.tolist(),
        available=np.ones(retina.directions.shape[:2]).tolist(),
        sensory_calibration=retina.calibration)
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in source['graph'].items()}
    torch.manual_seed(862813)
    model = MultiViewCircuitController(graph, **constructor)
    fresh, state = model.state_dict(), copy.deepcopy(source['model'])
    reset = []
    for key in state:
        if key.startswith('photometric_') or key in (
                'adapter.visual_adapter.indices', 'adapter.visual_adapter.centers',
                'adapter.visual_adapter.port_sources'):
            state[key] = fresh[key]
            reset.append(key)
    model.load_state_dict(state)
    kept = {key: equal_tensor(value, model.state_dict()[key])
            for key, value in source['model'].items() if key not in reset}
    if not all(kept.values()):
        raise ValueError('Registration changed undeclared source parameters/buffers')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862813,
        source_checkpoint_sha256=sha256(checkpoint), sensory_configuration=config,
        teacher_manifest_sha256=sha256(Path(args.teacher) / 'dataset.json')))
    payload = copy.deepcopy(source)
    payload.update(model=model.state_dict(), constructor=constructor,
        csp_input_registration=dict(configuration=config,
            source_checkpoint_sha256=sha256(checkpoint),
            reset_buffers=reset, physical_optics_calibrated=False, native_driving_verified=False),
        teacher_label_contract=teacher_manifest['teacher_label_contract'],
        AC_ready=False, driving_trained='new camera registration only; not trained on CSP images')
    torch.save(payload, run / 'checkpoint.pt')
    training = dict(source_checkpoint=str(run / 'checkpoint.pt'),
        source_checkpoint_sha256=sha256(run / 'checkpoint.pt'), seed=862814,
        action_profile='normalized_ac_absolute', epochs=8, chunk_length=16,
        learning_rate=.0001, brain_ticks_per_decision=4, control_dt_seconds=.04,
        trainable_groups=['visual', 'organs', 'readout'], pedal_mode='independent',
        organ_validity_mode='gate_only', readout_gain_parameterization='log',
        readout_gain_learning_rate=.03, activation_gradient='leaky_surrogate',
        surrogate_negative_slope=.05, condition_steering_gain=False,
        pedal_conditioning=dict(target_rms=[.2, .2], max_gain=1e10),
        native_steering_conditioning=dict(target_rms=.15, max_gain=1e10),
        photometric_normalization=dict(rgb_floor=.03, contrast_floor=.1),
        teacher_label_contract=teacher_manifest['teacher_label_contract'])
    write_json(run / 'training-config.json', training)
    write_json(run / 'result.json', dict(status='completed',
        checkpoint_sha256=sha256(run / 'checkpoint.pt'), reset_buffers=reset,
        undeclared_model_tensors_unchanged=True, recurrent_graph_unchanged=True,
        actor_view_order=retina.calibration['view_order'], camera_driving_validated=False,
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
