"""Register the new wheel input contract while preserving every graph and model tensor."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_pixel_retina import NativePixelRetina
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from scripts.audit_native_student_replay import equal_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-training-run', required=True)
    parser.add_argument('--retina-config', default='configs/ac-native-wheel-retina.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.source_training_run)
    report = json.loads((root / 'result.json').read_text(encoding='utf8'))
    checkpoint = root / 'checkpoint.pt'
    original_hash = sha256(checkpoint)
    if report['status'] != 'completed' or original_hash != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified source checkpoint required')
    front = NativePixelRetina(json.loads(Path(args.retina_config).read_text(encoding='utf8')))
    if front.wheel_spec is None:
        raise ValueError('Explicit wheel input contract required')
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    constructor = copy.deepcopy(source['constructor'])
    if (constructor.get('organ_validity_mode') != 'gate_only'
            or constructor['sensory_calibration']['organ_size'] != 132
            or not np.array_equal(constructor['directions'], front.local_addresses)):
        raise ValueError('Same local visual geometry, body width and gate-only validity required')
    constructor['sensory_calibration'] = front.calibration
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in source['graph'].items()}
    model = MultiViewCircuitController(graph, **constructor)
    model.load_state_dict(source['model'])
    if not all(equal_tensor(value, model.state_dict()[key])
               for key, value in source['model'].items()):
        raise ValueError('Registration changed model parameters or buffers')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862201,
        source_checkpoint_sha256=original_hash, new_pixel_configuration=front.config))
    payload = copy.deepcopy(source)
    payload.update(constructor=constructor, model=model.state_dict(), AC_ready=False,
        driving_trained='previous fixed-task labels; wheel-conditioned behavior not trained',
        native_wheel_registration=dict(source_checkpoint_sha256=original_hash,
            encoder_configuration=front.config, wheel_units_verified=False,
            measured_fly_receptor_tuning_verified=False))
    torch.save(payload, run / 'checkpoint.pt')
    if sha256(checkpoint) != original_hash:
        raise ValueError('Original checkpoint mutated')
    write_json(run / 'result.json', dict(status='completed', all_model_tensors_unchanged=True,
        graph_unchanged=True, original_checkpoint_unchanged=True,
        checkpoint_sha256=sha256(run / 'checkpoint.pt'),
        new_sensory_calibration=front.calibration, wheel_driving_validated=False,
        game_actions_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
