"""Offline original-start sensor ablations; no losses, driver access or safety claims."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_recording_provenance import recording_for_dataset
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes, replay_chunk


def prefix_variants(observation, scalar_offsets, maximum_age):
    modes = ['normal', 'vision_missing', 'body_missing', *scalar_offsets, 'age_zero']
    values = {k: v[:, None].repeat(1, len(modes), *([1]*(v.ndim-1)))
              for k, v in observation.items()}
    for column, mode in enumerate(modes):
        if mode == 'vision_missing':
            for key in ('view_rgb', 'view_contrast', 'view_valid', 'contrast_valid'):
                values[key][:, column] = 0
            values['view_age'][:, column] = maximum_age*10
        elif mode == 'body_missing':
            values['organs'][:, column] = 0
            values['organ_valid'][:, column] = 0
        elif mode in scalar_offsets:
            start = scalar_offsets[mode]
            values['organs'][:, column, start:start+36] = 0
            values['organ_valid'][:, column, start:start+36] = 0
        elif mode == 'age_zero':
            observed = values['view_valid'][:, column].any(-1)
            values['view_age'][:, column][observed] = 0
    return modes, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', action='append', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--retina-config', default='configs/ac-csp-yaw-retina.json')
    parser.add_argument('--decisions', type=int, default=64)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.decisions <= 128:
        raise ValueError('Finite original episode prefix of 1..128 decisions required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    retina_config = json.loads(Path(args.retina_config).read_text(encoding='utf8'))
    calibration = CSPCameraRetina(retina_config).calibration
    scalar_offsets = {spec['native_field']+'_missing': spec['feature_offset']
                      for spec in retina_config['native_retina']['scalar_organs']}
    if sorted(scalar_offsets.values()) != [18, 54, 90] or calibration['organ_size'] != 132:
        raise ValueError('Reviewed 132-slot wheel/suspension/temperature/load/yaw layout required')
    episodes, manifest = load_episodes(args.dataset, calibration)
    if len(episodes) != 1 or manifest.get('source_request_kind') != 'student':
        raise ValueError('One exactly replayed original student episode required')
    recorded = recording_for_dataset(args.dataset)
    recorded_manifest = json.loads((recorded/'manifest.json').read_text(encoding='utf8'))
    executed_hash = recorded_manifest['config']['checkpoint_sha256']
    prefix = {k: v[:args.decisions].cuda() for k, v in episodes[0]['observation'].items()}
    modes, observation = prefix_variants(prefix, scalar_offsets, calibration['max_view_age'])
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862953,
        entry_script_sha256=sha256(__file__),
        dataset_sha256=sha256(Path(args.dataset)/'dataset.json'),
        source_recording=str(recorded), executed_checkpoint_sha256=executed_hash,
        original_episode_prefix_only=True, actuator_messages_sent=0))
    rows = []
    with torch.no_grad():
        for index, directory in enumerate(args.training_run):
            root = Path(directory)
            report = json.loads((root/'result.json').read_text(encoding='utf8'))
            checkpoint = root/'checkpoint.pt'
            digest = sha256(checkpoint)
            if report['status'] != 'completed' or digest != report['checkpoint_sha256']:
                raise ValueError('Completed hash-verified checkpoint required')
            payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
            constructor = copy.deepcopy(payload['constructor'])
            if constructor['sensory_calibration'] != calibration:
                raise ValueError('Original actor calibration differs from model')
            constructor['sparse_backend'] = 'cached_csr_gpu'
            graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                        'weights': np.float32}.get(k)) for k, v in payload['graph'].items()}
            model = MultiViewCircuitController(graph, **constructor).cuda().eval()
            model.load_state_dict(payload['model'])
            model.chunk_validated_training = True
            prediction, _ = replay_chunk(model, observation,
                torch.ones(len(prefix['view_age']), len(modes), device='cuda'),
                model.initial_state(len(modes)),
                payload['training_configuration']['brain_ticks_per_decision'])
            predictions = prediction.cpu().numpy()
            replay_error = None
            if digest == executed_hash:
                replay_error = float(np.abs(predictions[:, 0]
                    - episodes[0]['action'][:len(predictions)].numpy()).max())
                if replay_error > 1e-6:
                    raise ValueError('Actual-prefix GPU replay differs from sealed actions')
            destination = run/f'model-{index:02d}-predictions.npz'
            np.savez_compressed(destination, predictions=predictions, modes=np.array(modes))
            rows.append(dict(training_run=directory, checkpoint_sha256=digest,
                actual_prefix_replay_max_error=replay_error,
                predictions_sha256=sha256(destination), decisions=len(predictions),
                modes={mode: dict(first_prediction=predictions[0, column].tolist(),
                    mean_prediction=predictions[:, column].mean(0).tolist(),
                    final_16_mean_prediction=predictions[-16:, column].mean(0).tolist(),
                    brake_over005=int((predictions[:, column, 2] > .05).sum()))
                    for column, mode in enumerate(modes)}))
            write_json(run/'progress.json', rows)
            del model
    torch.cuda.synchronize()
    write_json(run/'result.json', dict(status='completed', models=rows,
        scope='Original-start independent counterfactual sensor masks; not whole-episode '
              'performance, physical causality or braking safety',
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
