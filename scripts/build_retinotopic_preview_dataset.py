"""Paired raw-control/actual-neuron features with TRAIN-only calibration before spatial pooling."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_feature_scaling import ChannelMoments


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def activity(entry, source):
    root = Path(entry['output'])
    if sha256(root/'responses.npz') != entry['responses_sha256']:
        raise ValueError('Complete unchanged additional neuronal readout required')
    with load_array(root/'responses.npz', allow_pickle=False) as a:
        if a['additional_cell_types'].tolist() != source['plan']['types']:
            raise ValueError('Explicit original neuron channel order required')
        raw = np.concatenate([a['retina_intensity'][:, :, None],
            np.zeros((len(a['retina_intensity']), 4, 5, 721), np.float32),
            a['motion_activity']], axis=2)
        processed = np.concatenate([a['additional_activity'], a['motion_activity']], axis=2)
        clocks = a['decision_monotonic_seconds'].copy()
    return dict(raw_control=raw, processed_actual=processed), clocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan['readout_batch'])
    result, source = read(root/'result.json'), read(root/'manifest.json')['config']
    if result['status'] != 'completed_intermediate_visual_readout_batch':
        raise ValueError('Every original readout and whole-reference reproduction required')
    base = read(Path(plan['preview_data'])/'dataset.json')
    labels = {e['id']: e for e in base['episodes']}
    if set(labels) != {e['id'] for e in result['episodes']}:
        raise ValueError('Identical whole original histories and roles required')
    calibration = {name: ChannelMoments() for name in ('raw_control', 'processed_actual')}
    fitted_ids = []
    for entry in result['episodes']:
        if entry['role'] != labels[entry['id']]['role']:
            raise ValueError('Source role changed')
        if entry['role'] == 'train':
            data, _ = activity(entry, source)
            for name, values in data.items():
                calibration[name].add(values, role=entry['role'])
            fitted_ids.append(entry['id'])
    moments = {name: cal.finish(plan['channel_scale_floor']) for name, cal in calibration.items()}
    spec = read(plan['scene_spec'])
    if sha256(spec['scene_registration_file']) != spec['scene_registration_sha256']:
        raise ValueError('Registered static retinal geometry changed')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU readout pooling required')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as reg:
        registered = {k: reg[k].copy() for k in
                      ('image_uv', 'body_directions', 'view_order', 'motion_types')}
    pool = FlightScenePool(registered['image_uv'], spec['scene_basis']).to('cuda')
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan,
        plan_sha256=sha256(args.plan), program_sha256=sha256(__file__),
        readout_result_sha256=sha256(root/'result.json'),
        original_preview_dataset_sha256=sha256(Path(plan['preview_data'])/'dataset.json'),
        fitting_ids=fitted_ids, device=torch.cuda.get_device_name(0)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'channel-calibration.npz',
        **{name+'_'+key: value for name, pair in moments.items()
           for key, value in zip(('mean', 'scale'), pair, strict=True)})
    episodes = []
    for entry in result['episodes']:
        label = labels[entry['id']]
        if sha256(label['targets_file']) != label['targets_sha256']:
            raise ValueError('Privileged target arrays changed')
        with load_array(Path(entry['output'])/'retina-registration.npz', allow_pickle=False) as reg:
            for key, value in registered.items():
                np.testing.assert_array_equal(reg[key], value)
        data, clocks = activity(entry, source)
        with load_array(label['targets_file'], allow_pickle=False) as target:
            np.testing.assert_array_equal(clocks, target['observation_time'])
        features = {}
        for name, values in data.items():
            mean, scale = moments[name]
            with torch.no_grad():
                x = torch.tensor(values, device='cuda', dtype=torch.float32)
                m = torch.tensor(mean, device='cuda', dtype=torch.float32)[None, None, :, None]
                s = torch.tensor(scale, device='cuda', dtype=torch.float32)[None, None, :, None]
                features[name] = (torch.tanh((x-m)/s) @ pool.weight.T).flatten(1).cpu().numpy()
        path = run/(entry['id']+'.npz')
        np.savez_compressed(path, **features, observation_time=clocks)
        episodes.append(dict(label, features_file=str(path), features_sha256=sha256(path),
            source_responses_sha256=entry['responses_sha256']))
        write_json(run/'progress.json', dict(complete=len(episodes), total=10))
    write_json(run/'dataset.json', dict(schema='flybrain.retinotopic_preview.episodes.v1',
        episodes=episodes, conditions=['raw_control', 'processed_actual'],
        channel_calibration_sha256=sha256(run/'channel-calibration.npz'),
        processed_channels=source['plan']['types']+['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'],
        raw_control_channels=['retina_intensity']+['constant_zero']*5
            +['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'],
        processed_raw_intensity_input=False, calibrate_only_train=True, fitting_ids=fitted_ids,
        original_targets_roles_clocks_unchanged=True, body_or_oracle_inputs_added=False))
    write_json(run/'result.json', dict(status='completed_retinotopic_preview_dataset',
        episodes=len(episodes), readout_channels=14, optimizer_updates=0,
        author_weights_changed=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
