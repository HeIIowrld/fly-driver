"""Readout-window comparison on unchanged recorded fly neural responses."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_pooling import window_pool_weights
from scripts.build_retinotopic_preview_dataset import activity, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    previous = Path(plan['reference_data'])
    manifest = read(previous/'dataset.json')
    source_root = Path(plan['readout_batch'])
    result = read(source_root/'result.json')
    source = read(source_root/'manifest.json')['config']
    if result['status'] != 'completed_intermediate_visual_readout_batch':
        raise ValueError('Completed original neural histories required')
    sources = {entry['id']: entry for entry in result['episodes']}
    calibration = previous/'channel-calibration.npz'
    if sha256(calibration) != manifest['channel_calibration_sha256']:
        raise ValueError('Unchanged TRAIN-only channel calibration required')
    with load_array(calibration, allow_pickle=False) as arrays:
        mean, scale = arrays['processed_actual_mean'], arrays['processed_actual_scale']
    spec = read(plan['scene_spec'])
    if sha256(spec['scene_registration_file']) != spec['scene_registration_sha256']:
        raise ValueError('Static retinal registration changed')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as reg:
        uv, views = reg['image_uv'].copy(), reg['view_order'].tolist()
    if views[0] != 'front':
        raise ValueError('Explicit front-first registered view order required')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU pooling required')
    full = FlightScenePool(uv, spec['scene_basis']).to('cuda').weight.T
    window = torch.tensor(window_pool_weights(uv, bounds=plan['front_bounds']),
                          device='cuda').T
    m = torch.tensor(mean, device='cuda', dtype=torch.float32)[None, None, :, None]
    s = torch.tensor(scale, device='cuda', dtype=torch.float32)[None, None, :, None]
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), dataset_sha256=sha256(previous/'dataset.json'),
        device=torch.cuda.get_device_name(0)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    episodes, errors = [], []
    for entry in manifest['episodes']:
        original = sources[entry['id']]
        if original['role'] != entry['role']:
            raise ValueError('Original roles required')
        data, clocks = activity(original, source)
        if sha256(entry['features_file']) != entry['features_sha256']:
            raise ValueError('Original pooled reference changed')
        with torch.no_grad():
            neural = torch.tanh((torch.tensor(data['processed_actual'], device='cuda',
                                              dtype=torch.float32)-m)/s)
            baseline = neural @ full
            band = baseline.clone()
            band[:, 0] = neural[:, 0] @ window
            front_only = band.clone()
            front_only[:, 1:] = 0
            baseline = baseline.flatten(1).cpu().numpy()
            band = band.flatten(1).cpu().numpy()
            front_only = front_only.flatten(1).cpu().numpy()
        with load_array(entry['features_file'], allow_pickle=False) as old:
            np.testing.assert_array_equal(clocks, old['observation_time'])
            error = float(np.max(np.abs(baseline-old['processed_actual'])))
            if error > 2e-6:
                raise ValueError('Full-history original pooling not reproduced')
        path = run/(entry['id']+'.npz')
        np.savez_compressed(path, front_band_all_views=band,
                            front_band_only=front_only, observation_time=clocks)
        refs = {'full_field_all_views': dict(file=entry['features_file'],
            key='processed_actual', sha256=entry['features_sha256'])}
        for name in ('front_band_all_views', 'front_band_only'):
            refs[name] = dict(file=str(path), key=name, sha256=sha256(path))
        episodes.append(dict(entry, features_file=str(path), features_sha256=sha256(path),
                             feature_sources=refs))
        errors.append(dict(id=entry['id'], maximum_full_history_error=error))
        write_json(run/'progress.json', dict(complete=len(episodes), total=len(sources)))
    write_json(run/'dataset.json', dict(schema='flybrain.retinotopic_preview.episodes.v2',
        episodes=episodes, conditions=plan['conditions'],
        channel_calibration_file=str(calibration),
        channel_calibration_sha256=sha256(calibration),
        processed_channels=manifest['processed_channels'],
        pooling_contract=dict(front_bounds=plan['front_bounds'],
            bins=[4, 4], sigma_in_window=.18, outside_window_weight=0,
            engineering_readout_attention_not_anatomical_fovea=True,
            full_field_all_views='unchanged full-field four-view pooling',
            front_band_all_views='front window; other three full-field',
            front_band_only='front window; other three pooled blocks constant zero'),
        original_targets_roles_clocks_unchanged=True, actor_oracle_inputs=False))
    write_json(run/'result.json', dict(status='completed_front_band_preview_dataset',
        reproduction=errors, optimizer_updates=0, author_weights_changed=False,
        actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
