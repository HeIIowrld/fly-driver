"""TRAIN-only reflected retinal histories through unchanged author visual dynamics."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flyvis_camera import integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail
from flybrain_drive.graph import sha256
from flybrain_drive.retinal_reflection import reflect_retinal_history, reflection_permutation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, model_dir = Path(args.data), Path(args.model_dir)
    entries = [e for e in read(root/'dataset.json')['episodes'] if e['role'] == 'train']
    assert len(entries) == 6
    os.environ['FLYVIS_ROOT_DIR'] = str(Path(args.cache_root).resolve())
    os.environ['MPLBACKEND'] = 'Agg'
    install_datamate_cache_compat(args.cache_root)
    import flyvis

    torch.set_num_threads(1)
    torch.set_default_device('cuda')
    torch.manual_seed(863471)
    config = yaml.safe_load((model_dir/'_meta.yaml').read_text(encoding='utf8'))
    network = flyvis.Network(**config['config']['network']).to('cuda')
    weights = torch.load(model_dir/'best_chkpt', map_location='cpu', weights_only=True)['network']
    network.load_state_dict(weights, strict=True)
    network.eval().requires_grad_(False)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        dataset_sha256=sha256(root/'dataset.json'),
        author_checkpoint_sha256=sha256(model_dir/'best_chkpt'), seed=863471,
        baseline_reproduction_tolerance=2e-6, synthetic_augmentation_not_new_physical_data=True,
        development_or_evaluation_arrays_opened=False, optimizer_updates=0,
        state_policy='One grey initial state per complete history; original clocks and held input',
        biological_claim='Engineering optical reflection; no neuronal-type symmetry assumption'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    grey = network.steady_state(1., .005, batch_size=4, value=.5)
    backend = FractionalVisualTail(network, grey)
    records = []
    reference_uv = None
    baseline_max = 0.
    for entry in entries:
        visual = Path(entry['original']['scene_replay'])
        assert sha256(visual/'responses.npz') == entry['source_visual_sha256']
        original_manifest = read(visual/'manifest.json')
        assert original_manifest['config']['checkpoint_sha256'] == sha256(model_dir/'best_chkpt')
        with load_array(visual/'retina-registration.npz', allow_pickle=False) as a:
            uv, motion_indices = a['image_uv'].copy(), a['motion_indices'].copy()
            assert a['view_order'].tolist() == ['front', 'left', 'right', 'rear']
            assert a['motion_types'].tolist() == ['T4a', 'T4b', 'T4c', 'T4d',
                                                   'T5a', 'T5b', 'T5c', 'T5d']
        if reference_uv is None:
            reference_uv = uv
            permutation = reflection_permutation(uv)
            pool = FlightScenePool(uv, dict(columns=4, rows=4, sigma_image_uv=.18)).to('cuda')
        else:
            np.testing.assert_array_equal(uv, reference_uv)
        with load_array(visual/'responses.npz', allow_pickle=False) as a:
            retina, times = a['retina_intensity'].copy(), a['decision_monotonic_seconds'].copy()
            original_motion = a['motion_activity'].copy()
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as a:
            np.testing.assert_array_equal(times, a['observation_time'])
            original_scene = a['scene_features'].copy()
        mirrored = reflect_retinal_history(retina, permutation)
        np.testing.assert_array_equal(reflect_retinal_history(mirrored, permutation), retina)
        modes = ['baseline', 'mirror'] if not records else ['mirror']
        for mode in modes:
            inputs = retina if mode == 'baseline' else mirrored
            state, held, scenes, error = grey, None, [], 0.
            with torch.no_grad():
                for index in range(len(times)):
                    if index:
                        for dt in integration_intervals(float(times[index]-times[index-1])):
                            state = backend.simulate(held[:, None, None], dt, state,
                                                     as_states=True)[-1]
                    retinal = torch.tensor(inputs[index], device='cuda')
                    motion = state.nodes.activity[:, motion_indices]
                    assert bool(torch.isfinite(motion).all())
                    scene = torch.cat([retinal[:, None], motion], 1)
                    pooled = pool(scene[None])[0].cpu().numpy()
                    if mode == 'baseline':
                        error = max(error, float(np.abs(motion.cpu().numpy()
                                                       - original_motion[index]).max()))
                        error = max(error, float(np.abs(pooled-original_scene[index]).max()))
                    else:
                        scenes.append(pooled)
                    held = retinal
                    if index % 100 == 0:
                        write_json(run/'progress.json', dict(file=entry['file'], mode=mode,
                            observations=index+1, total=len(times), max_baseline_error=error))
            if mode == 'baseline':
                assert error <= 2e-6, f'Original neural/scene reproduction failed: {error}'
                baseline_max = error
        output = run/entry['file']
        np.savez_compressed(output, scene_features=np.array(scenes), observation_time=times)
        records.append(dict(file=entry['file'], role='derived_train',
            original_episode_sha256=entry['sha256'], observations=len(times),
            original_visual_sha256=entry['source_visual_sha256'], sha256=sha256(output)))
    assert all(torch.equal(value.cpu(), weights[key].cpu()) for key, value in
               network.state_dict().items() if key in weights)
    np.savez_compressed(run/'reflection-registration.npz', image_uv=reference_uv,
                        permutation=permutation, view_permutation=[0, 2, 1, 3])
    write_json(run/'result.json', dict(status='completed_mirrored_author_visual_replay',
        episodes=records, original_baseline_max_error=baseline_max, author_weights_unchanged=True,
        original_times_and_history_preserved=True, source_roles_unchanged=True,
        new_physical_observations=0, neuron_types_manually_swapped=False,
        actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
