"""Frozen fold CPU/GPU registered-feature and readout parity, zero author updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import read
from scripts.prepare_flyvis_preview_folds import TRAIN_IDS, registered_cpu
from scripts.probe_flyvis_registered_gradient import registered_features
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan['indices'] != [0, 31, 150, 738] or plan['train_ids'] != TRAIN_IDS
            or plan['maximum_feature_error'] != 2e-6
            or plan['maximum_raw_prediction_error'] != 1e-5):
        raise ValueError('Fixed36-case fold parity gate required')
    for file, digest in plan['input_sha256'].items():
        if sha256(file) != digest:
            raise ValueError('Declared immutable parity source changed: '+file)
    torch.set_num_threads(1)
    torch.set_default_device('cpu')
    if not torch.cuda.is_available():
        raise ValueError('Explicit GPU parity device required')
    setup = Path(plan['fold_setup'])
    completed = read(setup/'result.json')
    if completed['status'] != 'completed_cpu_train_history_fold_setup':
        raise ValueError('CompletedTRAIN-onlyfold setup required')
    references = {e['id']: e for e in completed['source_references']}
    episodes = {}
    indices = plan['indices']
    for name in TRAIN_IDS:
        reference = references[name]
        vision = Path(reference['source_responses']).parent
        with load_array(reference['source_responses'], allow_pickle=False) as a:
            clock = a['decision_monotonic_seconds'][indices].copy()
            activity = np.concatenate([a['additional_activity'][indices],
                                       a['motion_activity'][indices]], axis=2)
        with load_array(reference['source_body'], allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['decision_monotonic_seconds'][indices])
            body, mask = a['values'][indices].copy(), a['valid'][indices].copy()
        with load_array(vision/'retina-registration.npz', allow_pickle=False) as a:
            rays, origins = a['body_directions'].copy(), a['body_origins_m'].copy()
        episodes[name] = dict(clock=clock, activity=activity, body=body,
                              body_valid=mask, rays=rays, origins=origins)
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), optimizer_updates=0, author_network_loaded=False,
        known_development_arrays_opened=0, device=torch.cuda.get_device_name(0)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', optimizer_updates=0, author_network_loaded=False,
        known_development_arrays_opened=0, actuator_messages_sent=0,
        actor_deployed=False, independent_final_validation=False)
    records = []
    try:
        torch.cuda.reset_peak_memory_stats()
        for heldout in TRAIN_IDS:
            folder = setup/heldout
            with load_array(folder/'normalization.npz', allow_pickle=False) as a:
                norms = {key: a[key].copy() for key in a.files}
            saved = torch.load(folder/'vision_body.pt', map_location='cpu', weights_only=False)
            if (saved['heldout_history'] != heldout or heldout in saved['fitted_history_ids']
                    or set(saved['fitted_history_ids']) != set(TRAIN_IDS)-{heldout}):
                raise ValueError('WholeTRAINfold identity changed')
            for key in ('channel_mean', 'channel_scale', 'feature_mean', 'feature_scale',
                        'body_mean', 'body_scale', 'target_mean', 'target_scale'):
                np.testing.assert_array_equal(norms[key], saved[key])
            cpu_model = RegisteredPreview().cpu().eval().requires_grad_(False)
            cpu_model.load_state_dict(saved['model'], strict=True)
            gpu_model = RegisteredPreview().to('cuda').eval().requires_grad_(False)
            gpu_model.load_state_dict(saved['model'], strict=True)
            for history in TRAIN_IDS:
                e = episodes[history]
                with torch.device('cpu'):
                    cpu = registered_cpu(e, norms['channel_mean'], norms['channel_scale'],
                                         batch_size=4)['intact']
                features = []
                with torch.no_grad(), torch.device('cuda'):
                    for i in range(len(indices)):
                        selected = torch.tensor(e['activity'][i], dtype=torch.float32)
                        features.append(registered_features(selected, e['rays'], e['origins'],
                            e['body'][i:i+1], e['body_valid'][i:i+1], norms['channel_mean'],
                            norms['channel_scale']))
                    gpu_features = torch.cat(features, dim=0)
                    gpu = gpu_features.cpu().numpy()
                    normalized_gpu = ((gpu_features-torch.tensor(norms['feature_mean'],
                        dtype=torch.float32))/torch.tensor(norms['feature_scale'],
                        dtype=torch.float32)).clamp(-5, 5)
                normalized_cpu = ((cpu-norms['feature_mean'])/norms['feature_scale']).clip(
                    -5, 5).astype(np.float32)
                body = body_features(e['body'], e['body_valid'], norms['body_mean'],
                    norms['body_scale'], plan['body_route_indices'])
                with torch.no_grad():
                    pred_cpu = cpu_model(torch.tensor(normalized_cpu), torch.tensor(body)).numpy()
                    pred_gpu = gpu_model(normalized_gpu, torch.tensor(body, device='cuda')
                                         ).cpu().numpy()
                pred_cpu = pred_cpu*norms['target_scale']+norms['target_mean']
                pred_gpu = pred_gpu*norms['target_scale']+norms['target_mean']
                for i, source_index in enumerate(indices):
                    feature_error = float(np.abs(cpu[i]-gpu[i]).max())
                    prediction_error = float(np.abs(pred_cpu[i]-pred_gpu[i]).max())
                    normalized_error = float(np.abs(normalized_cpu[i]-
                        normalized_gpu[i].cpu().numpy()).max())
                    support_equal = bool(np.array_equal(cpu[i, 896:], gpu[i, 896:]))
                    records.append(dict(fold_heldout=heldout, history=history,
                        source_index=source_index, source_clock=float(e['clock'][i]),
                        maximum_feature_error=feature_error,
                        maximum_normalized_feature_error=normalized_error,
                        maximum_raw_prediction_error=prediction_error,
                        pose_support_exact=support_equal,
                        channel_source=str(folder/'normalization.npz'),
                        channel_source_sha256=sha256(folder/'normalization.npz'),
                        fitted_history_ids=saved['fitted_history_ids']))
                    write_json(run/'progress.json', dict(completed_cases=len(records),
                        records=records, expected_cases=36))
                    if (feature_error > plan['maximum_feature_error']
                            or prediction_error > plan['maximum_raw_prediction_error']
                            or not support_equal):
                        raise ValueError('Predeclared fold feature/readout parity gate failed')
            del cpu_model, gpu_model
        result.update(status='passed_fold_cpu_gpu_feature_readout_parity',
            cases=len(records), feature_error_max=max(r['maximum_feature_error'] for r in records),
            normalized_feature_error_max=max(r['maximum_normalized_feature_error']
                                             for r in records),
            raw_prediction_error_max=max(r['maximum_raw_prediction_error'] for r in records),
            all_pose_support_exact=True, all_gpu_routes_use_fold_calibration=True,
            original_fold_setup_result_sha256=sha256(setup/'result.json'),
            peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), completed_cases=len(records))
        raise
    finally:
        result['records'] = records
        write_json(run/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}))


if __name__ == '__main__':
    main()
