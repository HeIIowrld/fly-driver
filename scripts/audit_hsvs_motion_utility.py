"""Compare fixed direct-DN motion interventions against recorded steering labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--original-clamp', required=True)
    parser.add_argument('--candidate-clamp', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    dataset = json.loads((root/'dataset.json').read_text(encoding='utf8'))
    paths = [Path(args.comparison), Path(args.original_clamp), Path(args.candidate_clamp)]
    configs = [json.loads((p/'manifest.json').read_text(encoding='utf8'))['config'] for p in paths]
    if (configs[0]['original_checkpoint_sha256'] != configs[1]['checkpoint_sha256']
            or configs[0]['original_checkpoint_sha256'] != configs[2]['checkpoint_sha256']
            or configs[0]['candidate_checkpoint_sha256']
            != configs[2]['candidate_checkpoint_sha256']):
        raise ValueError('Original and candidate checkpoint bindings differ')
    episodes = {e['file'].removesuffix('.npz'): e for e in dataset['episodes']
                if e['role'] == 'development'}
    if sorted(episodes) != ['episode-02', 'episode-07']:
        raise ValueError('Two original known-development histories required')
    files = [Path(args.comparison)/'predictions.npz',
             Path(args.original_clamp)/'group-differences.npz',
             Path(args.candidate_clamp)/'group-differences.npz', root/'dataset.json']
    reports = [json.loads((p/'result.json').read_text(encoding='utf8')) for p in paths]
    expected = [reports[0]['predictions_sha256'], reports[1]['group_difference_sha256'],
                reports[2]['group_difference_sha256']]
    if any(sha256(path) != digest for path, digest in zip(files[:3], expected, strict=True)):
        raise ValueError('Saved prediction arrays do not match completed reports')
    pins = {str(p): sha256(p) for p in files}
    for p in paths:
        pins[str(p/'manifest.json')] = sha256(p/'manifest.json')
        pins[str(p/'result.json')] = sha256(p/'result.json')
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': pins,
        'known_development_not_blind': True, 'optimizer_updates': 0, 'actor_deployed': False})
    reports = []
    with (np.load(files[0], allow_pickle=False) as pred,
          np.load(files[1], allow_pickle=False) as old,
          np.load(files[2], allow_pickle=False) as new):
        for name, entry in episodes.items():
            path = root/entry['file']
            if sha256(path) != entry['sha256']:
                raise ValueError('Original role-bound label changed')
            with load_array(path, allow_pickle=False) as a:
                truth, valid = a['target'][:, 0].copy(), a['target_valid'][:, 0].copy()
            row = {'file': entry['file'], 'valid_rows': int(valid.sum()), 'models': {}}
            for label, source in [('original', old), ('candidate', new)]:
                # The stored deltas are intact-minus-mean and intact-minus-direct-clamped.
                contribution = source[name+'_intent'] - source[name+'_clamp_intent']
                intact = pred[name+'_'+label]
                no_motion = intact-contribution
                row['models'][label] = {
                    'motion_on_steering_rmse': float(np.sqrt(np.mean(
                        (intact[valid, 0]-truth[valid])**2))),
                    'motion_mean_steering_rmse': float(np.sqrt(np.mean(
                        (no_motion[valid, 0]-truth[valid])**2))),
                    'motion_effect_mean_abs_steering': float(np.abs(contribution[:, 0]).mean())}
            reports.append(row)
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('Source changed during audit')
    write_json(run/'result.json', {'status': 'completed_motion_utility_audit', 'reports': reports,
        'mean_intervention_not_actual_neuron_lesion_or_new_video': True,
        'direct_dn_input_fixed': True, 'known_development_not_final_test': True,
        'units': 'normalized neural steering readout, not wheel angle or track error',
        'optimizer_updates': 0, 'actor_deployed': False, 'goal_achieved': False})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
