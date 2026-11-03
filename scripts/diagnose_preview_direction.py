"""Inspect fixed preview readouts by teacher-path direction; posthoc, offline only."""

import argparse
import json

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_residual_preview import ResidualPreview
from scripts.train_visual_body_preview import body_features
from scripts.train_visual_residual_preview import open_episode, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256 or sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Fixed plan or checkpoint changed')
    plan = read(args.plan)
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Original experiment source changed: ' + path)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['development_used_for_fit'] or checkpoint['actor_qualified']:
        raise ValueError('Original unqualified fit expected')
    model = ResidualPreview().cpu().eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    entries = read(plan['dataset'])['episodes']
    run = begin_run(args.output, {'plan_sha256': args.plan_sha256,
        'checkpoint_sha256': args.checkpoint_sha256, 'program_sha256': sha256(__file__),
        'posthoc': True, 'actor_deployed': False, 'optimizer_updates': 0})
    reports = []
    # Thresholds are exploratory diagnostics, not qualified track/turn boundaries.
    threshold = .001
    for entry in entries:
        episode = open_episode(entry, plan['body_contract'])
        mask = episode['valid'][:, 5]
        b = body_features(episode['body'], episode['body_valid'], checkpoint['body_mean'],
                          checkpoint['body_scale'], checkpoint['body_route_indices'])
        visual = np.clip((episode['x']-checkpoint['feature_mean'])
                         / checkpoint['feature_scale'], -5, 5).astype(np.float32)
        with torch.no_grad():
            bt, vt = torch.tensor(b), torch.tensor(visual)
            body = model.body(bt).numpy()[:, 5]
            combined = model(vt, bt).numpy()[:, 5]
        truth = episode['y'][mask, 5]
        predictions = {'body_only': body[mask] * checkpoint['target_scale'][5]
                       + checkpoint['target_mean'][5],
                       'visual_residual': combined[mask] * checkpoint['target_scale'][5]
                       + checkpoint['target_mean'][5]}
        masks = {'negative': truth < -threshold, 'near_zero': np.abs(truth) <= threshold,
                 'positive': truth > threshold}
        groups = {}
        for name, chosen in masks.items():
            groups[name] = {'rows': int(chosen.sum())}
            for prediction, value in predictions.items():
                correct = np.sign(value[chosen]) == np.sign(truth[chosen])
                groups[name][prediction + '_same_sign_fraction'] = (
                    float(correct.mean()) if len(correct) else None)
                groups[name][prediction + '_rmse_per_m'] = (
                    float(np.sqrt(np.mean((value[chosen]-truth[chosen])**2)))
                    if len(correct) else None)
        reports.append({'id': entry['id'], 'role': entry['role'], 'valid_rows': int(mask.sum()),
                        'groups': groups})
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Original experiment source changed during diagnostic: ' + path)
    result = {'status': 'completed_posthoc', 'threshold_abs_curvature_per_m': threshold,
        'target': '20m teacher-path screen-right chord curvature, not a verified optimal line',
        'rows_are_not_independent': True, 'development_is_known_not_final_test': True,
        'corner_distance_not_measured': True, 'actor_deployed': False,
        'optimizer_updates': 0, 'reports': reports}
    write_json(run/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
