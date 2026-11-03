"""Posthoc distribution and residual-error diagnostic; never fit or select a model."""

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
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')
    plan = read(args.plan)
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Original experiment source changed: ' + path)
    dataset = read(plan['dataset'])
    entries = dataset['episodes']
    if [e['role'] for e in entries] != ['train', 'train', 'train', 'development']:
        raise ValueError('Original train/development roles required')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['development_used_for_fit'] or checkpoint['actor_qualified']:
        raise ValueError('Expected fixed, unqualified readout')
    model = ResidualPreview().cpu()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.requires_grad_(False)
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'plan_sha256': sha256(args.plan), 'checkpoint_sha256': args.checkpoint_sha256,
        'posthoc_diagnostic_only': True, 'optimizer_updates': 0, 'actor_deployed': False})
    reports = []
    for entry in entries:
        episode = open_episode(entry, plan['body_contract'])
        z = ((episode['x'][:, :896] - checkpoint['feature_mean'][:896])
             / checkpoint['feature_scale'][:896])
        clipped = np.clip(z, -5, 5).astype(np.float32)
        body = body_features(episode['body'], episode['body_valid'], checkpoint['body_mean'],
                             checkpoint['body_scale'], checkpoint['body_route_indices'])
        with torch.no_grad():
            baseline = model.body(torch.tensor(body)).numpy()[:, 5]
            addition = model.vision(torch.tensor(clipped)).numpy()[:, 5]
        mask = (episode['valid'][:, 5] & episode['body_valid'][:, 4]
                & episode['body_valid'][:, 16] & (episode['body'][:, 16] > 1))
        speed = episode['body'][mask, 16]
        truth = -speed * episode['y'][mask, 5]
        body_prediction = -speed * (baseline[mask] * checkpoint['target_scale'][5]
                                    + checkpoint['target_mean'][5])
        correction = -speed * addition[mask] * checkpoint['target_scale'][5]
        body_error = body_prediction - truth
        visual_error = body_error + correction
        groups = {}
        for name, sl in [('retinal', slice(0, 384)), ('motion', slice(384, 896))]:
            v = z[:, sl]
            groups[name] = {'mean_abs_standardized': float(np.mean(np.abs(v))),
                            'fraction_outside_clip': float(np.mean(np.abs(v) > 5)),
                            'fraction_nonfinite': float(np.mean(~np.isfinite(v)))}
        reports.append({'id': entry['id'], 'role': entry['role'], 'rows': int(mask.sum()),
            'features': groups, 'body_yaw_rmse_unclipped': float(np.sqrt(np.mean(body_error**2))),
            'visual_yaw_rmse_unclipped': float(np.sqrt(np.mean(visual_error**2))),
            'correction_rms': float(np.sqrt(np.mean(correction**2))),
            'body_error_correction_mean_product': float(np.mean(body_error * correction)),
            'visual_improves_row_fraction': float(np.mean(
                np.abs(visual_error) < np.abs(body_error)))})
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Original experiment source changed during diagnostic: ' + path)
    result = {'status': 'completed_posthoc_diagnostic', 'reports': reports,
        'development_is_known_not_independent': True, 'optimizer_updates': 0,
        'model_or_hyperparameter_selection': False, 'actor_deployed': False,
        'diagnostic_is_not_a_lap_or_real_steering_measurement': True}
    write_json(run / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
