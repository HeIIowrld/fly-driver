"""Evaluate fixed visual residual method with whole-history TRAIN-only folds."""

import argparse
import json

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_residual_preview import ResidualPreview
from scripts.train_visual_body_preview import body_features, body_moments
from scripts.train_visual_residual_preview import open_episode, read, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Experiment plan changed')
    plan = read(args.plan)
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Source changed: ' + path)
    entries = read(plan['dataset'])['episodes']
    entries = [entry for entry in entries if entry['role'] == 'train']
    if [e['id'] for e in entries] != ['episode-04-train', 'episode-05-train',
                                      'episode-06-train']:
        raise ValueError('Expected three original training histories')
    torch.set_num_threads(1)
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'plan_sha256': args.plan_sha256, 'development_opened': False,
        'fixed_algorithm_no_hyperparameter_selection': True, 'actor_deployed': False})
    episodes = [open_episode(e, plan['body_contract']) for e in entries]
    reports = []
    for held_index, held in enumerate(episodes):
        train = [e for i, e in enumerate(episodes) if i != held_index]
        xraw = np.concatenate([e['x'] for e in train]).astype(np.float64)
        target = np.concatenate([e['y'] for e in train]).astype(np.float64)
        valid = np.concatenate([e['valid'] for e in train])
        feature_mean = xraw.mean(axis=0)
        feature_scale = np.maximum(xraw.std(axis=0), 1e-5)
        feature_mean[896:], feature_scale[896:] = 0, 1
        body_mean, body_scale, _ = body_moments(train)
        target_mean = np.array([target[valid[:, h], h].mean() for h in range(6)])
        target_scale = np.maximum([target[valid[:, h], h].std() for h in range(6)],
                                  plan['target_scale_floor'])
        body = np.concatenate([body_features(e['body'], e['body_valid'], body_mean,
            body_scale, plan['body_route_indices']) for e in train])
        x = np.clip((xraw-feature_mean)/feature_scale, -5, 5).astype(np.float32)
        y = ((target-target_mean)/target_scale).astype(np.float32)
        weight = valid.astype(np.float32)/(6*valid.sum(axis=0))
        torch.manual_seed(plan['seed'])
        model = ResidualPreview().cpu()
        optimizer = torch.optim.AdamW(model.body.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['body_weight_decay'])
        xb, yt, wt = torch.tensor(body), torch.tensor(y), torch.tensor(weight)
        for _ in range(plan['body_updates']):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model.body(xb)-yt).square()*wt).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            baseline = model.body(xb).numpy()
        design = x[:, :896].astype(np.float64)
        coefficients = []
        for h in range(6):
            chosen = valid[:, h]
            d = design[chosen]
            gram = d.T @ d
            gram.flat[::len(gram)+1] += int(chosen.sum()) * plan['ridge_lambda_per_valid_cell']
            right = d.T @ (y[chosen, h]-baseline[chosen, h])
            coefficients.append(np.linalg.solve(gram, right))
        with torch.no_grad():
            model.vision.weight.copy_(torch.tensor(np.asarray(coefficients), dtype=torch.float32))
            held_body = body_features(held['body'], held['body_valid'], body_mean,
                                      body_scale, plan['body_route_indices'])
            held_x = np.clip((held['x']-feature_mean)/feature_scale, -5, 5).astype(np.float32)
            predictions = {'body_only': model.body(torch.tensor(held_body)).numpy(),
                'visual_residual': model(torch.tensor(held_x), torch.tensor(held_body)).numpy()}
        reports.append({'held_out': held['id'], 'fit_histories': [e['id'] for e in train],
                        'metrics': score(held, predictions, target_mean, target_scale)})
        print(json.dumps(reports[-1]), flush=True)
    for path, expected in plan['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Source changed during run: ' + path)
    write_json(run/'result.json', {'status': 'completed_train_only_three_history_folds',
        'reports': reports, 'body_optimizer_updates_per_fold': plan['body_updates'],
        'ridge_heads_per_fold': 6, 'original_author_optimizer_updates': 0,
        'development_opened': False, 'actor_deployed': False, 'goal_achieved': False,
        'folds_are_not_independent_final_tests': True,
        'teacher_path_is_certified_optimal': False})


if __name__ == '__main__':
    main()
