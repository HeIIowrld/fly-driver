"""One fixed TRAIN-only body/retinal-residual experiment; DEV is evaluation only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_residual_preview import ResidualPreview
from scripts.train_visual_body_preview import body_features, body_moments, direct_body_reference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def open_episode(entry, body_contract):
    for key in ('features', 'targets'):
        if sha256(entry[key + '_file']) != entry[key + '_sha256']:
            raise ValueError('Original role-bound feature or teacher target changed')
    with load_array(entry['features_file'], allow_pickle=False) as x:
        clock, features = x['observation_time'].copy(), x['registered'].copy()
    with load_array(entry['targets_file'], allow_pickle=False) as y:
        np.testing.assert_array_equal(clock, y['observation_time'])
        target = y['preview'].reshape(-1, 6).copy()
        valid = np.repeat(y['preview_valid'], 2, axis=1)
    body, body_valid, _ = direct_body_reference(entry, clock, sha256(body_contract))
    if (features.shape != (len(clock), 1024) or target.shape != (len(clock), 6)
            or valid.dtype != bool or not np.isfinite(features).all()
            or not np.isfinite(target).all() or len(clock) != entry['observations']):
        raise ValueError('Complete original registered history and target masks required')
    return dict(entry, clock=clock, x=features, y=target, valid=valid,
                body=body, body_valid=body_valid)


def score(episode, predictions, target_mean, target_scale):
    valid = episode['valid']
    body, body_valid = episode['body'], episode['body_valid']
    result = {}
    for name, normalized in predictions.items():
        physical = normalized * target_scale + target_mean
        standardized = [float(np.mean(((physical[valid[:, h], h] - episode['y'][valid[:, h], h])
                                        / target_scale[h]) ** 2)) for h in range(6)]
        mask = valid[:, 5] & body_valid[:, 4] & body_valid[:, 16] & (body[:, 16] > 1)
        reference = -body[mask, 16] * episode['y'][mask, 5]
        estimated = np.clip(-body[mask, 16] * physical[mask, 5], -2, 2)
        result[name] = dict(standardized_mse=float(np.mean(standardized)),
            valid_20m_yaw_rows=int(mask.sum()),
            teacher_surrogate_yaw_rmse_rad_s=float(np.sqrt(np.mean((estimated-reference)**2))),
            curvature20_rmse_per_m=float(np.sqrt(np.mean(
                (physical[valid[:, 5], 5]-episode['y'][valid[:, 5], 5])**2))),
            yaw_clipped_rows=int((np.abs(body[mask, 16] * physical[mask, 5]) > 2).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.expected_plan_sha256:
        raise ValueError('Exact preregistered single-experiment plan required')
    plan = read(args.plan)
    if (plan['body_updates'] != 200 or plan['seed'] != 863692
            or plan['ridge_lambda_per_valid_cell'] != 1.0
            or plan['body_route_indices'] != [0, 1, 4, 16]
            or plan['visual_residual_indices'] != [0, 896]
            or plan['development_role'] != 'development_evaluation_only'):
        raise ValueError('Declared fixed training, visual and role contracts required')
    for name, expected in plan['source_sha256'].items():
        if sha256(name) != expected:
            raise ValueError('Declared code or dataset metadata changed: ' + name)
    dataset = read(plan['dataset'])
    if (dataset.get('schema') != 'flybrain.body_registered_preview.v1'
            or dataset.get('feature_count') != 1024
            or dataset.get('neural_feature_count') != 896
            or dataset.get('support_feature_count') != 128
            or dataset.get('original_targets_clocks_roles_unchanged') is not True):
        raise ValueError('Exact original registered feature and role contract required')
    entries = dataset['episodes']
    train_entries = [e for e in entries if e['role'] == 'train']
    development = [e for e in entries if e['role'] == 'development']
    if ([e['id'] for e in train_entries] != [
            'episode-04-train', 'episode-05-train', 'episode-06-train']
            or len(development) != 1 or len(entries) != 4):
        raise ValueError('Exact original TRAIN3 and development roles required')
    torch.set_num_threads(1)
    started = time.perf_counter()
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), known_development_used_for_fit=False,
        original_payloads_copied=0, author_network_weights_changed=False,
        teacher_path_actor_input=False))
    result = dict(status='failed', role='experimental_preview_readout', actor_deployed=False,
        goal_achieved=False, original_author_optimizer_updates=0,
        known_development_used_for_fit=False, development_is_independent_final_test=False)
    try:
        # No development payload or moments are read before the model is fixed.
        train = [open_episode(e, plan['body_contract']) for e in train_entries]
        features = np.concatenate([e['x'] for e in train]).astype(np.float64)
        target = np.concatenate([e['y'] for e in train]).astype(np.float64)
        valid = np.concatenate([e['valid'] for e in train])
        if not valid.sum(axis=0).all():
            raise ValueError('Every original target head needs TRAIN support')
        feature_mean = features.mean(axis=0)
        feature_scale = np.maximum(features.std(axis=0), 1e-5)
        feature_mean[896:], feature_scale[896:] = 0, 1
        body_mean, body_scale, _ = body_moments(train)
        target_mean = np.array([target[valid[:, h], h].mean() for h in range(6)])
        target_scale = np.maximum([target[valid[:, h], h].std() for h in range(6)],
                                  plan['target_scale_floor'])
        inputs = np.concatenate([body_features(e['body'], e['body_valid'], body_mean,
            body_scale, plan['body_route_indices']) for e in train])
        x = np.clip((features-feature_mean)/feature_scale, -5, 5).astype(np.float32)
        y = ((target-target_mean)/target_scale).astype(np.float32)
        weight = valid.astype(np.float32)/(6*valid.sum(axis=0))
        torch.manual_seed(plan['seed'])
        model = ResidualPreview().cpu()
        optimizer = torch.optim.AdamW(model.body.parameters(), lr=plan['learning_rate'],
                                      weight_decay=plan['body_weight_decay'])
        xb, yt, wt = torch.tensor(inputs), torch.tensor(y), torch.tensor(weight)
        for index in range(plan['body_updates']):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model.body(xb)-yt).square()*wt).sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite original TRAIN body loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all()
                   for p in model.body.parameters()):
                raise ValueError('Missing or nonfinite body gradient')
            optimizer.step()
            if (index+1) % 50 == 0:
                print(json.dumps(dict(stage='body_fit', optimizer_updates=index+1)), flush=True)
        model.eval()
        with torch.no_grad():
            body_prediction = model.body(xb).numpy()
        # Fixed ridge penalty = number of valid TRAIN cells; no DEV selection.
        design = x[:, :896].astype(np.float64)
        coefficients, solve_errors = [], []
        for h in range(6):
            selected = valid[:, h]
            design_h = design[selected]
            gram = design_h.T @ design_h
            gram.flat[::len(gram)+1] += int(selected.sum()) * plan['ridge_lambda_per_valid_cell']
            right = design_h.T @ (y[selected, h]-body_prediction[selected, h])
            weights = np.linalg.solve(gram, right)
            solve_errors.append(float(np.linalg.norm(gram @ weights-right) /
                                      max(np.linalg.norm(right), 1e-12)))
            if not np.isfinite(weights).all() or solve_errors[-1] > 1e-8:
                raise ValueError('Finite accurate TRAIN-only ridge solve required')
            coefficients.append(weights)
        with torch.no_grad():
            model.vision.weight.copy_(torch.tensor(np.asarray(coefficients), dtype=torch.float32))
        model.requires_grad_(False)
        checkpoint = run/'preview-readout.pt'
        torch.save(dict(schema='flybrain.visual.body_residual_preview.v1',
            model=model.state_dict(), feature_mean=feature_mean, feature_scale=feature_scale,
            body_mean=body_mean, body_scale=body_scale, target_mean=target_mean,
            target_scale=target_scale, body_route_indices=plan['body_route_indices'],
            fitted_history_ids=[e['id'] for e in train], development_used_for_fit=False,
            original_author_weights_changed=False, actor_qualified=False,
            teacher_targets_are_optimal_racing_line=False), checkpoint)
        print(json.dumps(dict(stage='fixed_model', body_updates=200, ridge_heads=6)), flush=True)

        # The existing DEV role is opened only after all weights and moments are fixed.
        evaluation = open_episode(development[0], plan['body_contract'])
        reports = []
        for episode in [*train, evaluation]:
            b = body_features(episode['body'], episode['body_valid'], body_mean,
                              body_scale, plan['body_route_indices'])
            visual = np.clip((episode['x']-feature_mean)/feature_scale, -5, 5).astype(np.float32)
            zero = visual.copy()
            zero[:, :896] = 0
            motion = visual.copy()
            motion[:, 384:896] = 0
            with torch.no_grad():
                bt = torch.tensor(b)
                predictions = dict(body_only=model.body(bt).numpy(),
                    visual_residual=model(torch.tensor(visual), bt).numpy(),
                    neural_mean=model(torch.tensor(zero), bt).numpy(),
                    motion_at_mean=model(torch.tensor(motion), bt).numpy())
            if not np.array_equal(predictions['body_only'], predictions['neural_mean']):
                raise ValueError('Zero visual residual must exactly reproduce body baseline')
            reports.append(dict(id=episode['id'], role=episode['role'],
                observations=len(episode['clock']), metrics=score(
                    episode, predictions, target_mean, target_scale)))
        for name, expected in plan['source_sha256'].items():
            if sha256(name) != expected:
                raise ValueError('Source changed during fixed experiment: ' + name)
        result.update(status='completed_train_only_visual_residual_preview',
            original_external_roles_unchanged=True,
            observations_retained=sum(len(e['clock']) for e in train),
            body_optimizer_updates=plan['body_updates'], ridge_heads=6,
            ridge_relative_solve_errors=solve_errors, checkpoint_sha256=sha256(checkpoint),
            fitted_history_ids=[e['id'] for e in train],
            development_ids=[e['id'] for e in development],
            development_opened_only_after_fixed_model=True, target_clipping=False,
            teacher_path_is_optimal_racing_line=False, teacher_path_actor_input=False,
            visual_support_flags_used_by_residual=False, reports=reports,
            visual_readout_only_not_author_network_retraining=True,
            automatic_model_selection_or_retuning=False)
    except BaseException as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        write_json(run/'result.json', result)
    print(json.dumps(dict(status=result['status'], output=str(run),
                          result_sha256=sha256(run/'result.json'))), flush=True)


if __name__ == '__main__':
    main()
