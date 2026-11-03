"""Offline held-out-history causal probe, never an actuator or qualification."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT
from flybrain_drive.visual_yaw_bridge import RegisteredPreviewYaw
from scripts.prepare_flyvis_preview_folds import registered_cpu
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_pedal_b2 import OriginalGraphAdapter
from scripts.train_visual_body_preview import direct_body_reference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_pins(pins):
    for name, expected in pins.items():
        if sha256(name) != expected:
            raise ValueError('Original diagnostic input changed: ' + name)


def predictor(checkpoint):
    model = RegisteredPreview().cpu().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.requires_grad_(False)
    return RegisteredPreviewYaw(model, **{key: checkpoint[key] for key in (
        'feature_mean', 'feature_scale', 'body_mean', 'body_scale',
        'target_mean', 'target_scale', 'body_route_indices')}).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.expected_plan_sha256:
        raise ValueError('Exact declared diagnostic plan required')
    plan = read(args.plan)
    if (plan['heldout'] != 'episode-04-train' or plan['delay_seconds'] != 1.0
            or plan['desired_speed_m_s'] != 30 / 3.6
            or plan['role'] != 'internal_train_fold_holdout_no_actor_qualification'):
        raise ValueError('Exact original held-out history and diagnostic interventions required')
    verify_pins(plan['source_sha256'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = dict(status='failed', role=plan['role'], student_deployed=False,
                  optimizer_updates=0, actor_qualified=False, goal_achieved=False)
    try:
        dataset, replay = read(plan['dataset']), read(plan['sensor_replay'])
        entry = next(e for e in dataset['episodes'] if e['id'] == plan['heldout'])
        source = next(e for e in replay['episodes'] if e['id'] == plan['heldout'])
        if entry['role'] != 'train' or source['role'] != 'train':
            raise ValueError('Original TRAIN role required')
        fold = torch.load(plan['vision_checkpoint'], map_location='cpu', weights_only=False)
        body_only = torch.load(plan['body_checkpoint'], map_location='cpu', weights_only=False)
        if (fold['condition'] != 'vision_body' or body_only['condition'] != 'body_only'
                or fold['heldout_history'] != plan['heldout']
                or body_only['heldout_history'] != plan['heldout']
                or plan['heldout'] in fold['fitted_history_ids']
                or fold['fitted_history_ids'] != body_only['fitted_history_ids']
                or fold['actuator_policy'] is not False or body_only['actuator_policy'] is not False
                or fold['original_author_checkpoint_sha256'] != plan['author_checkpoint_sha256']):
            raise ValueError('Original fold-only calibrations and two fitted histories required')
        for name in ('feature_mean', 'feature_scale', 'body_mean', 'body_scale',
                     'target_mean', 'target_scale'):
            np.testing.assert_array_equal(fold[name], body_only[name])
        fit_plan = read(plan['pedal_fit_plan'])
        fit_result, pedal = read(plan['pedal_fit_result']), read(plan['pedal_checkpoint'])
        if (fit_result['status'] != 'completed_one_pass_pedal_component_fit'
                or fit_result['optimizer_updates'] != 123
                or pedal['schema'] != 'flybrain.two_DNg_final_candidate.v1'
                or pedal['bindings'] != fit_result['bindings']):
            raise ValueError('Exact completed original two-scalar pedal fit required')

        # Initialize the exact original graph/quiet reference before CPU pooling.
        adapter = OriginalGraphAdapter(fit_plan, lambda: None)
        with torch.no_grad():
            for name, parameter in adapter.parameters().items():
                parameter.copy_(torch.tensor(pedal['values'][name], device=parameter.device))
            quiet = adapter.initialize()
        expected_fixed = fit_result['fit']['final_fixed_rollout']['fixed_fingerprint']
        if adapter.fixed_fingerprint() != expected_fixed:
            raise ValueError('Original graph/readout and neutral reference changed')

        vision = Path(source['output'])
        if sha256(vision/'responses.npz') != source['responses_sha256']:
            raise ValueError('Original author visual response changed')
        with load_array(vision/'responses.npz', allow_pickle=False) as arrays:
            if arrays['additional_cell_types'].tolist() != ['L1', 'L2', 'Mi1', 'Tm3', 'Tm1', 'Tm9']:
                raise ValueError('Original six retinal and eight T4/T5 channels required')
            clocks = arrays['decision_monotonic_seconds'].copy()
            activity = np.concatenate(
                [arrays['additional_activity'], arrays['motion_activity']], axis=2)
        body, valid, _ = direct_body_reference(entry, clocks, sha256(plan['body_contract']))
        with load_array(vision/'retina-registration.npz', allow_pickle=False) as arrays:
            if arrays['view_order'].tolist() != ['front', 'left', 'right', 'rear']:
                raise ValueError('Original four view identities required')
            rays, origins = arrays['body_directions'].copy(), arrays['body_origins_m'].copy()
        with load_array(Path(source['recording'])/'observations.npz', allow_pickle=False) as arrays:
            np.testing.assert_array_equal(clocks, arrays['observation_time'])
            views = arrays['obs_view_valid'].all(axis=2)
        dt = np.diff(clocks)
        if (len(clocks) != entry['observations'] or len(clocks) != 739
                or not np.isfinite(dt).all() or np.any((dt <= 0) | (dt > .08))
                or not views.all() or not (valid[:, 4] & valid[:, 16]).all()):
            raise ValueError('Complete original clocks and required sensor validity required')
        episode = dict(activity=activity, clock=clocks, body=body, body_valid=valid,
                       rays=rays, origins=origins)

        def feature(values):
            return registered_cpu(dict(episode, activity=values), fold['channel_mean'],
                                  fold['channel_scale'], batch_size=8)['intact']

        intact = feature(activity)
        print(json.dumps(dict(stage='original_neural_features', rows=len(clocks))), flush=True)
        fixed = feature(np.broadcast_to(activity[:1], activity.shape))
        delay_indices = np.searchsorted(clocks, clocks - plan['delay_seconds'], side='right') - 1
        delayed = feature(activity[np.maximum(delay_indices, 0)])
        del activity
        print(json.dumps(dict(stage='visual_interventions', rows=len(clocks))), flush=True)

        visual, baseline = predictor(fold), predictor(body_only)
        body_t = torch.as_tensor(body, dtype=torch.float32)
        valid_t = torch.as_tensor(valid, dtype=torch.bool)
        views_t = torch.as_tensor(views, dtype=torch.bool)
        speed = torch.full((len(clocks),), plan['desired_speed_m_s'], dtype=torch.float32)
        goals = torch.tensor([[0., 1.]], dtype=torch.float32).repeat(len(clocks), 1)
        outputs = []
        with torch.no_grad():
            for values, intervention in ((intact, None), (fixed, None), (delayed, None),
                                          (intact, 'motion_channels')):
                outputs.append(visual(torch.tensor(values, dtype=torch.float32), body_t, valid_t,
                                      views_t, speed, goals, intervention=intervention))
            body_features = torch.as_tensor(
                np.broadcast_to(fold['feature_mean'], intact.shape).copy(), dtype=torch.float32)
            outputs.append(baseline(body_features, body_t, valid_t, views_t, speed, goals))
        commands = torch.stack([o['command'] for o in outputs], dim=1).cuda()
        body_gpu, valid_gpu, goals_gpu = body_t.cuda(), valid_t.cuda(), goals.cuda()
        nconditions = len(outputs)
        state = quiet.repeat(nconditions, 1)
        neural, dn = [], []
        with torch.no_grad():
            for index in range(len(clocks)):
                previous = max(0, index - 1)
                if index:
                    state = adapter.model.advance(commands[previous],
                        body_gpu[previous:previous+1].expand(nconditions, -1),
                        valid_gpu[previous:previous+1].expand(nconditions, -1),
                        goals_gpu[previous:previous+1].expand(nconditions, -1), state,
                        float(dt[index-1]), command_contract=MANEUVER_CONTRACT,
                        goal_contract=RACING_GOAL_CONTRACT)[2]
                currents, _ = adapter.model.command(commands[index],
                    body_gpu[index:index+1].expand(nconditions, -1),
                    valid_gpu[index:index+1].expand(nconditions, -1),
                    goals_gpu[index:index+1].expand(nconditions, -1),
                    command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
                dn.append(currents)
                neural.append(adapter.predict(state)['canonical'][:, 0])
                if index and index % 100 == 0:
                    print(json.dumps(dict(stage='flight_circuit', observations=index+1)),
                          flush=True)
        neural = torch.stack(neural).cpu().numpy()
        dn = torch.stack(dn).cpu().numpy()
        yaw = commands[:, :, 0].cpu().numpy()

        # Privileged teacher path is opened only after all actor predictions exist.
        if sha256(entry['targets_file']) != entry['targets_sha256']:
            raise ValueError('Original evaluation target changed')
        with load_array(entry['targets_file'], allow_pickle=False) as target:
            np.testing.assert_array_equal(target['observation_time'], clocks)
            curvature = target['preview'][:, 2, 1].copy()
            mask = target['preview_valid'][:, 2].copy()
        teacher_yaw = -body[:, 16] * curvature
        conditions = ['actual_visual', 'first_neural_state_held', 'past_neural_state_1s',
                      'motion_channels_at_fold_mean', 'body_only_readout']
        comparisons = {}
        for index, name in enumerate(conditions):
            comparisons[name] = dict(
                teacher_surrogate_yaw_rmse_rad_s=float(np.sqrt(np.mean(
                    (yaw[mask, index] - teacher_yaw[mask]) ** 2))),
                mean_absolute_yaw_difference_from_actual=float(np.mean(
                    np.abs(yaw[:, index] - yaw[:, 0]))),
                mean_absolute_DNa_current_difference_from_actual=float(np.mean(
                    np.abs(dn[:, index] - dn[:, 0]))),
                mean_absolute_neural_steering_difference_from_actual=float(np.mean(
                    np.abs(neural[:, index] - neural[:, 0]))))
        verify_pins(plan['source_sha256'])
        if adapter.fixed_fingerprint() != expected_fixed:
            raise ValueError('Original frozen motor model changed')
        result.update(status='completed_offline_visual_yaw_causal_replay', observations=len(clocks),
            original_intervals=len(dt), maximum_original_interval_seconds=float(dt.max()),
            valid_teacher_surrogate_labels=int(mask.sum()), conditions=conditions,
            comparisons=comparisons,
            clipping_counts=[int(o['yaw_was_clipped'].sum()) for o in outputs],
            yaw_left_rad_s=yaw.tolist(), neural_steering=neural.tolist(),
            source_clocks_sha256=hashlib.sha256(np.ascontiguousarray(clocks).tobytes()).hexdigest(),
            teacher_labels_opened_only_after_predictions=True, actor_truth_inputs_added=False,
            numerical_state_resets_within_history=0, original_pedal_scalars_preserved=True,
            trainable_weights_changed=0, exact_continuous_route_or_optimal_line_claimed=False,
            first_neural_state_held_is_not_a_first_image_resimulation=True,
            original_fold_models_were_not_qualified_as_driving_policies=True,
            plan_sha256=sha256(args.plan), program_sha256=sha256(__file__))
    except BaseException as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        result['wall_seconds'] = time.perf_counter() - start
        (out/'result.json').write_text(json.dumps(result, ensure_ascii=True, indent=2)+'\n',
                                       encoding='utf8')
    print(json.dumps(dict(status=result['status'], output=str(out),
                          result_sha256=sha256(out/'result.json'))), flush=True)


if __name__ == '__main__':
    main()
