"""Controlled R1/R2 flight-circuit experiment, offline and explicitly unqualified.

Fixed one-epoch continuation versus longer credit, then optional auxiliary loss.
Both preserve the six-muscle vehicle output, chemical circuit and body feedback.
Historical heldout is consumed development/regression, never a new test claim.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.anticipation_targets import future_command_targets, ridge_fit
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_anticipation_readout import FlightAnticipationReadout
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import (assemble_student, detach_recurrence,
                                               MuscleVehicleReadout, read)
from scripts.train_qualified_sac_unified import context, load_episode

BASE = Path('runs/teacher-synchronized-20261102')
CHECKPOINT_SHA = '24a8f4e03614636fe6eede71137faae161c8e82dcc1f212dc4bd4a75fb5203d4'


def command_metrics(prediction, target, mask):
    error = prediction[mask] - target[mask]
    return dict(weighted_mse=float(np.mean(error ** 2 * [4, 1, 1])),
                mae=np.mean(abs(error), 0).tolist(),
                rmse=np.sqrt(np.mean(error ** 2, 0)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=['trace', 'fit'], required=True)
    parser.add_argument('--trace', help='Completed trace run required for paired fit')
    parser.add_argument('--arm', choices=['continuation', 'credit32', 'auxiliary', 'all'], default='all')
    parser.add_argument('--plan', help='Immutable source/schedule plan required for fit')
    parser.add_argument('--auxiliary-initialization', choices=['zero', 'train_ridge'], default='zero')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(780304)
    np.random.seed(780304)
    checkpoint = BASE / 'student-fit-01/unified-sac-student.pt'
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('R0 checkpoint differs')
    entries = read(BASE / 'student-split-01.json')['episodes']
    if [e['role'] for e in entries] != ['train', 'train', 'heldout']:
        raise ValueError('Original whole-episode split required')
    items = [load_episode(e['data']) for e in entries]
    if any(e['arrays_sha256'] != evidence['arrays_sha256']
           or e['contract_sha256'] != evidence['contract_sha256']
           or e['episode_id'] != contract['episode_id']
           for e, (_, contract, evidence) in zip(entries, items)):
        raise ValueError('Exact chronological split evidence differs')
    if len({c['episode_id'] for _, c, _ in items}) != 3:
        raise ValueError('Independent episodes required')
    device = 'cpu'  # Existing working CPU runtime, no app-policy modifications.
    model, lineage = assemble_student()
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload['lineage'] != lineage:
        raise ValueError('Measured circuit/checkpoint lineage changed')
    model.load_state_dict(payload['model'], strict=True)
    _, muscle = model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(muscle.shape[-1])
    readout.load_state_dict(payload['readout'], strict=True)
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    goals = torch.tensor(racing_goal(300, 1))
    view = torch.ones((1, 4), dtype=torch.bool)
    weights = torch.tensor([4., 1., 1.])
    fixed = {k: v.detach().clone() for k, v in model.state_dict().items()
             if k not in {n for n, p in model.named_parameters() if p.requires_grad}}
    config = dict(arguments=vars(args), seed=780304, program_sha256=sha256(__file__),
        checkpoint_sha256=CHECKPOINT_SHA, split_sha256=sha256(BASE / 'student-split-01.json'),
        episodes=entries, device=device, lineage=lineage,
        dev_role='Historical consumed heldout, development/regression only',
        fixed_fit=dict(epochs=1, tbptt_events_by_arm=dict(continuation=8, credit32=32, auxiliary=32), adam_lr=.0001, gradient_clip=1.,
                       horizons_seconds=[.25, .5, 1., 2.], auxiliary_weight=.05,
                       auxiliary_scale_floor=1e-6),
        auxiliary_initialization=args.auxiliary_initialization,
        development_gate='When all arms are present, auxiliary weighted MSE at least 2% below credit32 and original; no axis RMSE >2% above credit32. Then complete vision/body intervention replays. Single arms require separate paired comparison. Not actor promotion.',
        actor_deployed=False, physiological_validation=False,
        prohibited_actor_inputs=['future labels', 'teacher controls', 'map', 'pose', 'world heading', 'absolute clocks'])
    if args.phase == 'fit':
        if not args.plan:
            raise ValueError('Sealed prefit source/schedule plan required')
        plan = read(args.plan)
        expected_window = 8 if args.arm == 'continuation' else 32
        if (plan['source_sha256'] != sha256(__file__) or args.arm not in plan['arms']
                or plan['arms'][args.arm] != expected_window or plan['epochs'] != 1
                or plan['adam_lr'] != .0001 or plan['seed'] != 780304
                or plan.get('auxiliary_initialization', 'zero') != args.auxiliary_initialization
                or plan['original_split_sha256'] != sha256(BASE / 'student-split-01.json')):
            raise ValueError('Sealed fit source/schedule/split differs')
        config['sealed_learning_plan_sha256'] = sha256(args.plan)
        if not args.trace:
            raise ValueError('Separate completed trace required')
        trace = Path(args.trace)
        trace_result = read(trace / 'result.json')
        trace_manifest = read(trace / 'manifest.json')
        if (trace_result['status'] != 'completed_continuous_circuit_trace'
                or sha256(trace / 'trace.npz') != trace_result['arrays_sha256']
                or trace_manifest['config']['checkpoint_sha256'] != CHECKPOINT_SHA
                or trace_manifest['config']['episodes'] != entries):
            raise ValueError('Original circuit trace provenance differs')
        with np.load(trace / 'trace.npz', allow_pickle=False) as replay, np.load(
                BASE / 'student-fit-01/heldout-predictions.npz', allow_pickle=False) as original:
            if not np.allclose(replay['baseline_2_prediction'], original['full_0'],
                               rtol=0., atol=2e-4):
                raise ValueError('R0 CPU replay differs from original beyond sealed 2e-4 command tolerance')
        config.update(trace_sha256=sha256(trace / 'trace.npz'),
                      trace_result_sha256=sha256(trace / 'result.json'))
    run = begin_run(args.output, config)
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False,
                  vehicle_control_sent=False, student_autonomous_lap_verified=False,
                  physiological_validation=False)
    outputs = {}
    start = time.monotonic()
    updates = 0

    def check_fixed():
        now = model.state_dict()
        for key, original in fixed.items():
            # Sparse tensors cannot be compared with torch.equal directly.
            value = now[key]
            if value.layout != torch.strided:
                a, b = value.to_sparse_coo().coalesce(), original.to_sparse_coo().coalesce()
                same = torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
            else:
                same = torch.equal(value, original)
            if not same:
                raise ValueError('Protected fixed state changed: ' + key)
        model.motor.backbone.assert_body_boundary_protected()

    def rollout(item, name, *, optimizer=None, auxiliary=None, calibration=None, intervention=None, window=8):
        nonlocal updates
        data, contract, _ = item
        model.reset_episode()
        scene = torch.tensor(data['scene'], dtype=torch.float32)
        label = torch.tensor(data['labels'], dtype=torch.float32)
        target, target_mask, target_index = future_command_targets(
            data['decision_time'], data['labels'], data['label_valid'], data['fit_mask'])
        future = torch.tensor(target, dtype=torch.float32)
        future_mask = torch.tensor(target_mask)
        predictions, muscles, plans, window_losses = [], [], [], []
        first, last = np.flatnonzero(data['fit_mask'])[[0, -1]]
        # Interventions are separate complete continuous replays, never fitted data.
        if intervention == 'vision':
            scene.zero_()
        body_hook = (model.body_feedback.register_forward_hook(
            lambda module, inputs, value: torch.zeros_like(value))
            if intervention == 'body_feedback' else None)
        try:
            for i in range(len(scene)):
                learning = bool(optimizer is not None and first <= i <= last)
                with torch.set_grad_enabled(learning):
                    fn = model.sensory_training_decision if learning else model.sensory_decision
                    _, activity = fn(decision_time=float(data['decision_time'][i]),
                        racing_goals=goals, scene_features=scene[i:i+1], scene_valid=view,
                        body_context=context(data, contract, i, intervention))
                    prediction = readout(activity)
                    if calibration is not None and data['fit_mask'][i]:
                        calibration.append(activity.detach().numpy().copy())
                    predictions.append(prediction[0].detach().numpy().copy())
                    if args.phase == 'trace':
                        muscles.append(activity[0].detach().numpy().copy())
                    if auxiliary is not None:
                        plan = auxiliary(activity)
                        plans.append(plan[0].detach().numpy().copy())
                    if learning and data['fit_mask'][i]:
                        loss = ((prediction[0] - label[i]).square() * weights).mean()
                        if auxiliary is not None and future_mask[i].any():
                            loss = loss + .05 * auxiliary.masked_loss(
                                plan, future[i:i+1], future_mask[i:i+1])
                        window_losses.append(loss)
                    if optimizer is not None and ((i + 1) % window == 0 or i == len(scene) - 1):
                        if window_losses:
                            optimizer.zero_grad(set_to_none=True)
                            torch.stack(window_losses).mean().backward()
                            params = [p for group in optimizer.param_groups for p in group['params']]
                            torch.nn.utils.clip_grad_norm_(params, 1.)
                            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                                raise ValueError('Nonfinite gradient')
                            optimizer.step()
                            updates += 1
                            window_losses.clear()
                        detach_recurrence(model)
                if i % 500 == 0:
                    write_json(run / 'progress.json', dict(phase=name, event=i, total=len(scene),
                        optimizer_updates=updates, episode=contract['episode_id'],
                        elapsed_seconds=time.monotonic() - start))
        finally:
            if body_hook is not None:
                body_hook.remove()
        prediction = np.asarray(predictions)
        outputs[name + '_prediction'] = prediction
        if muscles:
            outputs[name + '_muscles'] = np.asarray(muscles)
        metrics = command_metrics(prediction, data['labels'], data['fit_mask'])
        if plans:
            plans = np.asarray(plans).reshape(-1, 4, 3)
            outputs[name + '_future'] = plans
            metrics['future'] = [command_metrics(plans[:, h], target[:, h], target_mask[:, h])
                                 for h in range(4)]
        return metrics

    try:
        model.eval()
        readout.eval()
        if args.phase == 'trace':
            calibration, metrics = [], []
            for i, item in enumerate(items):
                metrics.append(rollout(item, f'baseline_{i}', calibration=calibration if i < 2 else None))
            activities = np.concatenate(calibration)
            std = activities.std(0, ddof=1)
            outputs['train_reference'], outputs['train_std'] = activities.mean(0), std
            check_fixed()
            np.savez_compressed(run / 'trace.npz', **outputs)
            result.update(status='completed_continuous_circuit_trace', metrics=metrics,
                train_muscle_std=std.tolist(),
                readout_scale_over_postfit_std=(readout.training_scale.numpy() / np.maximum(std, 1e-12)).tolist(),
                arrays_sha256=sha256(run / 'trace.npz'), original_checkpoint_unchanged=True,
                fixed_model_state_entries_verified=len(fixed), termination_reason='Three whole continuous episode replays completed')
        else:
            with np.load(trace / 'trace.npz', allow_pickle=False) as z:
                reference, scale = z['train_reference'].copy(), np.maximum(z['train_std'], 1e-6)
                baseline_muscles = [z[f'baseline_{i}_muscles'].copy() for i in range(2)]
            arms = {}
            for arm in (['continuation', 'credit32', 'auxiliary'] if args.arm == 'all' else [args.arm]):
                torch.manual_seed(780304)
                model.load_state_dict(payload['model'], strict=True)
                readout.load_state_dict(payload['readout'], strict=True)
                # Same existing circuit muscles only; this head never drives the car.
                auxiliary = (FlightAnticipationReadout(torch.tensor(reference), torch.tensor(scale))
                             if arm == 'auxiliary' else None)
                if auxiliary is not None and args.auxiliary_initialization == 'train_ridge':
                    # Fit the small prediction head on TRAIN baseline muscle states
                    # before neural optimization. This does not change actor output.
                    # Exact affine reparameterization uses the same train-only scale.
                    targets = [future_command_targets(d['decision_time'], d['labels'],
                               d['label_valid'], d['fit_mask']) for d, _, _ in items[:2]]
                    with torch.no_grad():
                        for h in range(4):
                            x = np.concatenate([m[t[1][:, h]] for m, t in zip(baseline_muscles, targets)])
                            y = np.concatenate([t[0][:, h][t[1][:, h]] for t in targets])
                            fitted = ridge_fit(x, y, .01)
                            weight = (scale / fitted['scale'])[:, None] * fitted['weights']
                            bias = fitted['target_mean'] + ((reference - fitted['mean']) / fitted['scale']) @ fitted['weights']
                            auxiliary.projection.weight[h*3:(h+1)*3].copy_(torch.tensor(weight.T, dtype=torch.float32))
                            auxiliary.projection.bias[h*3:(h+1)*3].copy_(torch.tensor(bias, dtype=torch.float32))
                params = [p for p in model.parameters() if p.requires_grad] + list(readout.parameters())
                if auxiliary is not None:
                    params += list(auxiliary.parameters())
                optimizer = torch.optim.Adam(params, lr=.0001)
                before = updates
                model.train()
                readout.train()
                train_metrics = [rollout(item, f'{arm}_train_{i}', optimizer=optimizer,
                                         auxiliary=auxiliary, window=8 if arm == 'continuation' else 32)
                                 for i, item in enumerate(items[:2])]
                model.eval()
                readout.eval()
                dev = rollout(items[2], f'{arm}_dev', auxiliary=auxiliary)
                check_fixed()
                target_path = run / f'{arm}.pt'
                torch.save(dict(schema='flybrain.anticipation_candidate.v1', model=model.state_dict(),
                    readout=readout.state_dict(), auxiliary=auxiliary.state_dict() if auxiliary is not None else None,
                    auxiliary_role='Training-only head from existing six flight muscles; never vehicle output',
                    lineage=lineage, source_checkpoint_sha256=CHECKPOINT_SHA,
                    manifest_sha256=sha256(run / 'manifest.json'), actor_qualified=False,
                    optimizer_updates=updates-before), target_path)
                arms[arm] = dict(train=train_metrics, dev=dev, updates=updates-before,
                    checkpoint_sha256=sha256(target_path),
                    sensory_max_absolute_changes={n: float((p.detach() - payload['model'][n]).abs().max())
                        for n, p in model.named_parameters() if p.requires_grad})
                write_json(run / 'arms.json', arms)
            gate = None
            if args.arm == 'all':
                control, candidate = arms['credit32']['dev'], arms['auxiliary']['dev']
                original_mse = trace_result['metrics'][2]['weighted_mse']
                gate = (candidate['weighted_mse'] <= .98 * min(control['weighted_mse'], original_mse)
                        and np.all(np.asarray(candidate['rmse']) <= 1.02 * np.asarray(control['rmse'])))
            interventions = {}
            if gate:
                for intervention in ['vision', 'body_feedback']:
                    interventions[intervention] = rollout(items[2], 'auxiliary_dev_' + intervention,
                        auxiliary=auxiliary, intervention=intervention)
            np.savez_compressed(run / 'predictions.npz', **outputs)
            result.update(status='completed_fixed_anticipation_learning', arms=arms,
                optimizer_updates=updates, arrays_sha256=sha256(run / 'predictions.npz'),
                fixed_model_state_entries_verified=len(fixed), original_checkpoint_unchanged=sha256(checkpoint)==CHECKPOINT_SHA,
                consumed_development_gate_passed=bool(gate) if gate is not None else None, interventions=interventions,
                termination_reason='Fixed paired one-epoch schedule completed',
                interpretation='Consumed single-track development only; report both arms and retain failures. No autonomous or physiological qualification.')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), optimizer_updates=updates,
                      termination_reason='Explicit failed experiment; not deployable')
    result['elapsed_seconds'] = time.monotonic() - start
    write_json(run / 'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
