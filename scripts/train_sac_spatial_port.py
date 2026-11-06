"""Paired offline residual-vision experiment on the fixed flight circuit.

Both arms refit the original six-muscle vehicle readout. Only the residual arm
learns a zero-initialized 256->42 current-camera adapter at existing visual ports.
All old sensory/body parameters and chemical graph remain EXACTLY unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.visual_descending_port import VisualDescendingPort
from flybrain_drive.sac_unified_student import assemble_student, MuscleVehicleReadout, detach_recurrence, read
from scripts.improve_sac_anticipation import command_metrics
from scripts.train_qualified_sac_unified import context, load_episode
from scripts.trace_sac_premuscle import CHECKPOINT, CHECKPOINT_SHA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['control', 'residual'], required=True)
    p.add_argument('--route', choices=['optical', 'descending'], default='optical')
    p.add_argument('--spatial', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(780405)
    np.random.seed(780405)
    plan = read(a.plan)
    if (plan['program_sha256'] != sha256(__file__) or plan['adapter_sha256'] != sha256('src/flybrain_drive/retinal_residual_port.py')
            or plan['checkpoint_sha256'] != CHECKPOINT_SHA or plan['seed'] != 780405
            or plan['window'] != 32 or plan['epochs'] != 1 or plan.get('route', 'optical') != a.route):
        raise ValueError('Sealed exact source and schedule required')
    if a.route == 'descending' and plan['descending_adapter_sha256'] != sha256('src/flybrain_drive/visual_descending_port.py'):
        raise ValueError('Sealed visual-intent boundary source differs')
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Original best candidate changed')
    entries = read('runs/teacher-synchronized-20261102/student-split-01.json')['episodes']
    items = [load_episode(e['data']) for e in entries]
    root = Path(a.spatial)
    spatial_result = read(root/'result.json')
    if spatial_result['status'] != 'completed_qualified_static_spatial_recovery' or sha256(root/'result.json') != plan['spatial_result_sha256']:
        raise ValueError('Qualified original retinal decomposition required')
    extras = []
    for i, ((d, c, proof), entry) in enumerate(zip(items, entries)):
        if proof['arrays_sha256'] != entry['arrays_sha256'] or c['episode_id'] != entry['episode_id']:
            raise ValueError('Original independent source split differs')
        file = root/f'spatial-{i}.npz'
        if sha256(file) != spatial_result['files_sha256'][file.name]:
            raise ValueError('Verified spatial source changed')
        with np.load(file, allow_pickle=False) as z:
            joined = z['event_to_camera']
            np.testing.assert_array_equal(z['camera_indices'][joined], d['camera_index'])
            residual = z['discarded_grid8'][joined].copy()
            if a.route == 'descending':
                intensity = d['scene'].reshape(-1, 4, 9, 16)[:, :, 0].reshape(-1, 64)
                residual = np.concatenate([intensity, residual], axis=1)
            extras.append(residual)
    fit_features = np.concatenate([x[d['fit_mask']] for x, (d, _, _) in zip(extras[:2], items[:2])])
    reference = torch.tensor(fit_features.mean(0), dtype=torch.float32)
    scale = torch.tensor(fit_features.std(0), dtype=torch.float32).clamp_min(1e-4)
    model, lineage = assemble_student()
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False)
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    _, muscles = model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(muscles.shape[-1])
    readout.load_state_dict(payload['readout'], strict=True)
    adapter = (RetinalResidualPort(reference, scale, len(model.motor.backbone.vision_ports))
               if a.route == 'optical' else VisualDescendingPort(reference, scale, model.motor.command.orientations,
                                                               maximum=model.motor.command.turn_gain))
    adapter.requires_grad_(a.arm == 'residual')
    parameters = list(readout.parameters()) + ([adapter.projection.weight] if a.arm == 'residual' else [])
    optimizer = torch.optim.Adam([dict(params=list(readout.parameters()), lr=.0001)] +
        ([dict(params=[adapter.projection.weight], lr=.001)] if a.arm == 'residual' else []))
    run = begin_run(a.output, dict(arguments=vars(a), seed=780405, program_sha256=sha256(__file__),
        plan_sha256=sha256(a.plan), checkpoint_sha256=CHECKPOINT_SHA, entries=entries, lineage=lineage,
        feature=('256 fixed grid8 narrow summaries of exact old-pool-nullspace static retinal intensity'
                 if a.route == 'optical' else '64 original static intensity plus256 discarded static spatial features'),
        route=a.route, tonic_current=.001 if a.route == 'descending' else None,
        target_port_ids=(model.motor.backbone.spec['vision_input_ids'] if a.route == 'optical'
                         else model.motor.backbone.spec['steering_input_ids']),
        hypothesis='Learned external visual intent at existing flight DN cut, not measured visual-DN connectivity' if a.route == 'descending' else 'Extra static detail at existing visual cut',
        train_only_mean=reference.tolist(), train_only_scale=scale.tolist(),
        readout_lr=.0001, adapter_lr=.001, epochs=1, tbptt_events=32, maximum_logit_delta=.5,
        existing_model_parameters='ALL frozen including old body and visual adapters',
        actor_deployed=False, new_biological_connections=False, vehicle_control_sent=False))
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False)
    goals, views = torch.tensor(racing_goal(300, 1)), torch.ones(1, 4, dtype=torch.bool)
    holder = dict(value=None)
    def augment(module, inputs, output):
        if holder['value'] is None:
            raise ValueError('Explicit original-current-camera extra features absent')
        if a.route == 'optical':
            return output + adapter(holder['value'])
        steering, power = output
        return steering + adapter(holder['value']), power
    hook = (model.visual if a.route == 'optical' else model.motor.command).register_forward_hook(augment)
    updates, start, saved = 0, time.monotonic(), {}

    def rollout(index, learn=False, stop=None):
        nonlocal updates
        d, contract, _ = items[index]
        x = torch.tensor(d['scene'], dtype=torch.float32)
        extra = torch.tensor(extras[index], dtype=torch.float32)
        label = torch.tensor(d['labels'], dtype=torch.float32)
        first, last = np.flatnonzero(d['fit_mask'])[[0, -1]]
        model.reset_episode()
        predictions, losses = [], []
        for i in range(len(x) if stop is None else stop):
            learning = bool(learn and first <= i <= last)
            holder['value'] = extra[i:i+1]
            with torch.set_grad_enabled(learning):
                fn = model.sensory_training_decision if learning else model.sensory_decision
                _, muscle = fn(decision_time=float(d['decision_time'][i]), racing_goals=goals,
                    scene_features=x[i:i+1], scene_valid=views, body_context=context(d, contract, i))
                pred = readout(muscle)
                predictions.append(pred[0].detach().numpy().copy())
                if learning and d['fit_mask'][i]:
                    losses.append(((pred[0]-label[i]).square()*pred.new_tensor([4., 1., 1.])).mean())
                if learn and ((i+1) % 32 == 0 or i == len(x)-1):
                    if losses:
                        optimizer.zero_grad(set_to_none=True)
                        torch.stack(losses).mean().backward()
                        torch.nn.utils.clip_grad_norm_(parameters, 1.)
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                            raise ValueError('Nonfinite gradient')
                        optimizer.step()
                        updates += 1
                        losses.clear()
                    detach_recurrence(model)
            if i % 500 == 0:
                write_json(run/'progress.json', dict(arm=a.arm, episode=index, event=i,
                    phase='training' if learn else 'evaluation', optimizer_updates=updates,
                    elapsed_seconds=time.monotonic()-start))
        holder['value'] = None
        return np.asarray(predictions)

    try:
        # Engineering zero-neutrality check uses no fit or threshold selection.
        initial = rollout(2, stop=64)
        with np.load('runs/visual-anticipation-20261103/continuation8-02/predictions.npz', allow_pickle=False) as z:
            error = float(abs(initial-z['continuation_dev_prediction'][:64]).max())
        if a.route == 'optical' and error > 2e-6:
            raise ValueError('Zero adapter changed original numerical prefix')
        reports = []
        for i in [0, 1]:
            prediction = rollout(i, True)
            saved[f'train_{i}'] = prediction
            reports.append(command_metrics(prediction, items[i][0]['labels'], items[i][0]['fit_mask']))
        prediction = rollout(2)
        saved['dev'] = prediction
        dev = command_metrics(prediction, items[2][0]['labels'], items[2][0]['fit_mask'])
        # Byte-exact all old model state: both learned boundaries and measured
        # chemical weights are preserved, not merely the protected body subset.
        for name, value in model.state_dict().items():
            original = payload['model'][name]
            if value.layout != torch.strided:
                x, y = value.to_sparse_coo().coalesce(), original.to_sparse_coo().coalesce()
                same = torch.equal(x.indices(), y.indices()) and torch.equal(x.values(), y.values())
            else:
                same = torch.equal(value, original)
            if not same:
                raise ValueError('Frozen original state changed: '+name)
        model.motor.backbone.assert_body_boundary_protected()
        checkpoint = run/'spatial-port.pt'
        torch.save(dict(schema='flybrain.static_retinal_residual_port.v1', arm=a.arm, route=a.route,
            baseline_checkpoint_sha256=CHECKPOINT_SHA, adapter=adapter.state_dict(),
            readout=readout.state_dict(), actor_qualified=False, optimizer_updates=updates,
            source_manifest_sha256=sha256(run/'manifest.json')), checkpoint)
        np.savez_compressed(run/'predictions.npz', **saved)
        result.update(status='completed_fixed_spatial_port_learning', optimizer_updates=updates,
            train=reports, dev=dev, checkpoint_sha256=sha256(checkpoint),
            predictions_sha256=sha256(run/'predictions.npz'),
            zero_neutral_prefix_max_error=error if a.route == 'optical' else None,
            initial_source_prefix_max_error=error,
            initial_prefix_prediction_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
            old_model_state_entries_exact=len(payload['model']), protected_body_and_all_old_adapters_unchanged=True,
            adapter_weight_max=float(adapter.projection.weight.detach().abs().max()),
            termination_reason='One fixed epoch on both train episodes and full consumed development replay completed')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), optimizer_updates=updates,
                      termination_reason='Explicit candidate failure')
    finally:
        hook.remove()
        holder['value'] = None
    result['elapsed_seconds'] = time.monotonic()-start
    write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
