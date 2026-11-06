"""Replay spatial/visual-intent candidates with explicit information interventions."""
import argparse
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
from flybrain_drive.sac_unified_student import assemble_student, MuscleVehicleReadout, read
from scripts.trace_sac_premuscle import CHECKPOINT, CHECKPOINT_SHA
from scripts.train_qualified_sac_unified import context, load_episode
from scripts.improve_sac_anticipation import command_metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', required=True)
    p.add_argument('--case', choices=['normal', 'new_path_neutral', 'camera_zero', 'body_feedback_zero', 'tilt_load_train_mean'], required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    root = Path(a.fit)
    result = read(root/'result.json')
    checkpoint = root/'spatial-port.pt'
    if result['status'] != 'completed_fixed_spatial_port_learning' or sha256(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Completed exact candidate required')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    route = payload.get('route', 'optical')
    if payload['baseline_checkpoint_sha256'] != CHECKPOINT_SHA or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Exact original source model required')
    cfg = read(root/'manifest.json')['config']
    entry = cfg['entries'][2]
    d, contract, evidence = load_episode(entry['data'])
    context_data = d
    if a.case == 'tilt_load_train_mean':
        context_data = {key: value.copy() for key, value in d.items()}
        indices = [0, 1, 8, 9, 10, 11]
        train = [load_episode(e['data'])[0] for e in cfg['entries'][:2]]
        replacement = np.concatenate([x['body'][x['fit_mask']][:, indices] for x in train]).mean(0)
        context_data['body'][:, indices] = replacement
        context_data['view_body'][:, :, indices] = replacement
    spatial = Path(cfg['arguments']['spatial'])
    sr = read(spatial/'result.json')
    file = spatial/'spatial-2.npz'
    if sha256(file) != sr['files_sha256'][file.name]:
        raise ValueError('Original retinal residual source differs')
    with np.load(file, allow_pickle=False) as z:
        joined = z['event_to_camera']
        np.testing.assert_array_equal(z['camera_indices'][joined], d['camera_index'])
        extra = z['discarded_grid8'][joined].copy()
    if route == 'descending':
        extra = np.concatenate([d['scene'].reshape(-1, 4, 9, 16)[:, :, 0].reshape(-1, 64), extra], axis=1)
    model, _ = assemble_student()
    base = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(base['model'], strict=True)
    model.eval().requires_grad_(False)
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    _, muscle = model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(muscle.shape[-1])
    readout.load_state_dict(payload['readout'], strict=True)
    readout.eval().requires_grad_(False)
    ac = payload['adapter']
    adapter = (RetinalResidualPort(ac['reference'], ac['scale'], len(model.motor.backbone.vision_ports))
               if route == 'optical' else VisualDescendingPort(ac['reference'], ac['scale'], ac['orientations']))
    adapter.load_state_dict(ac, strict=True)
    adapter.eval().requires_grad_(False)
    run = begin_run(a.output, dict(arguments=vars(a), seed=780408, program_sha256=sha256(__file__),
        checkpoint_sha256=sha256(checkpoint), fit_result_sha256=sha256(root/'result.json'), route=route,
        scope='Whole consumed development replay, information intervention only, not actual camera failure',
        optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False))
    out = dict(status='failed', optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False)
    holder = dict(value=None)
    def hook_fn(module, inputs, value):
        if holder['value'] is None:
            raise ValueError('Explicit original-current-camera extra input absent')
        if route == 'optical':
            delta = adapter(holder['value'])
            if a.case == 'new_path_neutral':
                delta.zero_()
            return value+delta
        steering, power = value
        current = adapter(holder['value'])
        if a.case == 'new_path_neutral':
            current = (adapter.maximum*adapter.tonic_logit.sigmoid()).expand_as(current)
        return steering+current, power
    hook = (model.visual if route == 'optical' else model.motor.command).register_forward_hook(hook_fn)
    body_hook = (model.body_feedback.register_forward_hook(lambda module, inputs, value: torch.zeros_like(value))
                 if a.case == 'body_feedback_zero' else None)
    start = time.monotonic()
    try:
        scene = torch.tensor(d['scene'], dtype=torch.float32)
        extra = torch.tensor(extra, dtype=torch.float32)
        if a.case == 'camera_zero':
            scene.zero_()
            extra.zero_()
        goals, views = torch.tensor(racing_goal(300, 1)), torch.ones(1, 4, dtype=torch.bool)
        predictions, dn_states, dn_currents = [], [], []
        model.reset_episode()
        for i in range(len(scene)):
            holder['value'] = extra[i:i+1]
            _, muscle = model.sensory_decision(decision_time=float(d['decision_time'][i]),
                racing_goals=goals, scene_features=scene[i:i+1], scene_valid=views,
                body_context=context(context_data, contract, i,
                    'body_feedback' if a.case == 'body_feedback_zero' else None))
            predictions.append(readout(muscle)[0].numpy().copy())
            if d['fit_mask'][i]:
                dn_states.append(model.state[0, model.motor.backbone.steering_ports].numpy().copy())
                dn_currents.append(model.previous_drive[0, model.motor.backbone.steering_ports].numpy().copy())
            if i % 500 == 0:
                write_json(run/'progress.json', dict(event=i, total=len(scene), elapsed_seconds=time.monotonic()-start))
        prediction = np.asarray(predictions)
        np.savez_compressed(run/'predictions.npz', prediction=prediction, dn_state=np.asarray(dn_states), dn_current=np.asarray(dn_currents))
        metrics = command_metrics(prediction, d['labels'], d['fit_mask'])
        if a.case == 'normal':
            with np.load(root/'predictions.npz', allow_pickle=False) as z:
                delta = float(abs(z['dev']-prediction).max())
            if delta > 2e-6:
                raise ValueError('Full original candidate replay differs')
        else:
            delta = None
        out.update(status='completed_spatial_port_information_replay', metrics=metrics,
            normal=result['dev'], weighted_mse_ratio=metrics['weighted_mse']/result['dev']['weighted_mse'],
            normal_replay_max_error=delta, predictions_sha256=sha256(run/'predictions.npz'),
            dn_model_rate_min=float(np.asarray(dn_states).min()), dn_model_rate_max=float(np.asarray(dn_states).max()),
            dn_fraction_above_1e_8=float((np.asarray(dn_states)>1e-8).mean()),
            interpretation='Recorded input-current and rate-surrogate response, not real neuron physiology or autonomous driving',
            termination_reason='Whole original episode replayed with one declared information condition')
    except Exception as error:
        out.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit replay failure')
    finally:
        hook.remove()
        if body_hook is not None:
            body_hook.remove()
        holder['value'] = None
    out['elapsed_seconds'] = time.monotonic()-start
    write_json(run/'result.json', out)
    print(json.dumps(out))
    if out['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
