"""Fit only the existing throttle decoder on cached, recurrently replayed output neurons."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes, replay_output_activity
from flybrain_drive.throttle_readout import tensors_identical, throttle_from_activity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 2000:
        raise ValueError('Finite1..2000 decoder iterations required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(862983)
    root = Path(args.training_run)
    report = json.loads((root/'result.json').read_text(encoding='utf8'))
    checkpoint = root/'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified source required')
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    constructor = copy.deepcopy(source['constructor'])
    if (constructor['pedal_mode'] != 'independent'
            or constructor['readout_gain_parameterization'] != 'log'):
        raise ValueError('Exact independent-pedal log-gain source codec required')
    episodes, manifest = load_episodes(args.dataset, constructor['sensory_calibration'])
    if manifest.get('teacher_label_contract') != source.get('teacher_label_contract'):
        raise ValueError('Matching applied-control task required')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862983,
        source_checkpoint_sha256=sha256(checkpoint), dataset_sha256=sha256(
            Path(args.dataset)/'dataset.json'), iterations=args.iterations,
        trainable_parameters=['readout.forward_weights', 'readout.pedal_log_gain[0]'],
        cache_contract='One recurrent replay per episode, four ticks/decision, no chunk resets. '
                       'Only the existing output-neuron activity reaches the fitted decoder.'))
    started = time.perf_counter()
    graph = {key: np.asarray(value, dtype={'pre': np.int64, 'post': np.int64,
                                         'weights': np.float32}.get(key))
             for key, value in source['graph'].items()}
    constructor['sparse_backend'] = 'cached_csr_gpu'
    model = MultiViewCircuitController(graph, **constructor).cuda().eval().requires_grad_(False)
    model.load_state_dict(source['model'])
    model.chunk_validated_training = False  # use the hook-visible exact runtime path once
    cached, targets, masks, row_weights, references = [], [], [], [], []
    for episode in episodes:
        episode['observation'] = {k: value.cuda() for k, value in episode['observation'].items()}
        episode['action'] = episode['action'].cuda()
        episode['action_valid'] = episode['action_valid'].cuda()
        final = []
        hook = model.readout.register_forward_hook(
            lambda module, inputs, output, final=final: final.append(output.detach().clone()))
        try:
            activity = replay_output_activity(model, [episode], 64, 4)
        finally:
            hook.remove()
        n = len(episode['action'])
        if len(activity) != n:
            raise ValueError('Cached neural activity/episode alignment differs')
        cached.append(activity)
        references.append(torch.cat(final[3::4])[:n])
        targets.append(episode['action'][:, 1].cuda())
        mask = episode['action_valid'][:, 1].cuda()
        masks.append(mask)
        weight = 3. if episode['source'].get('source_task_failed_partial_phase') else 1.
        row_weights.append(mask.float()*weight/max(1, int(mask.sum())))
        write_json(run/'heartbeat.json', dict(phase='cache_recurrent_outputs',
            completed_episodes=len(cached), total_episodes=len(episodes)))
    activity, target, valid = torch.cat(cached), torch.cat(targets), torch.cat(masks)
    if (not valid.any() or not (valid & (target <= .005)).any()
            or not (valid & (target > .05)).any()):
        raise ValueError('Both coast and positive-throttle supervised observations required')
    weights = torch.cat(row_weights)
    weights /= weights.sum()
    reference = torch.cat(references)
    with torch.no_grad():
        original = model.readout(activity)
    torch.testing.assert_close(original, reference, rtol=1e-5, atol=1e-7)
    np.savez_compressed(run/'output-neuron-cache.npz', activity=activity.cpu().numpy(),
                        target=target.cpu().numpy(), valid=valid.cpu().numpy())
    logits = torch.nn.Parameter(model.readout.forward_weights.detach().clone())
    gain = torch.nn.Parameter(model.readout.pedal_log_gain[0].detach().clone())
    optimizer = torch.optim.Adam([dict(params=[logits], lr=.05), dict(params=[gain], lr=.01)])

    def metrics(value):
        zero, drive = valid & (target <= .005), valid & (target > .05)
        return dict(weighted_mse=float(((value-target).square()*weights).sum()),
            rmse=float((value[valid]-target[valid]).square().mean().sqrt()),
            zero_target_mean=float(value[zero].mean()) if zero.any() else None,
            drive_target_rmse=float((value[drive]-target[drive]).square().mean().sqrt())
                if drive.any() else None)

    before = metrics(original[:, 1])
    best, best_iteration = before, 0
    best_parameters = (logits.detach().clone(), gain.detach().clone())
    history = []
    for iteration in range(1, args.iterations+1):
        optimizer.zero_grad()
        predicted = throttle_from_activity(activity, model.readout.forward_ports, logits, gain)
        loss = ((predicted-target).square()*weights).sum()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([logits, gain], 5.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite throttle decoder gradient')
        optimizer.step()
        with torch.no_grad():
            gain.clamp_(math.log(1e-6), math.log(1e10))
            value = throttle_from_activity(activity, model.readout.forward_ports, logits, gain)
            current = metrics(value)
            if current['weighted_mse'] < best['weighted_mse']:
                best, best_iteration = current, iteration
                best_parameters = (logits.detach().clone(), gain.detach().clone())
        if iteration % 100 == 0 or iteration == args.iterations:
            history.append(dict(iteration=iteration, **current))
            write_json(run/'history.json', history)
            write_json(run/'heartbeat.json', dict(phase='fit_throttle_decoder',
                                                 iteration=iteration))
    with torch.no_grad():
        model.readout.forward_weights.copy_(best_parameters[0])
        model.readout.pedal_log_gain[0].copy_(best_parameters[1])
        decoded = model.readout(activity)
    if not torch.equal(decoded[:, [0, 2]], original[:, [0, 2]]):
        raise ValueError('Steering/brake decoder changed on identical neural activity')
    learned = copy.deepcopy(source)
    learned['model']['readout.forward_weights'] = best_parameters[0].cpu()
    learned['model']['readout.pedal_log_gain'][0] = best_parameters[1].cpu()
    for key, tensor in source['model'].items():
        if key not in ('readout.forward_weights', 'readout.pedal_log_gain'):
            if not tensors_identical(tensor, model.state_dict()[key].cpu()):
                raise ValueError('Parameter outside declared throttle head changed')
    for key, tensor in learned['model'].items():
        if not tensors_identical(tensor, model.state_dict()[key].cpu()):
            raise ValueError('Saved candidate differs from fitted model')
    if not torch.equal(source['model']['readout.pedal_log_gain'][1],
                       learned['model']['readout.pedal_log_gain'][1]):
        raise ValueError('Brake gain changed')
    learned['training_configuration'] = dict(source['training_configuration'],
        seed=862983, learning_rate=.05, readout_gain_learning_rate=.01,
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha256(checkpoint),
        optimizer_update_scope='cached_throttle_fullbatch', optimizer_updates=args.iterations,
        trainable_groups=['readout'], epochs=1, decoder_iterations=args.iterations,
        chunk_validated_training=False, readout_only_cached_optimization=True,
        resource_experiment=dict(scope=__doc__, workers=1), scope=__doc__)
    learned['training_teacher_manifest_sha256'] = sha256(Path(args.dataset)/'dataset.json')
    torch.save(learned, run/'checkpoint.pt')
    eligible = (best['weighted_mse'] < before['weighted_mse']
        and best['zero_target_mean'] <= before['zero_target_mean']*.5
        and best['drive_target_rmse'] <= before['drive_target_rmse']*1.1)
    write_json(run/'result.json', dict(status='completed', before=before, after=best,
        selected_iteration=best_iteration, optimizer_updates=args.iterations,
        checkpoint_sha256=sha256(run/'checkpoint.pt'),
        cache_sha256=sha256(run/'output-neuron-cache.npz'),
        cache_prediction_max_error=float((original-reference).abs().max()),
        graph_and_sensory_parameters_unchanged=True, steering_and_brake_parameters_unchanged=True,
        candidate_eligible_for_bounded_trial=eligible, training_device='cuda:0',
        optimizer_parameter_devices=sorted({str(logits.device), str(gain.device)}),
        gradient_devices=sorted({str(logits.grad.device), str(gain.grad.device)}),
        cached_output_neuron_device=str(activity.device),
        frozen_parameter_gradients_absent=all(p.grad is None for p in model.parameters()),
        device_name=torch.cuda.get_device_name(0), seconds=time.perf_counter()-started,
        trainable_parameters=['readout.forward_weights', 'readout.pedal_log_gain[0]'],
        ui_actions_sent=0, actuator_messages_sent=0,
        closed_loop_verified=False, full_lap_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
