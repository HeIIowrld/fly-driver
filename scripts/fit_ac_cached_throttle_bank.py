"""Finite coast-loss comparison on verified output neurons; unchanged acceptance gates."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes
from flybrain_drive.throttle_readout import tensors_identical, throttle_bank_from_activity


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-run', required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 2000:
        raise ValueError('Finite1..2000 decoder updates required')
    torch.set_num_threads(1)
    torch.manual_seed(862988)
    torch.set_float32_matmul_precision('highest')
    cache_root = Path(args.cache_run)
    receipt = read(cache_root/'result.json')
    provenance = read(cache_root/'manifest.json')['config']
    cache_path = cache_root/'output-neuron-cache.npz'
    source_root = Path(provenance['arguments']['training_run'])
    dataset = Path(provenance['arguments']['dataset'])
    source_receipt = read(source_root/'result.json')
    checkpoint = source_root/'checkpoint.pt'
    if (receipt['status'] != 'completed' or receipt['cache_prediction_max_error'] != 0
            or sha256(cache_path) != receipt['cache_sha256']
            or source_receipt['status'] != 'completed'
            or sha256(checkpoint) != provenance['source_checkpoint_sha256']
            or sha256(checkpoint) != source_receipt['checkpoint_sha256']
            or sha256(dataset/'dataset.json') != provenance['dataset_sha256']):
        raise ValueError('Unchanged verified source, recurrent output cache and dataset required')
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    c = source['constructor']
    if c['pedal_mode'] != 'independent' or c['readout_gain_parameterization'] != 'log':
        raise ValueError('Existing independent log-gain throttle codec required')
    episodes, manifest = load_episodes(dataset, c['sensory_calibration'])
    if manifest['teacher_label_contract'] != source['teacher_label_contract']:
        raise ValueError('Same explicit experimental task required')
    with load_array(cache_path) as packed:
        activity, target, valid = [torch.from_numpy(packed[k].copy()).cuda()
                                   for k in ('activity', 'target', 'valid')]
    expected = torch.cat([e['action'][:, 1] for e in episodes]).cuda()
    expected_valid = torch.cat([e['action_valid'][:, 1] for e in episodes]).cuda()
    if not torch.equal(target, expected) or not torch.equal(valid, expected_valid):
        raise ValueError('Cache target order/masks differ from actual source episodes')
    if (activity.ndim != 2 or len(activity) != len(target)
            or not torch.isfinite(activity).all() or torch.any((activity < 0) | (activity > 1))):
        raise ValueError('Bounded actual graph output activity required')
    weights = torch.cat([e['action_valid'][:, 1].float()
        * (3. if e['source'].get('source_task_failed_partial_phase') else 1.)
        / max(1, int(e['action_valid'][:, 1].sum())) for e in episodes]).cuda()
    weights /= weights.sum()
    coast, drive = valid & (target <= .005), valid & (target > .05)
    if not coast.any() or not drive.any():
        raise ValueError('Actual coast and positive-drive labels required')
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in source['graph'].items()}
    constructor = copy.deepcopy(c)
    constructor['sparse_backend'] = 'cached_csr_gpu'
    model = MultiViewCircuitController(graph, **constructor).cuda().eval().requires_grad_(False)
    model.load_state_dict(source['model'])
    with torch.no_grad():
        original = model.readout(activity)
    coefficients = [0., .1, 1., 10., 100., 1000.]
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862988,
        cache_sha256=sha256(cache_path), cache_manifest_sha256=sha256(cache_root/'manifest.json'),
        source_checkpoint_sha256=sha256(checkpoint), dataset_manifest_sha256=sha256(
            dataset/'dataset.json'), coast_coefficients=coefficients,
        logits_learning_rate=.05, log_gain_learning_rate=.01,
        acceptance='same gates: baseMSE improved, coastMean <=0.5baseline, driveRMSE <=1.1baseline',
        source_cache_scope='One actual recurrent replay per explicit episode, four ticks/decision'))
    started = time.perf_counter()
    logits = torch.nn.Parameter(model.readout.forward_weights.detach().repeat(len(coefficients), 1))
    gain = torch.nn.Parameter(model.readout.pedal_log_gain[0].detach().repeat(len(coefficients)))
    optimizer = torch.optim.Adam([dict(params=[logits], lr=.05), dict(params=[gain], lr=.01)])
    coefficients_tensor = target.new_tensor(coefficients)

    def metrics(values):
        error = (values-target[:, None]).square()
        return torch.stack([(error*weights[:, None]).sum(0),
            error[valid].mean(0).sqrt(), values[coast].mean(0),
            error[drive].mean(0).sqrt()], dim=1)

    with torch.no_grad():
        initial = throttle_bank_from_activity(activity, model.readout.forward_ports, logits, gain)
        torch.testing.assert_close(initial, original[:, 1, None].expand_as(initial),
                                   rtol=1e-5, atol=1e-7)
        before = metrics(initial)[0].tolist()
    best, parameters, history = None, None, []
    for iteration in range(1, args.iterations+1):
        optimizer.zero_grad()
        value = throttle_bank_from_activity(activity, model.readout.forward_ports, logits, gain)
        base = ((value-target[:, None]).square()*weights[:, None]).sum(0)
        coast_mse = ((value[coast]-target[coast, None]).square()
                     * weights[coast, None]).sum(0)/weights[coast].sum()
        (base+coefficients_tensor*coast_mse).sum().backward()
        optimizer.step()
        with torch.no_grad():
            gain.clamp_(math.log(1e-6), math.log(1e10))
            value = throttle_bank_from_activity(activity, model.readout.forward_ports, logits, gain)
            current = metrics(value)
            eligible = (current[:, 0] < before[0]) & (current[:, 2] <= before[2]*.5) \
                & (current[:, 3] <= before[3]*1.1)
            scores = current[:, 0].masked_fill(~eligible, float('inf'))
            index = int(scores.argmin())
            if bool(eligible[index]) and (
                    best is None or float(scores[index]) < best['metrics'][0]):
                best = dict(iteration=iteration, candidate_index=index,
                            coast_coefficient=coefficients[index], metrics=current[index].tolist())
                parameters = logits[index].clone(), gain[index].clone()
        if iteration % 100 == 0 or iteration == args.iterations:
            history.append(dict(iteration=iteration, candidate_metrics=current.tolist()))
            write_json(run/'history.json', history)
            write_json(run/'heartbeat.json', dict(iteration=iteration, best_eligible=best))
    report = dict(status='completed', before=before, metric_order=[
        'weighted_mse', 'rmse', 'zero_target_mean', 'drive_target_rmse'],
        best_eligible=best, candidate_eligible_for_bounded_trial=best is not None,
        checkpoint_saved=False, optimizer_updates_per_candidate=args.iterations,
        candidates=len(coefficients), actual_device=str(activity.device),
        optimizer_parameter_devices=sorted({str(logits.device), str(gain.device)}),
        gradient_devices=sorted({str(logits.grad.device), str(gain.grad.device)}),
        frozen_parameter_gradients_absent=all(p.grad is None for p in model.parameters()),
        ui_actions_sent=0, actuator_messages_sent=0, closed_loop_verified=False,
        full_lap_verified=False, goal_achieved=False)
    if best is not None:
        with torch.no_grad():
            model.readout.forward_weights.copy_(parameters[0])
            model.readout.pedal_log_gain[0].copy_(parameters[1])
            decoded = model.readout(activity)
        if not torch.equal(decoded[:, [0, 2]], original[:, [0, 2]]):
            raise ValueError('Steering or brake changed on identical neural activity')
        learned = copy.deepcopy(source)
        learned['model']['readout.forward_weights'] = parameters[0].cpu()
        learned['model']['readout.pedal_log_gain'][0] = parameters[1].cpu()
        for name, value in model.state_dict().items():
            if not tensors_identical(value.cpu(), learned['model'][name]):
                raise ValueError('Parameter outside declared throttle decoder changed')
        learned['training_configuration'] = dict(source['training_configuration'], seed=862988,
            source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha256(checkpoint),
            learning_rate=.05, readout_gain_learning_rate=.01, trainable_groups=['readout'],
            epochs=1, optimizer_update_scope='verified_cached_throttle_bank',
            decoder_iterations=args.iterations, coast_coefficient=best['coast_coefficient'],
            readout_only_cached_optimization=True, scope=__doc__)
        learned['training_teacher_manifest_sha256'] = sha256(dataset/'dataset.json')
        torch.save(learned, run/'checkpoint.pt')
        report.update(checkpoint_saved=True, checkpoint_sha256=sha256(run/'checkpoint.pt'),
                      graph_and_sensory_parameters_unchanged=True,
                      steering_and_brake_parameters_unchanged=True)
    report['seconds'] = time.perf_counter()-started
    write_json(run/'result.json', report)


if __name__ == '__main__':
    main()
