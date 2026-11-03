"""Measure actual fly-model forward/backward on CPU and GPU, without optimizer/game actions."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import (
    brake_label_weights,
    configure_trainable_groups,
    episode_chunk,
    load_episodes,
    replay_chunk,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk-length', type=int, default=16)
    parser.add_argument('--chunk-validation', action='store_true')
    parser.add_argument('--gpu-backend', choices=['coo', 'cached_csr_gpu'], default='coo')
    parser.add_argument('--gpu-only', action='store_true')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--episodes-per-batch', type=int, default=4)
    parser.add_argument('--train-readout', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.chunk_length <= 256 or not 2 <= args.repeats <= 8:
        raise ValueError('Bounded chunk 1..256 and repetitions 2..8 required')
    if not 1 <= args.episodes_per_batch <= 64:
        raise ValueError('Bounded episode batch 1..64 required')
    checkpoint = Path(args.training_run)/'checkpoint.pt'
    report = json.loads((Path(args.training_run)/'result.json').read_text(encoding='utf8'))
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed verified checkpoint required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in source['graph'].items()}
    episodes, _ = load_episodes(args.dataset, source['constructor']['sensory_calibration'])
    episodes = episodes[:args.episodes_per_batch]
    groups = ['visual', 'organs'] + (['readout'] if args.train_readout else [])
    reference = MultiViewCircuitController(graph, **source['constructor'])
    reference.load_state_dict(source['model'])
    initial = reference.initial_state(len(episodes))
    with torch.no_grad():
        for offset in range(0, 112, 16):
            obs, _, _, active = episode_chunk(episodes, offset, 16)
            _, initial = replay_chunk(reference, obs, active, initial, 4)
    cpu_obs, cpu_labels, cpu_valid, cpu_active = episode_chunk(episodes, 112, args.chunk_length)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862899,
        entry_script_sha256=sha256(__file__), checkpoint_sha256=sha256(checkpoint),
        dataset_manifest_sha256=sha256(Path(args.dataset)/'dataset.json'),
        chunk_length=args.chunk_length, batch=len(episodes), ticks=4, offset=112,
        trainable_groups=groups,
        comparison='same recorded prefix state at each benchmark repetition; no in-chunk resets',
        torch_version=torch.__version__, hip_version=torch.version.hip))
    results, cpu_prediction = {}, None
    devices = [('cuda', args.gpu_backend)] if args.gpu_only else [
        ('cpu', 'cached_csr_cpu'), ('cuda', args.gpu_backend)]
    for device, backend in devices:
        try:
            constructor = copy.deepcopy(source['constructor'])
            constructor['sparse_backend'] = backend
            model = MultiViewCircuitController(graph, **constructor)
            model.load_state_dict(source['model'])
            model.to(device)
            model.chunk_validated_training = args.chunk_validation
            configure_trainable_groups(model, groups)
            obs = {k: v.to(device) for k, v in cpu_obs.items()}
            labels, valid, active = (v.to(device) for v in (cpu_labels, cpu_valid, cpu_active))
            times = []
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for repeat in range(args.repeats):
                model.zero_grad(set_to_none=True)
                state = initial.detach().to(device)
                if device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                predicted, _ = replay_chunk(model, obs, active, state, 4)
                weights = brake_label_weights(labels, valid, 20., .05)
                loss = ((predicted-labels).square()*weights).sum()/weights.sum()
                loss.backward()
                if device == 'cuda':
                    torch.cuda.synchronize()
                if repeat:
                    times.append(time.perf_counter()-started)
            value = predicted.detach().cpu().numpy()
            if device == 'cpu':
                cpu_prediction = value
            parameters = [p for p in model.parameters() if p.requires_grad]
            results[device] = dict(status='completed', sparse_backend=backend,
                parameter_devices=sorted({str(p.device) for p in parameters}),
                gradient_devices=sorted({str(p.grad.device) for p in parameters
                                        if p.grad is not None}),
                gradients_finite=all(bool(torch.isfinite(p.grad).all())
                                     for p in parameters if p.grad is not None),
                seconds=times, median_seconds=float(np.median(times)), loss=float(loss.detach()),
                prediction_max_error_from_cpu=(float(np.abs(value-cpu_prediction).max())
                                               if cpu_prediction is not None else None),
                valid_decisions=int(cpu_active.sum()),
                valid_decisions_per_second=float(cpu_active.sum())/float(np.median(times)),
                device_name=torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU 1 thread',
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                    if device == 'cuda' else None)
        except Exception as error:
            results[device] = dict(status='failed', error=type(error).__name__, reason=str(error))
        write_json(run/'progress.json', results)
    write_json(run/'result.json', dict(status='completed', devices=results,
        optimizer_updates=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
