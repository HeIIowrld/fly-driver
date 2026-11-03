"""Evaluate fixed learned temporal body context on known development histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import VisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.temporal_body_context import (
    missing_temporal_body_context,
    temporal_body_context,
)
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'opponent', 'body-context', 'learned', 'episode', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    training, opponent, context = map(Path, (args.training, args.opponent, args.body_context))
    tr, op, context_result = (read(path/'result.json') for path in (training, opponent, context))
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    root = Path(plan['data'])
    entry = next(e for e in read(root/'dataset.json')['episodes'] if e['file'] == args.episode)
    episode = root/entry['file']
    pairfile = context/'body-pairs.npz'
    if (entry['role'] != 'development' or sha256(episode) != entry['sha256']
            or context_result['status'] != 'completed_temporal_body_context'
            or context_result['training_admissible'] is not False
            or sha256(pairfile) != context_result['body_pairs_sha256']
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or sha256(source['checkpoint']) != source['checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or op['source_checkpoint_sha256'] != tr['checkpoint_sha256']
            or not torch.cuda.is_available()):
        raise ValueError('Exact saved TRAIN history, context and GPU graph required')
    source_config = read(context/'manifest.json')['config']
    if (source_config['events_sha256'] != entry['source_events_sha256']
            or Path(source_config['recording']) != Path(entry['original']['recording'])):
        raise ValueError('Context must use same original camera/body recording')
    with load_array(episode) as d, np.load(pairfile, allow_pickle=False) as p:
        times = d['observation_time'].copy()
        data = {k: torch.tensor(d[k], device='cuda') for k in
                ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')}
        values, valid, timing, packets = (p[k].copy() for k in
            ('values', 'valid', 'original_host_timing', 'packets'))
        if (not np.array_equal(times, p['decision_monotonic_seconds'])
                or not np.array_equal(d['body_valid'], valid[:, 0])
                or not np.allclose(d['body'], values[:, 0], atol=1e-6, rtol=0)):
            raise ValueError('Original cached body and decision times must match')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
        source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    cp = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)
    if cp['schema'] != model.schema or cp['spec'] != model.spec:
        raise ValueError('Original model schema/spec required')
    model.load_state_dict(cp['model'], strict=True)
    model.eval().requires_grad_(False)
    boundary = OpponentTemporalMotionBoundary(model.backbone.scene_command.mean,
        model.backbone.scene_command.scale, len(model.backbone.vision_ports)).to('cuda')
    saved = torch.load(opponent/'temporal-boundary.pt', map_location='cpu', weights_only=True)
    boundary.load_state_dict(saved['model'], strict=True)
    boundary.eval().requires_grad_(False)
    learned = Path(args.learned)
    learned_result = read(learned/'result.json')
    learned_manifest = read(learned/'manifest.json')['config']
    if (learned_result['status'] != 'completed_temporal_context_train_onepass'
            or learned_result['actor_deployed'] is not False
            or sha256(learned/'context-boundary.pt') != learned_result['checkpoint_sha256']
            or learned_manifest['source_checkpoint_sha256'] != tr['checkpoint_sha256']
            or learned_manifest['opponent_checkpoint_sha256'] != op['checkpoint_sha256']
            or learned_manifest['dataset_sha256'] != sha256(root/'dataset.json')):
        raise ValueError('Exact TRAIN-only learned candidate and lineage required')
    probe = VisualBodyContextBoundary(len(model.backbone.vision_ports), device='cuda')
    payload = torch.load(learned/'context-boundary.pt', map_location='cpu', weights_only=True)
    if payload['actor_qualified'] is not False:
        raise ValueError('Research-only learned checkpoint required')
    probe.load_state_dict(payload['model'], strict=True)
    probe.eval().requires_grad_(False)
    base = model._drive_tensor(*(data[k] for k in
        ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')))
    state, traces, held, outputs = model.initial_state(3), None, None, []
    scenes_valid = torch.ones((1, 4), dtype=torch.bool, device='cuda')
    with torch.no_grad():
        for i, now in enumerate(times):
            elapsed = None if i == 0 else float(now-times[i-1])
            if i:
                for dt in substeps(elapsed):
                    state = model._advance_elapsed_tensor(held, state, dt)
            outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
            old_time, lower, camera_time, current_time, decision = map(float, timing[i])
            current = NavigationBodyObservation('recorded', int(packets[i, 0]), current_time,
                decision, values[i, 0], valid[i, 0], 'original-contract')
            if packets[i, 1] < 0:
                body_context = missing_temporal_body_context(
                    current, decision_time=decision, reason='unavailable_historical_receipt')
            else:
                prior = NavigationBodyObservation('recorded', int(packets[i, 1]), old_time,
                    camera_time, values[i, 1], valid[i, 1], 'original-contract')
                body_context = temporal_body_context(prior, current,
                    camera_receipt_time=camera_time, render_start_lower=lower,
                    decision_time=decision)
            delta, traces = boundary.step(data['scene_features'][i:i+1], scenes_valid,
                                          traces, elapsed)
            baseline = boundary.modulate(base[i:i+1], model.backbone.vision_ports, delta)
            learned_delta = probe(traces, body_context)
            learned_drive = boundary.modulate(baseline, model.backbone.vision_ports, learned_delta)
            reversed_drive = boundary.modulate(baseline, model.backbone.vision_ports,
                                              -learned_delta)
            held = torch.cat((baseline, learned_drive, reversed_drive))
    outputs = np.stack(outputs)
    if not np.isfinite(outputs).all():
        raise ValueError('Finite development outputs required')
    with load_array(episode) as d:
        targets, target_valid = d['target'].copy(), d['target_valid'].copy().astype(bool)
    metrics = {}
    for axis, name in enumerate(('steering', 'forward')):
        mask = target_valid[:, axis]
        if not mask.any():
            raise ValueError('Original valid targets required')
        prediction = outputs[mask, :, axis]
        target = targets[mask, axis, None]
        metrics[name] = {'valid': int(mask.sum()),
            'rmse_existing_learned_reversed': np.sqrt(
                np.mean((prediction-target)**2, axis=0)).tolist(),
            'zero_rmse': float(np.sqrt(np.mean(target**2)))}
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'body_pairs_sha256': sha256(pairfile), 'dataset_sha256': sha256(root/'dataset.json'),
        'episode_sha256': sha256(episode), 'source_checkpoint_sha256': tr['checkpoint_sha256'],
        'opponent_checkpoint_sha256': op['checkpoint_sha256'],
        'conditions': ['existing', 'learned_context', 'reversed_context'],
        'learned_sha256': learned_result['checkpoint_sha256'],
        'role': 'known development, never optimization'})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'outputs.npz', times=times, outputs=outputs)
    difference = np.abs(outputs[:, 1]-outputs[:, 0])
    result = {'status': 'completed_temporal_context_known_development_evaluation',
        'observations': len(times),
        'context_present': int((packets[:, 1] >= 0).sum()),
        'context_missing': int((packets[:, 1] < 0).sum()), 'metrics': metrics,
        'learned_mean_abs_difference': difference.mean(axis=0).tolist(),
        'learned_max_abs_difference': difference.max(axis=0).tolist(),
        'outputs_sha256': sha256(run/'outputs.npz'), 'optimizer_updates': 0,
        'physical_sensor_sync_verified': False, 'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
