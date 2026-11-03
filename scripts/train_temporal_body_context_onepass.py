"""One-pass TRAIN-only context projection fit on the original recurrent flight graph."""

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
    for name in ('training', 'opponent', 'body-context', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    training, opponent, context = map(Path, (args.training, args.opponent, args.body_context))
    tr, op, context_result = (read(path/'result.json') for path in (training, opponent, context))
    plan = read(training/'manifest.json')['config']['plan']
    source = read(Path(plan['source'])/'result.json')
    root = Path(plan['data'])
    entry = next(e for e in read(root/'dataset.json')['episodes'] if e['file'] == 'episode-00.npz')
    episode = root/entry['file']
    pairfile = context/'body-pairs.npz'
    if (entry['role'] != 'train' or sha256(episode) != entry['sha256']
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
                ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features',
                 'target', 'target_valid')}
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
    extra = VisualBodyContextBoundary(len(model.backbone.vision_ports), device='cuda')
    optimizer = torch.optim.Adam(extra.parameters(), lr=.001, weight_decay=0,
                                 foreach=False, fused=False)
    with torch.no_grad():
        base = model._drive_tensor(*(data[k] for k in
            ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')))
    contexts = []
    for i in range(len(times)):
        old_time, lower, camera_time, current_time, decision = map(float, timing[i])
        current = NavigationBodyObservation('recorded', int(packets[i, 0]), current_time,
            decision, values[i, 0], valid[i, 0], 'original-contract')
        if packets[i, 1] < 0:
            ctx = missing_temporal_body_context(current, decision_time=decision,
                                                reason='unavailable_historical_receipt')
        else:
            prior = NavigationBodyObservation('recorded', int(packets[i, 1]), old_time,
                camera_time, values[i, 1], valid[i, 1], 'original-contract')
            ctx = temporal_body_context(prior, current, camera_receipt_time=camera_time,
                                        render_start_lower=lower, decision_time=decision)
        contexts.append(ctx)
    scenes_valid = torch.ones((1, 4), dtype=torch.bool, device='cuda')
    def rollout(train):
        state, traces, held = model.initial_state(1), None, None
        predictions, updates, loss_counts = [], 0, [0, 0]
        for start in range(0, len(times), 64):
            end = min(start+64, len(times))
            if train:
                optimizer.zero_grad(set_to_none=True)
            batch = []
            with torch.set_grad_enabled(train):
                for i in range(start, end):
                    elapsed = None if i == 0 else float(times[i]-times[i-1])
                    if i:
                        for dt in substeps(elapsed):
                            state = model._advance_elapsed_tensor(held, state, dt)
                    output = model._read_state_tensor(state)[0]
                    batch.append(output)
                    delta, traces = boundary.step(data['scene_features'][i:i+1],
                                                  scenes_valid, traces, elapsed)
                    existing = boundary.modulate(base[i:i+1], model.backbone.vision_ports, delta)
                    held = boundary.modulate(existing, model.backbone.vision_ports,
                                             extra(traces, contexts[i]))
                pred = torch.cat(batch, dim=0)
                predictions.append(pred.detach().cpu().numpy())
                target = data['target'][start:end]
                mask = data['target_valid'][start:end].bool()
                if pred.shape != target.shape or mask.shape != target.shape:
                    raise ValueError('Exact original target/mask shapes required')
                terms = []
                for axis in range(2):
                    n = int(mask[:, axis].sum().item())
                    loss_counts[axis] += n
                    if n:
                        terms.append((pred[mask[:, axis], axis]-target[mask[:, axis], axis])
                                     .square().mean())
                if train and terms:
                    loss = sum(terms)/len(terms)
                    if loss.requires_grad:
                        loss.backward()
                        grad = extra.projection.weight.grad
                        if grad is None or not torch.isfinite(grad).all():
                            raise ValueError('Finite context projection gradient required')
                        if grad.norm().item() > 0:
                            optimizer.step()
                            updates += 1
                state = state.detach()
                held = held.detach()
                traces = tuple(t.detach() for t in traces)
        pred = np.concatenate(predictions)
        target, mask = data['target'].cpu().numpy(), data['target_valid'].cpu().numpy().astype(bool)
        rmse = [float(np.sqrt(np.mean((pred[mask[:, j], j]-target[mask[:, j], j])**2)))
                if mask[:, j].any() else None for j in range(2)]
        return {'rmse': rmse, 'valid_counts': loss_counts, 'optimizer_updates': updates}, pred
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'body_pairs_sha256': sha256(pairfile), 'dataset_sha256': sha256(root/'dataset.json'),
        'episode_sha256': sha256(episode), 'source_checkpoint_sha256': tr['checkpoint_sha256'],
        'opponent_checkpoint_sha256': op['checkpoint_sha256'], 'role': 'original TRAIN only',
        'optimizer': 'Adam lr=0.001, TBPTT64, one pass, no reset at detach',
        'loss': 'mean valid steering/forward squared error', 'development_opened': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    baseline, baseline_pred = rollout(False)
    print(json.dumps({'stage': 'baseline', **baseline}), flush=True)
    training, _ = rollout(True)
    if not training['optimizer_updates']:
        raise ValueError('Actual nonzero updates required')
    final, final_pred = rollout(False)
    torch.save({'schema': 'flybrain.visual_body_context_projection.v1',
        'model': extra.state_dict(), 'actor_qualified': False}, run/'context-boundary.pt')
    np.savez_compressed(run/'predictions.npz', time=times, baseline=baseline_pred, final=final_pred)
    result = {'status': 'completed_temporal_context_train_onepass', 'baseline': baseline,
        'training': training, 'final': final,
        'checkpoint_sha256': sha256(run/'context-boundary.pt'),
        'predictions_sha256': sha256(run/'predictions.npz'), 'original_train_only': True,
        'known_development_evaluated': False, 'physical_sensor_sync_verified': False,
        'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
