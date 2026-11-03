"""Measure saved-scene historical-yaw interaction before the fixed flight graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import VisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.temporal_body_context import (
    missing_temporal_body_context,
    temporal_body_context,
)
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'learned', 'body-context', 'episode', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    training, learned, context = map(Path, (args.training, args.learned, args.body_context))
    tr, lr, cr = (read(path/'result.json') for path in (training, learned, context))
    plan = read(training/'manifest.json')['config']['plan']
    root = Path(plan['data'])
    entry = next(row for row in read(root/'dataset.json')['episodes']
                 if row['file'] == args.episode)
    episode, pairfile = root/entry['file'], context/'body-pairs.npz'
    context_manifest = read(context/'manifest.json')['config']
    learned_manifest = read(learned/'manifest.json')['config']
    if (sha256(episode) != entry['sha256'] or sha256(pairfile) != cr['body_pairs_sha256']
            or cr['status'] != 'completed_temporal_body_context'
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or sha256(learned/'context-boundary.pt') != lr['checkpoint_sha256']
            or learned_manifest['source_checkpoint_sha256'] != tr['checkpoint_sha256']
            or context_manifest['events_sha256'] != entry['source_events_sha256']):
        raise ValueError('Exact model, context and same source history required')
    with load_array(episode) as data, np.load(pairfile, allow_pickle=False) as pairs:
        times, scenes = data['observation_time'].copy(), data['scene_features'].copy()
        values, valid, timing, packets = (pairs[key].copy() for key in
            ('values', 'valid', 'original_host_timing', 'packets'))
        if not np.array_equal(times, pairs['decision_monotonic_seconds']):
            raise ValueError('Original decisions and context clocks differ')
    model = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model']
    mean, scale = model['backbone.scene_command.mean'], model['backbone.scene_command.scale']
    payload = torch.load(learned/'context-boundary.pt', map_location='cpu', weights_only=True)
    boundary = VisualBodyContextBoundary(payload['model']['projection.weight'].shape[0])
    boundary.load_state_dict(payload['model'], strict=True)
    temporal = OpponentTemporalMotionBoundary(mean, scale, boundary.projection.out_features)
    traces, rows = None, []
    with torch.no_grad():
        for i, now in enumerate(times):
            elapsed = None if i == 0 else float(now-times[i-1])
            _, traces = temporal.step(torch.tensor(scenes[i:i+1]), torch.ones(1, 4,
                                      dtype=torch.bool), traces, elapsed)
            old_time, lower, camera_time, current_time, decision = map(float, timing[i])
            current = NavigationBodyObservation('episode', int(packets[i, 0]), current_time,
                decision, values[i, 0], valid[i, 0], 'contract')
            if packets[i, 1] < 0:
                ctx = missing_temporal_body_context(current, decision_time=decision,
                                                    reason='unavailable_historical_receipt')
                yaw = 0.
            else:
                prior = NavigationBodyObservation('episode', int(packets[i, 1]), old_time,
                    camera_time, values[i, 1], valid[i, 1], 'contract')
                ctx = temporal_body_context(prior, current, camera_receipt_time=camera_time,
                                            render_start_lower=lower, decision_time=decision)
                yaw = float(prior.values[4]) if prior.valid[4] else 0.
            motion = (traces[0]-traces[1]).reshape(1, 4, 2, 4, 16)
            opponent = torch.stack((motion[:, :, :, 1]-motion[:, :, :, 0],
                                    motion[:, :, :, 2]-motion[:, :, :, 3]), dim=3)
            interaction = opponent * max(-1., min(1., yaw))
            delta = boundary(traces, ctx)
            rows.append([yaw, float(opponent.norm()), float(interaction.norm()),
                         float(delta.norm()), float(delta.abs().max())])
    array = np.asarray(rows)
    present = (packets[:, 1] >= 0) & valid[:, 1, 4]
    if not present.any() or not np.isfinite(array).all():
        raise ValueError('Finite present historical yaw required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'episode_sha256': sha256(episode), 'body_pairs_sha256': sha256(pairfile),
        'learned_checkpoint_sha256': lr['checkpoint_sha256'], 'role': entry['role'],
        'scope': 'fixed intermediate signals, no labels, no optimization'})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run/'signals.npz', times=times, signals=array, present=present)
    labels = ['yaw_rad_s', 'opponent_norm', 'interaction_norm', 'port_delta_norm',
              'max_abs_port_delta']
    quantiles = {name: np.percentile(np.abs(array[present, i]), [50, 95, 100]).tolist()
                 for i, name in enumerate(labels)}
    result = {'status': 'completed_temporal_context_signal_audit', 'observations': len(times),
        'context_valid': int(present.sum()), 'role': entry['role'], 'quantiles_abs': quantiles,
        'weight_l2': float(boundary.projection.weight.norm().detach()),
        'signal_sha256': sha256(run/'signals.npz'), 'labels_opened': False,
        'physical_sync_verified': False, 'optimizer_updates': 0, 'actor_deployed': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
