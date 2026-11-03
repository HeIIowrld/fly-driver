"""Test a posthoc past-only gate against hold and the original frozen forecast."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array, resolve_payload_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root = base/'20261029-t1434-body21-flight-cache'
    fit = base/'20261101-causal-motion-forecast-synthetic-fit'
    dataset = read(root/'dataset.json')
    receipt = read(fit/'result.json')
    checkpoint = fit/'motion-forecast.pt'
    if (sha256(root/'dataset.json') != read(root/'result.json')['dataset_manifest_sha256']
            or sha256(checkpoint) != receipt['checkpoint_sha256'] or receipt['actor_qualified'] is not False):
        raise ValueError('Original historical dataset and unqualified frozen predictor required')
    gain = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']['raw_gain'].tanh().numpy().reshape(1, 1, 8, 1)
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), dataset_sha256=sha256(root/'dataset.json'),
        checkpoint_sha256=sha256(checkpoint), gate='past two motion-change vectors cosine > 0.1',
        threshold_role='posthoc engineering choice, no independent selection or qualification',
        actor=False, optimizer_updates=0))
    groups = {key: [0, 0, 0., 0., 0.] for key in ('all', 'future_aligned', 'future_reversed', 'future_weak')}
    episodes = []
    for episode in dataset['episodes']:
        if sha256(resolve_payload_path(root/episode['file'])) != episode['sha256']:
            raise ValueError('Original episode payload changed')
        with load_array(root/episode['file'], allow_pickle=False) as data:
            scene, clocks = data['scene_features'].copy(), data['observation_time'].copy()
        if (scene.shape != (episode['observations'], 576) or len(clocks) != len(scene)
                or not np.isfinite(scene).all() or not np.isfinite(clocks).all()):
            raise ValueError('Original complete finite scene and clocks required')
        frame = scene.reshape(-1, 4, 9, 16)[:, :, 1:]
        earlier = frame[1:-2]-frame[:-3]
        current = frame[2:-1]-frame[1:-2]
        future = frame[3:]-frame[2:-1]
        dt = np.diff(clocks)
        ok = (dt[:-2] >= .036) & (dt[:-2] <= .044) & (dt[1:-1] >= .036) & (dt[1:-1] <= .044) & (dt[2:] >= .036) & (dt[2:] <= .044)

        def cosine(a, b):
            dot = np.einsum('nijk,nijk->n', a, b)
            magnitude = np.sqrt(np.sum(a*a, axis=(1, 2, 3))*np.sum(b*b, axis=(1, 2, 3)))
            return dot/np.maximum(magnitude, 1e-12)

        gate = cosine(earlier, current) > .1  # Uses only t-2, t-1, t.
        future_alignment = cosine(current, future)  # Evaluation labels only, never gate input.
        hold = np.square(future).mean(axis=(1, 2, 3), dtype=np.float64)
        prior = np.square(gain*current-future).mean(axis=(1, 2, 3), dtype=np.float64)
        gated = np.square(gain*current*gate[:, None, None, None]-future).mean(axis=(1, 2, 3), dtype=np.float64)
        masks = dict(all=ok, future_aligned=ok & (future_alignment > .1),
                     future_reversed=ok & (future_alignment < -.1), future_weak=ok & (np.abs(future_alignment) <= .1))
        for name, mask in masks.items():
            group = groups[name]
            group[0] += int(mask.sum())
            group[1] += int((gate & mask).sum())
            group[2] += float(hold[mask].sum())
            group[3] += float(prior[mask].sum())
            group[4] += float(gated[mask].sum())
        episodes.append(dict(file=episode['file'], role=episode['role'], eligible=int(ok.sum()),
                             excluded=int((~ok).sum())))
    summary = {}
    for name, (n, on, hold, prior, gated) in groups.items():
        summary[name] = dict(windows=n, gate_on=on, hold_mse=hold/n if n else None,
                             original_forecast_mse=prior/n if n else None,
                             gated_forecast_mse=gated/n if n else None,
                             original_relative_to_hold=prior/hold-1 if hold else None,
                             gated_relative_to_hold=gated/hold-1 if hold else None)
    result = dict(status='completed_posthoc_past_gate_diagnostic', groups=summary, episodes=episodes,
                  interpretation='Gate threshold chosen after looking at the histories; future direction only for '
                  'posthoc strata. No independent evaluation, fitting, deployment or driving claim.',
                  actor_deployed=False, training_performed=False, vehicle_control_sent=False,
                  physical_time_or_physiology_verified=False)
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
