"""Disaggregate frozen forecast error by episode start and motion-change direction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array, resolve_payload_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256
from scripts.train_causal_motion_forecast_synthetic import load_scenes


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    fit = base/'20261101-reversing-yaw-forecast-synthetic-fit'
    result = read(fit/'result.json')
    checkpoint = fit/'motion-forecast.pt'
    if result['status'] != 'completed_synthetic_reversal_forecast_training' or sha256(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Original frozen synthetic forecast required')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    gain = payload['model']['raw_gain'].tanh().numpy().reshape(1, 1, 8, 1)
    root = base/'20261029-t1434-body21-flight-cache'
    manifest = read(root/'dataset.json')
    if sha256(root/'dataset.json') != read(root/'result.json')['dataset_manifest_sha256']:
        raise ValueError('Original historical dataset manifest changed')
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), checkpoint_sha256=sha256(checkpoint),
        dataset_sha256=sha256(root/'dataset.json'), role='posthoc diagnostic, source fit forbidden',
        segments='episode first window vs rest; future change alignment is evaluation-only',
        actor=False, optimizer_updates=0))
    names = ('all', 'first_window', 'after_first', 'after_ten', 'aligned', 'reversed', 'weak')
    totals = {key: [0, 0., 0.] for key in names}
    episode_rows = []
    for episode in manifest['episodes']:
        if sha256(resolve_payload_path(root/episode['file'])) != episode['sha256']:
            raise ValueError('Historical episode payload changed')
        with load_array(root/episode['file'], allow_pickle=False) as data:
            scene, clock = data['scene_features'].copy(), data['observation_time'].copy()
        if scene.shape != (len(clock), 576) or not np.isfinite(scene).all() or not np.isfinite(clock).all():
            raise ValueError('Complete original finite scene and clock required')
        frame = scene.reshape(-1, 4, 9, 16)[:, :, 1:]
        previous_delta = frame[1:-1]-frame[:-2]
        next_delta = frame[2:]-frame[1:-1]
        dt = np.diff(clock)
        eligible = (dt[:-1] >= .036) & (dt[:-1] <= .044) & (dt[1:] >= .036) & (dt[1:] <= .044)
        hold = np.square(next_delta).mean(axis=(1, 2, 3), dtype=np.float64)
        forecast = np.square(gain*previous_delta-next_delta).mean(axis=(1, 2, 3), dtype=np.float64)
        dot = np.einsum('nijk,nijk->n', previous_delta, next_delta)
        magnitude = np.sqrt(np.sum(previous_delta**2, axis=(1, 2, 3))*np.sum(next_delta**2, axis=(1, 2, 3)))
        cosine = dot/np.maximum(magnitude, 1e-12)
        index = np.arange(len(eligible))
        masks = dict(all=eligible, first_window=eligible & (index == 0), after_first=eligible & (index >= 1),
                     after_ten=eligible & (index >= 10), aligned=eligible & (cosine > .1),
                     reversed=eligible & (cosine < -.1), weak=eligible & (np.abs(cosine) <= .1))
        for name, mask in masks.items():
            totals[name][0] += int(mask.sum())
            totals[name][1] += float(forecast[mask].sum())
            totals[name][2] += float(hold[mask].sum())
        episode_rows.append(dict(file=episode['file'], role=episode['role'], sha256=episode['sha256'],
                                 eligible=int(eligible.sum()), excluded=int((~eligible).sum())))

    def finish(values):
        count, predictor, hold = values
        return dict(windows=count, predictor_mse=predictor/count if count else None,
                    hold_mse=hold/count if count else None,
                    relative_change=(predictor/hold-1) if hold else None)

    # Retain the source's previously unseen synthetic validation, now disaggregated post hoc.
    torch.set_num_threads(1)
    spec = read('configs/flight-circuit-visual-v6-20261029.json')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as registration:
        pool = FlightScenePool(registration['image_uv'], spec['scene_basis'])
    scene = load_scenes(base/'20261101-flyvis-coherent-yaw-texture-879303-speed013', 879303, .13, pool).numpy()
    frame = scene.reshape(3, 41, 4, 9, 16)[:, :, :, 1:]
    previous_delta, next_delta = frame[:, 1:-1]-frame[:, :-2], frame[:, 2:]-frame[:, 1:-1]
    synthetic = {}
    for i, case in enumerate(('stationary', 'negative_yaw', 'positive_yaw')):
        h = np.square(next_delta[i]).mean(axis=(1, 2, 3), dtype=np.float64)
        p = np.square(gain*previous_delta[i]-next_delta[i]).mean(axis=(1, 2, 3), dtype=np.float64)
        synthetic[case] = {name: finish([int(mask.sum()), float(p[mask].sum()), float(h[mask].sum())])
                           for name, mask in [('initial_first_window', np.arange(39) == 0),
                                              ('later_stationary', (np.arange(39) >= 1) & (np.arange(39) <= 8)),
                                              ('after_motion_onset', np.arange(39) >= 9)]}
    record = dict(status='completed_forecast_transient_audit', historical={k: finish(v) for k, v in totals.items()},
                  episodes=episode_rows, synthetic_fresh_texture=synthetic,
                  interpretation='Posthoc temporal segments of previously examined real histories. '
                  'Future activity only defines evaluation strata, never predictor input. '
                  'Correlated windows are not independent trials.',
                  optimizer_updates=0, actor_deployed=False, training_admissibility_unchanged=True,
                  physical_time_or_physiology_verified=False, vehicle_control_sent=False)
    write_json(run/'result.json', record)
    print(json.dumps(record, allow_nan=False))


if __name__ == '__main__':
    main()
