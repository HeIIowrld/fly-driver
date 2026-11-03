"""Evaluate, never fit, a frozen synthetic predictor on original historical scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array, resolve_payload_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_motion_forecast import CausalMotionForecast
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    dataset = base/'20261029-t1434-body21-flight-cache'
    source_result = read(dataset/'result.json')
    manifest = read(dataset/'dataset.json')
    if (source_result['status'] != 'completed_explicit_body21_dataset_conversion'
            or sha256(dataset/'dataset.json') != source_result['dataset_manifest_sha256']):
        raise ValueError('Original historical manifest and lineage required')
    fit = base/'20261101-causal-motion-forecast-synthetic-fit'
    fit_result = read(fit/'result.json')
    checkpoint = fit/'motion-forecast.pt'
    if (fit_result['status'] != 'completed_synthetic_motion_forecast_training'
            or fit_result['actor_qualified'] is not False or sha256(checkpoint) != fit_result['checkpoint_sha256']):
        raise ValueError('Frozen, unqualified synthetic predictor checkpoint required')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['schema'] != 'flybrain.experimental.causal_motion_forecast.v1' or payload['actor_qualified'] is not False:
        raise ValueError('Explicit research-only prediction checkpoint required')
    model = CausalMotionForecast()
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    torch.set_num_threads(1)
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), dataset_sha256=sha256(dataset/'dataset.json'),
        checkpoint_sha256=sha256(checkpoint), scope='scene prediction only; original fit-forbidden source',
        target_used='next pooled T4/T5 activity, evaluation only', actor=False, optimizer_updates=0))
    rows = []
    for episode in manifest['episodes']:
        file = resolve_payload_path(dataset/episode['file'])
        if sha256(file) != episode['sha256']:
            raise ValueError('Original episode payload changed')
        with load_array(dataset/episode['file'], allow_pickle=False) as archive:
            scene = archive['scene_features'].copy()
            clock = archive['observation_time'].copy()
        if (scene.shape != (episode['observations'], 576) or len(clock) != len(scene)
                or not np.isfinite(scene).all() or not np.isfinite(clock).all()
                or np.any(np.diff(clock) <= 0)):
            raise ValueError('Complete finite original scene and clock history required')
        dt = np.diff(clock)
        eligible = (dt[:-1] >= .036) & (dt[:-1] <= .044) & (dt[1:] >= .036) & (dt[1:] <= .044)
        prev = torch.tensor(scene[:-2][eligible], dtype=torch.float32)
        current = torch.tensor(scene[1:-1][eligible], dtype=torch.float32)
        target = torch.tensor(scene[2:][eligible], dtype=torch.float32).reshape(-1, 4, 9, 16)[:, :, 1:]
        if not len(prev):
            raise ValueError('No consecutive cadence-qualified original observations')
        with torch.no_grad():
            predicted = model(current, prev, .04).reshape(-1, 4, 9, 16)[:, :, 1:]
            hold = current.reshape(-1, 4, 9, 16)[:, :, 1:]
            pred_mse = float((predicted-target).square().mean())
            hold_mse = float((hold-target).square().mean())
        original_recording = Path(episode['original']['recording'])
        original_receipt = read(original_recording/'result.json')
        rows.append(dict(file=episode['file'], role=episode['role'], episode_sha256=episode['sha256'],
                         source_training_admissible=original_receipt.get('training_admissible'),
                         total_windows=len(eligible), eligible_windows=int(eligible.sum()),
                         excluded_cadence_windows=int((~eligible).sum()),
                         predictor_mse=pred_mse, hold_mse=hold_mse,
                         relative_change=(pred_mse/hold_mse-1) if hold_mse else None))
    result = dict(status='completed_offline_historical_forecast_evaluation', episodes=rows,
                  optimizer_updates=0, actor_deployed=False, actor_qualified=False,
                  source_permissions_upgraded=False, physical_exposure_or_body_sync_verified=False,
                  physiology_or_driving_verified=False, vehicle_control_sent=False,
                  interpretation='Previously examined real histories; only temporal-neural prediction error, not road preview or driving performance.')
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
