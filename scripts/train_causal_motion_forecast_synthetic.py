"""Fit a causal shared motion-vision predictor on declared synthetic stimuli only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_motion_forecast import CausalMotionForecast
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_scenes(root, seed, amplitude, pool):
    receipt, manifest = read(root/'result.json'), read(root/'manifest.json')['config']
    payload = root/'responses.npz'
    if (receipt.get('status') != 'completed_coherent_four_view_yaw'
            or sha256(payload) != receipt['responses_sha256']
            or manifest['texture_seed'] != seed or receipt['cases'] != [0., -amplitude, amplitude]
            or manifest['stationary_seconds'] != .4):
        raise ValueError('Exact declared synthetic stimulus lineage required')
    with np.load(payload, allow_pickle=False) as data:
        retina, motion, times = data['retina'].copy(), data['motion'].copy(), data['times'].copy()
    if (retina.shape != (3, 41, 4, 721) or motion.shape != (3, 41, 4, 8, 721)
            or not np.allclose(times, np.arange(41)*.04, rtol=0, atol=1e-12)
            or not np.isfinite(retina).all() or not np.isfinite(motion).all()):
        raise ValueError('Complete finite original four-view synthetic activity required')
    scene = np.concatenate([retina[:, :, :, None], motion], axis=3)
    with torch.no_grad():
        features = pool(torch.tensor(scene.reshape(-1, 4, 9, 721), dtype=torch.float32)).reshape(3, 41, 576)
    return features


def examples(features):
    previous = features[:, :-2].reshape(-1, 576)
    current = features[:, 1:-1].reshape(-1, 576)
    future = features[:, 2:].reshape(-1, 4, 9, 16)[:, :, 1:].reshape(-1, 512)
    return previous, current, future


def prediction(model, current, previous):
    return model(current, previous, .04).reshape(-1, 4, 9, 16)[:, :, 1:].reshape(-1, 512)


def report(model, data):
    previous, current, future = examples(data)
    with torch.no_grad():
        output = prediction(model, current, previous)
        hold = current.reshape(-1, 4, 9, 16)[:, :, 1:].reshape(-1, 512)
        model_loss = (output-future).square().reshape(3, 39, 512).mean(dim=(1, 2)).tolist()
        hold_loss = (hold-future).square().reshape(3, 39, 512).mean(dim=(1, 2)).tolist()
    return dict(cases=['stationary', 'negative_yaw', 'positive_yaw'],
                predictor_mse=model_loss, hold_mse=hold_loss,
                aggregate_predictor_mse=sum(model_loss)/3, aggregate_hold_mse=sum(hold_loss)/3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    plan_path = base/'20261101-causal-motion-forecast-plan/plan.json'
    plan = read(plan_path)
    component = Path('src/flybrain_drive/flight_motion_forecast.py')
    if sha256(component) != plan['component_sha256']:
        raise ValueError('Causal predictor source changed after the declared plan')
    fit_specs = [(Path(item['path']), seed, item['responses_sha256'])
                 for item, seed in zip(plan['training_sources'], (863101, 863103, 863104), strict=True)]
    if any(sha256(path/'responses.npz') != digest for path, _, digest in fit_specs):
        raise ValueError('Declared training stimulus changed')
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    spec = read('configs/flight-circuit-visual-v6-20261029.json')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as registration:
        pool = FlightScenePool(registration['image_uv'], spec['scene_basis'])
    data = [load_scenes(path, seed, .23, pool) for path, seed, _ in fit_specs]
    model = CausalMotionForecast()
    initial = [report(model, item) for item in data]
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), plan_sha256=sha256(plan_path),
        scene_spec_sha256=sha256('configs/flight-circuit-visual-v6-20261029.json'),
        registration_sha256=sha256(spec['scene_registration_file']), seed=863009,
        source_responses=[digest for _, _, digest in fit_specs],
        actor=False, physical_sensor_or_physiology_verified=False))
    losses = []
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        terms = []
        for item in data:
            previous, current, target = examples(item)
            terms.append((prediction(model, current, previous)-target).square().mean())
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise ValueError('Finite training loss required')
        loss.backward()
        if model.raw_gain.grad is None or not torch.isfinite(model.raw_gain.grad).all():
            raise ValueError('Finite predictor gradient required')
        optimizer.step()
        losses.append(float(loss.detach()))
    fit_reports = [report(model, item) for item in data]
    # This declared fresh seed is not loaded until all optimizer updates are finished.
    validation_root = base/plan['validation_source']
    validation = load_scenes(validation_root, plan['fresh_validation_seed'], .13, pool)
    validation_report = report(model, validation)
    checkpoint = run/'motion-forecast.pt'
    torch.save(dict(schema='flybrain.experimental.causal_motion_forecast.v1',
                    model=model.state_dict(), actor_qualified=False,
                    training_sources=[digest for _, _, digest in fit_specs]), checkpoint)
    result = dict(status='completed_synthetic_motion_forecast_training', optimizer_updates=200,
                  training_loss_first=losses[0], training_loss_last=losses[-1],
                  gains=model.raw_gain.tanh().detach().tolist(), fit_before=initial, fit_after=fit_reports,
                  validation=validation_report, validation_source_sha256=sha256(validation_root/'responses.npz'),
                  validation_used_for_fitting=False, checkpoint_sha256=sha256(checkpoint),
                  only_synthetic_stimulus=True, actor_qualified=False, vehicle_control_sent=False,
                  actual_driving_or_physiology_verified=False)
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
