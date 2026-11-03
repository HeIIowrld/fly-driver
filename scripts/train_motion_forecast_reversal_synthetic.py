"""Train the existing causal visual forecast on synthetic bidirectional reversals."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.flight_motion_forecast import CausalMotionForecast
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_reversal(root, seed, amplitude, pool, generator_hash):
    receipt, config = read(root/'result.json'), read(root/'manifest.json')['config']
    path = root/'responses.npz'
    if (receipt['status'] != 'completed_coherent_four_view_reversing_yaw'
            or sha256(path) != receipt['responses_sha256'] or config['program_sha256'] != generator_hash
            or config['texture_seed'] != seed or config['reverse_after_motion_seconds'] != .6
            or config['stationary_seconds'] != .4 or receipt['cases'] != [0., -amplitude, amplitude]):
        raise ValueError('Exact declared reversal stimulus required')
    with np.load(path, allow_pickle=False) as f:
        retina, motion, times = [f[key].copy() for key in ('retina', 'motion', 'times')]
    if (retina.shape != (3, 41, 4, 721) or motion.shape != (3, 41, 4, 8, 721)
            or not np.allclose(times, np.arange(41)*.04, rtol=0, atol=1e-12)
            or not np.isfinite(retina).all() or not np.isfinite(motion).all()
            or not np.all(retina[:, :11] == retina[:, :1])
            or not np.array_equal(retina[:, -1], retina[:, 0])
            or np.array_equal(retina[1, 25], retina[2, 25])):
        raise ValueError('Complete finite original synthetic four-view episode required')
    full = np.concatenate([retina[:, :, :, None], motion], axis=3)
    with torch.no_grad():
        return pool(torch.tensor(full.reshape(-1, 4, 9, 721), dtype=torch.float32)).reshape(3, 41, 576)


def arrays(model, scene):
    previous, current = scene[:, :-2].reshape(-1, 576), scene[:, 1:-1].reshape(-1, 576)
    target = scene[:, 2:].reshape(3, 39, 4, 9, 16)[:, :, :, 1:]
    pred = model(current, previous, .04).reshape(3, 39, 4, 9, 16)[:, :, :, 1:]
    return pred, target


def mse_segments(model, scene):
    with torch.no_grad():
        prediction, target = arrays(model, scene)
        error = (prediction-target).square()
        around = error[1:, 23:27]
        other = torch.cat([error[1:, 9:23], error[1:, 27:39]], dim=1)
        return dict(reversal_mse=float(around.mean()), nonreversal_mse=float(other.mean()),
                    reversal_windows=8, nonreversal_windows=52)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    plan_path = base/'20261101-reversing-yaw-forecast-plan/plan.json'
    plan = read(plan_path)
    if (sha256('src/flybrain_drive/flight_motion_forecast.py') != plan['forecast_component_sha256']
            or sha256('scripts/replay_flyvis_coherent_yaw_reversal.py') != plan['generator_sha256']):
        raise ValueError('Declared forecast and generator sources must remain unchanged')
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    spec = read('configs/flight-circuit-visual-v6-20261029.json')
    with load_array(spec['scene_registration_file'], allow_pickle=False) as registration:
        pool = FlightScenePool(registration['image_uv'], spec['scene_basis'])
    sources = [base/f'20261101-flyvis-reversing-yaw-{seed}' for seed in plan['fit_texture_seeds']]
    fit = [load_reversal(path, seed, .23, pool, plan['generator_sha256'])
           for path, seed in zip(sources, plan['fit_texture_seeds'], strict=True)]
    model = CausalMotionForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), plan_sha256=sha256(plan_path),
        source_hashes=[sha256(p/'responses.npz') for p in sources], seed=863009,
        actor=False, real_history_used_for_fit=False))
    loss_history = []
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        terms = []
        for scene in fit:
            pred, target = arrays(model, scene)
            error = (pred-target).square()[1:]
            around = error[:, 23:27].mean()
            other = torch.cat([error[:, 9:23], error[:, 27:39]], dim=1).mean()
            terms.append(.5*around + .5*other)
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise ValueError('Finite declared weighted loss required')
        loss.backward()
        if model.raw_gain.grad is None or not torch.isfinite(model.raw_gain.grad).all():
            raise ValueError('Finite causal predictor gradient required')
        optimizer.step()
        loss_history.append(float(loss.detach()))
    # Only open the declared fresh validation response after the last update.
    validation_root = base/f"20261101-flyvis-reversing-yaw-{plan['validation_texture_seed']}"
    validation = load_reversal(validation_root, plan['validation_texture_seed'], .13, pool, plan['generator_sha256'])
    old_root = base/'20261101-causal-motion-forecast-synthetic-fit'
    old_result = read(old_root/'result.json')
    old_path = old_root/'motion-forecast.pt'
    if sha256(old_path) != old_result['checkpoint_sha256']:
        raise ValueError('Frozen original synthetic forecast changed')
    old = CausalMotionForecast()
    old.load_state_dict(torch.load(old_path, map_location='cpu', weights_only=True)['model'], strict=True)
    hold = CausalMotionForecast()
    report = dict(hold=mse_segments(hold, validation), old=mse_segments(old, validation),
                  reversal_trained=mse_segments(model, validation))
    with torch.no_grad():
        pred, target = arrays(model, validation)
        old_pred, _ = arrays(old, validation)
        hold_pred, _ = arrays(hold, validation)
    np.savez_compressed(run/'validation.npz', target=target.numpy(), prediction=pred.numpy(),
                        old_prediction=old_pred.numpy(), hold_prediction=hold_pred.numpy())
    path = run/'motion-forecast.pt'
    torch.save(dict(schema='flybrain.experimental.causal_motion_forecast.v1', model=model.state_dict(),
                    actor_qualified=False, synthetic_reversal_training=True), path)
    result = dict(status='completed_synthetic_reversal_forecast_training', optimizer_updates=200,
                  training_loss_first=loss_history[0], training_loss_last=loss_history[-1],
                  gains=model.raw_gain.tanh().detach().tolist(), validation=report,
                  validation_response_sha256=sha256(validation_root/'responses.npz'),
                  validation_used_for_fitting=False, validation_arrays_sha256=sha256(run/'validation.npz'),
                  checkpoint_sha256=sha256(path), old_checkpoint_sha256=sha256(old_path),
                  only_synthetic_stimulus=True, actor_qualified=False, vehicle_control_sent=False,
                  actual_driving_or_physiology_verified=False)
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
