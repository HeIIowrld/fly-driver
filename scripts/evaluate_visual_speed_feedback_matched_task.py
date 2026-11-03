"""Evaluate fixed stop boundary on known 30km/h sensory histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_sensory_stream import FlightSensoryStream
from flybrain_drive.flight_visual_body_stop import (
    ExperimentalVisualBodySpeedErrorStop,
    ExperimentalVisualBodyStop,
)
from flybrain_drive.graph import sha256
from flybrain_drive.relative_airflow_sensor import relative_airflow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'prior', 'candidate', 'transfer', 'output'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    training, prior = Path(a.training), Path(a.prior)
    tr, tm, pr = read(training / 'result.json'), read(training / 'manifest.json'), read(
        prior / 'result.json')
    plan = tm['config']['plan']
    prior_config = read(prior / 'manifest.json')['config']
    root = Path(plan['data'])
    dataset = read(root / 'dataset.json')
    source = read(Path(plan['source']) / 'result.json')
    prior_predictions = prior / 'development-predictions.npz'
    if (tr['status'] != 'completed_fixed_joint_command_training'
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or sha256(source['checkpoint']) != source['checkpoint_sha256']
            or pr['status'] != 'completed_fixed_embodied_development_evaluation'
            or prior_config['checkpoint_sha256'] != tr['checkpoint_sha256']
            or prior_config['training_result_sha256'] != sha256(training / 'result.json')
            or not torch.cuda.is_available()):
        raise ValueError('Exact completed original graph and saved development replay required')
    prior_hash = sha256(prior_predictions)
    entries = [e for e in dataset['episodes'] if e['role'] == 'development']
    if len(entries) != 2:
        raise ValueError('The original two known development episodes required')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
        source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    payload = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)
    if payload['schema'] != model.schema or payload['spec'] != model.spec:
        raise ValueError('Exact original model and schema required')
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    candidate, transfer = Path(a.candidate), Path(a.transfer)
    cr = read(candidate / 'result.json')
    xr = read(transfer / 'result.json')
    xc = read(transfer / 'manifest.json')['config']
    if (cr['status'] != 'completed_fixed_stop_strength_selection'
            or sha256(candidate / 'stop-task-adapter.pt') != cr['checkpoint_sha256']
            or xr['status'] != 'completed_cross_model_visual_stop_counterfactual'
            or xc['original_checkpoint_sha256'] != tr['checkpoint_sha256']
            or xc['stop_candidate_sha256'] != cr['checkpoint_sha256']
            or xc['same_graph_and_ordered_ports'] is not True):
        raise ValueError('Exact previous graph/port transfer evidence required')
    stop = torch.load(candidate / 'stop-task-adapter.pt', map_location='cpu', weights_only=True)
    if stop['actor_qualified'] is not False:
        raise ValueError('Offline research candidate required')
    neutral = ExperimentalVisualBodyStop(model).to('cuda').eval()
    fitted = ExperimentalVisualBodySpeedErrorStop(model).to('cuda').eval()
    fitted.stop.load_state_dict(stop['model'], strict=True)
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'dataset_sha256': sha256(root / 'dataset.json'),
        'source_checkpoint_sha256': tr['checkpoint_sha256'],
        'stop_candidate_sha256': cr['checkpoint_sha256'],
        'goal_gate': 'max(narrow_goal_gate, wide_goal_gate * tanh(positive_speed_error/2))',
        'transfer_result_sha256': sha256(transfer / 'result.json'),
        'prior_predictions_sha256': prior_hash,
        'prior_prediction_hash_previously_pinned': False,
        'known_development_not_blind': True, 'task_kmh': 30,
        'receipt_targets_not_actor_inputs': True, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, saved = [], {}
    for entry in entries:
        path = root / entry['file']
        contract = entry['teacher_label_contract']
        if (sha256(path) != entry['sha256'] or entry['desired_speed_kmh'] != 30
                or contract['ai_speed_hint_kmh'] != 30
                or contract['kind'] != 'engine_ai_future_applied_control_receipt_v1'):
            raise ValueError('Same 30km/h original hashed receipt dataset required')
        with load_array(path, allow_pickle=False) as data:
            time = data['observation_time'].copy()
            names = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')
            tensors = {key: torch.tensor(data[key], device='cuda') for key in names}
        complete_history(time, time)
        if (tensors['visual_valid'].any() or not torch.all(tensors['goals'][:, 0] == 0)
                or not torch.all(tensors['goals'][:, 1] == 1)):
            raise ValueError('Original 30km/h task and invalid visual6 mask required')
        streams = [FlightSensoryStream(neutral), FlightSensoryStream(fitted)]
        prediction = []
        for i, clock in enumerate(time.tolist()):
            airflow = relative_airflow(wind_world_m_s=[0., 0., 0.],
                vehicle_world_m_s=[0., 0., 0.], body_axes_world=np.eye(3),
                wind_receipt=clock, body_receipt=clock, decision=clock, source_verified=False)
            per_condition = []
            for stream in streams:
                intent, _ = stream.decision(decision_time=clock,
                    **{key: tensors[key][i:i+1] for key in names},
                    scene_valid=torch.ones((1, 4), dtype=torch.bool, device='cuda'),
                    airflow_observation=airflow)
                per_condition.append(intent[0].cpu().numpy())
            prediction.append(per_condition)
        prediction = np.stack(prediction)
        with load_array(prior_predictions, allow_pickle=False) as old:
            original = old[entry['file'].removesuffix('.npz') + '_intact'].copy()
        max_error = float(np.max(np.abs(prediction[:, 0] - original)))
        if max_error > 1e-6:
            raise ValueError('Stateful output must reproduce existing fixed full replay')
        # Privileged teacher targets are opened only after model inference finishes.
        with load_array(path, allow_pickle=False) as data:
            action, mask = data['actions'].copy(), data['action_valid'].copy()
        if (action.shape != (len(time), 3) or mask.shape != action.shape
                or mask.dtype != bool or not np.isfinite(action).all()):
            raise ValueError('Finite original canonical action labels and masks required')
        metrics = {}
        for condition, column in (('neutral', 0), ('speed_feedback_stop', 1)):
            controls = np.stack([canonical_controls(row) for row in prediction[:, column]])
            metrics[condition] = {}
            for index, name in enumerate(('steer', 'throttle', 'brake')):
                valid, target, out = mask[:, index], action[:, index], controls[:, index]
                if not valid.any():
                    raise ValueError('Nonempty original per-head label masks required')
                metrics[condition][name] = {'count': int(valid.sum()),
                    'model_rmse': float(np.sqrt(np.mean((out[valid] - target[valid]) ** 2))),
                    'zero_rmse': float(np.sqrt(np.mean(target[valid] ** 2))),
                    'model_positive_over_001': int((out[valid] > .01).sum())}
        reports.append({'file': entry['file'], 'metrics': metrics,
                        'max_error_vs_prior': max_error,
                        'physical_command_time_verified': False})
        name = entry['file'].removesuffix('.npz')
        saved[name + '_prediction'], saved[name + '_action'] = prediction, action
        saved[name + '_mask'], saved[name + '_time'] = mask, time
    if sha256(prior_predictions) != prior_hash:
        raise ValueError('Prior replay changed during evaluation')
    np.savez_compressed(run / 'comparison.npz', **saved)
    result = {'status': 'completed_speed_feedback_same_task_development_replay',
        'episodes': reports,
        'comparison_sha256': sha256(run / 'comparison.npz'),
        'known_development_not_independent': True, 'actor_deployed': False,
        'teacher_inputs': False, 'actual_airflow_valid': False, 'optimizer_updates': 0,
        'v5_candidate_to_v7_transfer_not_calibrated': True,
        'goal_achieved': False}
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
