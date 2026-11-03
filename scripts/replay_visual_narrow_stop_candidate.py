"""Probe cross-model stop ports with stored vision/body and synthetic stop goals."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_learning import complete_history
from flybrain_drive.flight_sensory_stream import FlightSensoryStream
from flybrain_drive.flight_visual_body_stop import (
    ExperimentalVisualBodyNarrowStop,
    ExperimentalVisualBodyStop,
)
from flybrain_drive.graph import sha256
from flybrain_drive.relative_airflow_sensor import relative_airflow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'candidate', 'stop-plan', 'audit', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    training, candidate = Path(a.training), Path(a.candidate)
    tr, tm, cr = read(training / 'result.json'), read(training / 'manifest.json'), read(
        candidate / 'result.json')
    plan = tm['config']['plan']
    old_plan = read(a.stop_plan)
    audit = Path(a.audit)
    audit_config = read(audit / 'manifest.json')['config']
    audit_result = read(audit / 'result.json')
    candidate_config = read(candidate / 'manifest.json')['config']
    spec, old_spec = read(plan['source_spec']), read(old_plan['model']['spec'])
    if (tr['status'] != 'completed_fixed_joint_command_training'
            or sha256(tr['checkpoint']) != tr['checkpoint_sha256']
            or cr['status'] != 'completed_fixed_stop_strength_selection'
            or cr['actor_deployed'] is not False
            or sha256(candidate / 'stop-task-adapter.pt') != cr['checkpoint_sha256']
            or audit_result['status'] != 'completed_stop_boundary_strength_audit'
            or candidate_config['training_comparison_result_sha256'] != sha256(
                audit / 'result.json')
            or audit_config['source_hashes'][a.stop_plan] != sha256(a.stop_plan)
            or candidate_config['original_checkpoint_sha256'] != (
                audit_result['candidate_checkpoint_sha256'])
            or not torch.cuda.is_available()):
        raise ValueError('Original checkpoint and fixed offline stop candidate required')
    for key in ('graph_sha256', 'goal_input_ids', 'body_input_ids', 'motor_groups'):
        if spec[key] != old_spec[key]:
            raise ValueError('Graph and ordered port/motor identities must match: ' + key)
    source = read(Path(plan['source']) / 'result.json')
    if sha256(source['checkpoint']) != source['checkpoint_sha256']:
        raise ValueError('Original v6 source checkpoint changed')
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
        source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    payload = torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)
    if payload['schema'] != model.schema or payload['spec'] != model.spec:
        raise ValueError('Original v7 specification required')
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    old = torch.load(candidate / 'stop-task-adapter.pt', map_location='cpu', weights_only=True)
    if (old['schema'] != 'flybrain.experimental_stop_vector_scaled.v1'
            or old['actor_qualified'] is not False
            or old['model']['stop_delta'].shape != (len(spec['goal_input_ids']),)):
        raise ValueError('Exactly ordered 29-port offline stop hypothesis required')
    root = Path(plan['data'])
    entries = [e for e in read(root / 'dataset.json')['episodes']
               if e['file'] == 'episode-02.npz' and e['role'] == 'development']
    if len(entries) != 1 or sha256(root / entries[0]['file']) != entries[0]['sha256']:
        raise ValueError('Exact known development sensory history required')
    names = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')
    with load_array(root / entries[0]['file'], allow_pickle=False) as data:
        time = data['observation_time'].copy()
        tensors = {key: torch.tensor(data[key], device='cuda') for key in names}
    complete_history(time, time)
    if len(time) <= 451 or not torch.all(tensors['goals'] == torch.tensor(
            [0., 1.], device='cuda')):
        raise ValueError('Original 30km/h goal and full history required')
    torch.set_num_threads(1)
    neutral, fitted = ExperimentalVisualBodyStop(model).to('cuda').eval(), (
        ExperimentalVisualBodyNarrowStop(model).to('cuda').eval())
    fitted.stop.load_state_dict(old['model'], strict=True)
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'original_checkpoint_sha256': tr['checkpoint_sha256'],
        'stop_candidate_sha256': cr['checkpoint_sha256'], 'stop_plan_sha256': sha256(a.stop_plan),
        'visual_spec_sha256': sha256(plan['source_spec']),
        'older_stop_spec_sha256': sha256(old_plan['model']['spec']),
        'history_sha256': entries[0]['sha256'], 'same_graph_and_ordered_ports': True,
        'goal_gate': 'exp(-desired_speed_m_s / 0.1)',
        'synthetic_stop_indices': [300, 450], 'initial_final_goal_kmh': 30,
        'known_development_not_blind': True, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    streams = [FlightSensoryStream(neutral), FlightSensoryStream(fitted)]
    output = []
    for i, clock in enumerate(time.tolist()):
        air = relative_airflow(wind_world_m_s=[0., 0., 0.], vehicle_world_m_s=[0., 0., 0.],
            body_axes_world=np.eye(3), wind_receipt=clock, body_receipt=clock,
            decision=clock, source_verified=False)
        goals = tensors['goals'][i:i+1].clone()
        if 300 <= i < 450:
            goals[:, 1] = 0
        rows = []
        for stream in streams:
            intent, _ = stream.decision(decision_time=clock,
                **{key: tensors[key][i:i+1] for key in names if key != 'goals'}, goals=goals,
                scene_valid=torch.ones((1, 4), dtype=torch.bool, device='cuda'),
                airflow_observation=air)
            rows.append(intent[0].cpu().numpy())
        output.append(rows)
    output = np.asarray(output)
    brake = np.maximum(-output[:, :, 1], 0)
    np.savez_compressed(run / 'predictions.npz', time=time, intent=output, brake=brake)
    result = {'status': 'completed_cross_model_narrow_visual_stop_counterfactual',
        'observations': len(time), 'stop_interval_observations': 150,
        'mean_brake_stop_last_50': brake[400:450].mean(0).tolist(),
        'mean_abs_steering_difference_stop': float(np.mean(np.abs(
            output[300:450, 1, 0] - output[300:450, 0, 0]))),
        'mean_brake_pre_stop_last_50': brake[250:300].mean(0).tolist(),
        'mean_brake_after_release_last_50': brake[500:550].mean(0).tolist(),
        'predictions_sha256': sha256(run / 'predictions.npz'),
        'original_body_was_not_response_to_synthetic_stop': True,
        'v5_trained_adapter_transferred_to_v7_not_calibrated': True,
        'brake_is_v7_signed_forward_codec_not_previous_v2_mn_decoder': True,
        'physical_synchronization_verified': False, 'actor_deployed': False,
        'optimizer_updates': 0, 'goal_achieved': False}
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
