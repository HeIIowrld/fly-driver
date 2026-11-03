"""Qualify joint visual/body/airflow input on the original fixed flight graph."""

import argparse
import json
from pathlib import Path

import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.graph import sha256
from flybrain_drive.relative_airflow_sensor import relative_airflow
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train = Path(args.training)
    receipt, manifest = read(train/'result.json'), read(train/'manifest.json')
    plan = manifest['config']['plan']
    if (receipt['status'] != 'completed_fixed_joint_command_training'
            or sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']):
        raise ValueError('Original fixed visual-body graph checkpoint required')
    source = read(Path(plan['source'])/'result.json')
    if sha256(source['checkpoint']) != source['checkpoint_sha256']:
        raise ValueError('Original graph source changed')
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [entry for entry in dataset['episodes'] if entry['role'] == 'train']
    if not entries:
        raise ValueError('Original training history required')
    path = root/entries[0]['file']
    if sha256(path) != entries[0]['sha256']:
        raise ValueError('Original training history changed')
    with load_array(path, allow_pickle=False) as a:
        arrays = {key: a[key].copy() for key in a.files}
    if len(arrays['observation_time']) < 9 or not torch.cuda.is_available():
        raise ValueError('Nine actual history states and GPU required')
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
        source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    payload = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    if payload['schema'] != model.schema or payload['spec'] != model.spec:
        raise ValueError('Original fixed topology/spec required')
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    joint = ExperimentalVisualBodyAirflow(model).to('cuda').eval()
    keys = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')
    t = {key: torch.tensor(arrays[key][:9], device='cuda') for key in keys}
    views = torch.ones((9, 4), dtype=torch.bool, device='cuda')
    wind = torch.tensor([[3., 5., -1.]], device='cuda').repeat(9, 1)
    valid = torch.ones((9, 3), dtype=torch.bool, device='cuda')
    with torch.no_grad():
        baseline = model._drive_tensor(*(t[key] for key in keys))
        initial = joint.drive(*(t[key] for key in keys), views, wind, valid)
    if not torch.equal(initial, baseline):
        raise ValueError('Joint neutral initialization must preserve original drive exactly')
    with torch.no_grad():
        changed_scene = t['scene_features'].clone()
        changed_scene[:, :] = model.backbone.scene_command.mean
        changed_body = t['body'].clone()
        changed_body[:, :2] = 0
        visual_test = joint.drive(t['visual'], t['visual_valid'], t['body'], t['body_valid'],
                                  t['goals'], changed_scene, views, wind, valid)
        tilt_test = joint.drive(t['visual'], t['visual_valid'], changed_body, t['body_valid'],
                                t['goals'], t['scene_features'], views, wind, valid)
        visual_change = float((initial-visual_test).abs().mean())
        tilt_change = float((initial-tilt_test).abs().mean())
        joint.airflow.projection.weight.fill_(.1)
        invalid = torch.zeros_like(valid)
        absent = joint.drive(*(t[key] for key in keys), views, wind, invalid)
        if not torch.equal(absent, baseline):
            raise ValueError('Unavailable airflow must remain neutral even after weight changes')
        joint.airflow.projection.weight.zero_()
    decisions = arrays['observation_time'][:9].tolist()
    def observation(decision, verified):
        return relative_airflow(
            wind_world_m_s=[3., 5., -1.], vehicle_world_m_s=[0., 0., 0.],
            body_axes_world=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            wind_receipt=decision, body_receipt=decision, decision=decision,
            source_verified=verified,
        )
    missing_obs = [observation(d, False) for d in decisions]
    present_obs = [observation(d, True) for d in decisions]
    with torch.no_grad():
        joint.airflow.projection.weight.fill_(.1)
        observed_missing = joint.drive_observed(
            *(t[key] for key in keys), views, missing_obs, decisions)
        observed_present = joint.drive_observed(
            *(t[key] for key in keys), views, present_obs, decisions)
        if not torch.equal(observed_missing, baseline):
            raise ValueError('Unverified observed airflow must remain neutral')
        if torch.equal(observed_present, baseline):
            raise ValueError('Synthetic verified fixture must reach original sensory ports')
        try:
            joint.drive_observed(*(t[key] for key in keys), views, present_obs,
                                 [d + .001 for d in decisions])
        except ValueError:
            pass
        else:
            raise ValueError('Airflow observation reuse must fail')
        joint.airflow.projection.weight.zero_()
    bad_views = views.clone()
    bad_views[0, 0] = False
    try:
        joint.drive(*(t[key] for key in keys), bad_views, wind, valid)
    except ValueError:
        camera_loss_rejected = True
    else:
        raise ValueError('Camera loss must not be silently repaired')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_checkpoint_sha256': receipt['checkpoint_sha256'], 'history_sha256': sha256(path),
        'dataset_sha256': sha256(root/'dataset.json'), 'role': 'original_train_first9',
        'airflow': 'synthetic body-frame [3,5,-1] m/s', 'optimizer_updates': 0})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    state = joint.initial_state(1)
    for i in range(8):
        drive = joint.drive(*(t[key][i:i+1] for key in keys), views[i:i+1],
                            wind[i:i+1], valid[i:i+1])
        for dt in substeps(float(arrays['observation_time'][i+1]-arrays['observation_time'][i])):
            (_, _), state = joint.advance(drive, state, dt)
    output = model._read_state_tensor(state)[0]
    output.square().sum().backward()
    gradient = joint.airflow.projection.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
        raise ValueError('Actual recurrent graph gradient to airflow boundary required')
    if sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']:
        raise ValueError('Original fixed checkpoint changed')
    result = {'status': 'qualified_joint_visual_body_synthetic_airflow_graph',
        'original_train_episode': entries[0]['file'], 'original_observations': 9,
        'actual_recurrent_intervals': 8, 'baseline_drive_exact': True,
        'camera_loss_rejected': camera_loss_rejected,
        'observed_unverified_airflow_neutral': True,
        'synthetic_observed_airflow_reaches_body_ports': True,
        'observed_reuse_next_decision_rejected': True,
        'scene_mean_replacement_drive_mean_abs': visual_change,
        'tilt_zero_replacement_drive_mean_abs': tilt_change,
        'missing_airflow_neutral_after_nonzero_weights': True,
        'synthetic_airflow_m_s': [3., 5., -1.],
        'airflow_gradient_norm': float(gradient.norm()), 'actual_airflow_recorded': False,
        'source_graph_frozen': True, 'optimizer_updates': 0, 'actor_deployed': False,
        'physiological_or_driving_validation': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
