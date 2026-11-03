"""Evaluate body feedback projection ablation inside the single flight graph on a historical development episode."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array, resolve_payload_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_motion_forecast import CausalMotionForecast
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    training = base/'20261029-t1437-embodied-joint-training'
    opponent = base/'20261031-opponent-temporal-graph-trained'
    context = base/'20261031-historical-train00-temporal-context-onepass'
    body_fit = base/'20261030-body-feedback-brake-steering-anchored'
    data_root = base/'20261029-t1434-body21-flight-cache'
    road_fit=base/'20261101-road-spatial-training-fit'
    road_result=read(road_fit/'result.json')
    if road_result['status']!='completed_synthetic_road_spatial_training' or sha256(road_fit/'visual-retina-weight.pt')!=road_result['checkpoint_sha256']:
        raise ValueError('Original synthetic road checkpoint required')
    visual_fit = base/'20261101-causal-yaw-multispeed-fit'
    visual_result = read(visual_fit/'result.json')
    if visual_result['status'] != 'completed_synthetic_multispeed_yaw_training' or sha256(visual_fit/'opponent-projection.pt') != visual_result['checkpoint_sha256']:
        raise ValueError('Exact synthetic visual candidate required')
    roll_fit = base/'20261101-speed-wheel-turn-control-fit'
    roll_result = read(roll_fit/'result.json')
    if roll_result['status'] != 'completed_synthetic_speed_wheel_turn_control_training' or sha256(roll_fit/'speed-body-adapter.pt') != roll_result['checkpoint_sha256']:
        raise ValueError('Original synthetic-only roll candidate required')
    forecast_root = base/'20261101-causal-motion-forecast-synthetic-fit'
    tr, op, cr, br, fr = [read(x/'result.json') for x in (training, opponent, context, body_fit, forecast_root)]
    data_manifest = read(data_root/'dataset.json')
    if (not torch.cuda.is_available() or sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or sha256(context/'context-boundary.pt') != cr['checkpoint_sha256']
            or sha256(body_fit/'body-feedback-adapter.pt') != br['checkpoint_sha256']
            or fr['status'] != 'completed_synthetic_motion_forecast_training'
            or fr['actor_qualified'] is not False
            or sha256(forecast_root/'motion-forecast.pt') != fr['checkpoint_sha256']
            or sha256(data_root/'dataset.json') != read(data_root/'result.json')['dataset_manifest_sha256']):
        raise ValueError('Original graph, dataset and frozen forecast lineage required')
    selected = [e for e in data_manifest['episodes'] if e['role'] == 'development']
    if [e['file'] for e in selected] != ['episode-02.npz', 'episode-07.npz']:
        raise ValueError('Exact original known development role required')
    selected = selected[:1]
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), dataset_sha256=sha256(data_root/'dataset.json'),
        forecast_disabled=True, forecast_sha256=fr['checkpoint_sha256'], source_visual_sha256=sha256(tr['checkpoint']),
        source_road_candidate_sha256=road_result['checkpoint_sha256'], source_visual_candidate_sha256=visual_result['checkpoint_sha256'], source_body_sha256=roll_result['checkpoint_sha256'], source_context_sha256=cr['checkpoint_sha256'],
        role='previously examined development; original source fit-forbidden',
        input='four-view scene and original body; all prior-view body explicitly missing',
        external_yaw_and_speed_error='neutral', seed=863009, optimizer_updates=0, actor_deployed=False))
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    spec = read(training/'manifest.json')['config']['plan']
    source = read(Path(spec['source'])/'result.json')
    visual = EmbodiedVisualFlightCircuit(spec['source_spec'], torch.load(source['checkpoint'],
        map_location='cpu', weights_only=True), sparse_backend='cached_csr_cpu')
    visual.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'])
    motor_plan = read(base/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model']
    motor = PhysicalManeuverCircuit(motor_plan['spec'], torch.load(motor_plan['checkpoint'],
        map_location='cpu', weights_only=True), sparse_backend='cached_csr_cpu')
    temporal = OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,
        visual.backbone.scene_command.scale, len(visual.backbone.vision_ports))
    temporal.load_state_dict(torch.load(opponent/'temporal-boundary.pt', map_location='cpu', weights_only=True)['model'])
    boundary = PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports))
    boundary.load_state_dict(torch.load(context/'context-boundary.pt', map_location='cpu', weights_only=True)['model'])
    model = UnifiedFlightSensorimotor(motor, visual.backbone, temporal, boundary,
                                     motion_forecast=CausalMotionForecast(),
                                     body_feedback=CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)))
    model.body_feedback.load_state_dict(torch.load(roll_fit/'speed-body-adapter.pt',
                                                   map_location='cpu', weights_only=True)['model'])
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_gpu'
    model = model.to('cuda').eval().requires_grad_(False)
    payload = torch.load(forecast_root/'motion-forecast.pt', map_location='cpu', weights_only=True)
    if payload['schema'] != 'flybrain.experimental.causal_motion_forecast.v1' or payload['actor_qualified'] is not False:
        raise ValueError('Explicit research-only forecast required')
    trained = payload['model']['raw_gain'].to('cuda')
    adapter_weight = model.body_feedback.projection.weight.detach().clone()
    original_temporal_weight = model.temporal.opponent_projection.weight.detach().clone()
    visual_payload = torch.load(visual_fit/'opponent-projection.pt',map_location='cpu',weights_only=True)
    if visual_payload['actor_qualified'] is not False or visual_payload['source_body_sha256'] != roll_result['checkpoint_sha256']:
        raise ValueError('Unqualified and matching synthetic visual candidate required')
    new_temporal_weight = visual_payload['weight'].to('cuda')
    road_payload=torch.load(road_fit/'visual-retina-weight.pt',map_location='cpu',weights_only=True)
    if road_payload['actor_qualified'] is not False or road_payload['body_sha256']!=roll_result['checkpoint_sha256'] or road_payload['temporal_sha256']!=visual_result['checkpoint_sha256']:
        raise ValueError('Synthetic road candidate lineage required')
    original_retina_weight=model.visual.weight.detach().clone()
    trained_retina_weight=road_payload['weight'].to('cuda')
    goals = torch.tensor(racing_goal(100, 1), device='cuda')
    view_valid = torch.ones((1, 4), dtype=torch.bool, device='cuda')
    reports = []
    for episode in selected:
        recording = Path(episode['original']['recording'])
        event_file = recording/'events.jsonl'
        if sha256(event_file) != episode['source_events_sha256'] or sha256(resolve_payload_path(data_root/episode['file'])) != episode['sha256']:
            raise ValueError('Original episode or event source changed')
        source_receipt = read(recording/'result.json')
        if source_receipt.get('training_admissible') is not False:
            raise ValueError('Original diagnostic-only source role must remain explicit')
        events = [json.loads(x) for x in event_file.read_text(encoding='utf8').splitlines()]
        identity = read(recording/'identity.json')
        with load_array(data_root/episode['file'], allow_pickle=False) as file:
            scenes, body_values, body_valid, clocks = [file[k].copy() for k in
                ('scene_features', 'body', 'body_valid', 'observation_time')]
        if (len(events) != len(clocks) or scenes.shape != (len(clocks), 576)
                or np.any(np.diff(clocks) < .036) or np.any(np.diff(clocks) > .044)):
            raise ValueError('Complete original decisions and 40ms cadence required')
        encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json', native_identity=identity)
        bodies = []
        for i, event in enumerate(events):
            native = event['native']
            raw = base64.b64decode(native['prefix_base64'], validate=True)
            if (event['index'] != i or event['decision_monotonic_seconds'] != clocks[i]
                    or hashlib.sha256(raw).hexdigest() != native['prefix_sha256']):
                raise ValueError('Original body bytes, indices and decision clock required')
            body = encoder.encode(decode_navigation_body_prefix(raw, float(native['received_monotonic_seconds'])),
                                  episode_id=episode['file'], decision_monotonic_seconds=float(clocks[i]))
            if not np.array_equal(body.valid, body_valid[i]) or not np.allclose(body.values, body_values[i], rtol=0, atol=1e-6):
                raise ValueError('Original body conversion does not reproduce cached source')
            bodies.append(body)
        outputs = []
        for feedback_enabled in (False, True):
            model.reset_episode()
            with torch.no_grad():
                model.motion_forecast.raw_gain.zero_()
                model.body_feedback.projection.weight.copy_(adapter_weight)
                model.temporal.opponent_projection.weight.copy_(new_temporal_weight)
                model.visual.weight.copy_(trained_retina_weight if feedback_enabled else original_retina_weight)
                episode_output = []
                for i, event in enumerate(events):
                    camera = event['camera']
                    if [v['view_id'] for v in camera['views']] != ['front', 'left', 'right', 'rear']:
                        raise ValueError('Four original view identities required')
                    ctx = per_view_temporal_body_context((None,)*4, bodies[i],
                        render_start_lowers=tuple(float(v['render_started_lower']) for v in camera['views']),
                        camera_receipt_time=float(camera['received']), decision_time=float(clocks[i]))
                    scene = torch.as_tensor(scenes[i:i+1], device='cuda')
                    intent, _ = model.sensory_decision(decision_time=float(clocks[i]), racing_goals=goals,
                        scene_features=scene, scene_valid=view_valid, body_context=ctx)
                    episode_output.append(intent.cpu().numpy()[0])
                    if i % 100 == 0:
                        write_json(run/'progress.json',dict(episode=episode['file'],feedback_enabled=feedback_enabled,decision=i,total=len(events)))
                outputs.append(np.asarray(episode_output))
        values = np.asarray(outputs)
        name = episode['file'].replace('.npz', '-body-ablation-intents.npz')
        np.savez_compressed(run/name, clocks=clocks, intents=values)
        change = np.abs(values[1]-values[0])
        reports.append(dict(episode=episode['file'], decisions=len(clocks), source_events_sha256=sha256(event_file),
                            output_sha256=sha256(run/name), mean_abs_intent_delta=change.mean(0).tolist(),
                            max_abs_intent_delta=change.max(0).tolist(),
                            turn_sign_changes=int((np.sign(values[0,:,0]) != np.sign(values[1,:,0])).sum()),
                            source_training_admissible=False))
    result = dict(status='completed_single_graph_synthetic_road_historical_replay', episodes=reports,
                  optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False,
                  historical_context_missing_all_views=True, physical_exposure_sync_verified=False,
                  physiology_or_driving_verified=False,
                  interpretation='Synthetic road-trained retinal projection versus original retinal projection, with identical speed/wheel and roll body adapter in the same graph; previously examined fit-forbidden historical episode. Previously examined development history; no causal claim about vehicle performance.')
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
