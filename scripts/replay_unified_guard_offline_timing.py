"""Evaluate body feedback projection ablation inside the single flight graph on a historical development episode."""
import argparse
import base64
import hashlib
import json
import time
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
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warm', action='store_true')
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    training = base/'20261029-t1437-embodied-joint-training'
    opponent = base/'20261031-opponent-temporal-graph-trained'
    context = base/'20261031-historical-train00-temporal-context-onepass'
    body_fit = base/'20261030-body-feedback-brake-steering-anchored'
    data_root = base/'20261029-t1434-body21-flight-cache'
    pattern_root=base/'20261101-adaptive-pattern-fit'
    pattern_result=read(pattern_root/'result.json')
    if pattern_result['status']!='completed_synthetic_adaptive_pattern_fit' or sha256(pattern_root/'adaptive-pattern-contrast.pt')!=pattern_result['checkpoint_sha256']:
        raise ValueError('exact experimental pattern checkpoint required')
    pattern=torch.load(pattern_root/'adaptive-pattern-contrast.pt',map_location='cpu',weights_only=True)
    if pattern['actor_qualified'] is not False:raise ValueError('unqualified pattern only')
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
    selected = [e for e in data_manifest['episodes'] if e['file'] == 'episode-06.npz']
    if len(selected) != 1:
        raise ValueError('Exact posthoc history required')
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), dataset_sha256=sha256(data_root/'dataset.json'),
        forecast_disabled=True, forecast_sha256=fr['checkpoint_sha256'], source_visual_sha256=sha256(tr['checkpoint']),
        source_pattern_sha256=pattern_result['checkpoint_sha256'], source_road_candidate_sha256=road_result['checkpoint_sha256'], source_visual_candidate_sha256=visual_result['checkpoint_sha256'], source_body_sha256=roll_result['checkpoint_sha256'], source_context_sha256=cr['checkpoint_sha256'],
        role='posthoc coverage-selected, train-labelled but original source fit-forbidden', warmup=args.warm,
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
    if pattern['body_sha256']!=roll_result['checkpoint_sha256'] or pattern['temporal_sha256']!=visual_result['checkpoint_sha256']:
        raise ValueError('pattern checkpoint lineage mismatch')
    model.spatial_contrast=RetinalSpatialContrastBoundary(len(motor.backbone.vision_ports),reference=pattern['reference'],max_contrast_rms=pattern['max_contrast_rms'])
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
        model.freshness_limits=HostFreshnessLimits(.05,.15)
        with torch.no_grad():
            model.motion_forecast.raw_gain.zero_();model.body_feedback.projection.weight.copy_(adapter_weight)
            model.temporal.opponent_projection.weight.copy_(new_temporal_weight);model.visual.weight.copy_(original_retina_weight)
            model.spatial_contrast.projection.weight.copy_(pattern['weight'].to('cuda'))
        def inputs(i):
            camera=events[i]['camera']
            if [v['view_id'] for v in camera['views']]!=['front','left','right','rear']:raise ValueError('four original views')
            ctx=per_view_temporal_body_context((None,)*4,bodies[i],render_start_lowers=tuple(float(v['render_started_lower']) for v in camera['views']),camera_receipt_time=float(camera['received']),decision_time=float(clocks[i]))
            return ctx,torch.as_tensor(scenes[i:i+1],device='cuda')
        if args.warm:
            model.reset_episode()
            for i in range(3):
                ctx,scene=inputs(i);model.sensory_decision(decision_time=float(clocks[i]),racing_goals=goals,scene_features=scene,scene_valid=view_valid,body_context=ctx)
            torch.cuda.synchronize();model.reset_episode()
        class ReplayClock:
            def anchor(self,recorded):self.recorded=recorded;self.started=time.perf_counter()
            def __call__(self):return self.recorded+(time.perf_counter()-self.started)
        clock=ReplayClock();guard=UnifiedHostGuard(model,clock=clock);published=[];durations=[];reason=None;failed_index=None
        for i in range(len(events)):
            ctx,scene=inputs(i);clock.anchor(float(clocks[i]))
            try:
                v=guard.step(decision_time=float(clocks[i]),racing_goals=goals,scene_features=scene,scene_valid=view_valid,body_context=ctx)
            except (ValueError,RuntimeError) as e:
                failed_index=i;reason=f'{type(e).__name__}: {e}';durations.append(time.perf_counter()-clock.started);break
            durations.append(time.perf_counter()-clock.started)
            if v is not None:published.append((i,v.intent.detach().cpu().numpy()[0]))
            if i%50==0:write_json(run/'progress.json',dict(decision=i,total=len(events),published=len(published)))
        elapsed=np.asarray(durations);indices=np.array([x[0] for x in published],dtype=np.int64);intents=np.asarray([x[1] for x in published],dtype=np.float32).reshape(-1,2)
        original_root=base/'20261101-adaptive-selected-history-replay';original_result=read(original_root/'result.json');original_file=original_root/'episode-06-body-ablation-intents.npz'
        if original_result['status']!='completed_synthetic_adaptive_selected_history_replay' or sha256(original_file)!=original_result['episodes'][0]['output_sha256']:raise ValueError('reference output mismatch')
        with np.load(original_file,allow_pickle=False) as z:reference=z['intents'][1].copy()
        error=float(np.max(np.abs(intents-reference[indices]))) if len(indices) else None
        output=run/'timings.npz';np.savez_compressed(output,elapsed_seconds=elapsed,indices=indices,intents=intents)
        result=dict(status='completed_guarded_offline_replay' if failed_index is None else 'stopped_on_host_freshness_rejection',warmup=args.warm,total_events=len(events),attempted=len(elapsed),published=len(published),priming_suppressed=True,failed_index=failed_index,reason=reason,guard_latched=guard.failed,compute_seconds_median=float(np.median(elapsed)),compute_seconds_p95=float(np.quantile(elapsed,.95)),compute_seconds_max=float(np.max(elapsed)),published_max_abs_reference_error=error,output_sha256=sha256(output),training_admissible=False,optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False,physical_exposure_sync_verified=False,interpretation='Historical host times with actual local measured computation and CUDA synchronization; per-decision clock anchored offline, not paced live; input preparation and later transfer excluded')
        write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
