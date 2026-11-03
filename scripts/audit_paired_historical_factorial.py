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
from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--cpu-backend', choices=('scipy_csr_cpu','numba_csr8_cpu'), required=True)
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
    if ((args.device == 'cuda' and not torch.cuda.is_available()) or sha256(tr['checkpoint']) != op['source_checkpoint_sha256']
            or sha256(opponent/'temporal-boundary.pt') != op['checkpoint_sha256']
            or sha256(context/'context-boundary.pt') != cr['checkpoint_sha256']
            or sha256(body_fit/'body-feedback-adapter.pt') != br['checkpoint_sha256']
            or fr['status'] != 'completed_synthetic_motion_forecast_training'
            or fr['actor_qualified'] is not False
            or sha256(forecast_root/'motion-forecast.pt') != fr['checkpoint_sha256']
            or sha256(data_root/'dataset.json') != read(data_root/'result.json')['dataset_manifest_sha256']):
        raise ValueError('Original graph, dataset and frozen forecast lineage required')
    selected = [e for e in data_manifest['episodes'] if e['file'] in ('episode-03.npz','episode-04.npz')]
    if len(selected) != 2:
        raise ValueError('Exact posthoc history required')
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), dataset_sha256=sha256(data_root/'dataset.json'),
        forecast_disabled=True, forecast_sha256=fr['checkpoint_sha256'], source_visual_sha256=sha256(tr['checkpoint']),
        source_pattern_sha256=pattern_result['checkpoint_sha256'], source_road_candidate_sha256=road_result['checkpoint_sha256'], source_visual_candidate_sha256=visual_result['checkpoint_sha256'], source_body_sha256=roll_result['checkpoint_sha256'], source_context_sha256=cr['checkpoint_sha256'],
        role='previously examined train/development roles, all original sources fit-forbidden; no independent holdout claim', device=args.device,
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
    model.motor.backbone.circuit.sparse_backend = args.cpu_backend if args.device=='cpu' else 'cached_csr_gpu'
    model = model.to(args.device).eval().requires_grad_(False)
    payload = torch.load(forecast_root/'motion-forecast.pt', map_location='cpu', weights_only=True)
    if payload['schema'] != 'flybrain.experimental.causal_motion_forecast.v1' or payload['actor_qualified'] is not False:
        raise ValueError('Explicit research-only forecast required')
    trained = payload['model']['raw_gain'].to(args.device)
    adapter_weight = model.body_feedback.projection.weight.detach().clone()
    original_temporal_weight = model.temporal.opponent_projection.weight.detach().clone()
    visual_payload = torch.load(visual_fit/'opponent-projection.pt',map_location='cpu',weights_only=True)
    if visual_payload['actor_qualified'] is not False or visual_payload['source_body_sha256'] != roll_result['checkpoint_sha256']:
        raise ValueError('Unqualified and matching synthetic visual candidate required')
    new_temporal_weight = visual_payload['weight'].to(args.device)
    road_payload=torch.load(road_fit/'visual-retina-weight.pt',map_location='cpu',weights_only=True)
    if road_payload['actor_qualified'] is not False or road_payload['body_sha256']!=roll_result['checkpoint_sha256'] or road_payload['temporal_sha256']!=visual_result['checkpoint_sha256']:
        raise ValueError('Synthetic road candidate lineage required')
    original_retina_weight=model.visual.weight.detach().clone()
    trained_retina_weight=road_payload['weight'].to(args.device)
    goals = torch.tensor(racing_goal(100, 1), device=args.device)
    view_valid = torch.ones((1, 4), dtype=torch.bool, device=args.device)
    reports = []
    starting_reference = model.motor.backbone.readout.lateral_reference.detach().clone()
    starting_gain = model.motor.backbone.readout.turn_log_gain.detach().clone()
    paired_root=base/'20261102-paired-direction-fit-1';paired_result=read(paired_root/'result.json');paired_file=paired_root/'candidate.pt';paired_plan=read(base/'20261102-paired-direction-fit-plan-1/plan.json')
    speed_file=base/'20261101-synthetic-state-speed-1/state-speed-candidate.pt';power_file=base/'20261101-synthetic-power-reference-closed-loop-1/power-reference-candidate.pt'
    if paired_result['status']!='completed_synthetic_paired_direction_history_fit' or paired_result['real_data_used_for_fit'] is not False or sha256(paired_file)!=paired_result['checkpoint_sha256'] or sha256(speed_file)!=paired_plan['speed_sha256'] or sha256(power_file)!=paired_plan['power_sha256']:raise ValueError('Exact synthetic candidate and source plan required')
    speed=torch.load(speed_file,map_location='cpu',weights_only=True);power=torch.load(power_file,map_location='cpu',weights_only=True)
    if speed['actor_qualified'] is not False or power['actor_qualified'] is not False:raise ValueError('Research candidates only')
    model.speed_command_boundary=FlightStateSpeedCommand(6).to(args.device);model.speed_command_boundary.load_state_dict(speed['model'],strict=True);model.speed_command_boundary.eval().requires_grad_(False)
    with torch.no_grad():model.motor.backbone.readout.power_reference.copy_(power['power_reference'].to(args.device))
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
        with torch.no_grad():
            model.motion_forecast.raw_gain.zero_();model.body_feedback.projection.weight.copy_(adapter_weight)
            model.temporal.opponent_projection.weight.copy_(new_temporal_weight);model.visual.weight.copy_(original_retina_weight)
            model.spatial_contrast.projection.weight.copy_(pattern['weight'].to(args.device))
        def inputs(i):
            camera=events[i]['camera']
            if [v['view_id'] for v in camera['views']]!=['front','left','right','rear']:raise ValueError('four original views')
            ctx=per_view_temporal_body_context((None,)*4,bodies[i],render_start_lowers=tuple(float(v['render_started_lower']) for v in camera['views']),camera_receipt_time=float(camera['received']),decision_time=float(clocks[i]))
            return ctx,torch.as_tensor(scenes[i:i+1],device=args.device)
        def decide(i):
            ctx,scene=inputs(i);start=time.perf_counter();intent,_=model.experimental_recurrent_speed_decision(decision_time=float(clocks[i]),racing_goals=goals,scene_features=scene,scene_valid=view_valid,body_context=ctx)
            if args.device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            return intent.detach().cpu().numpy()[0],elapsed
        candidate_root=paired_root;fit_result=paired_result;candidate_path=paired_file
        if fit_result['status']!='completed_synthetic_paired_direction_history_fit' or sha256(candidate_path)!=fit_result['checkpoint_sha256'] or fit_result['real_data_used_for_fit'] is not False:raise ValueError('Verified synthetic-only fit required')
        candidate=torch.load(candidate_path,map_location='cpu',weights_only=True)
        if candidate['actor_qualified'] is not False or candidate['synthetic_only'] is not True:raise ValueError('Unqualified synthetic research candidate required')
        projection=model.spatial_contrast.projection.weight;reference=model.motor.backbone.readout.lateral_reference;original_projection=projection.detach().clone();original_reference=starting_reference;scenarios=['observed','projection','reference','gain','proj_ref','proj_gain','ref_gain','synthetic_candidate'];outputs={}
        for scenario in scenarios:
            with torch.no_grad():
                use_proj=scenario in ('projection','proj_ref','proj_gain','synthetic_candidate');use_ref=scenario in ('reference','proj_ref','ref_gain','synthetic_candidate');use_gain=scenario in ('gain','proj_gain','ref_gain','synthetic_candidate')
                projection.copy_(candidate['spatial_projection'].to(args.device) if use_proj else original_projection);reference.copy_(candidate['lateral_reference'].to(args.device) if use_ref else original_reference);model.motor.backbone.readout.turn_log_gain.copy_(candidate['turn_log_gain'].to(args.device) if use_gain else starting_gain)
                if scenario=='observed' and not torch.equal(reference,starting_reference):raise ValueError('Baseline neutral reference changed between episodes')
            model.reset_episode();series=[]
            for i in range(len(events)):
                intent,_=decide(i);series.append(intent)
                if i%100==0:write_json(run/'progress.json',dict(episode=episode['file'],scenario=scenario,decision=i,total=len(events)))
            outputs[scenario]=np.stack(series)
        prior_root=base/'20261102-paired-historical-diagnostic-1';prior=read(prior_root/'result.json');prior_rows=[x for x in prior['episodes'] if x['episode']==episode['file']];prior_file=prior_root/episode['file'].replace('.npz','-ablation-intents.npz')
        if prior['status']!='completed_paired_candidate_historical_diagnostic' or len(prior_rows)!=1 or sha256(prior_file)!=prior_rows[0]['output_sha256']:raise ValueError('Original paired diagnostic required')
        with np.load(prior_file,allow_pickle=False) as z:replay_diff=max(float(np.max(np.abs(outputs[k]-z[k]))) for k in ('observed','synthetic_candidate'))
        if replay_diff>2e-5:raise ValueError('Original paired condition failed repeat threshold')
        # Privileged future path is opened only after all model decisions have been computed.
        coverage=base/'20261101-historical-future-path-coverage';cov=read(coverage/'result.json');matches=[x for x in cov['episodes'] if x['episode']==episode['file']]
        if cov['status']!='completed_historical_future_path_coverage' or len(matches)!=1 or matches[0]['original_fit_forbidden'] is not True or matches[0]['events_sha256']!=sha256(event_file):raise ValueError('Original future path role mismatch')
        path=coverage/episode['file'].replace('.npz','-path.npz')
        if sha256(path)!=matches[0]['path_sha256']:raise ValueError('Original future path hash mismatch')
        with np.load(path,allow_pickle=False) as z:curvature=z['curvature'].copy();valid=z['valid'].copy();speed=z['speed'].copy();future_clock=z['clock'].copy()
        np.testing.assert_array_equal(future_clock,clocks)
        baseline_difference=None  # Original comparison used another speed/readout contract.
        metrics={}
        for j,h in enumerate((5,10,20)):
            keep=valid[:,j]&(speed>1)&(np.abs(curvature[:,j])>=.001);truth=curvature[:,j];left=keep&(truth<0);right=keep&(truth>0);target=-np.sign(truth)
            metrics[str(h)]={'left':int(left.sum()),'right':int(right.sum()),'models':{}}
            for scenario,values in outputs.items():
                signs=np.sign(values[:,0]);lm=float(np.mean(signs[left]==target[left])) if left.any() else None;rm=float(np.mean(signs[right]==target[right])) if right.any() else None
                metrics[str(h)]['models'][scenario]={'left_recall':lm,'right_recall':rm,'balanced_accuracy':(lm+rm)/2 if lm is not None and rm is not None else None,'positive':int((signs[keep]>0).sum()),'negative':int((signs[keep]<0).sum()),'zero':int((signs[keep]==0).sum()),'turn_change_sign_vs_observed':int((signs[keep]!=np.sign(outputs['observed'][keep,0])).sum())}
        out=run/episode['file'].replace('.npz','-ablation-intents.npz');np.savez_compressed(out,clocks=clocks,**outputs)
        reports.append({'episode':episode['file'],'role':episode['role'],'decisions':len(events),'path_sha256':sha256(path),'events_sha256':sha256(event_file),'output_sha256':sha256(out),'baseline_reference_max_abs':baseline_difference,'paired_replay_max_abs':replay_diff,'horizons':metrics})
        write_json(run/'partial-results.json',{'episodes':reports})
    result={'status':'completed_paired_factorial_historical_diagnostic','episodes':reports,'fit_allowed':False,'source_training_admissible':False,'future_truth_actor_input':False,'candidate_sha256':sha256(paired_file),'speed_sha256':sha256(speed_file),'power_sha256':sha256(power_file),'optimizer_updates':0,'actor_deployed':False,'vehicle_control_sent':False,'historical_context_missing_all_views':True,'interpretation':'Previously examined coverage-selected fit-forbidden histories, correlated frames, future driven chord is post-hoc evaluation only; no claim of holdout, physiology, calibrated steering or driving performance.'}
    write_json(run/'result.json',result);print(json.dumps(result,allow_nan=False))



if __name__=='__main__':main()
