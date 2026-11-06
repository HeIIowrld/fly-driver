"""Replay raw student decisions and independently bind SDK attempts to native responses."""
import argparse,base64,hashlib,json,gc
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.current_flight_control_contract import FlightDecisionEnvelope,control_request
from flybrain_drive.current_flight_runtime import load_current_flight_runtime
from flybrain_drive.dual_reference_candidate import load_dual_reference_runtime
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,NavigationBodyObservation,decode_navigation_body_prefix
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation


def envelope(value):
    value=dict(value)
    for k in ('view_render_lowers','view_render_uppers','view_body_first_copy_starts','prediction'):value[k]=tuple(value[k])
    return FlightDecisionEnvelope(**value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.run);r=read(root/'result.json');m=read(root/'manifest.json');args=m['config']['arguments']
    out=begin_run(a.output,dict(arguments=vars(a),source_result_sha256=sha256(root/'result.json'),seed=None,vehicle_control_sent=False))
    result=dict(status='failed',whole_trial_passed=r['status']=='completed_finite_student_connection_trial',actual_student_motion_verified=r.get('actual_student_motion_verified',False),actor_qualified=False,vehicle_control_sent=False)
    runtime=None;before=gc.isenabled();gc.disable();pred=[]
    try:
        for name,digest in r['artifact_sha256'].items():
            if sha256(root/name)!=digest:raise ValueError('Original artifact changed '+name)
        preflight=read(root/'ready-preflight.json');identity=preflight['identity']
        ready=read(root/'worker-ready.json')['receipt'];actor=read(root/'actor/owner-terminal.json')
        if ready['checkpoint_sha256']!=args['checkpoint_sha256'] or ready!=actor['actor_receipt']:raise ValueError('Actor readiness/checkpoint differs')
        episode=read(root/'resume-identity.json')['request_id'];generation=preflight['session']['generation']
        scenes={e['capture_id']:e for e in read(root/'camera-events.json') if e.get('status')=='scene_published'}
        with np.load(root/'published-visual-inputs.npz') as z:arrays={k:z[k].copy() for k in z.files}
        positions={int(v):i for i,v in enumerate(arrays['capture_ids'])}
        if set(positions)!=set(scenes) or len(positions)!=len(arrays['capture_ids']):raise ValueError('Published scene coverage differs')
        packets={}
        for row in read(root/'actor/owner-native-copy.json'):
            auth=row['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('Original native bytes changed')
            packets.setdefault(auth['packet_id'],(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'],row['copy']['first_copy_start']))
        encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity)
        from flybrain_drive.launch_pedal_candidate import load_control_actor_runtime
        loader=lambda cp,digest:load_control_actor_runtime(args['actor_kind'],cp,digest)
        torch.set_num_threads(1);torch.set_flush_denormal(True)
        runtime,loaded=loader(args['checkpoint'],args['checkpoint_sha256']);runtime.begin_episode(episode)
        muscles=[];policy=[];power=[]
        runtime.handles.append(runtime.head.register_forward_pre_hook(lambda mod,inp:muscles.append(inp[0].detach().numpy().copy())))
        runtime.handles.append(runtime.speed_policy.register_forward_hook(lambda mod,inp,value:policy.append(np.concatenate([value.detach().numpy()[0],inp[1].detach().numpy()[0]]))))
        runtime.handles.append(runtime.model.motor.command.register_forward_hook(lambda mod,inp,value:power.append(np.concatenate([inp[0].detach().numpy()[0],value[1].detach().numpy()[0]]))))
        maximum=0.;events=read(root/'actor/owner-ticker-events.json');outputs=[e for e in events if e['status']=='shadow_output']
        for e in outputs:
            scene=scenes[e['capture_id']];j=positions[e['capture_id']];raw,received,_=packets[e['body_packet']]
            if received!=e['body_first_receipt'] or hashlib.sha256(arrays['motion'][j].tobytes()).hexdigest()!=e['motion_features_sha256']:raise ValueError('Body/motion original binding differs')
            current=encoder.encode(decode_navigation_body_prefix(raw,received),episode_id=episode,decision_monotonic_seconds=e['decision_time'])
            history=tuple(NavigationBodyObservation(episode,v['packet'],v['receipt'],scene['camera_received'],np.asarray(v['values']),np.asarray(v['valid']),encoder.contract_sha256) for v in scene['view_body'])
            context=per_view_temporal_body_context(history,current,render_start_lowers=tuple(scene['render_start_lowers']),camera_receipt_time=scene['camera_received'],decision_time=e['decision_time'])
            motion=MotionSpatialObservation(episode,scene['camera_received'],tuple(scene['render_start_lowers']),arrays['coarse'][j],arrays['motion'][j])
            decision=runtime.decide(decision_time=e['decision_time'],scene_features=arrays['scene'][j],static_features=arrays['static'][j],body_context=context,native_sample=decode_native_extension(raw,received),motion_observation=motion)
            maximum=max(maximum,float(abs(decision['prediction']-np.asarray(e['prediction'])).max()))
            if maximum>2e-6:raise ValueError('Live neural prediction not reproduced')
            value=envelope(e['decision_envelope'])
            value.validate(now=e['output_host_time'],run_id=actor['run_id'],episode_id=episode,generation=generation,checkpoint_sha256=args['checkpoint_sha256'])
            expected_starts=tuple(packets[v['packet']][2] for v in scene['view_body'])
            if value.view_body_first_copy_starts!=expected_starts:raise ValueError('Per-view source first copy binding differs')
            pred.append(decision['prediction'])
        mu=np.concatenate(muscles);po=np.asarray(policy);pw=np.asarray(power)
        np.savez_compressed(out/'mechanism.npz',muscles=mu,policy=po,power=pw)
        result['mechanism']=dict(policy_calls=len(po),policy_target_speed_range_m_s=[float(po[:,1].min()),float(po[:,1].max())],sensed_speed_range_m_s=[float(po[:,2].min()),float(po[:,2].max())],speed_error_range_m_s=[float((po[:,1]-po[:,2]).min()),float((po[:,1]-po[:,2]).max())],command_and_power_ranges=[pw.min(0).tolist(),pw.max(0).tolist()],arrays_sha256=sha256(out/'mechanism.npz'))
        if args['actor_kind'] in ['parent','dual_guard']:
            from flybrain_drive.learned_flight_pedal_readout import LearnedFlightPedalReadout
            fit=Path('runs/student-launch-learning-20261105')/(args['actor_kind']+'-pedal-fit-02');cp=torch.load(fit/'pedals.pt',map_location='cpu',weights_only=True)
            if cp['parent_checkpoint_sha256']!=args['checkpoint_sha256']:raise ValueError('Counterfactual pedal lineage differs')
            head=LearnedFlightPedalReadout(runtime.head);head.pedal_projection.load_state_dict(cp['pedal_projection']);head.verify()
            with torch.no_grad():counterfactual=head(torch.tensor(mu)).numpy()
            result['counterfactual_pedal_mean_on_same_muscles']=counterfactual[-64:,1:].mean(0).tolist()
            result['counterfactual_pedals_are_not_actual_requests']=True
        terminal=read(root/'actuator/actuator-terminal.json');feedback=read(root/'actuator/actuator-feedback.json')
        by_decision={e['decision_envelope']['decision_id']:e['decision_envelope'] for e in outputs}
        applied=[];last=0;max_body_age=max_view_age=0.
        from flybrain_drive.current_flight_native_control_feedback import calibration_native_steering
        native_contract=preflight.get('native_feedback') or calibration_native_steering('runs/thermal-driving-20261103/steering-calibration-02')
        gain=native_contract['native_raw_per_sdk_normalized_request']
        result['native_feedback_contract']=native_contract
        for event in terminal['events']:
            if event['kind']!='student_request':continue
            value=envelope(event['decision'])
            if by_decision.get(value.decision_id)!=event['decision']:raise ValueError('Applied decision differs from actor output')
            request=control_request(value.prediction)
            if request!=event['conversion']:raise ValueError('Canonical output altered before SDK request')
            receipt=event.get('receipt')
            if receipt is None:continue
            if receipt['requested_control']!=request['requested_control'] or not receipt['driver_accepted']:raise ValueError('SDK requested controls differ')
            end=receipt['driver_call_finished_perf_counter_seconds']
            value.validate(now=end,run_id=actor['run_id'],episode_id=episode,generation=generation,checkpoint_sha256=args['checkpoint_sha256'],after_decision=last)
            last=value.decision_id;max_body_age=max(max_body_age,end-value.body_first_receipt);max_view_age=max(max_view_age,end-min(value.view_render_lowers))
            expected=request['requested_control'];following=[v for v in feedback if end<=v['when']<=end+.1]
            errors=[(abs(v['fields']['steerAngle']-expected['steering']*gain),abs(v['fields']['gas']-expected['throttle']),abs(v['fields']['brake']-expected['brake'])) for v in following]
            applied.append(dict(decision_id=value.decision_id,request=expected,expected_native_steer=expected['steering']*gain,expected_sdk_steer_raw=expected['steering']*native_contract['sdk_raw_per_sdk_normalized_request'],following_native_packets=len(set(v['packet'] for v in following)),
                best_steer_gas_brake_errors=np.min(errors,axis=0).tolist() if errors else None))
        np.savez_compressed(out/'replayed-predictions.npz',prediction=np.asarray(pred));write_json(out/'request-feedback-comparison.json',applied)
        for row in feedback:
            if 'authority' in row:
                authority=row['authority'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
                if hashlib.sha256(raw).hexdigest()!=authority['prefix_sha256']:raise ValueError('Actuator native feedback bytes changed')
                from flybrain_drive.ac_native_physics import decode_physics_prefix
                decoded=decode_physics_prefix(raw[:PHYSICS_PREFIX.size],authority['received_monotonic_seconds'])
                expected={k:tuple(v) if isinstance(v,list) else v for k,v in row['fields'].items()}
                if dict(decoded.fields)!=expected or decoded.packet_id!=row['packet']:raise ValueError('Applied native feedback decode differs')
        stop=terminal.get('stop_feedback',[])
        strict_stop=(len({v['packet'] for v in stop})>=3 and all(abs(v['fields']['steerAngle'])<=native_contract['neutral_tolerance_native'] and abs(v['fields']['speedKmh'])<=.1 and v['fields']['gas']<=.02 and v['fields']['brake']>=.98 for v in stop))
        if terminal['stationary_feedback_verified'] and not strict_stop:raise ValueError('Native neutral stop not supported in calibrated units')
        result['native_neutral_stop_verified']=strict_stop
        result['actuator_feedback_raw_bytes_present']=all('authority' in v for v in feedback)
        actual_requests=[v for v in terminal['events'] if v['kind']=='student_request' and v.get('receipt',{}).get('driver_accepted')]
        if actual_requests:
            first=actual_requests[0]['receipt']['driver_call_started_perf_counter_seconds']
            following=next((v for v in feedback if v['first_receipt']>=first),None)
            result['first_post_request_native_temperatures_c']=None if following is None else following['fields']['tyreCoreTemperature']
            controls=np.array([[v['receipt']['requested_control'][k] for k in ('steering','throttle','brake')] for v in actual_requests])
            result['mean_requested_steering_throttle_brake']=controls.mean(0).tolist()
            result['pedal_coactivation_fraction_above02']=float(np.mean((controls[:,1]>.02)&(controls[:,2]>.02)))
            result['first_to_last_student_request_seconds']=actual_requests[-1]['when']-actual_requests[0]['when']
            from flybrain_drive.current_flight_launch_diagnostics import decode_launch_diagnostics
            active=[decode_launch_diagnostics(base64.b64decode(v['authority']['prefix_base64']),v['first_receipt']) for v in feedback if first<=v['first_receipt']<=actual_requests[-1]['when']]
            result['active_native_gear_values']=sorted({v['gear'] for v in active})
            result['active_native_rpm_range']=[min(v['rpms'] for v in active),max(v['rpms'] for v in active)]
            result['active_native_clutch_range']=[min(v['clutch'] for v in active),max(v['clutch'] for v in active)]
            result['launch_diagnostics_actor_input']=False
        fields=[v['fields'] for v in feedback]
        result.update(status='completed_independent_student_audit',neural_outputs_verified=len(pred),maximum_prediction_difference=maximum,
            sdk_requests_verified=len(applied),maximum_body_age_at_sdk_finish_seconds=max_body_age,maximum_view_age_at_sdk_finish_seconds=max_view_age,
            maximum_native_speed_kmh=max(abs(v['speedKmh']) for v in fields),maximum_native_gas=max(v['gas'] for v in fields),
            maximum_native_steer_raw=max(abs(v['steerAngle']) for v in fields),maximum_native_displacement_m=max(v.get('displacement_m',0) for v in feedback),
            four_wheel_start_temperatures_c=preflight['native_fields']['tyreCoreTemperature'],stationary_feedback_verified=terminal['stationary_feedback_verified'],
            device_released=terminal['device_released'],stop_reason=terminal['stop_reason'],failed_trial_preserved=not result['whole_trial_passed'],
            full_lap_verified=False,physiological_qualification=False)
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),neural_outputs_verified=len(pred))
    finally:
        if runtime is not None:runtime.close()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
        write_json(out/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
