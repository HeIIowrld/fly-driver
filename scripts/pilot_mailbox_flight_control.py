"""Finite current flight student trial via CSP camera/lifecycle and background vJoy only."""
import argparse,copy,json,os,subprocess,sys,time,uuid,threading
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.current_flight_control_contract import FEATURE_SELECTION_MAX_VIEW_AGE
from flybrain_drive.current_flight_control_owner import CurrentFlightControlOwnerProcess
from flybrain_drive.current_flight_actuator_process import CurrentFlightActuatorProcess
from flybrain_drive.current_flight_dispatcher import CurrentFlightDecisionDispatcher,CurrentFlightDecisionObserver
from flybrain_drive.current_flight_actuation import ready_to_arm
from flybrain_drive.flight_decision_mailbox import FlightDecisionMailbox,MailboxTrial
from flybrain_drive.owned_pause_process import OwnedPauseProcess
from flybrain_drive.current_flight_visual_setup import prepare
from flybrain_drive.ac_runtime_gc import defer_automatic_cyclic_gc
from flybrain_drive.ac_csp_vision import CSPVisionRecorder,NoFreshCSPFrame
from flybrain_drive.immutable_camera_group import freeze_camera_group,require_nonzero_camera_views
from flybrain_drive.camera_body_alignment import source_render_witness
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate
from scripts.record_native_process_unified_shadow_v2 import owned_resume
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat
from scripts.shadow_current_flight_teacher_motion import previous_camera
from scripts.preflight_current_flight_control import inspect,PIN

CPU=str(Path.home() / 'AppData/Local/Programs/Python/Python312/python.exe')
CPULIB=str(Path.home() / 'AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','checkpoint-sha256','candidate-plan','output','thermal-initialization-run'):p.add_argument('--'+key,required=True)
    p.add_argument('--actor-kind',choices=['parent','dual_guard','launch_parent','launch_dual_guard','balanced_parent','balanced_dual_guard','compact_parent','compact_dual_guard'],required=True)
    p.add_argument('--calibration-result',default='runs/thermal-driving-20261103/steering-calibration-02/result.json')
    p.add_argument('--planned-student-seconds',type=float,default=.8)
    p.add_argument('--completed-metadata-evidence');p.add_argument('--expected-lua-sha256',default=PIN);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:raise ValueError('Explicit finite student execution required')
    if not .1<=a.planned_student_seconds<=.8:raise ValueError('Planned observation must be0.1..0.8seconds, all independent upper limits unchanged')
    limits=dict(maximum_seconds=1.,maximum_decisions=100,maximum_speed_kmh=10.,maximum_travel_m=1.,planned_student_seconds=a.planned_student_seconds,minimum_motion_speed_kmh=.5,minimum_motion_displacement_m=.02)
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=781605,
        limits=limits,feature_selection_max_view_age_seconds=FEATURE_SELECTION_MAX_VIEW_AGE,teacher_action_used=False,actor_input='video and permitted body only',ui_actions_sent=0,
        qualification='Independent owner-to-actuator mailbox; learned research actor; no lap/limit claim'))
    result=dict(status='failed',student_requests=0,ui_actions_sent=0,full_lap_verified=False,actor_qualified=False)
    owner=camera=actuator=dispatcher=visual=observer=mailbox=pause_worker=None;events=[];visual_inputs=[];preparation=[];resumed=False;readiness=None
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain';status_path=local/'csp-session/status.json';camera_root=local/'csp-camera'
    gc_scope=defer_automatic_cyclic_gc();gc_receipt=gc_scope.__enter__()
    try:
        if sha256(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Exact checkpoint required')
        preflight=inspect(expected_lua_sha256=a.expected_lua_sha256,calibration_result=a.calibration_result);write_json(out/'preflight.json',preflight)
        if preflight['native_guard']['gear']<2 or preflight['native_guard']['numberOfTyresOut']!=0:raise ValueError('Stationary forward gear/zero tyres-out start required')
        from scripts.audit_initial_thermal_evidence import audit
        from flybrain_drive.thermal_teacher_start import admit_thermal_teacher_start
        thermal=audit(a.thermal_initialization_run)
        admit_thermal_teacher_start(thermal,preflight['session'],preflight['native_fields']['tyreCoreTemperature'],source_sha256=a.expected_lua_sha256)
        torch.set_num_threads(1);torch.manual_seed(781605);np.random.seed(781605);sys.setswitchinterval(.001)
        build,camera_config,visual_receipt=prepare(a.candidate_plan);write_json(out/'visual-setup.json',visual_receipt)
        result['motion_features_enabled']=True;write_json(out/'motion-preflight.json',visual_receipt['motion_proof'])
        # Recheck all external identities after expensive setup.
        preflight=inspect(expected_lua_sha256=a.expected_lua_sha256,calibration_result=a.calibration_result);write_json(out/'ready-preflight.json',preflight)
        admit_thermal_teacher_start(thermal,preflight['session'],preflight['native_fields']['tyreCoreTemperature'],source_sha256=a.expected_lua_sha256)
        state=preflight['session']
        from flybrain_drive.current_flight_camera_admission import admit
        state,gate,recovery=admit(state,previous_camera(camera_root),camera_root=camera_root,status_path=status_path,pin=a.expected_lua_sha256,completed_evidence=a.completed_metadata_evidence)
        if recovery is not None:write_json(out/'completed-metadata-diagnostic.json',recovery)
        mailbox=FlightDecisionMailbox()
        pause_worker=OwnedPauseProcess(out/'final-pause',read(out/'manifest.json'),a.expected_lua_sha256,CPU,CPULIB)
        write_json(out/'pause-worker-ready.json',pause_worker.ready)
        owner_dir=out/'actor';owner_dir.mkdir()
        owner=CurrentFlightControlOwnerProcess(checkpoint=a.checkpoint,checkpoint_sha256=a.checkpoint_sha256,
            directory=owner_dir,python_executable=CPU,library_directory=CPULIB,actor_kind=a.actor_kind,run_id=out.resolve().as_posix(),mailbox_spec=mailbox.spec)
        write_json(out/'worker-ready.json',owner.ready)
        start_gate=threading.Event();camera_prepared=threading.Event()
        camera=CSPVisionRecorder(preflight['window_handle'],camera_config,max_frames=240,period=.04,max_bytes=256*1024**2,
            image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,
            mmf_metadata=True,preserve_producer_time=True,expected_source_sha256=a.expected_lua_sha256,
            mmf_stop_signal=True,memory_only_receive_checks=True,start_gate=start_gate,prepared_event=camera_prepared)
        preparation_deadline=time.perf_counter()+30
        while not camera_prepared.wait(.01):
            if camera.error:raise RuntimeError(str(camera.error))
            if time.perf_counter()>preparation_deadline:raise TimeoutError("Owned camera preparation unknown")
        native_identity={k:preflight['identity'][k] for k in ('car_model','shared_memory_version','ac_version')}
        owner.begin_live(camera.request_id,state['generation'],native_identity)
        observer=CurrentFlightDecisionObserver(owner,read_latest=mailbox.read);observer.start()
        identity=dict(run_id=out.resolve().as_posix(),episode_id=camera.request_id,generation=state['generation'],checkpoint_sha256=a.checkpoint_sha256)
        start_gate.set();camera.wait_ready()
        owned_resume(out,gate,camera,status_path,camera_root);resumed=True
        resume_deadline=time.perf_counter()+1.
        while True:
            confirmed,modified,_=read_session_heartbeat(status_path);validate_heartbeat(confirmed,time.time()-modified,mutating=False)
            if (confirmed['generation']==state['generation'] and confirmed['epoch']==state['epoch'] and confirmed['live'] and not confirmed['paused'] and not confirmed.get('pending_receipt_id') and not confirmed.get('error')):break
            if time.perf_counter()>resume_deadline:raise TimeoutError('Owned resume heartbeat did not settle')
            time.sleep(.01)
        actuator=CurrentFlightActuatorProcess(directory=out/'actuator',axis=preflight['axis'],identity=identity,limits=limits,
            guard_config=dict(status_path=str(status_path),epoch=state['epoch'],window_handle=preflight['window_handle'],native_feedback=preflight['native_feedback']),
            python_executable=CPU,library_directory=CPULIB,mailbox_spec=mailbox.spec)
        write_json(out/'actuator-ready.json',actuator.ready)
        write_json(out/'resume-identity.json',dict(request_id=camera.request_id,generation=state['generation']))
        visual=build(camera.request_id,state['generation'])
        deadline=time.perf_counter()+4.;last=-1;first_publication=None;groups=0;last_preparation_id=0
        while True:
            now=time.perf_counter()
            if dispatcher is not None:
                if dispatcher.error:raise RuntimeError('Dispatcher failed: '+str(dispatcher.error))
                if not actuator.process.is_alive():break
                if now>=trial_end:break
            elif now>=deadline:raise TimeoutError('Student preparation did not settle within four seconds')
            fresh,modified,_=read_session_heartbeat(status_path,maximum_wait=.01);validate_heartbeat(fresh,time.time()-modified,mutating=False)
            if (fresh['generation']!=state['generation'] or fresh['epoch']!=state['epoch'] or fresh['paused']
                    or not fresh['live'] or fresh['camera_request_id']!=camera.request_id or fresh['ai_controlled']):
                raise ValueError('Owned live camera/session authority changed')
            try:frame,selected,_=camera.wait_for_candidate(max_age=FEATURE_SELECTION_MAX_VIEW_AGE,timeout=.01)
            except NoFreshCSPFrame:continue
            if frame['capture_id']==last:time.sleep(.001);continue
            last=frame['capture_id'];event=dict(capture_id=last,camera_received=frame['received'],render_start_lowers=[v['render_started_lower'] for v in frame['views']]);events.append(event)
            group=freeze_camera_group(frame,camera.samples_for(frame),selected_at=selected);require_nonzero_camera_views(group)
            bounds=tuple(event['render_start_lowers']);uppers=[];clock=camera.clock.record()
            for view in frame['views']:
                witness=view['source_render_metadata'];source_render_witness(witness)
                uppers.append(witness['render_started_clock']+clock['offset_upper_seconds']+clock.get('host_clock_quantization_seconds',0.))
            context=owner.context_for_camera(bounds=bounds,uppers=tuple(uppers),camera_received=frame['received'],generation=state['generation'],capture_id=last)
            if context['status']!='context':
                event.update(status='unqualified_initial_pixels',reason=context.get('reason'))
                if visual.previous is not None:raise ValueError('Source qualification lost inside episode')
                continue
            obs=visual.process(pixels=group['pixels'],render_start_lowers=bounds,camera_received=frame['received'],capture_id=last,generation=state['generation']);torch.cuda.synchronize()
            event.update(context['evidence']);event['source_render_gap_seconds']=context['gaps'];event['motion_features_sha256']=obs.motion_features_sha256
            event['publication']=owner.publish(obs,context['context_id']);event['status']='scene_published';groups+=1
            if first_publication is None:first_publication=event['publication']
            visual_inputs.append((last,obs.scene.copy(),obs.static.copy(),obs.motion_features.copy(),obs.coarse_motion_reference.copy()))
            visual.prepare_held_prefix();torch.cuda.synchronize()
            if dispatcher is None:
                wait_until=time.perf_counter()+.01
                decision=None
                while True:
                    if observer.error:raise RuntimeError('Preparation observer failed: '+str(observer.error))
                    decision=observer.latest_decision
                    if decision is not None:
                        for observed in tuple(observer.decisions):
                            value=observed['decision']
                            if last_preparation_id<value['decision_id']<=decision.decision_id:
                                preparation.append(value);last_preparation_id=value['decision_id']
                    if decision is not None and decision.publication_id==owner.publication_sequence:break
                    if time.perf_counter()>=wait_until:break
                    time.sleep(.001)
                if ready_to_arm(preparation,decision,now=time.perf_counter(),first_publication=first_publication,
                        valid_groups=groups,publication_id=owner.publication_sequence,capture_id=last):
                    if gate.phase=='camera_ready':
                        gate.resume_observed(fresh,read(camera_root/'status.json'),first_body_packet=preflight['native']['packet_id'],later_body_packet=decision.body_packet)
                    if observer.error:raise RuntimeError('Preparation observer failed: '+str(observer.error))
                    decision=observer.latest_decision
                    preparation=[v['decision'] for v in tuple(observer.decisions)
                                 if v['decision']['decision_id']<=decision.decision_id]
                    last_preparation_id=decision.decision_id
                    observed=time.perf_counter()
                    if not ready_to_arm(preparation,decision,now=observed,first_publication=first_publication,
                        valid_groups=groups,publication_id=owner.publication_sequence,capture_id=last):
                        event['arm_deferred']='Readiness expired during native and camera admission'
                        continue
                    decision.validate(now=observed,**identity)
                    readiness=dict(prepared_since=first_publication,valid_camera_groups=groups,
                        observed=time.perf_counter(),last_five_predictions=[v['prediction'] for v in preparation[-5:]],
                        decision_id=decision.decision_id,publication_id=decision.publication_id,capture_id=decision.capture_id,
                        body_age_before_arm=time.perf_counter()-decision.body_first_receipt,
                        view_age_before_arm=time.perf_counter()-min(decision.view_render_lowers),
                        student_controls_before_arm=0,numerical_reset_at_arm=False)
                    dispatcher=MailboxTrial(actuator,limits['planned_student_seconds'])
                    dispatcher.start(prepared_since=first_publication,valid_camera_groups=groups)
                    trial_end=time.perf_counter()+limits['planned_student_seconds']
        result['status']='trial_finished_pending_audit'
    except BaseException as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        if dispatcher is not None:
            try:dispatcher.stop();result['dispatcher_error']=dispatcher.error
            except Exception as error:result['dispatcher_close_error']=str(error)
        # Request braking first; stop inference immediately instead of inventing a stale tail.
        # The independent owner proves stop and releases while actor cleanup runs.
        if actuator is not None:
            try:result['brake_stop_ack']=actuator.begin_stop()
            except Exception as error:result['brake_stop_rpc_error']=str(error)
        if observer is not None:
            try:observer.stop();result['preparation_observer_terminal']=dict(error=observer.error,outputs=len(observer.decisions),thread_alive=observer.thread.is_alive())
            except Exception as error:result['preparation_observer_error']=str(error)
        if owner is not None:
            try:result['owner_process_terminal']=owner.stop();result['owner_terminal']=read(out/'actor/owner-terminal.json')
            except Exception as error:result['owner_close_error']=str(error)
        if actuator is not None:
            try:result['actuator_process_terminal']=actuator.stop()
            except Exception as error:result['actuator_close_error']=str(error)
            if (out/'actuator/actuator-terminal.json').exists():
                terminal=read(out/'actuator/actuator-terminal.json');result['actuator_terminal']=terminal
                result['student_requests']=terminal['student_requests']
        if dispatcher is not None:
            write_json(out/'decision-deliveries.json',dispatcher.decisions)
            write_json(out/'decision-attempts.json',dispatcher.attempts)
        if camera is not None:
            try:result['camera_close']=camera.close()
            except Exception as error:result['camera_close_error']=str(error)
        if pause_worker is not None:
            try:
                camera_admission=None
                terminal_camera=(result.get('camera_close') or {}).get('producer_terminal') or {}
                echo=terminal_camera.get('camera_stop_policy_echo') or {}
                if (camera is not None and terminal_camera.get('status')=='stopped'
                        and echo.get('stop_signal_observed') is True
                        and terminal_camera.get('request_id')==echo.get('request_id')==camera.request_id
                        and echo.get('generation')==state['generation']):
                    camera_admission=dict(generation=state['generation'],epoch=state['epoch'],
                                          request_id=camera.request_id)
                pause=pause_worker.pause(camera=camera_admission);result['pause_action']=pause;result['pause_returncode']=pause['exitcode']
                if pause['exitcode']!=0:result['pause_error']=pause['result'].get('reason','Verified pause failed')
            except Exception as error:result['pause_error']=str(error)
            try:result['pause_process_terminal']=pause_worker.close()
            except Exception as error:result['pause_worker_close_error']=str(error)
        if mailbox is not None:
            mailbox.close();mailbox.unlink()
        if camera is not None:
            try:result['camera_archive']=camera.archive_to(out/'vision',bundle=True)
            except Exception as error:result['archive_error']=str(error)
        if readiness is not None:write_json(out/'student-readiness.json',readiness)
        if observer is not None:write_json(out/'preparation-observer-decisions.json',observer.decisions)
        write_json(out/'camera-events.json',events);write_json(out/'preparation-decisions.json',preparation)
        if visual_inputs:
            np.savez_compressed(out/'published-visual-inputs.npz',capture_ids=np.array([x[0] for x in visual_inputs]),scene=np.array([x[1] for x in visual_inputs]),static=np.array([x[2] for x in visual_inputs]),motion=np.array([x[3] for x in visual_inputs]),coarse=np.array([x[4] for x in visual_inputs]))
        result['artifact_sha256']={str(f.relative_to(out)).replace('\\','/'):sha256(f) for f in out.rglob('*') if f.is_file() and f.suffix in ('.json','.npz') and f.name!='result.json'}
        if (out/'actuator/actuator-feedback.json').exists():
            observed_feedback=read(out/'actuator/actuator-feedback.json')
            if observed_feedback:
                maximum_speed=max(abs(v['fields']['speedKmh']) for v in observed_feedback)
                displacement=max(v.get('displacement_m',0.) for v in observed_feedback)
                result.update(maximum_native_speed_kmh=maximum_speed,maximum_native_displacement_m=displacement,actual_student_motion_verified=maximum_speed>=limits['minimum_motion_speed_kmh'] and displacement>=limits['minimum_motion_displacement_m'])
            result['independent_actuator_stop_reason']=result.get('actuator_terminal',{}).get('stop_reason')
        if result['status']=='trial_finished_pending_audit':
            terminal=result.get('actuator_terminal',{});owner_result=result.get('owner_terminal',{})
            if (result['student_requests']>0 and terminal.get('stationary_feedback_verified')
                    and terminal.get('device_released') and result.get('pause_returncode')==0
                    and terminal.get('stop_reason')=='planned_observation_complete'
                    and not owner_result.get('ticker_error') and not result.get('dispatcher_error')
                    and not any(k.endswith('_error') and v for k,v in result.items())):
                feedback=read(out/'actuator/actuator-feedback.json')
                maximum_speed=max(abs(v['fields']['speedKmh']) for v in feedback)
                displacement=max(v.get('displacement_m',0.) for v in feedback)
                result.update(maximum_native_speed_kmh=maximum_speed,maximum_native_displacement_m=displacement,actual_student_motion_verified=maximum_speed>=limits['minimum_motion_speed_kmh'] and displacement>=limits['minimum_motion_displacement_m'])
                result['status']='completed_finite_student_connection_trial' if result['actual_student_motion_verified'] else 'completed_student_actuation_without_motion'
            else:result['status']='failed'
        gc_scope.__exit__(None,None,None);result['gc_scope']=gc_receipt
        result['termination_reason']=result.get('reason',result.get('actuator_terminal',{}).get('stop_reason','preflight failure'))
        write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('actuator_terminal','owner_terminal','camera_archive')}))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
