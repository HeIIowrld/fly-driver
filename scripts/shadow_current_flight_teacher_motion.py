"""Bounded external-teacher motion with read-only native flight inference.

The privileged teacher alone owns vJoy. The student receives video and body
sensation and never sends controls. This is a moving timing experiment, not a
student lap, limit-driving or physiological qualification.
"""
import argparse,base64,copy,hashlib,json,os,shutil,subprocess,sys,time,yaml
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_runtime_gc import defer_automatic_cyclic_gc
from flybrain_drive.visual_phase_probe import VisualPhaseProbe
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_clock import CSPClockBounds,initialize_clock
import ctypes,threading,uuid
from flybrain_drive.ac_csp_vision import CSPVisionRecorder,NoFreshCSPFrame
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.camera_body_alignment import source_render_witness
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector,VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_camera_group import freeze_camera_group,require_nonzero_camera_views
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_flight_body_ticker import NativeFlightBodyTicker
from flybrain_drive.current_flight_gc_owner import CurrentFlightGCOwnerProcess
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder,decode_native_extension
from flybrain_drive.native_process_rows import NativeProcessRowsSampler
from flybrain_drive.online_body_shadow import OnlineBodyShadow
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate
from flybrain_drive.per_view_flyvis_stream import PerViewFlyvisStream
from flybrain_drive.retinal_residual_port import RetinalResidualPort
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat
from scripts.record_native_process_unified_shadow_v2 import owned_resume
from scripts.sac_steering_experiment import assembly
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate,require_motion_stream_parity
from flybrain_drive.motion_per_view_flyvis_stream import MotionPerViewFlyvisStream
from flybrain_drive.teacher_shadow_camera import TeacherShadowCameraCoordinator,require_teacher_motion_state

PIN='b2227739fb11d6ccce2020e976b2e0b3626d41ecd7dd8defb42ea8105a0495c2'
FIT=Path('runs/thermal-driving-20261103/native-fit-02')
EXT=Path('runs/thermal-driving-20261103/extension-data-01')
SPATIAL=Path('runs/thermal-driving-20261103/spatial-recovery-01')
AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'


def previous_camera(root):
    request=read(root/'request.json') if (root/'request.json').exists() else None
    status=read(root/'status.json') if (root/'status.json').exists() else None
    terminal=None
    if request is not None:
        import re
        rid=request['id'];assert re.fullmatch('[A-Za-z0-9_-]{1,80}',rid)
        path=root/rid/'result.json';terminal=read(path) if path.exists() else None
    return request,status,terminal


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-lua-sha256',default=PIN);p.add_argument('--teacher-lateral-calibration');p.add_argument('--flush-body-denormal',action='store_true');p.add_argument('--defer-body-gc',action='store_true');p.add_argument('--profile-visual',action='store_true');p.add_argument('--defer-visual-gc',action='store_true');p.add_argument('--completed-metadata-evidence');p.add_argument('--candidate-plan',required=True);p.add_argument('--current-checkpoint',required=True);p.add_argument('--current-checkpoint-sha256',required=True);p.add_argument('--thermal-initialization-run',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit bounded background lifecycle execution required')
    pin=a.expected_lua_sha256
    if sha256(a.current_checkpoint)!=a.current_checkpoint_sha256:raise ValueError('Exact current actor required')
    if not a.defer_body_gc:raise ValueError('This entrypoint explicitly requires finite CPU GC deferral')
    # Reject obsolete thermal evidence before expensive GPU initialization.
    from scripts.audit_initial_thermal_evidence import audit as audit_thermal
    from flybrain_drive.thermal_teacher_start import admit_thermal_teacher_start
    from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
    preflight_heartbeat,mtime,_=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
    validate_heartbeat(preflight_heartbeat,time.time()-mtime,mutating=False)
    thermal_reader=NativeTeacherPhysicsReader()
    try:
        thermal_sample=repeated_teacher_snapshot(thermal_reader.mapping)
        admit_thermal_teacher_start(audit_thermal(a.thermal_initialization_run),preflight_heartbeat,thermal_sample.physics.fields['tyreCoreTemperature'],source_sha256=pin)
    finally:thermal_reader.close()
    torch.set_num_threads(1);sys.setswitchinterval(.001)
    proof=Path('runs/native-body-multirate-20261103/source-prefix-parity-01');pr=read(proof/'result.json');pm=read(proof/'manifest.json')['config']
    assert pr['status']=='completed_per_view_flyvis_source_parity' and pm['visual_stream_sha256']==sha256('src/flybrain_drive/per_view_flyvis_stream.py')
    fr=read(FIT/'result.json');assert fr['status']=='completed_thermal_native_body_port_fit' and sha256(FIT/'body-extension.pt')==fr['checkpoint_sha256']
    cp=torch.load(FIT/'body-extension.pt',map_location='cpu',weights_only=True);source=Path(cp['source_fit_directory']);assert sha256(source/'steering.pt')==cp['source_checkpoint_sha256']
    assert sha256(EXT/'contract.json')==cp['extension_contract_sha256']
    candidate_plan=read(a.candidate_plan)
    (model,head,dn,optical,body,encoder),motion_port,candidate_receipt=load_readonly_motion_candidate(candidate_plan)
    motion_proof=require_motion_stream_parity(candidate_plan,candidate_receipt,SPATIAL/'pool-geometry.npz')
    visual_class=MotionPerViewFlyvisStream if motion_port is not None else PerViewFlyvisStream
    if candidate_receipt['native_checkpoint_sha256']!=fr['checkpoint_sha256']:raise ValueError('Moving probe original native fit differs')
    warm=prepare_fixed_cpu_backend(model)
    cfg=read('configs/flight-stream-batched-v1-20261029.json');md=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root']);assert sha256(md/'best_chkpt')==cfg['author_weights_sha256']
    os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
    import flyvis
    camera_config=read('configs/ac-csp-exterior-front-v3-retina.json');pool=copy.deepcopy(model.scene_pool).to('cuda').eval().requires_grad_(False)
    with np.load(SPATIAL/'pool-geometry.npz') as z:geometry={k:z[k].copy() for k in ['old','inverse','grid8_narrow']}
    with torch.device('cuda'):
        nc=yaml.safe_load((md/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**nc).to('cuda')
        network.load_state_dict(torch.load(md/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
        nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors]
        indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{k}{j}' for k in ('T4','T5') for j in 'abcd'])
        projector=FlyvisCameraProjector(camera_config,u,v);prior=network.steady_state(1.,.005,batch_size=4,value=.5)
    # Warm discarded computations only. The actual stream starts from its
    # original author .5 steady prior, not from these synthetic pixels/states.
    warm_visual=visual_class(network,prior,pool,projector,indices,geometry,episode_id='backend-warmup',generation='warmup',device='cuda')
    for i in range(3):
        warm_visual.process(pixels={n:np.full((180,320,3),128,dtype=np.uint8) for n in VIEW_ORDER},render_start_lowers=tuple(1+i*.04+j*.001 for j in range(4)),camera_received=1+i*.04+.01,capture_id=i,generation='warmup')
        warm_visual.prepare_held_prefix();torch.cuda.synchronize()
    del warm_visual
    with torch.device('cuda'):prior=network.steady_state(1.,.005,batch_size=4,value=.5)
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain';status_path=local/'csp-session/status.json';camera_root=local/'csp-camera'
    status,modified,_=read_session_heartbeat(status_path);validate_heartbeat(status,time.time()-modified,mutating=True)
    previous=previous_camera(camera_root);recovery=None
    if a.completed_metadata_evidence:
        old=Path(a.completed_metadata_evidence);old_result=read(old/'result.json');old_config=read(old/'manifest.json')['config']
        assert old_result['camera_close']['producer_terminal']==previous[2]
        archive=CameraArchive(old/'vision')
        try:actual_bundle=archive.bundle_sha256
        finally:archive.close()
        def exited(pid):
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.restype=ctypes.c_void_p
            handle=kernel.OpenProcess(0x1000,False,pid)
            if not handle:
                if ctypes.get_last_error()==87:return True
                raise OSError('Cannot verify previous owner process state')
            try:
                code=ctypes.c_ulong();kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
                if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError('Cannot query previous owner exit')
                return code.value!=259
            finally:
                kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)
        evidence=dict(producer_terminal=old_result['camera_close']['producer_terminal'],expected_bundle_sha256=old_result['camera_archive']['bundle_sha256'],
            archive_bundle_sha256=actual_bundle,archive_verified=old_result['camera_archive']['all_hashes_verified'],
            prior_sources_closed=all(exited(pid) for pid in [old_result['owner_terminal']['pid'],old_result['owner_summary']['native_producer_pid']]),
            source_sha256=old_config['source_sha256'],expected_source_sha256=pin)
        bounds=CSPClockBounds(status['generation']);initialize_clock(camera_root,bounds,int(status['window_handle'].removesuffix('ULL')),host_clock=time.perf_counter)
        fresh,modified,_=read_session_heartbeat(status_path);validate_heartbeat(fresh,time.time()-modified,mutating=True)
        if previous_camera(camera_root)[0]!=previous[0] or previous_camera(camera_root)[2]!=previous[2]:raise ValueError('Original recovery request/terminal changed')
        camera_status=read(camera_root/'status.json')
        gate=PausedMMFStartupGate.completed_metadata_diagnostic(fresh,previous[0],camera_status,previous[2],bounds.record(),evidence,now=time.perf_counter())
        recovery=dict(admission=gate.diagnostic_admission,session=fresh,request=previous[0],camera_status=camera_status,terminal=previous[2],clock=bounds.record(),evidence=evidence,
            previous_result_sha256=sha256(old/'result.json'),previous_result_directory=str(old));status=fresh
    else:gate=PausedMMFStartupGate(status,*previous)
    api=WindowsACWindow();windows=api.discover();assert len(windows)==1;window=windows[0]
    assert api.user.GetForegroundWindow()!=window.hwnd and int(status['window_handle'].removesuffix('ULL'))==window.hwnd
    with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
    encoder=NativeMultisensoryExtensionEncoder(EXT/'contract.json',native_identity={k:identity[k] for k in read(EXT/'contract.json')['native_identity']})
    with ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size) as mapping:initial=repeated_teacher_snapshot(mapping)
    assert not initial.builtin_ai_controlled
    axis=read(AXIS);assert sha256(axis['profile_path'])==axis['profile_sha256'];sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256']);assert sdk.GetVJDStatus(axis['device_id'])==1
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),visual_stream_sha256=sha256('src/flybrain_drive/per_view_flyvis_stream.py'),phase_probe_sha256=sha256('src/flybrain_drive/visual_phase_probe.py'),camera_collector_sha256=sha256('src/flybrain_drive/ac_csp_vision.py'),ticker_sha256=sha256('src/flybrain_drive/native_flight_body_ticker.py'),worker_sha256=sha256('src/flybrain_drive/current_flight_gc_owner.py'),base_worker_sha256=sha256('src/flybrain_drive/current_flight_owner.py'),current_actor_checkpoint_sha256=a.current_checkpoint_sha256,current_actor_loader_sha256=sha256('src/flybrain_drive/current_flight_runtime.py'),fit_checkpoint_sha256=fr['checkpoint_sha256'],visual_parity_sha256=sha256(proof/'result.json'),source_sha256=pin,identity=identity,preflight_session=status,candidate=candidate_receipt,teacher_speed_targets_kmh=[10,20],teacher_seconds_per_stage=2,hard_speed_ceiling_kmh=40,seed=None,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0,inference_end_contract='stop_owner_before_teacher_wait_and_archives_v1',scope=__doc__))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');write_json(run/'backend-initialization.json',warm)
    if motion_proof is not None:write_json(run/'motion-preflight.json',motion_proof)
    if recovery is not None:write_json(run/'completed-metadata-diagnostic.json',recovery)
    camera=sampler=runtime=ticker=None;coordinator=teacher=teacher_log=None;events=[];motion_states=[];visual_inputs=[];native_samples={};native_position=0;result=dict(status='failed',actor_qualified=False,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0,motion_features_enabled=motion_port is not None)
    gc_scope=defer_automatic_cyclic_gc() if a.defer_visual_gc else None
    gc_receipt=gc_scope.__enter__() if gc_scope else None
    try:
        runtime=CurrentFlightGCOwnerProcess(checkpoint=a.current_checkpoint,checkpoint_sha256=a.current_checkpoint_sha256,directory=run,python_executable='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe',library_directory='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages',flush_denormal=a.flush_body_denormal)
        write_json(run/'worker-ready.json',runtime.ready)
        if not runtime.ready['receipt']['motion_features_required']:raise ValueError('Current actor requires motion features')
        start_gate=threading.Event();prepared=threading.Event()
        camera=CSPVisionRecorder(window.hwnd,camera_config,max_frames=240,period=.04,max_bytes=256*1024**2,image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,expected_source_sha256=pin,mmf_stop_signal=True,memory_only_receive_checks=True,start_gate=start_gate,prepared_event=prepared)
        preparation_deadline=time.perf_counter()+30
        while not prepared.wait(.01):
            if camera.error:raise RuntimeError(str(camera.error))
            if time.perf_counter()>preparation_deadline:raise TimeoutError('Shadow camera preparation timed out')
        coordination=Path(os.environ['LOCALAPPDATA'])/'flybrain/camera-coordination'/uuid.uuid4().hex
        coordination.mkdir(parents=True,exist_ok=False)
        coordinator=TeacherShadowCameraCoordinator(camera,start_gate,coordination,window_handle=window.hwnd,source_sha256=pin,recording=run,native_ready_required=True)
        write_json(run/'camera-ownership-plan.json',coordinator.plan)
        cpu='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe'
        command=[cpu,'-m','scripts.evaluate_sac_teacher_curriculum','--axis-config',AXIS,
            '--calibration-result','runs/thermal-driving-20261103/steering-calibration-02/result.json',
            '--teacher-config','assets/external/acgym-teacher-candidates-20261102/step_07000000.json',
            '--transport','tcp','--source-native','--require-quiescent-file-bridge','--shift-handling','policy',
            '--targets','10,20','--stage-seconds','2','--final-seconds','2','--full-laps','1',
            '--camera-plan',str(coordination/'ready.json'),'--thermal-initialization-run',a.thermal_initialization_run,
            '--execute','--output',str(run/'teacher')]
        if a.teacher_lateral_calibration:command+=['--low-speed-lateral-calibration',a.teacher_lateral_calibration]
        teacher_env=os.environ.copy();teacher_env['PYTHONPATH']='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages;src;.';teacher_env['OPENBLAS_NUM_THREADS']='1'
        teacher_log=(run/'teacher.log').open('x',encoding='utf8')
        teacher=subprocess.Popen(command,cwd=Path.cwd(),env=teacher_env,stdout=teacher_log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'ABOVE_NORMAL_PRIORITY_CLASS',0))
        write_json(run/'teacher-process.json',dict(pid=teacher.pid,command=command,student_actuator_access=False))
        live_deadline=time.perf_counter()+90
        while True:
            coordinator.check()
            if teacher.poll() is not None:raise RuntimeError('Teacher ended before owned live resume; inspect teacher result')
            current,modified,_=read_session_heartbeat(status_path);validate_heartbeat(current,time.time()-modified,mutating=False)
            if (coordination/'trigger').exists() and runtime.episode is None:
                if not current['paused'] or current['camera_active']:raise ValueError('Native initialization must precede owned camera trigger/control')
                body_begin_started=time.perf_counter()
                runtime.begin_live(camera.request_id,status['generation'],{k:identity[k] for k in read(EXT/'contract.json')['native_identity']})
                admission=dict(started=body_begin_started,finished=time.perf_counter(),episode_id=camera.request_id,
                    request_id=camera.request_id,generation=status['generation'],prepared_session=current,
                    prior_actor_steps=0,native_start_phase='after_teacher_preflight_before_camera_trigger')
                write_json(run/'body-start-admission.json',admission)
                write_json(coordination/'native-ready.json',admission)
            if not current['paused'] and current['live']:
                require_teacher_motion_state(current,generation=status['generation'],epoch=status['epoch'],request_id=camera.request_id)
                write_json(run/'live-admission.json',dict(observed_host_time=time.perf_counter(),session=current));break
            if time.perf_counter()>live_deadline:raise TimeoutError('Teacher owned resume did not complete within90s')
            time.sleep(.01)
        if runtime.episode is None:raise ValueError('Native source readiness was not established before live camera/control')
        camera.wait_ready();write_json(run/'resume-identity.json',dict(request_id=camera.request_id,generation=status['generation']))
        ticker=runtime
        visual=visual_class(network,prior,pool,projector,indices,geometry,episode_id=camera.request_id,generation=status['generation'],device='cuda')
        phase_probe=VisualPhaseProbe(visual,synchronize=torch.cuda.synchronize,event_factory=lambda:torch.cuda.Event(enable_timing=True)) if a.profile_visual else None
        shadow_budget=8 if a.teacher_lateral_calibration else 6
        result['read_only_shadow_budget_seconds']=shadow_budget
        deadline=time.perf_counter()+shadow_budget;last=-1;maximum_speed=0.
        while time.perf_counter()<deadline:
            if ticker.error:raise RuntimeError(str(ticker.error))
            current,modified,_=read_session_heartbeat(status_path);validate_heartbeat(current,time.time()-modified,mutating=False)
            require_teacher_motion_state(current,generation=status['generation'],epoch=status['epoch'],request_id=camera.request_id)
            coordinator.check();maximum_speed=max(maximum_speed,current['speed_kmh'])
            motion_states.append(dict(observed=time.perf_counter(),heartbeat_modified=modified,state=current))
            if teacher.poll() is not None:break
            try:frame,selected,_=camera.wait_for_candidate(max_age=.15,timeout=.04)
            except NoFreshCSPFrame:continue
            if frame['capture_id']==last:time.sleep(.001);continue
            last=frame['capture_id'];event=dict(capture_id=last,camera_received=frame['received'],render_start_lowers=[v['render_started_lower'] for v in frame['views']]);events.append(event)
            group=freeze_camera_group(frame,camera.samples_for(frame),selected_at=selected);require_nonzero_camera_views(group)
            bounds=tuple(event['render_start_lowers']);admission=[]
            try:
                clock=camera.clock.record()
                for view,lower in zip(frame['views'],bounds):
                    witness=view['source_render_metadata'];source_render_witness(witness)
                    admission.append(witness['render_started_clock']+clock['offset_upper_seconds']+clock.get('host_clock_quantization_seconds',0.))
                context_reply=runtime.context_for_camera(bounds=bounds,uppers=tuple(admission),camera_received=frame['received'],generation=status['generation'])
                if context_reply['status']!='context':raise ValueError(context_reply.get('reason','Owner context response invalid'))
                ctx=context_reply['context'];event.update(context_reply['evidence']);admission=context_reply['gaps']
            except (KeyError,ValueError) as error:
                event.update(status='unqualified_warmup_pixels_skipped',reason=str(error))
                if visual.previous is not None:raise ValueError('Source qualification lost inside visual episode: '+str(error))
                continue
            arguments=dict(pixels=group['pixels'],render_start_lowers=bounds,camera_received=frame['received'],capture_id=last,generation=status['generation'])
            started=time.perf_counter()
            try:
                if phase_probe:
                    event['visual_profile']={};obs=phase_probe.run('process',event['visual_profile'],**arguments)
                else:
                    obs=visual.process(**arguments);torch.cuda.synchronize()
            finally:event['visual_seconds']=time.perf_counter()-started
            if ctx is None or any(v is None for v in ctx.visual_contexts):raise ValueError('Qualified visual group lacks original body history')
            publication=ticker.publish(obs,ctx)
            event.update(status='scene_published',publication=publication,source_render_gap_seconds=admission)
            if motion_port is not None:
                event['motion_features_sha256']=obs.motion_features_sha256
                visual_inputs.append((last,obs.scene.copy(),obs.static.copy(),obs.motion_features.copy(),obs.coarse_motion_reference.copy()))
            prefix_started=time.perf_counter()
            try:
                if phase_probe:
                    event['prefix_profile']={};phase_probe.run('prepare_held_prefix',event['prefix_profile'])
                else:
                    visual.prepare_held_prefix();torch.cuda.synchronize()
            finally:event['prefix_seconds']=time.perf_counter()-prefix_started
        count=ticker.output_count
        if ticker.error:raise RuntimeError(str(ticker.error))
        if not count:raise ValueError('No admitted end-to-end shadow outputs')
        # End read-only inference before waiting for teacher cleanup or formatting
        # large archives. Continuing on the last view would invent a stale tail.
        end_started=time.perf_counter()
        result['owner_terminal']=runtime.stop()
        result['inference_end']=dict(started=end_started,finished=time.perf_counter(),
            reason='teacher_process_terminal' if teacher.poll() is not None else 'fixed_eight_second_shadow_budget' if a.teacher_lateral_calibration else 'fixed_six_second_shadow_budget',
            source='Original synchronous CPU owner stop before teacher wait/large archive writes',
            numerical_reset=False,student_vehicle_control_sent=False)
        teacher.wait(timeout=5)
        tr=read(run/'teacher/result.json')
        if teacher.returncode!=0 or tr['status']!='curriculum_completed' or tr['actuator_messages_sent']<=0 or maximum_speed<2:
            raise ValueError('Finite teacher motion/clean completion not established')
        write_json(run/'motion-states.json',motion_states)
        result.update(status='completed_current_actor_teacher_moving_shadow',shadow_outputs=count,teacher_result_sha256=sha256(run/'teacher/result.json'),teacher_actuator_messages_sent=tr['actuator_messages_sent'],maximum_heartbeat_speed_kmh=maximum_speed,termination_reason='Finite external teacher motion with student read-only source-qualified inference; no student control, limit driving, lap or physiological qualification')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error),runtime_rejection=None,termination_reason='Explicit live source/timing/authority failure; no fallback or student control')
    finally:
        if result['status']=='failed' and teacher is not None and teacher.poll() is None and (run/'teacher').exists():
            (run/'teacher/stop').touch(exist_ok=True)
        if runtime is not None and not runtime.stopped:
            end_started=time.perf_counter()
            try:
                result['owner_terminal']=runtime.stop()
                result['inference_end']=dict(started=end_started,finished=time.perf_counter(),reason='failed_or_interrupted_shadow',
                    source='Original synchronous CPU owner stop before archive writes',numerical_reset=False,student_vehicle_control_sent=False)
            except Exception as error:result['owner_close_error']=str(error);result['status']='failed'
        write_json(run/'motion-states.json',motion_states);result['motion_states_sha256']=sha256(run/'motion-states.json')
        if teacher is not None:
            if teacher.poll() is None:
                if (run/'teacher').exists():(run/'teacher/stop').touch(exist_ok=True)
                try:teacher.wait(timeout=15)
                except subprocess.TimeoutExpired:result['teacher_still_running']=teacher.pid;result['status']='failed'
            result['teacher_exitcode']=teacher.poll()
            if (run/'teacher/result.json').exists():
                tr=read(run/'teacher/result.json');result['teacher_status']=tr['status'];result['teacher_actuator_messages_sent']=tr['actuator_messages_sent'];result['teacher_result_sha256']=sha256(run/'teacher/result.json')
        if coordinator is not None:
            try:coordinator.close(failed=result['status']=='failed')
            except Exception as error:result['coordinator_close_error']=str(error);result['status']='failed'
        if teacher_log is not None:teacher_log.close()
        if runtime is not None:
            try:
                result['owner_terminal']=runtime.stop();result['ticker_stopped']=not runtime.process.is_alive()
                owner_summary=read(run/'owner-terminal.json');result['ticker_error']=owner_summary.get('ticker_error')
                if owner_summary['mode'] is None:write_json(run/'owner-native-copy.json',[])
                shutil.copyfile(run/'owner-native-copy.json',run/'native-copy.json');result['native_sha256']=sha256(run/'native-copy.json');result['native_sampler_stopped']=owner_summary['mode'] is None or (owner_summary.get('native_producer_exitcode')==0 and owner_summary.get('sampler_error') is None)
                result['single_authoritative_native_source']=True
                result['owner_summary']=owner_summary
                if (result['ticker_error'] or not result['native_sampler_stopped'] or runtime.terminal['poisoned']
                        or runtime.terminal['forced_termination'] or runtime.terminal['exitcode']!=0):result['status']='failed'
                shutil.copyfile(run/'owner-ticker-events.json',run/'ticker-events.json');result['ticker_events_sha256']=sha256(run/'ticker-events.json')
            except Exception as error:result['owner_close_error']=str(error);result['status']='failed'
        if camera is not None:
            try:result['camera_close']=camera.close()
            except Exception as error:result['camera_close_error']=str(error)
        if sampler is not None:
            try:sampler.stop();result['native_sampler_stopped']=True
            except Exception as error:result['native_sampler_close_error']=str(error)
            write_json(run/'native-copy.json',sampler.rows);result['native_sha256']=sha256(run/'native-copy.json')
        if camera is not None:
            try:
                pause_env=os.environ.copy();pause_env['PYTHONPATH']='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages;src;.'
                pause=subprocess.run(['C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe','-m','scripts.control_ac_session','--command','pause','--expected-lua-sha256',pin,'--execute','--output',str(run/'final-pause')],env=pause_env,capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));result['pause_returncode']=pause.returncode;write_json(run/'pause-process.json',dict(returncode=pause.returncode,stdout=pause.stdout,stderr=pause.stderr))
            except Exception as error:result['pause_error']=str(error)
        if camera is not None:
            try:result['camera_archive']=camera.archive_to(run/'vision',bundle=True)
            except Exception as error:result['archive_error']=str(error)
        result['shadow_outputs']=read(run/'owner-terminal.json').get('outputs',0) if (run/'owner-terminal.json').exists() else 0
        if result.get('pause_returncode')!=0 or result.get('native_sampler_stopped') is not True or result.get('teacher_exitcode')!=0 or any(key in result for key in ['camera_close_error','pause_error','archive_error','owner_close_error','coordinator_close_error','teacher_still_running']):
            result['status']='failed';result['cleanup_verified']=False
        else:result['cleanup_verified']=True
        if gc_scope is not None:
            gc_scope.__exit__(None,None,None);result['gc_scope']=gc_receipt
        if result['status']=='failed':result['termination_reason']='Explicit source/timing/authority or resource/end-boundary failure; original failures retained, no retry/reset/fallback or student actuation'
        if motion_port is not None and visual_inputs:
            np.savez_compressed(run/'published-visual-inputs.npz',capture_ids=np.asarray([x[0] for x in visual_inputs]),scene=np.asarray([x[1] for x in visual_inputs]),static=np.asarray([x[2] for x in visual_inputs]),motion=np.asarray([x[3] for x in visual_inputs]),coarse=np.asarray([x[4] for x in visual_inputs]))
            result['published_visual_inputs_sha256']=sha256(run/'published-visual-inputs.npz');result['published_motion_groups']=len(visual_inputs)
        write_json(run/'events.json',events);result['events_sha256']=sha256(run/'events.json');write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
