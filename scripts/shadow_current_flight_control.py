"""External-teacher shadow and optional calibrated moving student interval."""
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
from flybrain_drive.held_visual_prefix import HeldVisualPrefix
from flybrain_drive.immutable_camera_group import freeze_camera_group,require_nonzero_camera_views
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_flight_body_ticker import NativeFlightBodyTicker
from flybrain_drive.current_flight_control_contract import FEATURE_SELECTION_MAX_VIEW_AGE
from flybrain_drive.current_flight_control_owner import CurrentFlightControlOwnerProcess
from flybrain_drive.current_flight_dispatcher import CurrentFlightDecisionObserver
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

from scripts.preflight_current_flight_control import PIN
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actor-kind',choices=['parent','dual_guard','balanced_parent','balanced_dual_guard','compact_parent','compact_dual_guard'],required=True);p.add_argument('--expected-lua-sha256',default=PIN);p.add_argument('--teacher-lateral-calibration');p.add_argument('--flush-body-denormal',action='store_true');p.add_argument('--defer-body-gc',action='store_true');p.add_argument('--profile-visual',action='store_true');p.add_argument('--defer-visual-gc',action='store_true');p.add_argument('--completed-metadata-evidence');p.add_argument('--candidate-plan',required=True);p.add_argument('--current-checkpoint',required=True);p.add_argument('--current-checkpoint-sha256',required=True);p.add_argument('--thermal-initialization-run');p.add_argument('--magione-stage-audit');p.add_argument('--magione-geometry');p.add_argument('--axis-config',default=AXIS);p.add_argument('--calibration-result',default='runs/thermal-driving-20261103/steering-calibration-02/result.json');p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--moving-reference');p.add_argument('--moving-stop-audit');p.add_argument('--moving-camera-ingress',action='store_true');p.add_argument('--moving-camera-period',type=float,choices=(.06,.065,.07));p.add_argument('--moving-body-threads',type=int,choices=(1,2,4),default=1);p.add_argument('--moving-nonblocking-heartbeat',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit bounded background lifecycle execution required')
    moving=a.moving_reference is not None
    magione=bool(a.magione_stage_audit)
    if magione != bool(a.magione_geometry) or magione == bool(a.thermal_initialization_run):
        raise ValueError('One explicit thermal preparation or independently audited Magione stage required')
    if not magione and a.axis_config!=AXIS:
        raise ValueError('Explicit alternate axis evidence is limited to a staged Magione trial')
    if magione and (not moving or not a.teacher_lateral_calibration):
        raise ValueError('Staged Magione requires the moving authority and measured geometric steering')
    if moving!=bool(a.moving_stop_audit) or (moving and a.teacher_lateral_calibration and not magione):
        raise ValueError('Moving authority requires a matched stop audit and original pacing')
    if (a.moving_camera_ingress or a.moving_camera_period is not None or a.moving_body_threads!=1 or a.moving_nonblocking_heartbeat) and not moving:
        raise ValueError('Independent camera ingress requires a calibrated moving trial')
    capped=False
    if moving:
        from flybrain_drive.moving_flight_trial import admit_calibration
        reference,_=admit_calibration(a.moving_reference,a.moving_stop_audit)
        capped=reference.get('teacher_pacing_profile')=='monza_straight_120_v1'
        from flybrain_drive.magione_stage import MAGIONE_MOVING_STAGE_PROFILES
        if magione != (reference.get('teacher_pacing_profile') in MAGIONE_MOVING_STAGE_PROFILES):
            raise ValueError('Staging and verified corner reference track differ')
        if magione and (sha256(a.magione_geometry)!=reference['geometry_result_sha256']
                or a.expected_lua_sha256!=reference['installed_lua_sha256']):
            raise ValueError('Exact staged Magione route and installation required')
    frames,period=(1200,a.moving_camera_period or .06) if moving else (240,.04)
    speed_ceiling=40 if magione else 140 if capped else 300 if moving else 40
    pin=a.expected_lua_sha256
    if sha256(a.current_checkpoint)!=a.current_checkpoint_sha256:raise ValueError('Exact current actor required')
    if not a.defer_body_gc:raise ValueError('This entrypoint explicitly requires finite CPU GC deferral')
    # Reject obsolete thermal evidence before expensive GPU initialization.
    from scripts.audit_initial_thermal_evidence import audit as audit_thermal
    from flybrain_drive.thermal_teacher_start import admit_thermal_teacher_start
    from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
    preflight_heartbeat,mtime,_=read_session_heartbeat(Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
    validate_heartbeat(preflight_heartbeat,time.time()-mtime,mutating=False)
    stage = None
    if magione:
        from scripts.preflight_current_flight_control import inspect
        stage = inspect(expected_lua_sha256=pin,calibration_result=a.calibration_result,
            axis_config=a.axis_config,magione_stage_audit=a.magione_stage_audit)['magione_staged_start']
        if (stage['geometry_result_sha256']!=sha256(a.magione_geometry)
                or stage['staging_profile']!=MAGIONE_MOVING_STAGE_PROFILES[reference['teacher_pacing_profile']]):
            raise ValueError('Stationary staging and corner route differ')
    else:
        thermal_reader=NativeTeacherPhysicsReader()
        try:
            thermal_sample=repeated_teacher_snapshot(thermal_reader.mapping)
            admit_thermal_teacher_start(audit_thermal(a.thermal_initialization_run),preflight_heartbeat,
                thermal_sample.physics.fields['tyreCoreTemperature'],source_sha256=pin)
        finally:thermal_reader.close()
    early_root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera'
    PausedMMFStartupGate(preflight_heartbeat,*previous_camera(early_root))
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
    from flybrain_drive.current_flight_camera_admission import admit
    status,gate,recovery=admit(status,previous_camera(camera_root),camera_root=camera_root,status_path=status_path,pin=pin,completed_evidence=a.completed_metadata_evidence)
    api=WindowsACWindow();windows=api.discover();assert len(windows)==1;window=windows[0]
    assert api.user.GetForegroundWindow()!=window.hwnd and int(status['window_handle'].removesuffix('ULL'))==window.hwnd
    with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
    encoder=NativeMultisensoryExtensionEncoder(EXT/'contract.json',native_identity={k:identity[k] for k in read(EXT/'contract.json')['native_identity']})
    with ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size) as mapping:initial=repeated_teacher_snapshot(mapping)
    assert not initial.builtin_ai_controlled
    axis=read(a.axis_config);assert sha256(axis['profile_path'])==axis['profile_sha256'];sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256']);assert sdk.GetVJDStatus(axis['device_id'])==1
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),visual_stream_sha256=sha256('src/flybrain_drive/per_view_flyvis_stream.py'),feature_selection_max_view_age_seconds=FEATURE_SELECTION_MAX_VIEW_AGE,phase_probe_sha256=sha256('src/flybrain_drive/visual_phase_probe.py'),camera_collector_sha256=sha256('src/flybrain_drive/ac_csp_vision.py'),ticker_sha256=sha256('src/flybrain_drive/native_flight_body_ticker.py'),worker_sha256=sha256('src/flybrain_drive/current_flight_control_owner.py'),base_worker_sha256=sha256('src/flybrain_drive/current_flight_owner.py'),current_actor_checkpoint_sha256=a.current_checkpoint_sha256,current_actor_loader_sha256=sha256('src/flybrain_drive/current_flight_runtime.py'),fit_checkpoint_sha256=fr['checkpoint_sha256'],visual_parity_sha256=sha256(proof/'result.json'),source_sha256=pin,identity=identity,preflight_session=status,candidate=candidate_receipt,teacher_speed_targets_kmh=[20,25,30] if magione else [20,40,60,80,119] if capped else [20,40,60,80,120,160,220,280] if moving else [10,20],teacher_seconds_per_stage=3 if moving else 2,hard_speed_ceiling_kmh=speed_ceiling,seed=None,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0,inference_end_contract='stop_owner_before_teacher_wait_and_archives_v1',scope=__doc__))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');write_json(run/'backend-initialization.json',warm)
    if stage is not None:write_json(run/'magione-stage-admission.json',stage)
    if motion_proof is not None:write_json(run/'motion-preflight.json',motion_proof)
    if recovery is not None:write_json(run/'completed-metadata-diagnostic.json',recovery)
    camera=sampler=runtime=ticker=observer=ingress=None;coordinator=teacher=teacher_log=mailbox=pause_worker=None;events=[];deliveries=[];motion_states=[];heartbeat_reads=[];visual_inputs=[];native_samples={};native_position=0;result=dict(status='failed',actor_qualified=False,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0,motion_features_enabled=motion_port is not None)
    gc_scope=defer_automatic_cyclic_gc() if a.defer_visual_gc else None
    gc_receipt=gc_scope.__enter__() if gc_scope else None
    try:
        cpu='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe'
        library='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages'
        if moving:
            from flybrain_drive.flight_decision_mailbox import FlightDecisionMailbox
            from flybrain_drive.owned_pause_process import OwnedPauseProcess
            mailbox=FlightDecisionMailbox()
            pause_worker=OwnedPauseProcess(run/'final-pause',read(run/'manifest.json'),pin,cpu,library,heldout_magione=magione)
            write_json(run/'pause-worker-ready.json',pause_worker.ready)
        runtime=CurrentFlightControlOwnerProcess(actor_kind=a.actor_kind,run_id=run.resolve().as_posix(),checkpoint=a.current_checkpoint,checkpoint_sha256=a.current_checkpoint_sha256,directory=run,python_executable='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe',library_directory='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages',flush_denormal=a.flush_body_denormal,mailbox_spec=mailbox.spec if moving else None,native_sample_budget=45000 if moving else 20000,body_threads=a.moving_body_threads)
        write_json(run/'worker-ready.json',runtime.ready)
        if not runtime.ready['receipt']['motion_features_required']:raise ValueError('Current actor requires motion features')
        start_gate=threading.Event();prepared=threading.Event()
        camera=CSPVisionRecorder(window.hwnd,camera_config,max_frames=frames,period=period,max_bytes=(1024 if moving else 256)*1024**2,image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,expected_source_sha256=pin,mmf_stop_signal=True,memory_only_receive_checks=True,start_gate=start_gate,prepared_event=prepared,
            expected_track='magione' if magione else 'monza')
        preparation_deadline=time.perf_counter()+30
        while not prepared.wait(.01):
            if camera.error:raise RuntimeError(str(camera.error))
            if time.perf_counter()>preparation_deadline:raise TimeoutError('Shadow camera preparation timed out')
        coordination=Path(os.environ['LOCALAPPDATA'])/'flybrain/camera-coordination'/uuid.uuid4().hex
        coordination.mkdir(parents=True,exist_ok=False)
        coordinator=TeacherShadowCameraCoordinator(camera,start_gate,coordination,window_handle=window.hwnd,source_sha256=pin,recording=run,native_ready_required=True,frames=frames,period=period)
        write_json(run/'camera-ownership-plan.json',coordinator.plan)
        terminal_path=coordination/'student-terminal.json'
        command=[cpu,'-m','scripts.evaluate_sac_teacher_curriculum','--axis-config',a.axis_config,
            '--calibration-result',a.calibration_result,
            '--teacher-config','assets/external/acgym-teacher-candidates-20261102/step_07000000.json',
            '--transport','tcp','--source-native','--require-quiescent-file-bridge','--shift-handling','policy',
            '--full-laps','1',
            '--camera-plan',str(coordination/'ready.json'),
            '--execute','--output',str(run/'teacher')]
        if magione:
            command+=['--heldout-magione-geometry',a.magione_geometry,'--magione-stage-audit',a.magione_stage_audit,
                      '--heldout-geometric-speed']
        else:
            command+=['--thermal-initialization-run',a.thermal_initialization_run]
        if moving:
            student_plan=dict(schema='flybrain.moving_student_trial.v1',mailbox=mailbox.spec,
                identity=dict(run_id=run.resolve().as_posix(),episode_id=camera.request_id,
                              generation=status['generation'],checkpoint_sha256=a.current_checkpoint_sha256),
                terminal_path=str(terminal_path),reference_sha256=sha256(a.moving_reference),
                audit_sha256=sha256(a.moving_stop_audit))
            plan_path=coordination/'student-plan.json'
            write_json(plan_path,student_plan)
            write_json(run/'moving-student-plan.json',student_plan)
            command+=['--max-gap-m','3' if magione else '12','--stage-seconds','3',
                      '--moving-stop-reference',a.moving_reference,'--moving-stop-audit',a.moving_stop_audit,
                      '--moving-student-plan',str(plan_path)]
            # Use the existing teacher pacing cap with a 1km/h reserve below
            # the measured calibration profile. No student output is changed.
            if magione:command+=['--targets','20,25,30','--final-seconds','55']
            elif capped:command+=['--targets','20,40,60,80,119','--final-seconds','55']
            else:command.append('--unlimited-final-stage')
        else:
            command+=['--targets','10,20','--stage-seconds','2','--final-seconds','2']
        if a.teacher_lateral_calibration:command+=['--low-speed-lateral-calibration',a.teacher_lateral_calibration]
        teacher_env=os.environ.copy();teacher_env['PYTHONPATH']='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages;src;.';teacher_env['OPENBLAS_NUM_THREADS']='1'
        teacher_log=(run/'teacher.log').open('x',encoding='utf8')
        teacher=subprocess.Popen(command,cwd=Path.cwd(),env=teacher_env,stdout=teacher_log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'ABOVE_NORMAL_PRIORITY_CLASS',0))
        write_json(run/'teacher-process.json',dict(pid=teacher.pid,command=command,student_actuator_access=moving))
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
                require_teacher_motion_state(current,generation=status['generation'],epoch=status['epoch'],request_id=camera.request_id,maximum_speed_kmh=speed_ceiling)
                write_json(run/'live-admission.json',dict(observed_host_time=time.perf_counter(),session=current));break
            if time.perf_counter()>live_deadline:raise TimeoutError('Teacher owned resume did not complete within90s')
            time.sleep(.01)
        if runtime.episode is None:raise ValueError('Native source readiness was not established before live camera/control')
        camera.wait_ready();write_json(run/'resume-identity.json',dict(request_id=camera.request_id,generation=status['generation']))
        ticker=runtime
        if a.moving_camera_ingress:
            from flybrain_drive.flight_camera_ingress import FlightCameraIngress
            ingress=FlightCameraIngress(camera)
        if not moving:
            observer=CurrentFlightDecisionObserver(runtime)
            observer.start()
        visual=visual_class(network,prior,pool,projector,indices,geometry,episode_id=camera.request_id,generation=status['generation'],device='cuda')
        phase_probe=VisualPhaseProbe(visual,synchronize=torch.cuda.synchronize,event_factory=lambda:torch.cuda.Event(enable_timing=True)) if a.profile_visual else None
        shadow_budget=69 if moving else 8 if a.teacher_lateral_calibration else 6
        result['visual_processing_budget_seconds' if moving else 'read_only_shadow_budget_seconds']=shadow_budget
        deadline=time.perf_counter()+shadow_budget;last=-1;maximum_speed=0.
        if observer is not None:deliveries=observer.decisions
        next_owner_check=0.
        next_heartbeat_read=0.
        while time.perf_counter()<deadline:
            if moving and terminal_path.exists():
                result['moving_interval_terminal']=read(terminal_path)
                break
            if observer is not None and observer.error:raise RuntimeError(str(observer.error))
            if not moving or time.perf_counter()>=next_owner_check:
                owner_error=ticker.error
                if owner_error:raise RuntimeError(str(owner_error))
                next_owner_check=time.perf_counter()+.2
            if a.moving_nonblocking_heartbeat:
                if time.perf_counter()>=next_heartbeat_read:
                    started=time.perf_counter()
                    try:
                        current,modified,retries=read_session_heartbeat(status_path,maximum_wait=.005)
                    except (FileNotFoundError,PermissionError) as error:
                        if isinstance(error,PermissionError) and getattr(error,'winerror',None) not in (32,33) and error.errno!=13:
                            raise
                        # The teacher independently checks native/session authority
                        # before each SDK request. Keep only the same original
                        # already-read heartbeat, with its unmodified mtime.
                        heartbeat_reads.append(dict(started=started,finished=time.perf_counter(),read='transient',original_mtime=modified))
                        next_heartbeat_read=started+.01
                    else:
                        heartbeat_reads.append(dict(started=started,finished=time.perf_counter(),read='current',original_mtime=modified,retries=retries))
                        next_heartbeat_read=started+.1
            else:
                current,modified,_=read_session_heartbeat(status_path)
            validate_heartbeat(current,time.time()-modified,mutating=False)
            require_teacher_motion_state(current,generation=status['generation'],epoch=status['epoch'],request_id=camera.request_id,maximum_speed_kmh=speed_ceiling)
            coordinator.check();maximum_speed=max(maximum_speed,current['speed_kmh'])
            motion_states.append(dict(observed=time.perf_counter(),heartbeat_modified=modified,state=current))
            if teacher.poll() is not None:break
            try:
                if ingress is not None:
                    frame,group=ingress.receive()
                    selected=group['selected_at']
                else:
                    frame,selected,_=camera.wait_for_candidate(max_age=FEATURE_SELECTION_MAX_VIEW_AGE,timeout=.04)
                    if frame['capture_id']==last:time.sleep(.001);continue
                    group=freeze_camera_group(frame,camera.samples_for(frame),selected_at=selected)
                    require_nonzero_camera_views(group)
            except NoFreshCSPFrame:continue
            last=frame['capture_id'];event=dict(capture_id=last,camera_received=frame['received'],
                selected_at=selected,dispatched_at=time.perf_counter(),
                render_start_lowers=[v['render_started_lower'] for v in frame['views']]);events.append(event)
            bounds=tuple(event['render_start_lowers']);admission=[]
            try:
                clock=camera.clock.record()
                for view,lower in zip(frame['views'],bounds):
                    witness=view['source_render_metadata'];source_render_witness(witness)
                    admission.append(witness['render_started_clock']+clock['offset_upper_seconds']+clock.get('host_clock_quantization_seconds',0.))
                context_reply=runtime.context_for_camera(bounds=bounds,uppers=tuple(admission),camera_received=frame['received'],generation=status['generation'],capture_id=last)
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
            publication=ticker.publish(obs,context_reply['context_id'])
            event.update(status='scene_published',publication=publication,source_render_gap_seconds=admission)
            if motion_port is not None:
                event['motion_features_sha256']=obs.motion_features_sha256
                visual_inputs.append((last,obs.scene.copy(),obs.static.copy(),obs.motion_features.copy(),obs.coarse_motion_reference.copy()))
            prefix_started=time.perf_counter()
            try:
                if phase_probe:
                    event['prefix_profile']={};phase_probe.run('prepare_held_prefix',event['prefix_profile'])
                elif moving:
                    # Keep the 5ms integration and exact commit. Recompute only the
                    # unprepared tail of the actual interval inside prefix.commit().
                    # Subsequent prefix consumption and GPU transfers are on
                    # this same stream. Let the host admit the next real group
                    # while speculative CUDA work is still queued.
                    visual.prefix=HeldVisualPrefix(visual.network,visual.state,visual.held,steps=11)
                else:
                    visual.prepare_held_prefix();torch.cuda.synchronize()
            finally:
                event['prefix_seconds']=time.perf_counter()-prefix_started
                event['prefix_device_synchronized']=bool(phase_probe or not moving)

        if ingress is not None:ingress.close()
        if observer is not None:
            observer.stop()
            if observer.error:raise RuntimeError(str(observer.error))
            write_json(run/'decision-deliveries.json',deliveries)
            if not deliveries:raise ValueError('No source-bound decisions delivered')
        count=ticker.output_count
        if ticker.error:raise RuntimeError(str(ticker.error))
        if not count:raise ValueError('No admitted end-to-end shadow outputs')
        # End read-only inference before waiting for teacher cleanup or formatting
        # large archives. Continuing on the last view would invent a stale tail.
        end_started=time.perf_counter()
        result['owner_terminal']=runtime.stop()
        result['inference_end']=dict(started=end_started,finished=time.perf_counter(),
            reason='moving_interval_terminal' if moving else 'teacher_process_terminal' if teacher.poll() is not None else 'fixed_eight_second_shadow_budget' if a.teacher_lateral_calibration else 'fixed_six_second_shadow_budget',
            source='Original synchronous CPU owner stop before teacher wait/large archive writes',
            numerical_reset=False,student_vehicle_control_sent=None if moving else False)
        if moving and not terminal_path.exists():raise ValueError('No owned moving interval terminal')
        teacher.wait(timeout=25 if moving else 5)
        tr=read(run/'teacher/result.json')
        expected='completed_moving_student_trial' if moving else 'curriculum_completed'
        if teacher.returncode!=0 or tr['status']!=expected or tr['actuator_messages_sent']<=0 or maximum_speed<2:
            raise ValueError('Finite teacher motion/clean completion not established')
        write_json(run/'motion-states.json',motion_states)
        result.update(status=expected if moving else 'completed_source_bound_actor_teacher_moving_shadow',shadow_outputs=count,teacher_result_sha256=sha256(run/'teacher/result.json'),teacher_actuator_messages_sent=tr['actuator_messages_sent'],maximum_heartbeat_speed_kmh=maximum_speed,termination_reason='Calibrated moving interval ended' if moving else 'Finite external teacher motion with read-only student inference')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error),runtime_rejection=None,termination_reason='Explicit live source, timing or authority failure')
    finally:
        if moving and result['status']=='failed' and mailbox is not None:mailbox.request_stop()
        if ingress is not None:
            try:ingress.close()
            except Exception as error:result['camera_ingress_close_error']=str(error)
        if observer is not None:
            try:observer.stop();deliveries=observer.decisions
            except Exception as error:result['observer_close_error']=str(error);result['status']='failed'
        if result['status']=='failed' and teacher is not None and teacher.poll() is None and (run/'teacher').exists():
            (run/'teacher/stop').touch(exist_ok=True)
        if runtime is not None and not runtime.stopped:
            end_started=time.perf_counter()
            try:
                result['owner_terminal']=runtime.stop()
                result['inference_end']=dict(started=end_started,finished=time.perf_counter(),reason='failed_or_interrupted_shadow',
                    source='Original synchronous CPU owner stop before archive writes',numerical_reset=False,student_vehicle_control_sent=None if moving else False)
            except Exception as error:result['owner_close_error']=str(error);result['status']='failed'
        write_json(run/'decision-deliveries.json',deliveries)
        if a.moving_nonblocking_heartbeat:
            write_json(run/'heartbeat-read-timing.json',heartbeat_reads)
            result['heartbeat_read_timing_sha256']=sha256(run/'heartbeat-read-timing.json')
        write_json(run/'motion-states.json',motion_states);result['motion_states_sha256']=sha256(run/'motion-states.json')
        if teacher is not None:
            if teacher.poll() is None:
                if (run/'teacher').exists():(run/'teacher/stop').touch(exist_ok=True)
                try:teacher.wait(timeout=15)
                except subprocess.TimeoutExpired:result['teacher_still_running']=teacher.pid;result['status']='failed'
            result['teacher_exitcode']=teacher.poll()
            if (run/'teacher/result.json').exists():
                tr=read(run/'teacher/result.json');result['moving_student_trial']=tr.get('moving_student_trial') if moving else None;result['student_requests']=tr.get('student_requests',0);result['teacher_status']=tr['status'];result['teacher_actuator_messages_sent']=tr['actuator_messages_sent'];result['teacher_result_sha256']=sha256(run/'teacher/result.json')
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
        if pause_worker is not None:
            try:
                if teacher is not None and teacher.poll() is None:
                    raise RuntimeError('Pacing owner still active at pause boundary')
                terminal_camera=(result.get('camera_close') or {}).get('producer_terminal') or {}
                echo=terminal_camera.get('camera_stop_policy_echo') or {}
                camera_admission=None
                if (camera is not None and terminal_camera.get('status')=='stopped'
                        and echo.get('stop_signal_observed') is True
                        and terminal_camera.get('request_id')==echo.get('request_id')==camera.request_id
                        and echo.get('generation')==status['generation']):
                    camera_admission=dict(generation=status['generation'],epoch=status['epoch'],
                                          request_id=camera.request_id)
                already_paused = False
                if magione and not (run/'live-admission.json').exists():
                    final_session,final_mtime,_=read_session_heartbeat(status_path)
                    validate_heartbeat(final_session,time.time()-final_mtime,mutating=True)
                    already_paused = final_session['paused'] is True
                if already_paused:
                    from scripts.preflight_current_flight_control import inspect
                    final=inspect(expected_lua_sha256=pin,calibration_result=a.calibration_result,
                                  axis_config=a.axis_config)
                    verified=final['session']
                    if (verified['generation']!=status['generation'] or verified['epoch']!=status['epoch']
                            or not verified['paused'] or verified['camera_active']):
                        raise ValueError('Original staged pause did not remain intact')
                    write_json(run/'already-paused-native-verification.json',final)
                    result['pause_returncode']=0
                    result['pause_action']=dict(status='already_paused_verified',lifecycle_requests_sent=0)
                else:
                    pause=pause_worker.pause(camera=camera_admission)
                    result['pause_action']=pause
                    result['pause_returncode']=pause['exitcode']
            except Exception as error:result['pause_error']=str(error)
            try:result['pause_process_terminal']=pause_worker.close()
            except Exception as error:result['pause_worker_close_error']=str(error)
        elif camera is not None:
            try:
                pause_env=os.environ.copy();pause_env['PYTHONPATH']='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages;src;.'
                pause=subprocess.run(['C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe','-m','scripts.control_ac_session','--command','pause','--expected-lua-sha256',pin,'--execute','--output',str(run/'final-pause')],env=pause_env,capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));result['pause_returncode']=pause.returncode;write_json(run/'pause-process.json',dict(returncode=pause.returncode,stdout=pause.stdout,stderr=pause.stderr))
            except Exception as error:result['pause_error']=str(error)
        if camera is not None:
            try:result['camera_archive']=camera.archive_to(run/'vision',bundle=True)
            except Exception as error:result['archive_error']=str(error)
        if mailbox is not None:
            mailbox.close()
            mailbox.unlink()
        if moving:
            interval=result.get('moving_student_trial') or {}
            result['student_vehicle_control_sent']=bool(result.get('student_requests',interval.get('student_requests')))
        result['shadow_outputs']=read(run/'owner-terminal.json').get('outputs',0) if (run/'owner-terminal.json').exists() else 0
        if result.get('pause_returncode')!=0 or result.get('native_sampler_stopped') is not True or result.get('teacher_exitcode')!=0 or any(key in result for key in ['camera_close_error','pause_error','archive_error','owner_close_error','coordinator_close_error','teacher_still_running','pause_worker_close_error','camera_ingress_close_error']):
            result['status']='failed';result['cleanup_verified']=False
        else:result['cleanup_verified']=True
        if gc_scope is not None:
            gc_scope.__exit__(None,None,None);result['gc_scope']=gc_receipt
        if result['status']=='failed':result['termination_reason']='Explicit source, authority, or cleanup failure; original evidence retained'
        if motion_port is not None and visual_inputs:
            np.savez_compressed(run/'published-visual-inputs.npz',capture_ids=np.asarray([x[0] for x in visual_inputs]),scene=np.asarray([x[1] for x in visual_inputs]),static=np.asarray([x[2] for x in visual_inputs]),motion=np.asarray([x[3] for x in visual_inputs]),coarse=np.asarray([x[4] for x in visual_inputs]))
            result['published_visual_inputs_sha256']=sha256(run/'published-visual-inputs.npz');result['published_motion_groups']=len(visual_inputs)
        write_json(run/'events.json',events);result['events_sha256']=sha256(run/'events.json');write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
