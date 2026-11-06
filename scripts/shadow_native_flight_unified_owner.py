"""Finite one-native-source isolated body-owner/source-prefix camera/native flight shadow; no controls."""
import argparse,base64,copy,hashlib,json,os,shutil,subprocess,sys,time,yaml
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_clock import CSPClockBounds,initialize_clock
import ctypes
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
from flybrain_drive.native_flight_body_owner_process import NativeFlightBodyOwnerProcess
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--completed-metadata-evidence');p.add_argument('--seconds',type=int,choices=[3,5,8],default=3);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit bounded background lifecycle execution required')
    torch.set_num_threads(1);sys.setswitchinterval(.001)
    proof=Path('runs/native-body-multirate-20261103/source-prefix-parity-01');pr=read(proof/'result.json');pm=read(proof/'manifest.json')['config']
    assert pr['status']=='completed_per_view_flyvis_source_parity' and pm['visual_stream_sha256']==sha256('src/flybrain_drive/per_view_flyvis_stream.py')
    fr=read(FIT/'result.json');assert fr['status']=='completed_thermal_native_body_port_fit' and sha256(FIT/'body-extension.pt')==fr['checkpoint_sha256']
    cp=torch.load(FIT/'body-extension.pt',map_location='cpu',weights_only=True);source=Path(cp['source_fit_directory']);assert sha256(source/'steering.pt')==cp['source_checkpoint_sha256']
    assert sha256(EXT/'contract.json')==cp['extension_contract_sha256']
    model,head,dn,_,_=assembly(source);pc=torch.load(source/'steering.pt',map_location='cpu',weights_only=True);s=pc['optical']
    optical=RetinalResidualPort(s['reference'],s['scale'],len(model.motor.backbone.vision_ports));optical.load_state_dict(s)
    s=cp['body_port'];body=MultisensoryBodyFeedbackPort(s['reference'],s['scale'],len(model.motor.backbone.body_ports));body.load_state_dict(s)
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
    warm_visual=PerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id='backend-warmup',generation='warmup',device='cuda')
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
            source_sha256=old_config['source_sha256'],expected_source_sha256=PIN)
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
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),visual_stream_sha256=sha256('src/flybrain_drive/per_view_flyvis_stream.py'),camera_collector_sha256=sha256('src/flybrain_drive/ac_csp_vision.py'),ticker_sha256=sha256('src/flybrain_drive/native_flight_body_ticker.py'),worker_sha256=sha256('src/flybrain_drive/native_flight_body_owner_process.py'),fit_checkpoint_sha256=fr['checkpoint_sha256'],visual_parity_sha256=sha256(proof/'result.json'),source_sha256=PIN,identity=identity,preflight_session=status,seed=None,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0,scope=__doc__))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');write_json(run/'backend-initialization.json',warm)
    if recovery is not None:write_json(run/'completed-metadata-diagnostic.json',recovery)
    camera=sampler=runtime=ticker=None;events=[];native_samples={};native_position=0;result=dict(status='failed',actor_qualified=False,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0)
    try:
        runtime=NativeFlightBodyOwnerProcess(model,head,dn,optical,body,encoder,native_identity={k:identity[k] for k in read(EXT/'contract.json')['native_identity']},directory=run,python_executable='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe',library_directory='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages')
        write_json(run/'worker-ready.json',runtime.ready)
        camera=CSPVisionRecorder(window.hwnd,camera_config,max_frames=240,period=.04,max_bytes=256*1024**2,image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,expected_source_sha256=PIN,mmf_stop_signal=True,memory_only_receive_checks=True)
        runtime.begin_episode(camera.request_id,status['generation'])
        camera.wait_ready();write_json(run/'resume-identity.json',dict(request_id=camera.request_id,generation=status['generation']))
        owned_resume(run,gate,camera,status_path,camera_root)
        ticker=runtime
        visual=PerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id=camera.request_id,generation=status['generation'],device='cuda')
        deadline=time.perf_counter()+a.seconds;last=-1
        while time.perf_counter()<deadline:
            if ticker.error:raise RuntimeError(str(ticker.error))
            current,modified,_=read_session_heartbeat(status_path);validate_heartbeat(current,time.time()-modified,mutating=False)
            if current['generation']!=status['generation'] or current['paused'] or not current['live'] or current['ai_controlled'] or abs(current['speed_kmh'])>=.1 or current['gas']>=.02 or current['brake']<=.98:raise ValueError('Stationary live AI-off full-brake state changed')
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
            started=time.perf_counter();obs=visual.process(pixels=group['pixels'],render_start_lowers=bounds,camera_received=frame['received'],capture_id=last,generation=status['generation']);torch.cuda.synchronize();event['visual_seconds']=time.perf_counter()-started
            if ctx is None or any(v is None for v in ctx.visual_contexts):raise ValueError('Qualified visual group lacks original body history')
            publication=ticker.publish(obs,ctx)
            event.update(status='scene_published',publication=publication,source_render_gap_seconds=admission)
            prefix_started=time.perf_counter();visual.prepare_held_prefix();torch.cuda.synchronize();event['prefix_seconds']=time.perf_counter()-prefix_started
        count=ticker.output_count
        if ticker.error:raise RuntimeError(str(ticker.error))
        if not count:raise ValueError('No admitted end-to-end shadow outputs')
        result.update(status='completed_stationary_native_flight_shadow',shadow_outputs=count,termination_reason='Finite stationary source-qualified per-view author vision plus current native flight inference, read-only; no moving/closed-loop/physiological qualification')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error),runtime_rejection=None,termination_reason='Explicit live source/timing/authority failure; no fallback or student control')
    finally:
        if runtime is not None:
            try:
                result['owner_terminal']=runtime.stop();result['ticker_stopped']=not runtime.process.is_alive()
                owner_summary=read(run/'owner-terminal.json');result['ticker_error']=owner_summary.get('ticker_error')
                shutil.copyfile(run/'owner-native-copy.json',run/'native-copy.json');result['native_sha256']=sha256(run/'native-copy.json');result['native_sampler_stopped']=owner_summary.get('sampler_stopped') is True
                result['single_authoritative_native_source']=True
                result['owner_summary']=owner_summary
                if (result['ticker_error'] or not owner_summary.get('sampler_stopped') or runtime.terminal['poisoned']
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
                pause=subprocess.run([sys.executable,'-m','scripts.control_ac_session','--command','pause','--expected-lua-sha256',PIN,'--execute','--output',str(run/'final-pause')],capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));result['pause_returncode']=pause.returncode;write_json(run/'pause-process.json',dict(returncode=pause.returncode,stdout=pause.stdout,stderr=pause.stderr))
            except Exception as error:result['pause_error']=str(error)
        if camera is not None:
            try:result['camera_archive']=camera.archive_to(run/'vision',bundle=True)
            except Exception as error:result['archive_error']=str(error)
        result['shadow_outputs']=sum(e.get('status')=='shadow_output' for e in runtime.events) if runtime is not None else 0
        if result.get('pause_returncode')!=0 or result.get('native_sampler_stopped') is not True or any(key in result for key in ['camera_close_error','pause_error','archive_error','owner_close_error']):
            result['status']='failed';result['cleanup_verified']=False
        else:result['cleanup_verified']=True
        write_json(run/'events.json',events);result['events_sha256']=sha256(run/'events.json');write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
