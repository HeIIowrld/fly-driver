"""Finite background isolated-process multirate camera/native flight shadow; no controls."""
import argparse,base64,copy,hashlib,json,os,shutil,subprocess,sys,time,yaml
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_capture import WindowsACWindow
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
from flybrain_drive.native_flight_process_runtime import NativeFlightProcessRuntime
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise ValueError('Explicit bounded background lifecycle execution required')
    torch.set_num_threads(1);sys.setswitchinterval(.001)
    proof=Path('runs/per-view-visual-runtime-20261103/source-parity-01');pr=read(proof/'result.json');pm=read(proof/'manifest.json')['config']
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
    del warm_visual
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain';status_path=local/'csp-session/status.json';camera_root=local/'csp-camera'
    status,modified,_=read_session_heartbeat(status_path);validate_heartbeat(status,time.time()-modified,mutating=True)
    gate=PausedMMFStartupGate(status,*previous_camera(camera_root))
    api=WindowsACWindow();windows=api.discover();assert len(windows)==1;window=windows[0]
    assert api.user.GetForegroundWindow()!=window.hwnd and int(status['window_handle'].removesuffix('ULL'))==window.hwnd
    with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
    encoder=NativeMultisensoryExtensionEncoder(EXT/'contract.json',native_identity={k:identity[k] for k in read(EXT/'contract.json')['native_identity']})
    with ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size) as mapping:initial=repeated_teacher_snapshot(mapping)
    assert not initial.builtin_ai_controlled
    axis=read(AXIS);assert sha256(axis['profile_path'])==axis['profile_sha256'];sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256']);assert sdk.GetVJDStatus(axis['device_id'])==1
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),runtime_sha256=sha256('src/flybrain_drive/native_flight_decision_runtime.py'),visual_stream_sha256=sha256('src/flybrain_drive/per_view_flyvis_stream.py'),camera_collector_sha256=sha256('src/flybrain_drive/ac_csp_vision.py'),ticker_sha256=sha256('src/flybrain_drive/native_flight_body_ticker.py'),worker_sha256=sha256('src/flybrain_drive/native_flight_process_runtime.py'),fit_checkpoint_sha256=fr['checkpoint_sha256'],visual_parity_sha256=sha256(proof/'result.json'),source_sha256=PIN,identity=identity,preflight_session=status,seed=None,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0,scope=__doc__))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');write_json(run/'backend-initialization.json',warm)
    camera=sampler=runtime=ticker=None;events=[];native_samples={};native_position=0;result=dict(status='failed',actor_qualified=False,optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0)
    try:
        runtime=NativeFlightProcessRuntime(model,head,dn,optical,body,encoder,python_executable='C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe',library_directory='C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages')
        write_json(run/'worker-ready.json',runtime.ready)
        sampler=NativeProcessRowsSampler(maximum_samples=20000);sampler.start()
        camera=CSPVisionRecorder(window.hwnd,camera_config,max_frames=160,period=.04,max_bytes=256*1024**2,image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,expected_source_sha256=PIN,mmf_stop_signal=True,memory_only_receive_checks=True)
        camera.wait_ready();write_json(run/'resume-identity.json',dict(request_id=camera.request_id,generation=status['generation']))
        owned_resume(run,gate,camera,status_path,camera_root)
        selector=OnlineBodyShadow(directory='.',request_id=camera.request_id,generation=status['generation'],offset_lower=0.,native_rows=sampler.rows,native_identity={k:identity[k] for k in read(EXT/'contract.json')['native_identity']})
        runtime.begin_episode(camera.request_id)
        ticker=NativeFlightBodyTicker(runtime,native_rows=sampler.rows,native_identity={k:identity[k] for k in read(EXT/'contract.json')['native_identity']},episode_id=camera.request_id);ticker.start()
        visual=PerViewFlyvisStream(network,prior,pool,projector,indices,geometry,episode_id=camera.request_id,generation=status['generation'],device='cuda')
        deadline=time.perf_counter()+3.;last=-1
        while time.perf_counter()<deadline:
            if sampler.error:raise RuntimeError(sampler.error)
            if ticker.error:raise RuntimeError(str(ticker.error))
            current,modified,_=read_session_heartbeat(status_path);validate_heartbeat(current,time.time()-modified,mutating=False)
            if current['generation']!=status['generation'] or current['paused'] or not current['live'] or current['ai_controlled'] or abs(current['speed_kmh'])>=.1 or current['gas']>=.02 or current['brake']<=.98:raise ValueError('Stationary live AI-off full-brake state changed')
            try:frame,selected,_=camera.wait_for_candidate(max_age=.15,timeout=.04)
            except NoFreshCSPFrame:continue
            if frame['capture_id']==last:time.sleep(.001);continue
            last=frame['capture_id'];event=dict(capture_id=last,camera_received=frame['received'],render_start_lowers=[v['render_started_lower'] for v in frame['views']]);events.append(event)
            group=freeze_camera_group(frame,camera.samples_for(frame),selected_at=selected);require_nonzero_camera_views(group)
            selector._drain_native()
            bounds=tuple(event['render_start_lowers']);admission=[]
            try:
                clock=camera.clock.record()
                for view,lower in zip(frame['views'],bounds):
                    witness=view['source_render_metadata'];source_render_witness(witness)
                    j=int(np.searchsorted(selector.history.receipts,lower,side='left')-1)
                    if j<1:raise ValueError('No noninitial prior native view receipt')
                    upper=witness['render_started_clock']+clock['offset_upper_seconds']+clock.get('host_clock_quantization_seconds',0.)
                    row=next(r for r in sampler.rows[:selector.position] if r['authority']['packet_id']==selector.history.samples[j].legacy.packet_id)
                    gap=upper-row['copy']['first_copy_start']
                    if not 0<=gap<=.025:raise ValueError('Original source-to-render bound exceeds25ms')
                    admission.append(gap)
            except (KeyError,ValueError) as error:
                event.update(status='unqualified_warmup_pixels_skipped',reason=str(error))
                if visual.previous is not None:raise ValueError('Source qualification lost inside visual episode: '+str(error))
                continue
            started=time.perf_counter();obs=visual.process(pixels=group['pixels'],render_start_lowers=bounds,camera_received=frame['received'],capture_id=last,generation=status['generation']);torch.cuda.synchronize();event['visual_seconds']=time.perf_counter()-started
            ctx=selector._select_context(event,camera_receipt=obs.camera_received,bounds=obs.render_start_lowers)
            if ctx is None or any(v is None for v in ctx.visual_contexts):raise ValueError('Qualified visual group lacks original body history')
            publication=ticker.publish(obs,ctx)
            event.update(status='scene_published',publication=publication,source_render_gap_seconds=admission)
        count=sum(e.get('status')=='shadow_output' for e in ticker.events)
        if ticker.error:raise RuntimeError(str(ticker.error))
        if not count:raise ValueError('No admitted end-to-end shadow outputs')
        result.update(status='completed_stationary_native_flight_shadow',shadow_outputs=count,termination_reason='Finite stationary source-qualified per-view author vision plus current native flight inference, read-only; no moving/closed-loop/physiological qualification')
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error),runtime_rejection=runtime.last_rejection if runtime else None,termination_reason='Explicit live source/timing/authority failure; no fallback or student control')
    finally:
        if ticker is not None:
            try:ticker.stop();result['ticker_stopped']=True
            except Exception as error:result['ticker_close_error']=str(error)
            result['ticker_error']=ticker.error
            write_json(run/'ticker-events.json',ticker.events);result['ticker_events_sha256']=sha256(run/'ticker-events.json')
            if ticker.error or not result.get('ticker_stopped'):result['status']='failed'
        if runtime is not None:
            try:result['worker_terminal']=runtime.close()
            except Exception as error:result['worker_close_error']=str(error);result['status']='failed'
        if camera is not None:
            try:result['camera_close']=camera.close()
            except Exception as error:result['camera_close_error']=str(error)
            try:
                pause=subprocess.run([sys.executable,'-m','scripts.control_ac_session','--command','pause','--expected-lua-sha256',PIN,'--execute','--output',str(run/'final-pause')],capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));result['pause_returncode']=pause.returncode;write_json(run/'pause-process.json',dict(returncode=pause.returncode,stdout=pause.stdout,stderr=pause.stderr))
            except Exception as error:result['pause_error']=str(error)
        if sampler is not None:
            try:sampler.stop();result['native_sampler_stopped']=True
            except Exception as error:result['native_sampler_close_error']=str(error)
            write_json(run/'native-copy.json',sampler.rows);result['native_sha256']=sha256(run/'native-copy.json')
        if camera is not None:
            try:result['camera_archive']=camera.archive_to(run/'vision',bundle=True)
            except Exception as error:result['archive_error']=str(error)
        result['shadow_outputs']=sum(e.get('status')=='shadow_output' for e in ticker.events) if ticker is not None else 0
        if result.get('pause_returncode')!=0 or result.get('native_sampler_stopped') is not True or any(key in result for key in ['camera_close_error','pause_error','archive_error']):
            result['status']='failed';result['cleanup_verified']=False
        else:result['cleanup_verified']=True
        write_json(run/'events.json',events);result['events_sha256']=sha256(run/'events.json');write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
