"""Stationary background four-view and single-circuit multirate shadow; never control."""
import argparse,json,copy,os,time,yaml,uuid,subprocess,sys
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.graph import sha256
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.stationary_camera_admission import validate_stationary_file_camera_admission, validate_known_stopped_metadata_failure_probe
from flybrain_drive.ac_csp_clock import CSPClockBounds, initialize_clock
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder, NoFreshCSPFrame
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.immutable_camera_group import freeze_camera_group, require_nonzero_camera_views
from flybrain_drive.online_body_shadow import OnlineBodyShadow
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.unified_multirate_shadow import UnifiedMultirateShadow
from flybrain_drive.unified_circuit_worker import UnifiedCircuitWorker
from flybrain_drive.process_unified_multirate_shadow import ProcessUnifiedMultirateShadow
from flybrain_drive.unified_live_ticker import UnifiedLiveBodyTicker
from flybrain_drive.held_visual_prefix import HeldVisualPrefix
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from flybrain_drive.flight_stream import held_input_batches
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat, terminal_receipt
from scripts.isolated_camera_diagnostic import IsolatedCamera
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.native_process_rows import NativeProcessRowsSampler
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
PIN='78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6'
def owned_resume(run,gate,camera,status_path,camera_root):
 local=Path(os.environ['LOCALAPPDATA'])/'flybrain';deadline=time.monotonic()+2
 while True:
  st,modified,_=read_session_heartbeat(status_path);validate_heartbeat(st,time.time()-modified,mutating=True);cam=read(camera_root/'status.json')
  if st.get('camera_active') is True and st.get('camera_request_id')==camera.request_id and cam.get('request_id')==camera.request_id and cam.get('state')=='capturing':break
  if camera.error or not camera.thread.is_alive():raise RuntimeError('Owned camera ended before ready')
  if time.monotonic()>deadline:raise TimeoutError('Owned camera ready not observed')
  time.sleep(.01)
 gate.camera_ready(st,cam,camera.request_id)
 request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,command='resume',generation=st['generation'],epoch=st['epoch'],expires_clock=st['script_clock']+3,camera_request_id=camera.request_id,owned_mmf_resume=True)
 journal=local/'csp-session/intent-journal';journal.mkdir(exist_ok=True);write_json(journal/(request['id']+'.json'),dict(request=request,session=st,camera=cam,role='isolated_unified_stationary_shadow'))
 fresh,modified,_=read_session_heartbeat(status_path);validate_heartbeat(fresh,time.time()-modified,mutating=True)
 if fresh.get('camera_request_id')!=camera.request_id:raise ValueError('Camera ownership changed')
 request=fresh_unpublished_request(st,fresh,request,publication_started=False);write_json(run/'resume-request.json',request);write_json(local/'csp-session/request.json',request)
 deadline=time.monotonic()+4;receipt=None
 while time.monotonic()<deadline:
  receipt=terminal_receipt(local/'csp-session',request)
  if receipt is not None:break
  time.sleep(.01)
 if receipt is None:raise TimeoutError('Resume outcome unknown; no retry')
 write_json(run/'resume-receipt.json',receipt)
 if receipt.get('status')!='api_returned_true':raise ValueError('Resume rejected')
 return request['id']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--recover-known-terminal-error',action='store_true');ap.add_argument('--held-prefix',action='store_true');ap.add_argument('--prefix-steps',type=int,choices=(8,16),default=16);a=ap.parse_args();planpath=B/'20261101-adaptive-contrast-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 execution_plan=Path('D:/CodexResearchRuns/flybrain-native-process-shadow-plan-20261102-02/plan.json');declared=read(execution_plan)
 if declared['program_sha256']!=sha256(__file__) or declared['sampler_sha256']!=sha256('src/flybrain_drive/native_process_sampler.py') or declared['adapter_sha256']!=sha256('src/flybrain_drive/native_process_rows.py'):raise ValueError('declared process-sampler source changed')
 run=begin_run(a.output,dict(execution_plan_sha256=sha256(execution_plan),sampler_sha256=declared['sampler_sha256'],adapter_sha256=declared['adapter_sha256'],program_sha256=sha256(__file__),ticker_sha256=sha256('src/flybrain_drive/unified_live_ticker.py'),guard_sha256=sha256('src/flybrain_drive/unified_host_guard.py'),kernel_sha256=sha256('src/flybrain_drive/parallel_csr_cpu.py'),process_worker_sha256=sha256('src/flybrain_drive/unified_circuit_worker.py'),process_shadow_sha256=sha256('src/flybrain_drive/process_unified_multirate_shadow.py'),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=0,actor=False,real_data_used_for_fit=False))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False);worker=None
 try:
  if a.held_prefix:
   proofroot=B/('20261101-flyvis-held-prefix8-parity' if a.prefix_steps==8 else '20261101-flyvis-held-prefix-parity');proof=read(proofroot/'result.json');proof_manifest=read(proofroot/'manifest.json')
   if proof['status']!='completed_archived_rgb_prefix_parity' or proof_manifest['config']['prefix_sha256']!=sha256('src/flybrain_drive/held_visual_prefix.py') or proof_manifest['config']['program_sha256']!=sha256('scripts/verify_flyvis_held_visual_prefix_8.py' if a.prefix_steps==8 else 'scripts/verify_flyvis_held_visual_prefix.py'):raise ValueError('Exact prefix parity proof required')
   result.update(held_prefix=True,prefix_steps=a.prefix_steps,held_prefix_source_sha256=sha256('src/flybrain_drive/held_visual_prefix.py'),held_prefix_proof_sha256=sha256(proofroot/'result.json'))
  torch.set_num_threads(1);torch.manual_seed(863009)
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  spec=read(trdir/'manifest.json')['config']['plan'];source=read(Path(spec['source'])/'result.json')
  visual=EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(source['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
  mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];motor=PhysicalManeuverCircuit(mp['spec'],torch.load(mp['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
  temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,visual.backbone.scene_command.scale,len(visual.backbone.vision_ports));temporal.load_state_dict(torch.load(opdir/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'])
  boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports));boundary.load_state_dict(torch.load(ctxdir/'context-boundary.pt',map_location='cpu',weights_only=True)['model'])
  model=UnifiedFlightSensorimotor(motor,visual.backbone,temporal,boundary,body_feedback=CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)))
  model.motor.backbone.circuit.sparse_backend='numba_csr8_cpu';model=model.to('cpu').eval().requires_grad_(False)
  fit=B/'20261101-speed-wheel-turn-control-fit';fr=read(fit/'result.json');f=fit/'speed-body-adapter.pt'
  if fr['status']!='completed_synthetic_speed_wheel_turn_control_training' or sha256(f)!=fr['checkpoint_sha256']:raise ValueError('body checkpoint')
  payload=torch.load(f,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('body qualification')
  model.body_feedback.load_state_dict(payload['model'],strict=True);parameter=model.temporal.opponent_projection.weight;parameter.requires_grad_(False);initial=parameter.detach().clone()
  goals=torch.tensor(racing_goal(100,1),device='cpu');view=torch.ones((1,4),dtype=torch.bool,device='cpu')
  def load_scene(seed,amplitude,root):
   r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
   if r['status']!='completed_causal_four_view_constant_yaw' or sha256(f)!=r['responses_sha256'] or cfg['texture_seed']!=seed or cfg['program_sha256']!=plan['generator_sha256'] or r['cases']!=[0.,-amplitude,amplitude]:raise ValueError('declared causal synthetic stimulus required')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();mot=z['motion'].copy();times=z['times'].copy()
   if ret.shape!=(3,41,4,721) or mot.shape!=(3,41,4,8,721) or not np.isfinite(ret).all() or not np.isfinite(mot).all() or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12) or not np.all(ret[:,:11]==ret[:,:1]):raise ValueError('synthetic stimulus shape')
   full=np.concatenate([ret[:,:,:,None],mot],axis=3)
   with torch.no_grad():sc=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device='cpu')).reshape(3,41,576)
   return sc,times,sha256(f)
  def rollout(scene,times,learn):
   cases=[]
   for case in (1,2):
    model.reset_episode();values=np.zeros(21);values[16]=5.;values[17:21]=5./.29;valid=np.ones(21,dtype=bool);outputs=[]
    for i,t in enumerate(times):
     now=1.+float(t);body=NavigationBodyObservation('synthetic-yaw',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only');ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
     fn=model.sensory_training_decision if learn else model.sensory_decision
     intent,_=fn(decision_time=now,racing_goals=goals,scene_features=scene[case,i:i+1],scene_valid=view,body_context=ctx);outputs.append(intent[0])
    cases.append(torch.stack(outputs))
   return torch.stack(cases)
  if sha256('src/flybrain_drive/synthetic_road_visual.py')!=plan['road_source_sha256'] or sha256('scripts/replay_flyvis_synthetic_road_varied.py')!=plan['generator_sha256']:raise ValueError('registered generator changed')
  ckroot=B/'20261101-causal-yaw-multispeed-fit';crs=read(ckroot/'result.json');ck=ckroot/'opponent-projection.pt'
  if crs['status']!='completed_synthetic_multispeed_yaw_training' or sha256(ck)!=crs['checkpoint_sha256']:raise ValueError('visual checkpoint')
  candidate=torch.load(ck,map_location='cpu',weights_only=True)
  if candidate['actor_qualified'] is not False or candidate['source_body_sha256']!=sha256(f):raise ValueError('fixed body lineage')
  with torch.no_grad():model.temporal.opponent_projection.weight.copy_(candidate['weight'].to('cpu'))
  model.spatial_contrast=RetinalSpatialContrastBoundary(len(model.motor.backbone.vision_ports)).to('cpu')
  weight=model.spatial_contrast.projection.weight;initial_weight=weight.detach().clone()
  optim=torch.optim.Adam([weight],lr=.003,eps=1e-12)
  def load_source(folder,seed,curvature,start):
   root=B/folder;rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];src=root/'responses.npz'
   if rr['status']!='completed_causal_four_view_synthetic_road' or sha256(src)!=rr['responses_sha256'] or cfg['road_generator_sha256']!=plan['road_source_sha256'] or cfg['texture_seed']!=seed:raise ValueError('road source provenance')
   if seed!=879701 and (cfg['program_sha256']!=plan['generator_sha256'] or cfg['curvature']!=curvature or cfg['bend_start_m']!=start):raise ValueError('registered varied road')
   if seed==879701 and (curvature!=.018 or start!=12):raise ValueError('original fixed road parameters')
   with np.load(src,allow_pickle=False) as z:ret=z['retina'].copy();mot=z['motion'].copy();times=z['times'].copy();bends=z['bend'].copy()
   if ret.shape!=(3,41,4,721) or mot.shape!=(3,41,4,8,721) or not np.isfinite(ret).all() or not np.isfinite(mot).all() or not np.array_equal(bends,[0,-1,1]) or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12):raise ValueError('road sample shape')
   image=np.concatenate([ret[:,:,:,None],mot],axis=3)
   with torch.no_grad():scene=model.scene_pool(torch.tensor(image.reshape(-1,4,9,721),dtype=torch.float32,device='cpu')).reshape(3,41,576)
   return scene,times,sha256(src)
  def rollout(scene,times,learn):
   cases=[]
   for case in range(3):
    model.reset_episode();outputs=[]
    for i,t in enumerate(times):
     now=1.+float(t);speed=0. if t<=.4 else 3.;values=np.zeros(21);values[16]=speed;values[17:21]=speed/.29;valid=np.ones(21,dtype=bool)
     body=NavigationBodyObservation('synthetic-road',i,now-.01,now,values,valid,'synthetic-only');ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
     fn=model.sensory_training_decision if learn else model.sensory_decision
     intent,_=fn(decision_time=now,racing_goals=goals,scene_features=scene[case,i:i+1],scene_valid=view,body_context=ctx);outputs.append(intent[0])
    cases.append(torch.stack(outputs))
   return torch.stack(cases)
  fit=B/'20261101-adaptive-pattern-fit';fr=read(fit/'result.json');candidate_file=fit/'adaptive-pattern-contrast.pt'
  if fr['status']!='completed_synthetic_adaptive_pattern_fit' or sha256(candidate_file)!=fr['checkpoint_sha256']:raise ValueError('candidate mismatch')
  candidate=torch.load(candidate_file,map_location='cpu',weights_only=True)
  if candidate['actor_qualified'] is not False or candidate['body_sha256']!=sha256(f) or candidate['temporal_sha256']!=sha256(ck):raise ValueError('candidate lineage')
  with torch.no_grad():weight.copy_(candidate['weight']);model.spatial_contrast.reference.copy_(candidate['reference']);model.spatial_contrast.max_contrast_rms.fill_(candidate['max_contrast_rms'])
  # Compile and exercise the fixed CPU graph before acquiring a camera, then explicitly reset.
  sc,t,_=load_source(*plan['fit_sources'][0]);rollout(sc,t,False);model.reset_episode();model.freshness_limits=HostFreshnessLimits(.05,.15)
  parity=Path('D:/CodexResearchRuns/flybrain-unified-process-parity-20261102-04');proof=read(parity/'result.json')
  if proof.get('status')!='completed_unified_process_recorded_parity' or proof.get('maximum_abs_difference',1)>2e-5 or proof.get('worker_source_sha256')!=sha256('src/flybrain_drive/unified_circuit_worker.py') or sha256(parity/'differences.npz')!=proof.get('output_sha256') or proof.get('warmup_explicit_reset') is not True:raise ValueError('Exact warmup process parity proof required')
  result['process_parity_sha256']=sha256(parity/'result.json')
  worker=UnifiedCircuitWorker(model,goals,timeout=45.);result['worker_pid']=worker.process.pid
  warm_values=np.zeros(21);warm_valid=np.ones(21,dtype=bool)
  for i in range(2):
   now=10.+.04*i;body=NavigationBodyObservation('explicit-worker-warmup',i,now-.01,now,warm_values.copy(),warm_valid.copy(),'warmup-only');ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now);worker.step(when=now,scene=np.zeros((1,576),dtype=np.float32),context=ctx)
  result['worker_warmup_reset']=worker.reset();worker.timeout=.25
  cfg=read('configs/flight-stream-batched-v1-20261029.json');camera_config=read(cfg['camera_config']);modeldir=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root'])
  if sha256(cfg['camera_config'])!=cfg['camera_config_sha256'] or sha256(modeldir/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('visual configuration mismatch')
  os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  with torch.device('cuda'):
   netcfg=yaml.safe_load((modeldir/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];network=flyvis.Network(**netcfg).to('cuda');network.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
   nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];u,v=nodes.u[:][receptors],nodes.v[:][receptors];indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,[f'{kind}{d}' for kind in ('T4','T5') for d in 'abcd']);projector=FlyvisCameraProjector(camera_config,u,v);network_state=network.steady_state(1.,.005,batch_size=4,value=.5);gpu_pool=copy.deepcopy(model.scene_pool).to('cuda')
   # Warm exact pixel projection, recurrent FlyVis and pooled transfer on synthetic gray input.
   gray=np.zeros((180,320,3),dtype=np.uint8);warm=np.stack([projector.sample(gray) for _ in range(4)]);warm_gpu=torch.as_tensor(warm,device='cuda')
   with torch.no_grad():
    temporary=network.simulate(warm_gpu[:,None,None,:].expand(-1,8,-1,-1),dt=.005,initial_state=network_state,as_states=True)[-1]
    warm_motion=temporary.nodes.activity[:,indices];gpu_pool(torch.cat((warm_gpu[:,None,:],warm_motion),dim=1)[None]).to('cpu');torch.cuda.synchronize()
    network_state=network.steady_state(1.,.005,batch_size=4,value=.5);torch.cuda.synchronize()
  # Preserve and specifically validate a known stopped-result/failed-heartbeat conflict.
  recovery_original=None
  if a.recover_known_terminal_error:
   local=Path(os.environ['LOCALAPPDATA'])/'flybrain';cr=local/'csp-camera';sp=local/'csp-session/status.json';st,mtime,_=read_session_heartbeat(sp);validate_heartbeat(st,time.time()-mtime,mutating=False);prev=read(cr/'request.json');import re
   if not isinstance(prev.get('id'),str) or re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',prev['id']) is None:raise ValueError('Safe original camera request identity required')
   prior=read(cr/prev['id']/'result.json');camera_state=read(cr/'status.json');write_json(run/'recovery-before.json',dict(session=st,request=prev,terminal=prior,camera=camera_state));bounds=CSPClockBounds(st['generation'],maximum_width=.01);initialize_clock(cr,bounds,int(st['window_handle'].removesuffix('ULL')),host_clock=time.perf_counter);fresh,mtime,_=read_session_heartbeat(sp);validate_heartbeat(fresh,time.time()-mtime,mutating=False);cam=read(cr/'status.json');clock=bounds.record()
   if read(cr/'request.json')!=prev or read(cr/prev['id']/'result.json')!=prior:raise ValueError('Original recovery evidence changed')
   validate_known_stopped_metadata_failure_probe(fresh,prev,cam,prior,clock,now=time.perf_counter());recovery_original=dict(generation=st['generation'],request=prev,terminal=prior);write_json(run/'recovery-admission.json',dict(session=fresh,request=prev,terminal=prior,camera=cam,clock=clock,interpretation='Specific diagnostic only; previous mismatch remains'))
  # Refuse contradictory native authority before changing session state.
  with ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size) as mapping:preflight_native=repeated_teacher_snapshot(mapping)
  if preflight_native.builtin_ai_controlled:raise ValueError('native AI must be off before session pause')
  # Pause only through the background CSP lifecycle, after all warmup.
  pause_path=run/'pause';pause=subprocess.run([sys.executable,'scripts/control_ac_session.py','--command','pause','--execute','--expected-lua-sha256',PIN,'--output',str(pause_path)],capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  write_json(run/'pause-process.json',dict(returncode=pause.returncode,stdout=pause.stdout,stderr=pause.stderr))
  if pause.returncode!=0 or read(pause_path/'result.json').get('status')!='completed':raise RuntimeError('Pause failed or unknown; no camera request')
  result['pause_completed']=True
  status_path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json';refuse_leased_transport('stationary unified shadow',status_path=status_path);status,modified,_=read_session_heartbeat(status_path);validate_heartbeat(status,time.time()-modified,mutating=False)
  camera_root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera';request_path=camera_root/'request.json';camera_status_path=camera_root/'status.json'
  previous_request=read(request_path) if request_path.exists() else None
  camera_status=read(camera_status_path) if camera_status_path.exists() else None
  # Resolve only a syntactically safe owned ID before consulting its terminal receipt.
  import re
  rid=previous_request.get('id') if isinstance(previous_request,dict) else None
  terminal_path=camera_root/rid/'result.json' if isinstance(rid,str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',rid) else None
  terminal=read(terminal_path) if terminal_path is not None and terminal_path.exists() else None
  if recovery_original is not None:
   if status['generation']!=recovery_original['generation'] or previous_request!=recovery_original['request'] or terminal!=recovery_original['terminal'] or camera_status.get('state')!='failed' or 'Cannot commit camera metadata' not in camera_status.get('reason',''):raise ValueError('Recovery evidence changed during pause')
   gate=PausedMMFStartupGate(status,previous_request,None,terminal);result['recovery_admission_sha256']=sha256(run/'recovery-admission.json')
  else:gate=PausedMMFStartupGate(status,previous_request,camera_status,terminal)
  windows=WindowsACWindow();matches=windows.discover()
  if len(matches)!=1 or windows.user.GetForegroundWindow()==matches[0].hwnd or int(status['window_handle'].removesuffix('ULL'))!=matches[0].hwnd:raise ValueError('matching background engine required')
  with ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size) as mapping:native=repeated_teacher_snapshot(mapping)
  if native.builtin_ai_controlled:raise ValueError('native AI must be off')
  axis=read('configs/ac-axis-probe.json');sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
  if sdk.GetVJDStatus(axis['device_id'])!=1:raise ValueError('Released virtual device required')
  with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
  camera=sampler=ticker=None;events=[];stop_reason=None;shadow=None;archive=None
  try:
   actual_config=read('configs/ac-csp-exterior-front-v3-retina.json')
   for name in ('front','left','right','rear'):
    for key in ('projection_x','projection_y','forward_body','right_body','up_body'):
     if actual_config['views'][name][key]!=camera_config['views'][name][key]:raise ValueError('Camera projector geometry mismatch')
   result['actual_camera_config_sha256']=sha256('configs/ac-csp-exterior-front-v3-retina.json')
   camera=IsolatedCamera(matches[0].hwnd,actual_config,PIN,'configs/ac-frame-persistence-buffered32-20261030.json');result['camera_request_id']=camera.request_id;camera.wait_ready()
   result['resume_request_id']=owned_resume(run,gate,camera,status_path,camera_root);result['lifecycle_requests_sent']=2
   deadline=time.monotonic()+1.
   while True:
    live,modified,_=read_session_heartbeat(status_path);validate_heartbeat(live,time.time()-modified,mutating=False)
    if live.get('generation')==status['generation'] and live.get('paused') is False and live.get('live') is True and live.get('camera_active') is True and live.get('camera_request_id')==camera.request_id:break
    if camera.error or not camera.thread.is_alive() or time.monotonic()>=deadline:raise RuntimeError('Owned live camera state not observed after resume')
    time.sleep(.01)
   sampler=NativeProcessRowsSampler();sampler.start();selector=OnlineBodyShadow(directory='.',request_id=camera.request_id,generation=status['generation'],offset_lower=0.,native_rows=sampler.rows,native_identity=identity);encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity)
   shadow=ProcessUnifiedMultirateShadow(UnifiedHostGuard(model),worker,goals=goals);ticker=UnifiedLiveBodyTicker(shadow,native_rows=sampler.rows,native_identity=identity,episode_id=camera.request_id);ticker.start();deadline=time.monotonic()+8;last_capture=None;last_visual=None;held=None;prefix=None
   while time.monotonic()<deadline:
    if sampler.error:raise RuntimeError('native sampler failed: '+str(sampler.error))
    if ticker.error or not ticker.thread.is_alive():raise RuntimeError('body ticker stopped: '+str(ticker.error))
    if camera.error:raise RuntimeError('camera reader failed: '+str(camera.error))
    if not camera.thread.is_alive():raise RuntimeError('Camera ended before shadow completion')
    heartbeat,modified,_=read_session_heartbeat(status_path);validate_heartbeat(heartbeat,time.time()-modified,mutating=False)
    if heartbeat.get('generation')!=status['generation'] or heartbeat.get('paused') is not False or not heartbeat.get('live') or heartbeat.get('ai_controlled') or heartbeat.get('expert_probe_active') or abs(heartbeat.get('speed_kmh',999))>.5:raise ValueError('session changed during observation')
    try:frame,group=camera.pixel_candidate();selected=group['selected_at']
    except NoFreshCSPFrame:continue
    if frame['capture_id']==last_capture:time.sleep(.002);continue
    event=dict(capture_id=frame['capture_id'],camera_received=frame['received'],render_start_lower=[v['render_started_lower'] for v in frame['views']],vehicle_control=False)
    try:require_nonzero_camera_views(group)
    except ValueError as error:
     event.update(status='invalid_camera_view',reason=str(error));events.append(event);raise
    # The prior-view body context only depends on receipts before the original
    # render bounds; select it before visual compute. The ticker separately selects
    # a fresh decision body, so this never promotes this earlier body to an output.
    ctx=selector._select_context(event,camera_receipt=frame['received'],bounds=tuple(event['render_start_lower']))
    visual_snapshot=time.perf_counter();sampled=np.stack([projector.sample(group['pixels'][name]) for name in ('front','left','right','rear')]);sampling_finished=time.perf_counter()
    with torch.no_grad(),torch.device('cuda'):
     current=torch.as_tensor(sampled,device='cuda')
     if last_visual is not None:
      if not 0<visual_snapshot-last_visual<=.15:raise ValueError('visual processing gap too large')
      if a.held_prefix:
       if prefix is None:raise ValueError('Original held prefix missing')
       network_state=prefix.commit(visual_snapshot-last_visual);prefix=None
      else:
       for dt,count in held_input_batches(visual_snapshot-last_visual):
        while count:
         n=min(count,16);network_state=network.simulate(held[:,None,None,:].expand(-1,n,-1,-1),dt=dt,initial_state=network_state,as_states=True)[-1];count-=n
     torch.cuda.synchronize();motion_finished=time.perf_counter();motion=network_state.nodes.activity[:,indices];image=torch.cat((current[:,None,:],motion),dim=1)[None];scene=gpu_pool(image).to('cpu');torch.cuda.synchronize()
    visual_finished=time.perf_counter();held=current.detach().clone();last_visual=visual_snapshot;last_capture=frame['capture_id'];event.update(visual_snapshot=visual_snapshot,sampling_finished=sampling_finished,motion_finished=motion_finished,visual_finished=visual_finished)
    if ctx is None:
     event['status']='missing_current_body';events.append(event)
     if a.held_prefix:
      event['prefix_started']=time.perf_counter()
      with torch.no_grad(),torch.device('cuda'):
       prefix=HeldVisualPrefix(network,network_state,held,steps=a.prefix_steps);torch.cuda.synchronize()
      event['prefix_finished']=time.perf_counter()
     continue
    try:
     publication=ticker.publish(scene,ctx,last_capture)
    except (ValueError,RuntimeError,TimeoutError) as error:
     event.update(status='publication_rejected',reason=type(error).__name__+': '+str(error));events.append(event);stop_reason=event['reason'];break
    event.update(status='scene_published',publication=publication);events.append(event)
    if a.held_prefix:
     event['prefix_started']=time.perf_counter()
     with torch.no_grad(),torch.device('cuda'):
      prefix=HeldVisualPrefix(network,network_state,held,steps=a.prefix_steps);torch.cuda.synchronize()
     event['prefix_finished']=time.perf_counter()
   if ticker.error:stop_reason='body ticker stopped: '+str(ticker.error)
   result.update(status='observation_finished' if stop_reason is None else 'stopped_on_error',stop_reason=stop_reason,camera_request_id=camera.request_id,groups=len(events),ui_actions_sent=0,lifecycle_requests_sent=2,actuator_messages_sent=0,training_admissible=False,physical_exposure_sync_verified=False,interpretation='stationary isolated MMF multirate shadow with process sampler; no driving, physical timing or actor qualification')
  finally:
   if ticker is not None:
    try:
     ticker.stop();write_json(run/'ticker-events.json',ticker.events);result['ticker_events_sha256']=sha256(run/'ticker-events.json');write_json(run/'worker-timings.json',shadow.worker_observations);result['worker_timings_sha256']=sha256(run/'worker-timings.json');result['ticker_error']=ticker.error;result['published']=sum(e['status']=='shadow_output' for e in ticker.events)
     if ticker.error:result['status']='stopped_on_error';result['stop_reason']='body ticker stopped: '+str(ticker.error)
     elif result.get('status')=='observation_finished' and not result['published']:result['status']='incomplete_no_shadow_output'
    except Exception as error:result['ticker_close_error']=type(error).__name__+': '+str(error);result['status']='stopped_on_error'
   if camera is not None:
    try:result['camera_close']=camera.close()
    except Exception as error:result['camera_close_error']=type(error).__name__+': '+str(error);result['status']='stopped_on_error'
   if sampler is not None:
    try:sampler.stop();write_json(run/'native-copy.json',sampler.rows);result['native_sha256']=sha256(run/'native-copy.json');result['native_error']=sampler.error
    except Exception as error:result['sampler_close_error']=type(error).__name__+': '+str(error);result['status']='stopped_on_error'
   write_json(run/'events.json',events);result['events_sha256']=sha256(run/'events.json')
   if camera is not None:
    try:
     close=result.get('camera_close',{});terminal=close.get('producer_terminal',{})
     if close.get('status')!='stopped' or close.get('worker_alive') is not False or terminal.get('request_id')!=camera.request_id or terminal.get('status') not in ('stopped','completed'):raise ValueError('Matching clean terminal required')
     result['archive']=camera.archive_to(run/'vision',bundle=True)
     if result['archive']['count']!=close['frames']:raise ValueError('Archive count mismatch')
    except Exception as error:result['archive_error']=type(error).__name__+': '+str(error);result['status']='stopped_on_error'
    try:camera.shutdown()
    except Exception as error:result['owner_close_error']=type(error).__name__+': '+str(error);result['status']='stopped_on_error'
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if worker is not None:
   try:result['worker_terminal']=worker.close()
   except Exception as e:result['worker_close_error']=type(e).__name__+': '+str(e);result['worker_process_alive']=worker.process.is_alive();result['status']='stopped_on_error'
  write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
