"""Archived RGB and live native receipts in one non-controlling circuit; timing is synthetic."""
import argparse,json,copy,os,time,yaml
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
from flybrain_drive.stationary_camera_admission import validate_stationary_file_camera_admission
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder, NoFreshCSPFrame
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.immutable_camera_group import freeze_camera_group
from flybrain_drive.online_body_shadow import OnlineBodyShadow
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.unified_multirate_shadow import UnifiedMultirateShadow
from flybrain_drive.unified_live_ticker import UnifiedLiveBodyTicker
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.virtual_navigation_body import decode_navigation_body_prefix
import base64,hashlib,io,zipfile
from PIL import Image
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from flybrain_drive.flight_stream import held_input_batches
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-adaptive-contrast-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),ticker_sha256=sha256('src/flybrain_drive/unified_live_ticker.py'),kernel_sha256=sha256('src/flybrain_drive/parallel_csr_cpu.py'),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=0,actor=False,real_data_used_for_fit=False))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
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
  # Pixels are old recordings. Synthetic host bounds are only scheduler inputs;
  # they do not claim current camera exposure or synchronization with live body.
  archive=B/'20261101-stationary-unified-file-shadow-warm/vision';bundle=read(archive/'bundle.json');manifest=read(archive/'archive.json');zip_path=archive/'camera.zip'
  if sha256(zip_path)!=bundle['sha256'] or sha256(archive/'archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('Original camera archive changed')
  expected={item['file']:item for item in manifest};groups=[]
  with zipfile.ZipFile(zip_path) as z:
   for i in range(16):
    group=[]
    for name in ('front','left','right','rear'):
     filename=f'{i:04d}-{name}.png';blob=z.read(filename);row=expected[filename]
     if len(blob)!=row['bytes'] or hashlib.sha256(blob).hexdigest()!=row['sha256']:raise ValueError('Original pixel hash mismatch')
     group.append(np.asarray(Image.open(io.BytesIO(blob)).convert('RGB')).copy())
    groups.append(group)
  with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
  sampler=ticker=None;events=[]
  try:
   sampler=NativeCopySampler(lambda:ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size));sampler.start()
   shadow=UnifiedMultirateShadow(UnifiedHostGuard(model),goals=goals)
   ticker=UnifiedLiveBodyTicker(shadow,native_rows=sampler.rows,native_identity=identity,episode_id='archive-scene-live-body');ticker.start()
   encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);held=None;last_visual=None
   for index,group in enumerate(groups):
    if sampler.error or ticker.error:raise RuntimeError('sampler/ticker: '+str((sampler.error,ticker.error)))
    started=time.perf_counter();sampled=np.stack([projector.sample(pixels) for pixels in group]);sampling_finished=time.perf_counter()
    with torch.no_grad(),torch.device('cuda'):
     current=torch.as_tensor(sampled,device='cuda')
     if held is not None:
      elapsed=started-last_visual
      if not 0<elapsed<=.15:raise ValueError('Replay processing gap too large')
      for dt,count in held_input_batches(elapsed):
       while count:
        n=min(count,16);network_state=network.simulate(held[:,None,None,:].expand(-1,n,-1,-1),dt=dt,initial_state=network_state,as_states=True)[-1];count-=n
     torch.cuda.synchronize();motion_finished=time.perf_counter();motion=network_state.nodes.activity[:,indices];scene=gpu_pool(torch.cat((current[:,None,:],motion),dim=1)[None]).to('cpu');torch.cuda.synchronize()
    held=current.detach().clone();last_visual=started;completed=time.perf_counter()
    row=sampler.rows[-1];raw=base64.b64decode(row['authority']['prefix_base64'],validate=True)
    sample=decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size],row['copy']['second_copy_end']);now=time.perf_counter()
    body=encoder.encode(sample,episode_id='archive-scene-live-body',decision_monotonic_seconds=now)
    # The .02/.01 offsets are synthetic; they intentionally certify no actual camera.
    ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.02,)*4,camera_receipt_time=now-.01,decision_time=now)
    publication=ticker.publish(scene,ctx,index)
    events.append(dict(index=index,visual_started=started,sampling_finished=sampling_finished,motion_finished=motion_finished,visual_finished=completed,scene_published=publication,archived_pixels=True,synthetic_render_bounds=True,vehicle_control=False))
    # Pace close to camera's 40ms without asserting original acquisition timing.
    remaining=.04-(time.perf_counter()-started)
    if remaining>0:time.sleep(remaining)
   time.sleep(.02)
   if ticker.error:raise RuntimeError('ticker: '+str(ticker.error))
   result.update(status='archived_visual_live_body_finished',groups=len(events),archive_sha256=sha256(zip_path))
  finally:
   if ticker is not None:
    try:ticker.stop()
    except Exception as e:result.update(status='failed',ticker_stop_error=str(e))
    write_json(run/'ticker-events.json',ticker.events);result.update(ticker_error=ticker.error,ticker_events_sha256=sha256(run/'ticker-events.json'),shadow_outputs=sum(x['status']=='shadow_output' for x in ticker.events))
    if ticker.error:result['status']='failed'
   if sampler is not None:
    try:sampler.stop();write_json(run/'native-copy.json',sampler.rows);result.update(native_error=sampler.error,native_sha256=sha256(run/'native-copy.json'))
    except Exception as e:result.update(status='failed',sampler_stop_error=str(e))
   write_json(run/'visual-events.json',events);result['visual_events_sha256']=sha256(run/'visual-events.json')
  result.update(camera_request_sent=False,archived_pixels=True,synthetic_render_bounds=True,training_admissible=False,physical_exposure_sync_verified=False,interpretation='Archived four-view pixels and live native body with artificial host bounds; no real camera synchronization, qualification or control')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
