"""Offline replay of recorded unified shadow using reconstructed selected visual features."""
import argparse,json,copy,gc,os,sys,time,yaml
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
from flybrain_drive.ac_native_physics import decode_static_identity_prefix, PHYSICS_PREFIX
from flybrain_drive.virtual_navigation_body import decode_navigation_body_prefix
from scripts.replay_process_shadow_live_original import body_from_saved
import base64
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.immutable_camera_group import freeze_camera_group
from flybrain_drive.online_body_shadow import OnlineBodyShadow
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.unified_multirate_shadow import UnifiedMultirateShadow
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
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=0,actor=False,real_data_used_for_fit=False))
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
  src=Path('D:/CodexResearchRuns/flybrain-body-owner-live-20261102-06');live=read(src/'result.json');manifest=read(src/'manifest.json');recon=Path('D:/CodexResearchRuns/flybrain-body-owner-visual-reconstruction-20261102-01');rr=read(recon/'result.json');rf=recon/'visual-responses.npz'
  if live['status']!='observation_finished' or rr['status']!='completed_original_visual_reconstruction' or rr['source_run_sha256']!=sha256(src/'result.json') or read(recon/'exit-receipt.json').get('exitcode')!=0 or read(recon/'exit-receipt.json').get('result_sha256')!=sha256(recon/'result.json') or sha256(rf)!=rr['responses_sha256']:raise ValueError('Original completed run and visual reconstruction required')
  for filename,expected_hash in [('events.json',live['events_sha256']),('body-owner/ticker-events.json',live['worker_terminal']['events_sha256']),('body-owner/native-copy.json',live['worker_terminal']['native_sha256'])]:
   if sha256(src/filename)!=expected_hash:raise ValueError('Original recording hash differs')
  for path in ['src/flybrain_drive/unified_flight_sensorimotor.py','src/flybrain_drive/vehicle_circuit.py','src/flybrain_drive/parallel_csr_cpu.py','src/flybrain_drive/retinal_spatial_contrast_boundary.py','src/flybrain_drive/unified_multirate_shadow.py','src/flybrain_drive/flight_body_feedback_adapter.py','src/flybrain_drive/flight_opponent_temporal_boundary.py','src/flybrain_drive/flight_visual_body_temporal_context.py','src/flybrain_drive/virtual_navigation_body.py']:
   if manifest['source_sha256'].get(path)!=sha256(path):raise ValueError('Live source implementation differs: '+path)
  rows=read(src/'events.json');ticks=read(src/'body-owner/ticker-events.json');native=read(src/'body-owner/native-copy.json');published={x['capture_id']:x for x in rows if x['status']=='scene_published'}
  packets={}
  for row in native:
   auth=row['authority'];packets.setdefault(auth['packet_id'],row)
  with np.load(rf,allow_pickle=False) as z:ids=z['capture_ids'].copy();retina=z['retina'].copy();motion=z['motion'].copy()
  if set(map(int,ids))!=set(published) or retina.shape!=(len(ids),4,721) or motion.shape!=(len(ids),4,8,721):raise ValueError('All original selected scenes required')
  with torch.no_grad():
   pool=copy.deepcopy(model.scene_pool).to('cuda');image=torch.as_tensor(np.concatenate((retina[:,:,None,:],motion),axis=2),device='cuda');scene_values=pool(image).cpu().numpy()
  del pool,image;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  scenes={int(k):scene_values[i:i+1] for i,k in enumerate(ids)}
  identity=read('configs/ac-virtual-body-v1.json')['native_identity'];encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity);episode=live['camera_request_id'];contract=sha256('configs/ac-virtual-navigation-body-v2.json')
  class IdealClock:
   now=0.
   def __call__(self):return self.now
  model.reset_episode();clock=IdealClock();shadow=UnifiedMultirateShadow(UnifiedHostGuard(model,clock=clock),goals=goals);diffs=[];recorded=[];replayed=[];priming=0;last_capture=None
  for index,tick in enumerate(ticks):
   row=packets[tick['packet']];auth=row['authority'];receipt=row['copy']['second_copy_end'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
   if receipt!=tick['body_receipt'] or receipt!=auth['received_monotonic_seconds']:raise ValueError('Original first packet receipt mismatch')
   when=float(tick['decision']);sample=decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size],receipt);body=encoder.encode(sample,episode_id=episode,decision_monotonic_seconds=when);clock.now=when
   if tick['visual_replaced']:
    event=published[tick['capture_id']];previous=tuple(body_from_saved(x,episode=episode,decision=event['camera_received'],contract=contract) for x in event['view_body']);frame_body=body_from_saved(event['decision_body'],episode=episode,decision=event['decision'],contract=contract)
    frame=per_view_temporal_body_context(previous,frame_body,render_start_lowers=tuple(event['render_start_lower']),camera_receipt_time=event['camera_received'],decision_time=event['decision'])
    output=shadow.update(when=when,body=body,new_scene=torch.as_tensor(scenes[tick['capture_id']]),new_context=frame,published_at=tick['publication']);last_capture=tick['capture_id']
   else:
    if last_capture!=tick['capture_id']:raise ValueError('Held original capture changed')
    output=shadow.update(when=when,body=body)
   if tick['status']=='priming':
    if output is not None:raise ValueError('Original priming mismatch')
    priming+=1
   elif tick['status']=='shadow_output':
    if output is None:raise ValueError('Original output missing')
    actual=output.intent.detach().cpu().numpy()[0];reference=np.asarray(tick['intent']);recorded.append(reference);replayed.append(actual);diffs.append(float(np.max(np.abs(actual-reference))))
   else:raise ValueError('Unexpected recorded ticker event')
  if len(diffs)!=live['published'] or priming!=1:raise ValueError('Output accounting mismatch')
  out=run/'replay.npz';np.savez_compressed(out,recorded=np.stack(recorded),replayed=np.stack(replayed),max_error=np.array(diffs));max_error=max(diffs)
  result.update(status='completed' if max_error<=2e-5 else 'output_mismatch',decisions=len(ticks),priming=priming,outputs=len(diffs),max_intent_abs_difference=max_error,output_sha256=sha256(out),visual_reconstruction_sha256=sha256(rf),source_result_sha256=sha256(src/'result.json'),training_admissible=False,physical_exposure_sync_verified=False,interpretation='Recorded selected RGB, body and decision clocks replayed with ideal zero-compute guard clock; not live latency, physiology or driving')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
