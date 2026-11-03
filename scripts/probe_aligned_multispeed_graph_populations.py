"""Fit only existing visual boundary and readout reference on predeclared synthetic roads."""
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
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
from flybrain_drive.flight_stream import held_input_batches
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
from flybrain_drive.view_yaw_interval import ViewYawInterval
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
 experiment_plan=B/'20261101-paired-bend-fit-plan/plan.json'; experiment=read(experiment_plan)
 if experiment['schema']!='flybrain.diagnostic.paired-bend-plan.v1' or experiment['real_data_used_for_fit'] is not False or experiment['actor_qualified'] is not False:raise ValueError('exact synthetic-only protocol required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),experiment_plan_sha256=sha256(experiment_plan),graph_sha256=sha256(tr['checkpoint']),optimizer_updates_planned=0,actor=False,real_data_used_for_fit=False))
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
  model.motor.backbone.circuit.sparse_backend='cached_csr_gpu';model=model.to('cuda').eval().requires_grad_(False)
  fit=B/'20261101-speed-wheel-turn-control-fit';fr=read(fit/'result.json');f=fit/'speed-body-adapter.pt'
  if fr['status']!='completed_synthetic_speed_wheel_turn_control_training' or sha256(f)!=fr['checkpoint_sha256']:raise ValueError('body checkpoint')
  payload=torch.load(f,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('body qualification')
  model.body_feedback.load_state_dict(payload['model'],strict=True);parameter=model.temporal.opponent_projection.weight;parameter.requires_grad_(False);initial=parameter.detach().clone()
  goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda')
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
  with torch.no_grad():model.temporal.opponent_projection.weight.copy_(candidate['weight'].to('cuda'))
  model.spatial_contrast=RetinalSpatialContrastBoundary(len(model.motor.backbone.vision_ports)).to('cuda')
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
   with torch.no_grad():scene=model.scene_pool(torch.tensor(image.reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,41,576)
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
  def load_combined(path,expected):
   path=Path(path);r=read(path.parent/'result.json');cfg=read(path.parent/'manifest.json')['config']
   if r['status']!='completed_causal_synthetic_road_with_yaw' or sha256(path)!=expected or sha256(path)!=r['responses_sha256'] or cfg['road_generator_sha256']!=sha256('src/flybrain_drive/synthetic_yaw_road_visual.py'):raise ValueError('verified synthetic source required')
   with np.load(path,allow_pickle=False) as z:ret=z['retina'].copy();motion=z['motion'].copy();times=z['times'].copy();bends=z['bend'].copy();yaws=z['yaw_rate'].copy()
   if ret.shape!=(3,3,41,4,721) or motion.shape!=(3,3,41,4,8,721) or not np.array_equal(bends,[0,-1,1]) or not np.allclose(yaws,[0.,-.12,.12]) or not np.isfinite(ret).all() or not np.isfinite(motion).all():raise ValueError('complete source grid required')
   full=np.concatenate((ret[:,:,:,:,None],motion),axis=4)
   with torch.no_grad():sc=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,3,41,576)
   return sc,times,bends,yaws
  probeplan=B/'20261101-flight-state-corner-probe-plan/plan.json';ppn=read(probeplan);source_model=B/'20261101-synthetic-state-speed-1/state-speed-candidate.pt';sr=read(source_model.parent/'result.json');power=B/'20261101-synthetic-power-reference-closed-loop-1/power-reference-candidate.pt'
  if sha256(source_model)!=sr['candidate_sha256'] or sha256(power)!=sr['power_reference_sha256']:raise ValueError('Exact fixed synthetic candidates required')
  cp=torch.load(source_model,map_location='cpu',weights_only=True);pp=torch.load(power,map_location='cpu',weights_only=True)
  if cp['actor_qualified'] is not False or pp['actor_qualified'] is not False:raise ValueError('Diagnostic candidates only')
  model.speed_command_boundary=FlightStateSpeedCommand(6).to('cuda');model.speed_command_boundary.load_state_dict(cp['model'],strict=True)
  with torch.no_grad():model.motor.backbone.readout.power_reference.copy_(pp['power_reference'].to('cuda'))
  fs=read(mp['spec']);graph_file=Path(fs['graph_file'])
  if sha256(graph_file)!=fs['graph_sha256']:raise ValueError('Exact flight graph required')
  with np.load(graph_file,allow_pickle=False) as g:
   types=g['types'].copy();roles=g['roles'].copy();ids=g['ids'].copy()
  if len(ids)!=model.motor.backbone.neurons:raise ValueError('Graph ordering dimensions required')
  groups={k:np.flatnonzero(np.char.startswith(types,k)) for k in ('HS','VS','DNg02','DNa04','DNa05')};groups['wing_outputs']=np.flatnonzero(roles=='output')
  if [len(groups[k]) for k in ('HS','VS','DNg02','DNa04','DNa05','wing_outputs')]!=[8,34,29,2,2,67]:raise ValueError('Expected named graph populations required')
  groups['muscles6']=None;spatial=model.spatial_contrast.projection.weight.detach().clone()
  def source(seed,validation=False):
   root=B/f'20261101-synthetic-road-yaw-{seed}';rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];file=root/'responses.npz'
   if cfg['texture_seed']!=seed or (validation and (cfg['curvature']!=ppn['validation_curvature'] or cfg['bend_start_m']!=ppn['validation_bend_start_m'])):raise ValueError('Declared synthetic source required')
   sc,t,_,_=load_combined(file,rr['responses_sha256']);return sc,t,sha256(file)
  def extract(sc,t,body_speed):
   output={k:[] for k in groups};labels=[]
   for b in range(3):
    model.reset_episode();per={k:[] for k in groups};prior=None
    with torch.no_grad():
     for i,ts in enumerate(t):
      now=1.+float(ts);v=np.zeros(21);v[16]=body_speed if ts>.4 else 0.;v[17:21]=v[16]/.29;valid=np.zeros(21,dtype=bool);valid[[4,16,17,18,19,20]]=True;body=NavigationBodyObservation('synthetic-probe',i,now,now,v.copy(),valid.copy(),'synthetic-only');history=(None,)*4 if prior is None else (NavigationBodyObservation('synthetic-probe',i-1,now-.04,now,prior.copy(),valid.copy(),'synthetic-only'),)*4
      ctx=per_view_temporal_body_context(history,body,render_start_lowers=(now,)*4,camera_receipt_time=now,decision_time=now);feature=sc[b,0,i:i+1];model.experimental_recurrent_speed_decision(decision_time=now,racing_goals=goals,scene_features=feature,scene_valid=view,body_context=ctx)
      if i>=31:
       state=model.state.detach().cpu().numpy()[0];muscles=model.motor.backbone._read_state_tensor(model.state)[1].detach().cpu().numpy()[0]
       for k,indices in groups.items():per[k].append(muscles.copy() if indices is None else state[indices].copy())
      prior=v
    for k in groups:output[k].extend(per[k])
    labels.extend([1 if b==0 else -1]*10)
   return {k:np.stack(v) for k,v in output.items()},np.asarray(labels)
  speeds=(1.5,3.,4.5);sets={};hashes={}
  for role in ('train','evaluation'):
   for speed in speeds:
    folder=(f'20261101-synthetic-matched-speed-{speed:g}' if speed!=3. else '20261101-synthetic-road-yaw-881403') if role=='train' else f'20261101-multispeed-eval-881511-{speed:g}'
    root=B/folder;rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];file=root/'responses.npz'
    if cfg['texture_seed']!=(881403 if role=='train' else 881511) or cfg['curvature']!=.022 or cfg['bend_start_m']!=11. or (speed!=3. and cfg['speed_m_s']!=speed):raise ValueError('Exact declared source geometry required')
    sc,t,_,_=load_combined(file,rr['responses_sha256'])
    with torch.no_grad():model.spatial_contrast.projection.weight.copy_(spatial)
    sets[(role,speed)],labels=extract(sc,t,speed);hashes[f'{role}_{speed:g}']=sha256(file)
    if role=='evaluation':
     with torch.no_grad():model.spatial_contrast.projection.weight.zero_()
     sets[('spatial_off',speed)],off_labels=extract(sc,t,speed)
     if not np.array_equal(labels,off_labels):raise ValueError('Aligned intervention labels required')
    write_json(run/'progress.json',dict(role=role,speed=speed))
  targets=np.tile(np.repeat([3.5,2.5,2.5],10),3);findings={};saved={}
  for name in ('VS','DNg02','wing_outputs','muscles6'):
   train=np.vstack([sets[('train',v)][name] for v in speeds]).astype(np.float64);evaluation=np.vstack([sets[('evaluation',v)][name] for v in speeds]).astype(np.float64)
   mu=train.mean(0);scale=np.maximum(train.std(0),1e-3);speed_feature=np.repeat(speeds,30)[:,None]/30.
   X=np.column_stack(((train-mu)/scale,speed_feature,np.ones(90)));E=np.column_stack(((evaluation-mu)/scale,speed_feature,np.ones(90)))
   off=np.vstack([sets[('spatial_off',v)][name] for v in speeds]).astype(np.float64);O=np.column_stack(((off-mu)/scale,speed_feature,np.ones(90)))
   reg=np.eye(X.shape[1]);reg[-1,-1]=0;w=np.linalg.solve(X.T@X+reg,X.T@targets);raw=E@w;pred=np.clip(raw,0.,5.);offpred=np.clip(O@w,0.,5.);rows=[]
   for j,v in enumerate(speeds):
    a=pred[j*30:(j+1)*30].reshape(3,10).mean(1);offmeans=offpred[j*30:(j+1)*30].reshape(3,10).mean(1);rows.append(dict(speed_m_s=v,means=a.tolist(),straight_minus_bend=float(a[0]-a[1:].mean()),worst_margin=float(min(a[0]-a[1],a[0]-a[2])),spatial_off_means=offmeans.tolist(),spatial_off_straight_minus_bend=float(offmeans[0]-offmeans[1:].mean())))
   findings[name]=dict(train_rms=float(np.sqrt(np.mean((X@w-targets)**2))),evaluation_rms=float(np.sqrt(np.mean((pred-targets)**2))),evaluation=rows,raw_weight_norm=float(np.linalg.norm(w[:-1])),floored_features=int(np.count_nonzero(train.std(0)<1e-3)),max_raw_perturbation_per_1e_6=float(np.max(np.abs(w[:len(mu)]/scale))*1e-6),clipped_frames=int(np.count_nonzero(raw!=pred)))
   saved[name+'_train']=train;saved[name+'_evaluation']=evaluation;saved[name+'_spatial_off']=off;saved[name+'_weights']=w;saved[name+'_mean']=mu;saved[name+'_scale']=scale
  out=run/'population-states.npz';np.savez_compressed(out,**saved)
  result.update(status='completed_synthetic_aligned_multispeed_population_diagnostic',conditions=findings,source_hashes=hashes,output_sha256=sha256(out),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,validation_previously_examined=True,interpretation='Post-hoc synthetic fixed-state ridge diagnostics on one training and one evaluation texture, correlated frames and single geometry, spatial removal counterfactual; no actor, physiology or closed-loop proof')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),optimizer_updates=0)
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_synthetic_aligned_multispeed_population_diagnostic':raise SystemExit(1)
if __name__=='__main__':main()
