"""Factorize the frozen joint candidate on previously evaluated synthetic scenes."""
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
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),joint_plan_sha256=sha256(B/'20261101-joint-vision-yaw-new-plan/plan.json'),fit_protocol_sha256=sha256(B/'20261101-joint-vision-yaw-new-plan/fit-protocol.json'),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates_planned=0,actor=False,real_data_used_for_fit=False))
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
  newplanpath=B/'20261101-joint-vision-yaw-new-plan/plan.json';protocolpath=newplanpath.parent/'fit-protocol.json';newplan=read(newplanpath);protocol=read(protocolpath);source_root=B/'20261101-joint-vision-yaw-new-sources';generated=read(source_root/'result.json')
  if sha256(newplanpath)!='c771c163269730cdadbdd8204d851553e03c696685015120e70dbea3888aa650' or sha256(protocolpath)!='0de411ef5b86b1983939b861a632c1f30b634b4ce36ea3068521a7862286b1d8' or protocol['parent_plan_sha256']!=sha256(newplanpath) or generated['status']!='completed_disjoint_synthetic_sources' or generated['plan_sha256']!=sha256(newplanpath):raise ValueError('Exact predeclared plan and generated sources required')
  candidate_root=B/'20261101-joint-vision-yaw-synthetic-fit-v2';fit_result=read(candidate_root/'result.json');checkpoint=candidate_root/'candidate.pt'
  if fit_result['status']!='completed_synthetic_joint_vision_yaw_fit' or sha256(checkpoint)!=fit_result['checkpoint_sha256'] or fit_result['fit_plan_sha256']!=sha256(newplanpath) or fit_result['validation_accessed'] is not False:raise ValueError('Exact fit-only candidate required')
  trained=torch.load(checkpoint,map_location='cpu',weights_only=True)
  if trained['actor_qualified'] is not False or trained['source_pattern_sha256']!=sha256(candidate_file) or trained['fit_protocol_sha256']!=sha256(protocolpath):raise ValueError('Unqualified candidate lineage required')
  projection=model.spatial_contrast.projection.weight;reference=model.motor.backbone.readout.lateral_reference;original_projection=projection.detach().clone();original_reference=reference.detach().clone();validation=[]
  for entry in newplan['validation']:
   root=source_root/('validation-'+str(entry['seed']));rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];response=root/'responses.npz'
   if rr['status']!='completed_causal_four_view_synthetic_road' or sha256(response)!=rr['responses_sha256'] or cfg['texture_seed']!=entry['seed'] or cfg['curvature']!=entry['curvature'] or cfg['bend_start_m']!=entry['bend_start_m'] or cfg['program_sha256']!=newplan['generator_sha256']:raise ValueError('Exact separate validation source required')
   with np.load(response,allow_pickle=False) as z:retina=z['retina'].copy();motion=z['motion'].copy();times=z['times'].copy();bends=z['bend'].copy()
   if retina.shape!=(3,41,4,721) or motion.shape!=(3,41,4,8,721) or not np.array_equal(bends,[0,-1,1]) or not np.allclose(times,np.arange(41)*.04,atol=1e-12,rtol=0):raise ValueError('Original synthetic scene shape required')
   with torch.no_grad():scene=model.scene_pool(torch.tensor(np.concatenate((retina[:,:,:,None],motion),axis=3).reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,41,576)
   validation.append((entry,scene,times,sha256(response)))
  all_results=[];arrays=[]
  for entry,scene,times,source_hash in validation:
   models=[]
   for mode in ('baseline','projection_only','reference_only','candidate'):
    with torch.no_grad():
     projection.copy_(trained['spatial_projection'].to('cuda') if mode in ('projection_only','candidate') else original_projection);reference.copy_(trained['lateral_reference'].to('cuda') if mode in ('reference_only','candidate') else original_reference)
    results=[]
    for bend_index,bend in enumerate((0,-1,1)):
     yaw_rows=[]
     for yaw in newplan['body_yaw_rad_s']:
      model.reset_episode();trajectory=[]
      for i,t in enumerate(times):
       now=1.+float(t);speed=0. if t<=.4 else 3.;values=np.zeros(21);values[16]=speed;values[17:21]=speed/.29;values[4]=yaw if t>.4 else 0.;valid=np.ones(21,dtype=bool)
       body=NavigationBodyObservation('synthetic-val',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only');old=values.copy();old[4]=yaw if t-.03>.4 else 0.;prior=NavigationBodyObservation('synthetic-val',i,now-.05,now-.02,old,valid.copy(),'synthetic-only')
       ctx=per_view_temporal_body_context((prior,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
       intent,_=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=scene[bend_index,i:i+1],scene_valid=view,body_context=ctx);trajectory.append(intent[0].detach().cpu().numpy().copy())
      yaw_rows.append(np.stack(trajectory))
     results.append(np.stack(yaw_rows))
    models.append(np.stack(results))
   array=np.stack(models);arrays.append(array);targets=np.array([[.02*b-.10*y for y in newplan['body_yaw_rad_s']] for b in (0,-1,1)])
   scores={}
   for index,mode in enumerate(('baseline','projection_only','reference_only','candidate')):
    means=array[index,:,:,-5:,0].mean(axis=2);yaw_delta=means[:,2]-means[:,0]
    scores[mode]={'turn_means':means.tolist(),'target_hypothesis':targets.tolist(),'mean_squared_target_error':float(np.mean((means-targets)**2)),'yaw_positive_minus_negative_by_bend':yaw_delta.tolist(),'yaw_opposes_rotation_all_bends':bool(np.all(yaw_delta<0)),'zero_yaw_bend_minus_and_plus':means[1:,1].tolist(),'zero_yaw_bend_signs_match_hypothesis':bool(means[1,1]<0 and means[2,1]>0),'straight_zero_yaw_turn':float(means[0,1])}
   all_results.append({'seed':entry['seed'],'source_sha256':source_hash,'scores':scores})
  out=run/'validation-trajectories.npz';np.savez_compressed(out,trajectories=np.stack(arrays),times=times,bends=np.array([0,-1,1]),yaws=np.array(newplan['body_yaw_rad_s']));result.update(status='completed_synthetic_joint_factorization',validation=all_results,output_sha256=sha256(out),candidate_sha256=sha256(checkpoint),fit_plan_sha256=sha256(newplanpath),optimizer_updates=0,actor_qualified=False,real_data_used_for_fit=False,physical_or_physiology_verified=False,interpretation='Previously evaluated synthetic validation scenes, four fixed parameter combinations; no fitting or new validation claim. Targets are engineering hypotheses, not real-road, physiology or actor validation')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_synthetic_joint_factorization':raise SystemExit(1)
if __name__=='__main__':main()
