"""Replay controlled synthetic yaw in unified circuit, no optimization."""
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
 experiment_plan=Path('D:/CodexResearchRuns/flybrain-spatial-bounded-eval-plan-20261102-02/plan.json'); experiment=read(experiment_plan)
 if experiment['schema']!='flybrain.diagnostic.combined-road-yaw-plan.v1' or experiment['real_data_used_for_fit'] is not False or experiment['actor_qualified'] is not False or experiment['program_sha256']!=sha256(__file__) or experiment['fit_seed']==experiment['eval_seed']:raise ValueError('exact synthetic-only held-out protocol required')
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
   if r['status']!='completed_causal_four_view_constant_yaw' or sha256(f)!=r['responses_sha256'] or cfg['texture_seed']!=seed or cfg['program_sha256']!=experiment['yaw_generator_sha256'] or r['cases']!=[0.,-amplitude,amplitude]:raise ValueError('declared causal synthetic stimulus required')
   with np.load(f,allow_pickle=False) as z:ret=z['retina'].copy();mot=z['motion'].copy();times=z['times'].copy()
   if ret.shape!=(3,41,4,721) or mot.shape!=(3,41,4,8,721) or not np.isfinite(ret).all() or not np.isfinite(mot).all() or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12) or not np.all(ret[:,:11]==ret[:,:1]):raise ValueError('synthetic stimulus shape')
   full=np.concatenate([ret[:,:,:,None],mot],axis=3)
   with torch.no_grad():sc=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,41,576)
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
  from flybrain_drive.flight_pedal_learning_v2 import bind_registered_reference
  from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
  binding=bind_registered_reference(Path(mp['cpu_reference']).read_bytes(),Path(mp['cpu_readout']).read_bytes());rd=model.motor.backbone.readout
  if not torch.equal(rd.turn_log_gain.cpu(),torch.as_tensor(binding.readout_state['turn_log_gain']).cpu()) or not torch.equal(rd.steering_mix.cpu(),torch.as_tensor(binding.readout_state['steering_mix']).cpu()):raise ValueError('reference lineage')
  with torch.no_grad():rd.lateral_reference.copy_(torch.as_tensor(binding.readout_state['lateral_reference'],device='cuda'))
  oldfile=B/'20261101-synthetic-state-speed-1/state-speed-candidate.pt';power=B/'20261101-synthetic-power-reference-closed-loop-1/power-reference-candidate.pt';old=torch.load(oldfile,map_location='cpu',weights_only=True);pp=torch.load(power,map_location='cpu',weights_only=True)
  if sha256(oldfile)!=experiment['speed_sha256'] or sha256(power)!=experiment['power_sha256'] or old['actor_qualified'] is not False or pp['actor_qualified'] is not False:raise ValueError('fixed control lineage')
  model.speed_command_boundary=FlightStateSpeedCommand(6).to('cuda');model.speed_command_boundary.load_state_dict(old['model'],strict=True);model.speed_command_boundary.eval().requires_grad_(False)
  with torch.no_grad():rd.power_reference.copy_(pp['power_reference'].to('cuda'))
  cp=Path(experiment['candidate']);cr=read(cp.parent/'result.json')
  if cr['status']!='completed_bounded_synthetic_projection' or sha256(cp)!=experiment['candidate_sha256'] or sha256(cp)!=cr['checkpoint_sha256']:raise ValueError('candidate provenance')
  payload=torch.load(cp,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False or payload['source_weight_sha256']!=sha256(candidate_file):raise ValueError('candidate lineage')
  base_weight=weight.detach().clone();new_weight=payload['spatial_projection'].to('cuda');arrays={};rows=[]
  def runseq(scene,values,valid):
   model.reset_episode();outputs=[];previous=None
   with torch.no_grad():
    for i in range(len(scene)):
     now=1.+i*.04;v=values[i];body=NavigationBodyObservation('bounded-spatial-eval',i,now,now,v.copy(),valid.copy(),'synthetic-only');history=(None,)*4 if previous is None else (NavigationBodyObservation('bounded-spatial-eval',i-1,now-.04,now,previous.copy(),valid.copy(),'synthetic-only'),)*4;ctx=per_view_temporal_body_context(history,body,render_start_lowers=(now,)*4,camera_receipt_time=now,decision_time=now)
     intent,_=model.experimental_recurrent_speed_decision(decision_time=now,racing_goals=goals,scene_features=scene[i:i+1],scene_valid=view,body_context=ctx);outputs.append(intent[0].cpu().numpy());previous=v.copy()
   return np.stack(outputs)
  for folder,seed,amp in experiment['yaw_sources']:
   sc,times,digest=load_scene(seed,amp,B/folder)
   if digest!=experiment['yaw_source_sha256'][folder]:raise ValueError('source mismatch')
   valid=np.zeros(21,dtype=bool);valid[[4,16,17,18,19,20]]=True;values=np.zeros((41,21));values[:,16]=5.;values[:,17:21]=5./.29
   for mode in ('baseline','candidate'):
    with torch.no_grad():weight.copy_(base_weight if mode=='baseline' else new_weight)
    for case in range(3):arrays[f'yaw_{seed}_{amp}_{mode}_{case}']=runseq(sc[case],values,valid)
    a=np.stack([arrays[f'yaw_{seed}_{amp}_{mode}_{case}'] for case in range(3)]);last=a[:,-5:,0].mean(axis=1);rows.append(dict(kind='yaw',seed=seed,amplitude=amp,mode=mode,negative_minus_positive=float(last[1]-last[2]),negative_minus_zero=float(last[1]-last[0]),positive_minus_zero=float(last[2]-last[0])))
   write_json(run/'progress.json',dict(kind='yaw',seed=seed,amplitude=amp))
  sp=Path(experiment['road_scenes']);tp=Path(experiment['road_trajectory'])
  if sha256(sp)!=experiment['scene_sha256'] or sha256(tp)!=experiment['trajectory_sha256']:raise ValueError('road lineage')
  with np.load(sp,allow_pickle=False) as z:scenes={k:z[k].copy() for k in z.files}
  with np.load(tp,allow_pickle=False) as z:trajectories={k:z[k].copy() for k in z.files}
  for seed in experiment['road_eval_seeds']:
   for bend in (-1,0,1):
    k=f'baseline_{seed}_{bend}';a=trajectories[k];sc=torch.tensor(scenes[k],dtype=torch.float32,device='cuda');values=np.zeros((len(a),21));values[:,4]=a[:,5];values[:,16]=a[:,4];values[:,17:21]=a[:,4,None]/.29;valid=np.zeros(21,dtype=bool);valid[[4,16,17,18,19,20]]=True
    for mode in ('baseline','candidate'):
     with torch.no_grad():weight.copy_(base_weight if mode=='baseline' else new_weight)
     arrays[f'road_{seed}_{bend}_{mode}']=runseq(sc,values,valid)
    for win,lo,hi in (('before',9.,11.5),('after',13.,16.)):
     mask=(a[:,1]>=lo)&(a[:,1]<hi);d=arrays[f'road_{seed}_{bend}_candidate'][mask,0]-arrays[f'road_{seed}_{bend}_baseline'][mask,0];rows.append(dict(kind='road',seed=seed,bend=bend,window=win,mean_abs_turn_change=float(np.mean(np.abs(d))),max_abs_turn_change=float(np.max(np.abs(d)))))
    write_json(run/'progress.json',dict(kind='road',seed=seed,bend=bend))
  out=run/'outputs.npz';np.savez_compressed(out,**arrays);result.update(status='completed_bounded_spatial_recurrent_eval',rows=rows,output_sha256=sha256(out),candidate_sha256=sha256(cp),optimizer_updates=0,actor_qualified=False,training_admissible=False,physical_time_verified=False,interpretation='Synthetic previously examined textures; yaw body mismatch, fixed-history road not closed loop or physiological validation.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),optimizer_updates=0)
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_bounded_spatial_recurrent_eval':raise SystemExit(1)
if __name__=='__main__':main()
