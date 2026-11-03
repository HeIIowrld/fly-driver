"""Offline replay of recorded unified shadow using reconstructed selected visual features."""
import argparse,json,copy,gc,os,sys,time,yaml
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter
from flybrain_drive.synthetic_acceleration_feedback import SyntheticAccelerationBodyFeedbackAdapter
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
from flybrain_drive.flight_airflow_feedback_adapter import AirflowFeedbackAdapter
from flybrain_drive.relative_airflow_sensor import missing_airflow,relative_airflow
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
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates_planned=2,actor=False,real_data_used_for_fit=False))
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
  model.body_feedback.load_state_dict(payload['model'],strict=True);old_body=model.body_feedback;model.body_feedback=SyntheticAccelerationBodyFeedbackAdapter(len(model.motor.backbone.body_ports)).to('cuda');missing,unexpected=model.body_feedback.load_state_dict(old_body.state_dict(),strict=False)
  if set(missing)!={'acceleration_scales','acceleration_projection.weight'} or unexpected:raise ValueError('base body checkpoint mismatch')
  model.body_feedback.requires_grad_(False);model.body_feedback.acceleration_projection.weight.requires_grad_(True);parameter=model.temporal.opponent_projection.weight;parameter.requires_grad_(False);initial=parameter.detach().clone()
  goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda')
  def load_scene(seed,amplitude,root):
   r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
   if r['status']!='completed_causal_four_view_constant_yaw' or sha256(f)!=r['responses_sha256'] or cfg['texture_seed']!=seed or cfg['program_sha256']!=plan['generator_sha256'] or r['cases']!=[0.,-amplitude,amplitude]:raise ValueError('declared causal synthetic stimulus required')
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
  sealed=Path('D:/CodexResearchRuns/flybrain-accel-decoupled-plan-20261102-01/plan.json');pplan=read(sealed)
  if pplan['program_sha256']!=sha256(__file__) or pplan['adapter_sha256']!=sha256('src/flybrain_drive/synthetic_acceleration_feedback.py') or pplan['unified_sha256']!=sha256('src/flybrain_drive/unified_flight_sensorimotor.py'):raise ValueError('sealed source mismatch')
  def source(item):
   path=B/item['folder']/'responses.npz'
   if sha256(path)!=item['response_sha256']:raise ValueError('synthetic source hash changed')
   return load_source(item['folder'],item['seed'],item['curvature'],item['bend_start_m'])[0]
  fit=[source(x) for x in pplan['fit_sources']];validation=[source(x) for x in pplan['validation_sources']]
  w=model.body_feedback.acceleration_projection.weight;before_base=model.body_feedback.projection.weight.detach().clone();opt=torch.optim.Adam([w],lr=.001)
  values_by_kind={'neutral':(0.,0.),'forward':(3.,0.),'braking':(-3.,0.),'left':(0.,3.),'right':(0.,-3.)}
  targets={'forward':(0.,-.002),'braking':(0.,.002),'left':(.003,0.),'right':(-.003,0.)}
  def trajectory(scene,kind,training,mask_accel=False):
   model.reset_episode();rows=[]
   for i in range(41):
    now=1.+i*.04;values=np.zeros(21);values[5:7]=values_by_kind[kind];values[8:12]=.25;values[16]=0. if i<=10 else 3.;values[17:21]=values[16]/.29;valid=np.ones(21,dtype=bool)
    if mask_accel:valid[5:8]=False
    body=NavigationBodyObservation('synthetic-accel',i,now-.005,now,values,valid,'synthetic-only');ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
    fn=model.sensory_training_decision if training else model.sensory_decision;intent,_=fn(decision_time=now,racing_goals=goals,scene_features=scene[i:i+1],scene_valid=view,body_context=ctx)
    if training and i<35:model.detach_training_state()
    if i>=36:rows.append(intent[0])
   return torch.stack(rows)
  baselines=[[trajectory(sc[b],'neutral',False).detach() for b in (1,2)] for sc in fit];losses=[];updates=0
  for update in range(pplan['updates']):
   opt.zero_grad();total=0.
   for k,sc in enumerate(fit):
    for j,b in enumerate((1,2)):
     for kind,(turn,forward) in targets.items():
      out=trajectory(sc[b],kind,True);delta=out-baselines[k][j];target=torch.tensor([turn,forward],device='cuda');loss=(((delta-target)/torch.tensor([.003,.002],device='cuda'))**2).mean()/(len(fit)*2*len(targets));loss.backward();total+=float(loss.detach())
   if w.grad is None or not torch.isfinite(w.grad).all():raise ValueError('finite gradient required')
   opt.step();updates+=1
   if not torch.equal(model.body_feedback.projection.weight,before_base):raise ValueError('base body weights changed')
   losses.append(total);write_json(run/'progress.json',dict(completed_updates=updates,total_updates=pplan['updates'],loss=total))
  out=run/'synthetic-accel-adapter.pt';torch.save(dict(schema='flybrain.synthetic.acceleration_adapter.v1',model=model.body_feedback.state_dict(),base_sha256=sha256(f),plan_sha256=sha256(sealed),synthetic_only=True,actor_qualified=False),out)
  reports=[];arr={};mask_max=0.
  for k,sc in enumerate(validation):
   for b in (1,2):
    base=trajectory(sc[b],'neutral',False).detach().cpu().numpy();masked=trajectory(sc[b],'forward',False,mask_accel=True).detach().cpu().numpy();mask_max=max(mask_max,float(np.max(np.abs(masked-base))))
    for kind,target in targets.items():
     val=trajectory(sc[b],kind,False).detach().cpu().numpy();delta=val-base;arr[f'{k}_{b}_{kind}']=delta;reports.append(dict(seed=pplan['validation_sources'][k]['seed'],bend=b,kind=kind,mean_delta=delta.mean(axis=0).tolist(),target=list(target)))
    write_json(run/'progress.json',dict(completed_updates=updates,total_updates=pplan['updates'],validation_scene=k,validation_bend=b))
  checks=[]
  for item in reports:
   target=np.array(item['target']);got=np.array(item['mean_delta']);idx=int(np.flatnonzero(target)[0]);other=1-idx;checks.append(bool(got[idx]*target[idx]>0 and abs(got[idx])>=.1*abs(target[idx]) and abs(got[other])<=.25*[.003,.002][other]))
  screen=bool(all(checks) and mask_max<=1e-6);z=run/'validation.npz';np.savez_compressed(z,**arr);result.update(status='completed_decoupled_acceleration_training',optimizer_updates=updates,losses=losses,validation=reports,predeclared_screen_passed=screen,conditions_passed=sum(checks),conditions_total=len(checks),masked_max_difference=mask_max,checkpoint_sha256=sha256(out),output_sha256=sha256(z),plan_sha256=sha256(sealed),actor_qualified=False,training_admissible=False,synthetic_only=True,interpretation='Only new masked acceleration projection trained with hypothetical stabilizing steering/forward targets on synthetic scenes; no physiological or real driving proof')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),optimizer_updates=locals().get('updates',0))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_decoupled_acceleration_training':raise SystemExit(1)
if __name__=='__main__':
 main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
