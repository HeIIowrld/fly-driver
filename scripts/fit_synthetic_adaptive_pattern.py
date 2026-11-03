"""Synthetic speed and wheel relative response training in one frozen flight graph; not driving."""
import argparse,json
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
from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
from flybrain_drive.racing_goal import racing_goal
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
  diagnostic_planpath=B/'20261101-road-signal-decodability-plan/plan.json';diagnostic_plan=read(diagnostic_planpath);camera=read('configs/ac-csp-exterior-front-v3-retina.json')
  if sha256('src/flybrain_drive/synthetic_road_visual.py')!=diagnostic_plan['road_source_sha256'] or sha256('configs/ac-csp-exterior-front-v3-retina.json')!=diagnostic_plan['camera_sha256'] or sha256('src/flybrain_drive/flyvis_camera.py')!=diagnostic_plan['projector_sha256']:raise ValueError('training image source mismatch')
  uv=np.array([(q,r) for q in range(-15,16) for r in range(max(-15,-15-q),min(15,15-q)+1)]);projector=FlyvisCameraProjector(camera,*uv.T);pool=model.scene_pool.weight.detach().cpu().numpy().astype(np.float64);X=[];labels=[]
  for i,seed in enumerate(diagnostic_plan['train_seeds']):
   curvature,start=diagnostic_plan['train_geometry'][i%len(diagnostic_plan['train_geometry'])];road=SyntheticRoadVisual(camera,curvature=curvature,bend_start_m=start)
   for bend in diagnostic_plan['bend_labels']:
    rgb=road.render(bend=bend,forward_m=diagnostic_plan['position_m'],texture_seed=seed);ret=np.stack([projector.sample(v) for v in rgb]).astype(np.float64);pooled=np.tanh(ret)@pool.T;X.append(pooled-pooled.mean(-1,keepdims=True));labels.append(bend)
  X=np.stack(X);y=np.array(labels,dtype=np.float64);ref=X.mean(0);Z=(X-ref).reshape(len(X),-1);K=Z@Z.T;ridge=.01*float(np.trace(K)/len(K));pattern=Z.T@np.linalg.solve(K+ridge*np.eye(len(K)),y);pattern_norm=float(np.linalg.norm(pattern))
  if pattern_norm<1e-15:raise ValueError('degenerate training pattern')
  pattern/=pattern_norm;model.spatial_contrast.reference.copy_(torch.tensor(ref,device='cuda',dtype=weight.dtype));contrast_limit=float(np.sqrt(np.mean((X-ref)**2)));model.spatial_contrast.max_contrast_rms.fill_(contrast_limit)
  if not np.isfinite(contrast_limit) or contrast_limit<=0:raise ValueError('synthetic limit')
  sensitivity_root=B/'20261101-visual-port-sensitivity';sensitivity=read(sensitivity_root/'result.json');sensitivity_file=sensitivity_root/'port-gradient.npz'
  if sensitivity['status']!='completed_synthetic_visual_port_sensitivity' or sha256(sensitivity_file)!=sensitivity['output_sha256']:raise ValueError('port calibration provenance')
  with np.load(sensitivity_file,allow_pickle=False) as z:
   if not np.array_equal(z['port_ids'],model.motor.backbone.vision_ports.cpu().numpy()):raise ValueError('port ids')
   direction=z['gradient'].astype(np.float64)
  direction/=np.linalg.norm(direction);unit=torch.tensor(np.outer(direction,pattern),device='cuda',dtype=weight.dtype)
  data=[load_source(*x) for x in plan['fit_sources']];gains=[0.,1.,10.,100.,1000.];trials=[]
  for gain in gains:
   with torch.no_grad():weight.copy_(unit*gain)
   vals=[rollout(sc,t,False).cpu().numpy() for sc,t,_ in data];loss=0.
   for arr in vals:
    m=arr[:,-10:,0].mean(1);loss+=float((m[2]-m[0]-.002)**2+(m[1]-m[0]+.002)**2)
   trials.append(dict(gain=gain,training_loss=loss))
  best=int(np.argmin([v['training_loss'] for v in trials]));final=unit*gains[best];rows=[];valid=[]
  prior_root=B/'20261101-pattern-port-fit';prior_result=read(prior_root/'result.json');prior_file=prior_root/'pattern-contrast.pt'
  if prior_result['status']!='completed_synthetic_pattern_port_fit' or sha256(prior_file)!=prior_result['checkpoint_sha256']:raise ValueError('previous candidate provenance')
  prior=torch.load(prior_file,map_location='cuda',weights_only=True)
  if prior['actor_qualified'] is not False or not torch.allclose(prior['reference'],model.spatial_contrast.reference,atol=1e-7):raise ValueError('previous candidate reference')
  for source in plan['validation_sources']:
   sc,t,h=load_source(*source);vals=[]
   for w,limit in ((initial_weight,0.),(prior['weight'],0.),(final,contrast_limit)):
    with torch.no_grad():weight.copy_(w);model.spatial_contrast.max_contrast_rms.fill_(limit)
    vals.append(rollout(sc,t,False).cpu().numpy())
   vals=np.stack(vals);valid.append(vals);means=vals[:,:,-10:,0].mean(2)
   rows.append(dict(seed=source[1],source_sha256=h,labels=['boundary_ablated','previous_unlimited','adaptive_fitted'],left_minus_right=(means[:,2]-means[:,1]).tolist(),mean_turn=means.tolist()))
  dest=run/'adaptive-pattern-contrast.pt';torch.save(dict(schema='flybrain.experimental.adaptive_pattern_contrast.v1',weight=final.cpu(),reference=model.spatial_contrast.reference.cpu(),max_contrast_rms=contrast_limit,actor_qualified=False,body_sha256=sha256(f),temporal_sha256=sha256(ck)),dest)
  output=run/'outputs.npz';np.savez_compressed(output,training_images=X,training_labels=y,reference=ref,pattern=pattern,port_direction=direction,validation=np.stack(valid),times=t)
  result.update(status='completed_synthetic_adaptive_pattern_fit',contrast_limit=contrast_limit,previous_candidate_sha256=sha256(prior_file),parameter_fit=True,optimizer_updates=0,teacher_training_images=len(y),teacher_training_accuracy=float(np.mean(np.sign(Z@pattern)==y)),teacher_plan_sha256=sha256(diagnostic_planpath),calibration_sha256=sha256(sensitivity_file),training_source_sha256=[x[2] for x in data],trials=trials,selected_gain=gains[best],validation=rows,validation_used_for_fit=False,checkpoint_sha256=sha256(dest),outputs_sha256=sha256(output),graph_body_temporal_visual_weights_frozen=True,road_truth_actor_input=False,physical_or_physiology_verified=False)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
