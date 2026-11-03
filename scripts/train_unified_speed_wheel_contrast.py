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
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-speed-wheel-contrast-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=20,actor=False,real_data_used_for_fit=False))
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
  source_fit=B/'20261101-centered-roll-contrast-fit';source_result=read(source_fit/'result.json');source_ck=source_fit/'centered-body-adapter.pt'
  if source_result['status']!='completed_synthetic_centered_roll_contrast_training' or sha256(source_ck)!=source_result['checkpoint_sha256']:raise ValueError('roll candidate provenance')
  payload=torch.load(source_ck,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False:raise ValueError('unqualified research checkpoint required')
  adapter=model.body_feedback;weight=adapter.projection.weight;original=payload['model'];old_indices=original['indices'].tolist();new_indices=adapter.indices.tolist()
  with torch.no_grad():
   for j,feature in enumerate(old_indices):
    k=new_indices.index(feature);weight[:,k].copy_(original['projection.weight'][:,j]);weight[:,len(new_indices)+k].copy_(original['projection.weight'][:,len(old_indices)+j])
  before_weights=weight.detach().clone();weight.requires_grad_(True)
  train_columns=[new_indices.index(i) for i in (16,17,18,19,20)]
  def scene(root):
   with np.load(root/'responses.npz',allow_pickle=False) as z:ret=z['retina'][0,20].copy();mot=z['motion'][0,20].copy()
   with torch.no_grad():return model.scene_pool(torch.tensor(np.concatenate([ret[:,None],mot],axis=1)[None],dtype=torch.float32,device='cuda'))
  train=scene(roots[0]);goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda');optim=torch.optim.Adam([weight],lr=.005)
  def rollout(visual_scene,speeds,radius,learn,missing_wheels=False,roll=0.):
   outputs=[]
   for speed in speeds:
    model.reset_episode();values=np.zeros(21);values[1]=roll;values[16]=speed;values[17:21]=speed/radius;valid=np.ones(21,dtype=bool)
    if missing_wheels:valid[17:21]=False
    for i in range(12):
     now=1.+i*.04;body=NavigationBodyObservation('synthetic-speed',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only')
     ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
     fn=model.sensory_training_decision if learn else model.sensory_decision
     intent,_=fn(decision_time=now,racing_goals=goals,scene_features=visual_scene,scene_valid=view,body_context=ctx)
    outputs.append(intent[0])
   return torch.stack(outputs)
  baseline=rollout(train,plan['training_speeds_m_s'],.33,False).cpu().numpy();losses=[]
  for i in range(20):
   optim.zero_grad(set_to_none=True);out=rollout(train,plan['training_speeds_m_s'],.33,True);diff=out[[0,2],1]-out[1,1];target=torch.tensor([.004,-.004],device='cuda');loss=(diff-target).square().mean();loss.backward()
   if weight.grad is None or not torch.isfinite(weight.grad).all():raise ValueError('invalid gradient')
   mask=torch.zeros_like(weight.grad);mask[:,train_columns]=1.;weight.grad.mul_(mask);optim.step();losses.append(float(loss.detach().cpu()));result['optimizer_updates']=i+1;write_json(run/'progress.json',dict(update=i+1,loss=losses[-1]))
  keep=[i for i in range(weight.shape[1]) if i not in train_columns]
  if not torch.equal(weight[:,keep],before_weights[:,keep]):raise ValueError('frozen columns changed')
  after=rollout(train,plan['training_speeds_m_s'],.33,False).cpu().numpy();validation=scene(roots[1]);val=rollout(validation,plan['validation_speeds_m_s'],.29,False).cpu().numpy();missing=rollout(validation,plan['validation_speeds_m_s'],.29,False,True).cpu().numpy();roll=rollout(validation,[5.,5.,5.],.29,False,False,.025).cpu().numpy()
  ck=run/'speed-body-adapter.pt';torch.save(dict(schema='flybrain.experimental.centered_speed_wheel_contrast.v1',model=adapter.state_dict(),actor_qualified=False),ck)
  out=run/'outputs.npz';np.savez_compressed(out,train_before=baseline,train_after=after,validation=val,validation_wheels_missing=missing,roll_probe=roll)
  result.update(status='completed_synthetic_speed_wheel_contrast_training',training_loss_first=losses[0],training_loss_last=losses[-1],training_loss_history=losses,train_before=baseline.tolist(),train_after=after.tolist(),validation=val.tolist(),validation_relative_power=(val[[0,2],1]-val[1,1]).tolist(),validation_wheels_missing=missing.tolist(),checkpoint_sha256=sha256(ck),outputs_sha256=sha256(out),roll_source_sha256=sha256(source_ck),validation_used_for_fit=False,synthetic_hypothesis_not_calibrated=True,graph_frozen=True,frozen_columns_unchanged=True)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
