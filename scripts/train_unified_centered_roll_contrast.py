"""Synthetic relative roll-response training in one frozen flight graph; not driving."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flight_body_feedback_adapter import CenteredVehicleBodyFeedbackAdapter
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
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-centered-roll-contrast-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=12,actor=False,real_data_used_for_fit=False))
 result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  torch.set_num_threads(1);torch.manual_seed(863009)
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  spec=read(trdir/'manifest.json')['config']['plan'];source=read(Path(spec['source'])/'result.json')
  visual=EmbodiedVisualFlightCircuit(spec['source_spec'],torch.load(source['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu');visual.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True)['model'])
  mp=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];motor=PhysicalManeuverCircuit(mp['spec'],torch.load(mp['checkpoint'],map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu')
  temporal=OpponentTemporalMotionBoundary(visual.backbone.scene_command.mean,visual.backbone.scene_command.scale,len(visual.backbone.vision_ports));temporal.load_state_dict(torch.load(opdir/'temporal-boundary.pt',map_location='cpu',weights_only=True)['model'])
  boundary=PerViewVisualBodyContextBoundary(len(visual.backbone.vision_ports));boundary.load_state_dict(torch.load(ctxdir/'context-boundary.pt',map_location='cpu',weights_only=True)['model'])
  model=UnifiedFlightSensorimotor(motor,visual.backbone,temporal,boundary,body_feedback=CenteredVehicleBodyFeedbackAdapter(len(motor.backbone.body_ports)))
  model.motor.backbone.circuit.sparse_backend='cached_csr_gpu';model=model.to('cuda').eval().requires_grad_(False);weight=model.body_feedback.projection.weight;weight.requires_grad_(True)
  def scene(root):
   with np.load(root/'responses.npz',allow_pickle=False) as z:ret=z['retina'][0,20].copy();mot=z['motion'][0,20].copy()
   with torch.no_grad():return model.scene_pool(torch.tensor(np.concatenate([ret[:,None],mot],axis=1)[None],dtype=torch.float32,device='cuda'))
  train=scene(roots[0]);goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda');optim=torch.optim.Adam([weight],lr=.1)
  def rollout(visual_scene,rolls,learn,missing=False):
   outputs=[]
   for roll in rolls:
    model.reset_episode();values=np.zeros(21);values[1]=roll;values[16]=5.;valid=np.ones(21,dtype=bool)
    if missing:valid[1]=False
    for i in range(12):
     now=1.+i*.04;body=NavigationBodyObservation('synthetic-roll',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only')
     ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
     fn=model.sensory_training_decision if learn else model.sensory_decision
     intent,_=fn(decision_time=now,racing_goals=goals,scene_features=visual_scene,scene_valid=view,body_context=ctx)
    outputs.append(intent[0,0])
   return torch.stack(outputs)
  baseline=rollout(train,plan['train_roll_rad'],False).cpu().numpy();losses=[]
  for i in range(12):
   optim.zero_grad(set_to_none=True);out=rollout(train,plan['train_roll_rad'],True);diff=out[[0,2]]-out[1];target=torch.tensor([.004,-.004],device='cuda');loss=(diff-target).square().mean();loss.backward()
   if weight.grad is None or not torch.isfinite(weight.grad).all():raise ValueError('invalid gradient')
   weight.grad[:,14:]=0.;optim.step();losses.append(float(loss.detach().cpu()));result['optimizer_updates']=i+1;write_json(run/'progress.json',dict(update=i+1,loss=losses[-1]))
  after=rollout(train,plan['train_roll_rad'],False).cpu().numpy()
  validation=scene(roots[1]);val=rollout(validation,plan['validation_roll_rad'],False).cpu().numpy();missing=rollout(validation,plan['validation_roll_rad'],False,True).cpu().numpy()
  if not torch.equal(weight[:,14:],torch.zeros_like(weight[:,14:])):raise ValueError('availability weights changed')
  f=run/'centered-body-adapter.pt';torch.save(dict(schema='flybrain.experimental.centered_body_roll_contrast.v1',model=model.body_feedback.state_dict(),actor_qualified=False),f)
  out=run/'outputs.npz';np.savez_compressed(out,train_before=baseline,train_after=after,validation=val,missing=missing)
  result.update(status='completed_synthetic_centered_roll_contrast_training',training_loss_first=losses[0],training_loss_last=losses[-1],train_before=baseline.tolist(),train_after=after.tolist(),validation=val.tolist(),validation_relative=(val[[0,2]]-val[1]).tolist(),missing=missing.tolist(),missing_roll_spread=float(np.ptp(missing)),checkpoint_sha256=sha256(f),outputs_sha256=sha256(out),validation_used_for_fit=False,synthetic_hypothesis_not_calibrated=True,graph_frozen=True)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
