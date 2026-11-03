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
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-causal-yaw-multispeed-plan/plan.json';plan=read(planpath)
 trdir=B/'20261029-t1437-embodied-joint-training';opdir=B/'20261031-opponent-temporal-graph-trained';ctxdir=B/'20261031-historical-train00-temporal-context-onepass';tr,op,cr=[read(x/'result.json') for x in (trdir,opdir,ctxdir)]
 if sha256(tr['checkpoint'])!=op['source_checkpoint_sha256'] or sha256(opdir/'temporal-boundary.pt')!=op['checkpoint_sha256'] or sha256(ctxdir/'context-boundary.pt')!=cr['checkpoint_sha256']:raise ValueError('fixed graph lineage')
 roots=[B/f'20261101-flyvis-varied-causal-{seed}' for seed in (879511,879512)];hashes=[]
 for root,seed in zip(roots,(879511,879512),strict=True):
  rr=read(root/'result.json');cfg=read(root/'manifest.json')['config'];f=root/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(f)!=rr['responses_sha256'] or cfg['texture_seed']!=seed:raise ValueError('synthetic source')
  hashes.append(sha256(f))
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(planpath),source_sha256=hashes,graph_sha256=sha256(tr['checkpoint']),optimizer_updates=10,actor=False,real_data_used_for_fit=False))
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
  model.body_feedback.load_state_dict(payload['model'],strict=True);parameter=model.temporal.opponent_projection.weight;parameter.requires_grad_(True);initial=parameter.detach().clone()
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
  candidate=B/'20261101-causal-yaw-opponent-fit';crs=read(candidate/'result.json');ck=candidate/'opponent-projection.pt'
  if crs['status']!='completed_synthetic_causal_yaw_opponent_training' or sha256(ck)!=crs['checkpoint_sha256']:raise ValueError('prior candidate')
  payload=torch.load(ck,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False or payload['source_body_sha256']!=sha256(f):raise ValueError('candidate/body contract')
  old=payload['weight'].to('cuda');optim=torch.optim.Adam([parameter],lr=.002)
  data=[(*load_scene(seed,amp,B/folder),amp) for folder,seed,amp in plan['training_sources']]
  before=[rollout(sc,t,False).cpu().numpy() for sc,t,_,_ in data];losses=[]
  for i in range(10):
   optim.zero_grad(set_to_none=True);vals=[]
   for sc,t,digest,amp in data:
    pred=rollout(sc,t,True);delta=pred[0,-5:,0].mean()-pred[1,-5:,0].mean();term=(delta-.008*amp/.23).square()/len(data)
    if not torch.isfinite(term):raise ValueError('nonfinite loss')
    term.backward();vals.append(float(term.detach().cpu()));del pred,term
   if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise ValueError('nonfinite gradient')
   optim.step();losses.append(sum(vals));result['optimizer_updates']=i+1;write_json(run/'progress.json',dict(update=i+1,loss=losses[-1]))
  final=parameter.detach().clone();after=[rollout(sc,t,False).cpu().numpy() for sc,t,_,_ in data];valid=[];valid_rows=[]
  for folder,seed,amp in plan['validation_sources']:
   sc,t,digest=load_scene(seed,amp,B/folder);conditions=[]
   for w in (initial,old,final):
    with torch.no_grad():parameter.copy_(w)
    conditions.append(rollout(sc,t,False).cpu().numpy())
   val=np.stack(conditions);valid.append(val);deltas=val[:,:,-5:,0].mean(2);deltas=deltas[:,0]-deltas[:,1]
   valid_rows.append(dict(seed=seed,amplitude=amp,source_sha256=digest,original=float(deltas[0]),previous_candidate=float(deltas[1]),multispeed_candidate=float(deltas[2]),target=.008*amp/.23))
  with torch.no_grad():parameter.copy_(final)
  dest=run/'opponent-projection.pt';torch.save(dict(schema='flybrain.experimental.causal_yaw_opponent.v1',weight=final.cpu(),actor_qualified=False,source_body_sha256=sha256(f),multispeed=True),dest)
  out=run/'outputs.npz';np.savez_compressed(out,train_before=np.stack(before),train_after=np.stack(after),validation=np.stack(valid),times=t)
  result.update(status='completed_synthetic_multispeed_yaw_training',training_loss=losses,training_source_sha256=[x[2] for x in data],validation=valid_rows,validation_used_for_fit=False,checkpoint_sha256=sha256(dest),previous_candidate_sha256=sha256(ck),body_checkpoint_sha256=sha256(f),outputs_sha256=sha256(out),graph_and_body_frozen=True,physical_or_physiology_verified=False)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
