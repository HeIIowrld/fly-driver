"""Synthetic relative roll-response training in one frozen flight graph; not driving."""
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
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-unified-neutral-stability-plan/plan.json';plan=read(planpath)
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
  speed_index = model.body_feedback.indices.tolist().index(16)
  with torch.no_grad():
   model.body_feedback.projection.weight[:,speed_index].fill_(.25)
  def scene(root):
   with np.load(root/'responses.npz',allow_pickle=False) as z:ret=z['retina'][0,20].copy();mot=z['motion'][0,20].copy()
   with torch.no_grad():return model.scene_pool(torch.tensor(np.concatenate([ret[:,None],mot],axis=1)[None],dtype=torch.float32,device='cuda'))
  goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda')
  def rollout(visual_scene,roll,speed,label):
   model.reset_episode();values=np.zeros(21);values[1]=roll;values[16]=speed;valid=np.ones(21,dtype=bool);outs=[];muscles=[]
   for i in range(101):
    now=1.+i*.04;body=NavigationBodyObservation('synthetic-neutral',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only')
    ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
    intent,muscle=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=visual_scene,scene_valid=view,body_context=ctx)
    outs.append(intent[0].cpu().numpy());muscles.append(muscle[0].cpu().numpy())
   write_json(run/'progress.json',dict(completed=label))
   return np.stack(outs),np.stack(muscles)
  conditions=[('anchor',0,0.,0.),('held_texture_quiet',1,0.,0.),('held_texture_speed',1,0.,5.),('held_texture_faster',1,0.,10.)]
  scenes=[scene(x) for x in roots];outs=[];muscles=[]
  for label,source,roll,speed in conditions:
   o,m=rollout(scenes[source],roll,speed,label);outs.append(o);muscles.append(m)
  outs=np.stack(outs);muscles=np.stack(muscles);readout=model.motor.backbone.readout
  ref_lateral=np.array([muscles[0,100,1]-muscles[0,100,0],muscles[0,100,3]-muscles[0,100,2]])
  ref_power=(muscles[0,100,4]+muscles[0,100,5])*.5
  mix=readout.steering_mix.softmax(0).detach().cpu().numpy();tg=float(readout.turn_log_gain.exp().cpu());pg=float(readout.forward_log_gain.exp().cpu())
  lateral=np.stack([muscles[:,:,1]-muscles[:,:,0],muscles[:,:,3]-muscles[:,:,2]],axis=-1);power=(muscles[:,:,4]+muscles[:,:,5])*.5
  alt=np.tanh(np.stack([tg*((lateral-ref_lateral)@mix),pg*(power-ref_power)],axis=-1))
  p=run/'outputs.npz';np.savez_compressed(p,intents=outs,muscles=muscles,neutral_counterfactual=alt)
  results={label:dict(original_at_12=outs[j,11].tolist(),original_at_101=outs[j,100].tolist(),counterfactual_at_12=alt[j,11].tolist(),counterfactual_at_101=alt[j,100].tolist()) for j,(label,*_) in enumerate(conditions)}
  result.update(status='completed_unified_synthetic_speed_path_probe',conditions=results,anchor_lateral=ref_lateral.tolist(),anchor_power=float(ref_power),anchor_counterfactual=alt[0,100].tolist(),outputs_sha256=sha256(p),engineering_speed_weight=.25,model_mutated=True,reference_is_physical_calibration=False)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
