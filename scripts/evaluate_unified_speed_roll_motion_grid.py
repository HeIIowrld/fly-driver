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
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();planpath=B/'20261101-speed-roll-motion-grid-plan/plan.json';plan=read(planpath)
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
  d=B/'20261101-flyvis-varied-causal-879513';rr=read(d/'result.json');cfg=read(d/'manifest.json')['config'];source=d/'responses.npz'
  if rr['status']!='completed_causal_four_view_varied_synthetic_motion' or sha256(source)!=rr['responses_sha256'] or cfg['texture_seed']!=879513:raise ValueError('causal source')
  with np.load(source,allow_pickle=False) as z: retina=z['retina'].copy();motion=z['motion'].copy();times=z['times'].copy()
  if retina.shape!=(3,41,4,721) or motion.shape!=(3,41,4,8,721) or not np.allclose(times,np.arange(41)*.04,rtol=0,atol=1e-12):raise ValueError('source shape')
  full=np.concatenate([retina[:,:,:,None],motion],axis=3)
  with torch.no_grad():scenes=model.scene_pool(torch.tensor(full.reshape(-1,4,9,721),dtype=torch.float32,device='cuda')).reshape(3,41,576)
  goals=torch.tensor(racing_goal(100,1),device='cuda');view=torch.ones((1,4),dtype=torch.bool,device='cuda')
  candidates=[('roll_only','20261101-centered-roll-contrast-fit','centered-body-adapter.pt','completed_synthetic_centered_roll_contrast_training'),('power_only','20261101-speed-wheel-contrast-fit','speed-body-adapter.pt','completed_synthetic_speed_wheel_contrast_training'),('turn_control','20261101-speed-wheel-turn-control-fit','speed-body-adapter.pt','completed_synthetic_speed_wheel_turn_control_training')]
  def load_candidate(name,folder,filename,status):
   root=B/folder;r=read(root/'result.json');f=root/filename
   if r['status']!=status or sha256(f)!=r['checkpoint_sha256']:raise ValueError('checkpoint provenance')
   payload=torch.load(f,map_location='cpu',weights_only=True)
   if payload['actor_qualified'] is not False:raise ValueError('qualification')
   if name!='roll_only':model.body_feedback.load_state_dict(payload['model'],strict=True)
   else:
    state=payload['model'];old=state['indices'].tolist();new=model.body_feedback.indices.tolist()
    with torch.no_grad():
     model.body_feedback.projection.weight.zero_()
     for j,idx in enumerate(old):
      k=new.index(idx);model.body_feedback.projection.weight[:,k].copy_(state['projection.weight'][:,j]);model.body_feedback.projection.weight[:,len(new)+k].copy_(state['projection.weight'][:,len(old)+j])
   return sha256(f)
  arrays=[];pins={}
  for name,folder,filename,status in candidates:
   pins[name]=load_candidate(name,folder,filename,status);case_outputs=[]
   for visual_case in (1,2):
    speeds=[]
    for speed in (2.,8.):
     rolls=[]
     for roll in (-.025,.025):
      model.reset_episode();values=np.zeros(21);values[1]=roll;values[16]=speed;values[17:21]=speed/.29;valid=np.ones(21,dtype=bool);out=[]
      for i,t in enumerate(times):
       now=1.+float(t);body=NavigationBodyObservation('synthetic-combined',i,now-.01,now,values.copy(),valid.copy(),'synthetic-only')
       ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,camera_receipt_time=now-.02,decision_time=now)
       intent,_=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=scenes[visual_case,i:i+1],scene_valid=view,body_context=ctx)
       out.append(intent[0].cpu().numpy())
      rolls.append(np.stack(out))
     speeds.append(np.stack(rolls))
    case_outputs.append(np.stack(speeds));write_json(run/'progress.json',dict(candidate=name,visual_case=visual_case))
   arrays.append(np.stack(case_outputs))
  arr=np.stack(arrays);f=run/'outputs.npz';np.savez_compressed(f,intents=arr,times=times)
  reports={}
  for j,(name,*_) in enumerate(candidates):
   x=arr[j];speed=x[:,1,:,:,:]-x[:,0,:,:,:];roll=x[:,:,1,:,:]-x[:,:,0,:,:]
   reports[name]=dict(speed_turn_mean_abs=float(np.abs(speed[...,0]).mean()),speed_power_mean_abs=float(np.abs(speed[...,1]).mean()),speed_power_signed_mean=float(speed[...,1].mean()),roll_turn_signed_mean=float(roll[...,0].mean()),roll_turn_mean_abs=float(np.abs(roll[...,0]).mean()),positive_intent_counts=(x>0).sum(axis=(0,1,2,3)).tolist())
  result.update(status='completed_synthetic_combined_vision_body_grid',candidates=reports,checkpoints_sha256=pins,source_sha256=sha256(source),outputs_sha256=sha256(f),case_dimensions=['candidate','visual_case','speed','roll','decision','intent'],synthetic_body_not_physically_coupled=True,physiological_or_driving_verified=False)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise RuntimeError(result['reason'])
if __name__=='__main__':main()
