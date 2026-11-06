"""Five visual-route factorial on declared TRAIN exchanges and fixed candidate prefixes."""
import gc,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fixture_adjoint_speed_experiment import FixtureAdjointSpeedExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.train_qualified_sac_unified import context
from scripts.intervene_motion_spatial_same_body import state_digest

ROOT=Path('runs/visual-route-factorial-20261105')
PARENT=Path('runs/fixture-adjoint-learning-20261104')
ROUTES=['scene','optical_static','dn_static','dn_motion','visual_motion']

class RouteExperiment(FixtureAdjointSpeedExperiment):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  if self.model.motion_forecast is not None:raise ValueError('Direct visual splitting requires absent motion forecast')
  self.hooks.append(self.model.visual.register_forward_pre_hook(self._direct_scene))
 def _direct_scene(self,module,args):
  h=self.holder;e=h['episode'];i=h['index']
  if h['direct_donor']:e,i=h['donor']
  return (torch.cat([args[0][:,:12],torch.tensor(self.items[e][0]['scene'][i:i+1])],1),)
 def route(self,bit):
  h=self.holder;e=h['episode'];i=h['index']
  if h['route_mask']&(1<<bit):e,i=h['donor']
  j=int(self.maps[e][i]);return self.extras[e][i:i+1],self.features[e][j:j+1]
 def _visual(self,module,args,value):
  static,_=self.route(1);_,fine=self.route(4);return value+self.optical(static[:,64:])+self.motion(fine)
 def _descending(self,module,args,value):
  static,_=self.route(2);_,fine=self.route(3);steer,power=value;current=self.calibration(static,fine,self.dn(static));self.last_current=current.detach().clone();self.last_descending=(steer+current).detach().clone();return steer+current,power
 def step(self,i,learn=False):
  if learn:raise ValueError('Route factorial is read-only')
  h=self.holder;h['index']=i;e=h['episode'];d,c,_=self.items[e];de,di=h['donor'] if h['route_mask']&1 else (e,i)
  with torch.no_grad():
   _,mu=self.model.experimental_recurrent_speed_decision(decision_time=float(d['decision_time'][i]),racing_goals=self.goals,scene_features=torch.tensor(self.items[de][0]['scene'][di:di+1]),scene_valid=self.views,body_context=context(d,c,i));return self.head(mu)[0]


def main():
 torch.set_num_threads(1);plan=read(PARENT/'plan-01.json');checkpoint=PARENT/'input_adjoint-fit-01/ports.pt';assert sha256(checkpoint)==read(checkpoint.parent/'result.json')['checkpoint_sha256'];e=RouteExperiment(plan,fixture_input_adjoint=True);cp=torch.load(checkpoint,map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);e.speed_policy.requires_grad_(False);evaluation=PARENT/'input_adjoint-evaluation-01';er=read(evaluation/'result.json');assert sha256(evaluation/'candidate-prefix-states.pt')==er['prefix_states_sha256'];states=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);tasks=plan['image_tasks'];assert len(tasks)==10 and all(t['episode'] in [0,1] for t in tasks)
 out=begin_run(ROOT/'direct-scene-split-01',dict(program_sha256=sha256(__file__),parent_checkpoint_sha256=sha256(checkpoint),prefix_states_sha256=er['prefix_states_sha256'],routes=ROUTES,tasks=tasks,seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Five-route2^5 factorial. Same recipient body/native/clocks/downstream recurrent prefix. Artificial mixes of donor features retain donor upstream history; neither physical counterfactual truth nor unseen-track evaluation. TRAIN tasks only; no new reserved texture read.'));enabled=gc.isenabled();gc.disable();saved={};reports=[];identity_error=0.;identity_lr=0.;start=time.perf_counter();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
 try:
  with np.load(evaluation/'predictions.npz') as original:
   for number,task in enumerate(tasks):
    episode,begin,end=task['episode'],task['begin'],task['end'];prefix=states[episode,begin];prefix_hash=state_digest(prefix);rows=[]
    for mask in range(4):
     pred=[];currents=[];descending=[];muscles=[]
     for donor in [task['left_donor'],task['right_donor']]:
      restore(e.model,prefix);assert state_digest(snapshot(e.model))==prefix_hash;e.holder.clear();e.holder.update(mode='real',episode=episode,counterfactual_images=True,route_mask=30+(mask&1),direct_donor=bool(mask&2));branch=[];bc=[];bd=[];bm=[]
      for offset,i in enumerate(range(begin,end)):
       e.holder['donor']=(episode,donor+offset);branch.append(e.step(i).numpy());bc.append(e.last_current[0].numpy());bd.append(e.last_descending[0].numpy());bm.append(e.last_muscles[0].numpy())
      pred.append(branch);currents.append(bc);descending.append(bd);muscles.append(bm)
     prediction=np.asarray(pred);current=np.asarray(currents);mu=np.asarray(muscles);key=f'group{number}_mask{mask}';saved[key+'_prediction']=prediction;saved[key+'_current']=current;saved[key+'_descending']=np.asarray(descending);saved[key+'_muscles']=mu;difference=prediction[0]-prediction[1];rows.append(dict(mask=mask,routes=['other_scene_donor' if mask&1 else 'other_scene_recipient','direct_scene_donor' if mask&2 else 'direct_scene_recipient'],steering_margin=float(-difference[:,0].mean()/2),mean_current_left_minus_right=(current[0]-current[1]).mean(0).tolist(),mean_muscle_left_minus_right=(mu[0]-mu[1]).mean(0).tolist()))
     if mask in [0,3]:
      prior_mask=30 if mask==0 else 31
      with np.load(ROOT/'diagnostic-01/trajectories.npz') as prior:identity_error=max(identity_error,float(abs(prediction-prior[f'group{number}_mask{prior_mask}_prediction']).max()))
    reports.append(dict(group=number,task=task,prefix_sha256=prefix_hash,masks=rows));write_json(out/'progress.json',dict(groups=len(reports),total_groups=len(tasks),elapsed_seconds=time.perf_counter()-start))
  assert identity_error<2e-6;e.verify_fixed();np.savez_compressed(out/'trajectories.npz',**saved);write_json(out/'groups.json',reports);margins=np.array([[r['steering_margin'] for r in g['masks']] for g in reports]);result.update(status='completed_train_direct_scene_split',groups=len(reports),branches=len(reports)*8,maximum_parent_factorial_replay_difference=identity_error,mean_signed_margins_by_mask=margins.mean(0).tolist(),direction_counts_by_mask=(margins>0).sum(0).tolist(),target_counts_by_mask=(margins>=.002).sum(0).tolist(),groups_sha256=sha256(out/'groups.json'),trajectories_sha256=sha256(out/'trajectories.npz'),actual_driving_improvement_verified=False,termination_reason='All other image routes exchanged. Direct visual linear input and other scene-derived temporal/spatial/body context crossed independently; no new weights or physical truth claims')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),groups=len(reports));raise
 finally:e.close();gc.enable() if enabled else gc.disable();result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)

if __name__=='__main__':main()
