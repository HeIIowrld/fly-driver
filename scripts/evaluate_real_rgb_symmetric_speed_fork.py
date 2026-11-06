"""Same full upstream visual state and downstream student prefix, actual RGB donors."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.symmetric_speed_experiment import SymmetricSpeedExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.train_qualified_sac_unified import context
ROOT=Path('runs/symmetric-speed-learning-20261104')
class RGBForkExperiment(SymmetricSpeedExperiment):
 def inputs(self):
  normal=super().inputs();h=self.holder
  if 'rgb_fork' not in h:return normal
  row=int(h['event_map'][h['index']-h['aligned_begin']]);x=h['rgb_fork'];branch=h['branch'];return torch.tensor(x['static'][branch,row:row+1]),torch.tensor(x['motion'][branch,row:row+1]),normal[2],normal[3]
 def step(self,i,learn):
  if 'rgb_fork' not in self.holder:return super().step(i,learn)
  if learn:raise ValueError('Real RGB fork is diagnostic only')
  h=self.holder;h['index']=i;row=int(h['event_map'][i-h['aligned_begin']]);d,c,_=self.items[0]
  with torch.no_grad():
   _,mu=self.model.experimental_recurrent_speed_decision(decision_time=float(d['decision_time'][i]),racing_goals=self.goals,scene_features=torch.tensor(h['rgb_fork']['scene'][h['branch'],row:row+1]),scene_valid=self.views,body_context=context(d,c,i));return self.head(mu)[0]

def main():
 torch.set_num_threads(1);source=Path('runs/spatial-motion-dn-learning-20261104/real-rgb-fork-02');r=read(source/'result.json');assert r['status']=='completed_original_rgb_identical_flyvis_forks' and r['upstream_flyvis_state_rebranched'] and sha256(source/'features.npz')==r['features_sha256'];branches=read(source/'branches.json');assert len({v['initial_full_visual_state_sha256'] for v in branches})==1;assert len({v['initial_held_retina_sha256'] for v in branches})==1
 with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
 p=read(ROOT/'plan-01.json');e=RGBForkExperiment(p,learn_speed=True);fit=ROOT/'speed_policy-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);e.speed_policy.requires_grad_(False);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);evaluation=ROOT/'speed_policy-evaluation-01';er=read(evaluation/'result.json');assert sha256(evaluation/'candidate-prefix-states.pt')==er['prefix_states_sha256'];states=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);begin,end=r['aligned_begin'],r['end'];np.testing.assert_array_equal(x['decision_indices'],np.arange(begin,end));np.testing.assert_array_equal(x['camera_indices'][x['event_to_camera']],e.items[0][0]['camera_index'][begin:end]);restore(e.model,states[0,r['original_student_prefix_begin']]);e.holder.update(mode='real',episode=0);prefix_replay=[]
 for i in range(r['original_student_prefix_begin'],begin):prefix_replay.append(e.step(i,False).numpy())
 prefix=snapshot(e.model);prefix_hash=state_digest(prefix);out=begin_run(ROOT/'real-rgb-student-fork-01',dict(program_sha256=sha256(__file__),upstream_result_sha256=sha256(source/'result.json'),checkpoint_sha256=sha256(fit/'ports.pt'),source_prefix_sha256=er['prefix_states_sha256'],aligned_begin=begin,end=end,downstream_prefix_sha256=prefix_hash,upstream_full_visual_state_sha256=r['prefix_full_visual_state_sha256'],seed=None,optimizer_updates=0,actor_qualified=False,training_admissible=False,vehicle_control_sent=False,scope='One TRAIN recipient prefix with recipient pixels and opposite-corner donor pixels; complete upstream visual state and held retina cloned, same downstream flight state/body/native/clock restored. Retimed diagnostic, not physical action truth, new-track or actual driving.'));saved={};reports={};identity_error=0.;repeat_error=0.
 try:
  with np.load(evaluation/'predictions.npz') as z:
   prefix_error=float(abs(np.asarray(prefix_replay)-z['prediction_0'][r['original_student_prefix_begin']:begin]).max()) if len(prefix_replay) else 0.;assert prefix_error==0
   for mode in ['fine','visual_change_off','native_change_off','speed_policy_off']:
    values=[]
    for branch in range(3):
     restore(e.model,prefix);assert state_digest(snapshot(e.model))==prefix_hash;e.holder.clear();e.holder.update(mode='real',episode=0,counterfactual_images=True,rgb_fork=x,branch=branch,event_map=x['event_to_camera'],aligned_begin=begin,visual_change_off=mode=='visual_change_off',native_change_off=mode=='native_change_off',speed_policy_off=mode=='speed_policy_off');v=np.asarray([e.step(i,False).numpy() for i in range(begin,end)]);values.append(v)
    values=np.asarray(values);saved[mode+'_prediction']=values;repeat_error=max(repeat_error,float(abs(values[0]-values[2]).max()));margin=float(-(values[1,:,0]-values[0,:,0]).mean()/2);reports[mode]=dict(signed_teacher_bend_margin=margin,direction_correct=margin>0,target_margin_passed=margin>=.002,maximum_left_right_action_difference=float(abs(values[1]-values[0]).max()))
    if mode=='fine':identity_error=float(abs(values[0]-z['prediction_0'][begin:end]).max())
  assert identity_error<2e-6 and repeat_error<2e-6;e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_same_upstream_and_student_real_rgb_fork',reports=reports,prefix_replay_maximum_difference=prefix_error,identity_factual_prediction_maximum_difference=identity_error,identity_repeat_prediction_maximum_difference=repeat_error,upstream_flyvis_state_rebranched=True,downstream_prefix_identical=True,groups=1,decisions=end-begin,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,training_admissible=False,physical_counterfactual_verified=False,vehicle_control_sent=False))
 finally:e.close()
if __name__=='__main__':main()
