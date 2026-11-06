"""Original fine baseline sensed-body feedback, not desired-command injection."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.functional_priority_experiment import Experiment
from scripts.probe_spatial_motion_dn_feedback import FeedbackExperiment
from scripts.sac_steering_experiment import snapshot,restore
ROOT=Path('runs/spatial-motion-dn-learning-20261104')
class OriginalFeedback(Experiment):
 _hold_command=FeedbackExperiment._hold_command
 step=FeedbackExperiment.step
 def __init__(self,plan):
  super().__init__(plan);self.feedback_yaw=0.;self.feedback_speed=0.;self.fixed_speed=None;self.command_error=0.;self.hooks.append(self.model.motor.command.register_forward_pre_hook(self._hold_command))

def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');old=read(read(p['base_port_plan'])['parent_fit_plan']);e=OriginalFeedback(old);tasks=[t for t in p['physical_tasks'] if t['episode']<2];candidate=read(old['candidate_plan']);fine=Path(candidate['motion']['fit']);out=begin_run(ROOT/'original-fine-feedback-01',dict(program_sha256=sha256(__file__),original_candidate=e.parent,original_predictions_sha256=sha256(fine/'predictions.npz'),audited_prefix_sha256=sha256(e.prefix_file),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='All30 TRAIN command windows; actual sensed yaw/speed perturbations with fixed desired target, images/native and same original-fine recurrence. Original audited chronological prefix at or before each case is advanced factually. Model-local signs, not physical vehicle stability.'));saved={};reports=[];maximum=0.;advance_error=0.;counts=dict(total=0,yaw_damping=0,speed_damping=0,yaw_antidamping=0,speed_antidamping=0)
 try:
  with np.load(fine/'predictions.npz') as z:
   for j,t in enumerate(tasks):
    episode,begin,end=t['episode'],t['begin'],t['end'];nearest=max(k[1] for k in e.states if k[0]==episode and k[1]<=begin);restore(e.model,e.states[episode,nearest]);e.holder.clear();e.holder.update(mode='real',episode=episode);e.feedback_yaw=e.feedback_speed=0.
    for i in range(nearest,begin):advance_error=max(advance_error,float(abs(e.step(i,False).numpy()-z[f'prediction_{episode}'][i]).max()))
    prefix=snapshot(e.model);values={}
    for name,yaw,speed in [('neutral',0.,0.),('yaw_minus',-.01,0.),('yaw_plus',.01,0.),('speed_minus',0.,-.01),('speed_plus',0.,.01)]:
     restore(e.model,prefix);e.feedback_yaw=yaw;e.feedback_speed=speed;values[name]=np.asarray([e.step(i,False).numpy() for i in range(begin,end)]);saved[f'case{j}_{name}_prediction']=values[name]
    maximum=max(maximum,float(abs(values['neutral']-z[f'prediction_{episode}'][begin:end]).max()));row=dict(task=t,audited_prefix_begin=nearest,feedback={});counts['total']+=1
    for field in ['yaw','speed']:
     axis=lambda x:x[:,0] if field=='yaw' else x[:,1]-x[:,2];minus=axis(values[field+'_minus'])-axis(values['neutral']);plus=axis(values[field+'_plus'])-axis(values['neutral']);sign=1 if field=='yaw' else -1;damping=bool(sign*minus.mean()<0 and sign*plus.mean()>0);anti=bool(sign*minus.mean()>0 and sign*plus.mean()<0);counts[field+'_damping']+=int(damping);counts[field+'_antidamping']+=int(anti);row['feedback'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_both_damping=damping,mean_both_antidamping=anti)
    reports.append(row)
  assert maximum<2e-6 and advance_error<2e-6 and e.command_error==0;e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_original_fine_sensed_feedback_probe',counts=counts,reports=reports,maximum_original_neutral_difference=maximum,maximum_audited_prefix_advance_difference=advance_error,maximum_desired_command_error=e.command_error,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,physical_feedback_stability_verified=False));print(counts)
 finally:e.close()
if __name__=='__main__':main()
