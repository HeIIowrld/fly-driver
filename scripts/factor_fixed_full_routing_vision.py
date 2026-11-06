"""Same-body/prefix coarse-static versus spatial T4/T5 image-channel exchange."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment
from scripts.sac_steering_experiment import restore
from scripts.train_qualified_sac_unified import context

ROOT=Path('runs/fixed-full-routing-learning-20261104')
class FactorExperiment(FixedFullRoutingExperiment):
 def inputs(self):
  static,fine,native,valid=super().inputs();h=self.holder;factor=h.get('factor','full')
  if factor not in ['full','coarse_static','fine_motion','identity']:raise ValueError('Explicit channel factor required')
  if factor in ['fine_motion','identity']:static=self.extras[h['episode']][h['index']:h['index']+1]
  if factor in ['coarse_static','identity']:fine=self.features[h['episode']][int(self.maps[h['episode']][h['index']]):int(self.maps[h['episode']][h['index']])+1]
  return static,fine,native,valid
 def step(self,i,learn):
  if self.holder.get('factor','full') in ['full','coarse_static']:return super().step(i,learn)
  if learn:raise ValueError('Channel factorial is read-only')
  self.holder['index']=i;d,c,_=self.items[self.holder['episode']]
  with torch.no_grad():
   _,mu=self.model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=self.goals,scene_features=torch.tensor(d['scene'][i:i+1]),scene_valid=self.views,body_context=context(d,c,i));return self.head(mu)[0]

def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=FactorExperiment(p,train_body=True);fit=ROOT/'visual_native-fit-01';fr=read(fit/'result.json');assert sha256(fit/'ports.pt')==fr['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);evaluation=ROOT/'visual_native-evaluation-01';r=read(evaluation/'result.json');assert sha256(evaluation/'candidate-prefix-states.pt')==r['prefix_states_sha256'];states=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);groups=read(evaluation/'real-image-groups.json');out=begin_run(ROOT/'visual-channel-factorial-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'ports.pt'),prefix_states_sha256=r['prefix_states_sha256'],seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Retimed consumed-development diagnostic: donor coarse scene+static320 and donor spatial T4/T5 2048 are crossed. Recipient body/native/clocks and same recurrent prefix retained. Artificial channel mixtures are not physical action truth, biological ablation or new-track evidence. Same prefix refers to downstream student recurrence. Donor FlyVis features retain their original upstream visual history; the FlyVis circuit is not rebranched from identical real RGB prefix.'));saved={};reports=[];maximum_full_error=0.;maximum_identity_difference=0.;maximum_identity_replay_error=0.
 try:
  with np.load(evaluation/'predictions.npz') as z:
   for row in groups:
    j=row['group'];recipient=row['recipient'];begin,end=recipient['begin'],recipient['end'];report=dict(group=j,recipient=recipient,left_donor=row['left_donor'],right_donor=row['right_donor'],factors={})
    for factor in ['full','coarse_static','fine_motion','identity']:
     branches=[]
     for donor in [row['left_donor'],row['right_donor']]:
      restore(e.model,states[2,begin]);e.holder.clear();e.holder.update(mode='real',episode=2,counterfactual_images=True,factor=factor);sequence=[]
      for offset,i in enumerate(range(begin,end)):e.holder['donor']=(2,donor['begin']+offset);sequence.append(e.step(i,False).numpy())
      branches.append(sequence)
     value=np.asarray(branches);saved[f'group{j}_{factor}_prediction']=value;difference=value[0]-value[1];report['factors'][factor]=dict(signed_teacher_bend_margin=float(-difference[:,0].mean()/2),maximum_left_right_action_difference=float(abs(difference).max()))
     if factor=='full':maximum_full_error=max(maximum_full_error,float(abs(value-z[f'image{j}_fine_prediction']).max()))
     if factor=='identity':
      maximum_identity_difference=max(maximum_identity_difference,float(abs(difference).max()));maximum_identity_replay_error=max(maximum_identity_replay_error,float(abs(value-z['prediction_2'][begin:end][None]).max()))
    reports.append(report)
  assert maximum_full_error<2e-6 and maximum_identity_difference==0 and maximum_identity_replay_error<2e-6;e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_readonly_visual_channel_factorial',upstream_flyvis_state_rebranched=False,counterfactual_real_rgb_rollout_verified=False,reports=reports,maximum_full_replay_difference=maximum_full_error,maximum_identity_left_right_difference=maximum_identity_difference,maximum_identity_replay_difference=maximum_identity_replay_error,direction_counts={f:sum(r['factors'][f]['signed_teacher_bend_margin']>0 for r in reports) for f in ['full','coarse_static','fine_motion','identity']},target_counts={f:sum(r['factors'][f]['signed_teacher_bend_margin']>=.002 for r in reports) for f in ['full','coarse_static','fine_motion','identity']},mean_signed_margins={f:float(np.mean([r['factors'][f]['signed_teacher_bend_margin'] for r in reports])) for f in ['full','coarse_static','fine_motion','identity']},predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False))
 finally:e.close()
if __name__=='__main__':main()
