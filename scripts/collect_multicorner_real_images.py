"""Shared actual pre-corner image muscles and explicit native19-port ablation."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from scripts.multicorner_command_data import ROOT
from scripts.evaluate_real_image_exchange import ImageExchangeExperiment,select_donors
from scripts.sac_steering_experiment import restore,snapshot
from scripts.intervene_motion_spatial_same_body import state_digest

class NativeCutImageExchange(ImageExchangeExperiment):
 def _body(self,module,args,value):
  if self.holder.get('native_port_off',False):return value
  return super()._body(module,args,value)

def main():
 torch.set_num_threads(1);protocol=read(ROOT/'protocol-01.json');parent=torch.load(protocol['parent_checkpoint'],map_location='cpu',weights_only=True);plan=read(protocol['parent_fit_plan']);e=NativeCutImageExchange(plan,4,parent['readout']);e.calibration.load_state_dict(parent['current_calibration']);original=Path(protocol['parent_evaluation']);assert sha256(original/'candidate-prefix-states.pt')==read(original/'result.json')['prefix_states_sha256'];states=torch.load(original/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);donors=select_donors(e.items[2][0]);tasks=[v for v in read(Path(plan['yaw_probes'])/'result.json')['cases'] if v['role']=='development'];capture=[];hook=e.head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().numpy().copy()));saved={};groups=[];heads={'parent':e.head}
 for variant in ['affine','quadratic']:heads[variant]=load_muscle_interaction_head(ROOT/f'{variant}-fit-01/decoder.pt')[0]
 out=begin_run(ROOT/'real-images-01',dict(program_sha256=sha256(__file__),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],source_prefix_sha256=sha256(original/'candidate-prefix-states.pt'),donors=donors,scope='One consumed actual donor bend pair at three recipient body/circuit prefixes; reassigned image clocks are an offline intervention, not physically valid replay or true counterfactual vehicle action. Native19 ablation removes only its learned port addition after prefix; legacy body21 remains.',seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False))
 try:
  with np.load(ROOT/'episode2-probes-01/neutral-full.npz') as z:neutral_full=z['muscles'].copy()
  for task in tasks:
   t=task['task'];begin,end=t['begin'],t['end'];prefix=states[2,begin];row=dict(case=task['case'],task=t,prefix_sha256=state_digest(prefix),arms={});e.holder.update(mode='real',episode=2,calibration_off=False,motion_off=False,native_port_off=False);restore(e.model,prefix);capture.clear()
   for i in range(begin,end):e.holder['donor']=(2,i);e.step(i,False)
   identity=float(abs(np.concatenate(capture)-neutral_full[begin:end]).max());assert identity<2e-6;row['identity_muscle_difference']=identity
   for arm in ['fine','calibration_off_after_prefix','native_port_off_after_prefix']:
    branches=[]
    for donor in [donors['left'],donors['right']]:
     restore(e.model,prefix);assert state_digest(snapshot(e.model))==row['prefix_sha256'];e.holder.update(calibration_off=arm=='calibration_off_after_prefix',native_port_off=arm=='native_port_off_after_prefix');capture.clear()
     for offset,i in enumerate(range(begin,end)):e.holder['donor']=(2,donor['begin']+offset);e.step(i,False)
     branches.append(np.concatenate(capture))
    mu=np.asarray(branches);saved[f'case{task["case"]}_{arm}_muscles']=mu;row['arms'][arm]={}
    with torch.no_grad():
     for name,head in heads.items():
      pred=head(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,32,3);saved[f'case{task["case"]}_{arm}_{name}_prediction']=pred;row['arms'][arm][name]=dict(signed_teacher_bend_margin=float(-np.mean(pred[0,:,0]-pred[1,:,0])/2),mean_action_difference=(pred[0]-pred[1]).mean(0).tolist())
   groups.append(row)
  e.holder.pop('donor',None);e.verify_fixed();np.savez_compressed(out/'responses.npz',**saved);write_json(out/'result.json',dict(status='completed_shared_real_image_exchange',groups=groups,responses_sha256=sha256(out/'responses.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False,termination_reason='Three identical-prefix body-held exchanges of one actual left/right image pair, calibration/native19 after-prefix cuts and stateless decoder comparison completed.'))
 finally:hook.remove();e.close()
if __name__=='__main__':main()
