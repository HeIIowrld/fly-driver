"""Recollect exact candidate yaw muscles and split steering-logit contributions."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from scripts.sac_steering_experiment import restore

def main():
 torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');fit=root/'rank4-fit-01';evaluation=root/'rank4-evaluation-01';assert sha256(fit/'joint-dn-decoder.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'joint-dn-decoder.pt',map_location='cpu',weights_only=True);e=JointDNDecoderExperiment(plan,4,cp['readout']);e.calibration.load_state_dict(cp['current_calibration']);er=read(evaluation/'result.json');assert sha256(evaluation/'candidate-prefix-states.pt')==er['prefix_states_sha256'];states=torch.load(evaluation/'candidate-prefix-states.pt',map_location='cpu',weights_only=True);original=np.load(evaluation/'predictions.npz');raw=e.head.projection.weight.detach().numpy().astype(float)/e.head.training_scale.numpy();saved={};reports=[];capture=[];hook=e.head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().numpy().copy()));out=begin_run(root/'yaw-decomposition-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'joint-dn-decoder.pt'),prefix_states_sha256=sha256(evaluation/'candidate-prefix-states.pt'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False))
 try:
  for case in er['command_probes']:
   t=case['task'];episode,begin,end=t['episode'],t['begin'],t['end'];e.holder.update(mode='real',episode=episode,calibration_off=False,motion_off=False);pairs=[];pred=[];key=f"{case['role']}_{case['case']}"
   for sign in [-1,1]:
    restore(e.model,states[episode,begin]);e.yaw_offset=sign*.01;capture.clear();pred.append(np.asarray([e.step(i,False).numpy() for i in range(begin,end)]));pairs.append(np.concatenate(capture))
   e.yaw_offset=0.;pred=np.asarray(pred);mu=np.asarray(pairs);neutral=original[f'muscles_{episode}'][begin:end];parity=float(abs(pred-original[key+'_yaw_prediction']).max());assert parity<2e-6;delta=mu.astype(float)-neutral;contribution=delta*raw[0];common=.5*(delta[:,:,4]+delta[:,:,5])*(raw[0,4]+raw[0,5]);difference=.5*(delta[:,:,5]-delta[:,:,4])*(raw[0,5]-raw[0,4]);assert abs(common+difference-contribution[:,:,4:].sum(-1)).max()<1e-10;total=contribution.sum(-1);without_common=total-common;saved[key+'_muscles']=mu;saved[key+'_neutral_muscles']=neutral;saved[key+'_logit_contributions']=contribution;saved[key+'_common_power_logit_change']=common;reports.append(dict(role=case['role'],case=case['case'],task=t,maximum_prediction_parity_difference=parity,mean_steering_logit_change_minus_plus=total.mean(1).tolist(),mean_common_power_logit_change_minus_plus=common.mean(1).tolist(),mean_power_difference_logit_change_minus_plus=difference.mean(1).tolist(),mean_logit_change_without_common_power_minus_plus=without_common.mean(1).tolist(),mean_proxy_contributions_minus_plus=contribution.mean(1).tolist()))
  np.savez_compressed(out/'muscle-probes.npz',**saved);write_json(out/'result.json',dict(status='completed_candidate_yaw_logit_decomposition',reports=reports,common_power_steering_raw_logit_derivative=float(raw[0,4]+raw[0,5]),muscle_probes_sha256=sha256(out/'muscle-probes.npz'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Exact affine logit decomposition on recollected candidate counterfactuals. Removing a component algebraically is not retraining, a physiological lesion or measured vehicle effect.'))
 finally:hook.remove();e.yaw_offset=0.;e.close();original.close()
if __name__=='__main__':main()
