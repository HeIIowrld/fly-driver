"""Float64 NumPy audit of all cached neural outputs and reported evaluation gates."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data
from scripts.improve_sac_anticipation import command_metrics

def main():
 root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');_,items,_,_,pre=load_data(read(plan['paired_data_plan']));reports={}
 for rank in [1,4]:
  fit=root/f'rank{rank}-fit-01';evaluation=root/f'rank{rank}-evaluation-01';fr=read(fit/'result.json');er=read(evaluation/'result.json');assert sha256(fit/'joint-dn-decoder.pt')==fr['checkpoint_sha256'] and sha256(evaluation/'predictions.npz')==er['predictions_sha256'] and sha256(evaluation/'groups.json')==er['groups_sha256'];state=torch.load(fit/'joint-dn-decoder.pt',map_location='cpu',weights_only=True)['readout'];s={k:v.numpy().astype(float) for k,v in state.items()};errors=[];metricerrors=[]
  def decode(mu):
   z=((mu.astype(float)-s['training_reference'])/s['training_scale'])@s['projection.weight'].T+s['projection.bias'];return np.concatenate([np.tanh(z[...,:1]),1/(1+np.exp(-z[...,1:]))],-1)
  with np.load(evaluation/'predictions.npz') as z:
   for e in range(3):
    pred=decode(z[f'muscles_{e}']);errors.append(float(abs(pred-z[f'prediction_{e}']).max()));d=items[e][0]
    for role,mask in [('all',d['fit_mask']),('pre',pre[e])]:
     metric=command_metrics(pred,d['labels'],mask);reported=er['actual_recordings'][e]['metrics'][role]['joint'];metricerrors.append(float(abs(np.array(metric['rmse'])-reported['rmse']).max()))
   counts={k:0 for k in er['reserved_direction_counts']};targets=0;maximum_margin_error=0.
   for g in read(evaluation/'groups.json'):
    for arm in counts:
     p=z[f"group_{g['group']}_{arm}_prediction"];margin=float(-np.mean(p[1,14:,0]-p[0,14:,0])/2);maximum_margin_error=max(maximum_margin_error,abs(margin-g['arms'][arm]['signed_margin']));counts[arm]+=int(margin>.0001);targets+=int(arm=='fine' and margin>=plan['target_margin'])
   assert counts==er['reserved_direction_counts'] and targets==er['reserved_target_pairs'] and maximum_margin_error<1e-12
   commanderrors=[];one_sided=[]
   for case in er['command_probes']:
    for field in ['yaw','speed']:
     p=z[f"{case['role']}_{case['case']}_{field}_prediction"];d=(p[1].astype(float)-p[0])/.02;commanderrors.append(float(abs(d.mean(0)-case['models'][field]['vehicle_derivative']).max()));task=case['task'];neutral=z[f"prediction_{task['episode']}"][task['begin']:task['end']];value=p[:,:,0] if field=='yaw' else p[:,:,1]-p[:,:,2];base=neutral[:,0] if field=='yaw' else neutral[:,1]-neutral[:,2];sign=-1 if field=='yaw' else 1;minus=value[0]-base;plus=value[1]-base;one_sided.append(dict(role=case['role'],case=case['case'],field=field,mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),correct_minus_fraction=float((sign*minus< -1e-7).mean()),correct_plus_fraction=float((sign*plus>1e-7).mean())))
  assert max(errors)<2e-5 and max(metricerrors)<2e-6 and max(commanderrors)<1e-12
  journal=read(fit/'state-journal.json');assert sha256(fit/'state-journal.json')==fr['state_journal_sha256'];assert len(journal)==1020 and sum(v['learn'] for v in journal)==93 and sum(v['end']-v['begin'] for v in journal)==32597 and not any(v['parent_state_injected'] for v in journal)
  for prev,current in zip(journal,journal[1:]):
   if prev['episode']==current['episode']:assert prev['after_sha256']==current['incoming_sha256'] and prev['end']==current['begin']
  records=read(fit/'training-records.json');assert sha256(fit/'training-records.json')==fr['training_records_sha256'] and len(records)==147 and sum(v['dynamic_physical'] is not None for v in records)==2
  preservation=all(all(row['metrics']['all']['joint']['rmse'][i]<=plan['actual_recording_preservation_maximum_ratio']*row['metrics']['all']['parent']['rmse'][i] for i in range(3)) and row['metrics']['pre']['joint']['rmse'][0]<=plan['actual_recording_preservation_maximum_ratio']*row['metrics']['pre']['parent']['rmse'][0] for row in er['actual_recordings']);reports[f'rank{rank}']=dict(maximum_numpy_prediction_errors=errors,maximum_rmse_difference=max(metricerrors),maximum_command_derivative_difference=max(commanderrors),reserved_direction_counts=counts,reserved_target_pairs=targets,yaw_direction_gate=er['yaw_direction_gate'],speed_direction_gate=er['speed_direction_gate'],one_sided_command_probes=one_sided,numerical_chain_windows=len(journal),dynamic_physical_updates=2,actual_recording_preservation_gate=preservation,steering_pre_rmse_reduction_percent=[100*(1-v['metrics']['pre']['joint']['rmse'][0]/v['metrics']['pre']['parent']['rmse'][0]) for v in er['actual_recordings']],power_gas_raw_logit_derivatives=(s['projection.weight'][1,4:]/s['training_scale'][4:]).tolist(),power_brake_raw_logit_derivatives=(s['projection.weight'][2,4:]/s['training_scale'][4:]).tolist(),common_power_steering_raw_logit_derivative=float((s['projection.weight'][0,4:]/s['training_scale'][4:]).sum()))
 write_json(root/'independent-audit-01.json',dict(status='completed_independent_joint_decoder_audit',models=reports,actor_qualified=False,actual_driving_improvement_verified=False));print(reports)
if __name__=='__main__':main()
