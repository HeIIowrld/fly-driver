"""Fixed decoded trajectories, all forty-five command windows and reserved images."""
import argparse
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read,MuscleVehicleReadout
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.muscle_interaction_vehicle_readout import PAIRS
from scripts.multicorner_command_data import ROOT,collections
from scripts.paired_decoder_data import load_data
from scripts.improve_sac_anticipation import command_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['affine','quadratic'],required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(ROOT/'protocol-01.json');head,receipt=load_muscle_interaction_head(ROOT/f'{a.variant}-fit-01/decoder.pt');source=collections(plan);_,items,_,baseline,pre=load_data(read(plan['data_plan']));saved={};metrics=[];reports=[];counts={role:dict(total=0,yaw=0,speed=0,yaw_responsive_frames=0) for role in ['TRAIN','TRAIN_diagnostic','development']};all_mu=[]
 for path,r in source:
  with np.load(path/'neutral-full.npz') as z:all_mu.append(z['muscles'].copy())
 train=np.concatenate([all_mu[e][items[e][0]['fit_mask']] for e in [0,1]]).astype(float)
 if head.degree==2:
  z=(train-head.base.training_reference.numpy().astype(float))/head.base.training_scale.numpy().astype(float);q=np.stack([z[:,i]*z[:,j] for i,j in PAIRS],1);np.testing.assert_array_equal(head.quadratic_reference.numpy(),q.mean(0));np.testing.assert_array_equal(head.quadratic_scale.numpy(),np.maximum(q.std(0),1e-6))
 with torch.no_grad():
  for episode,(path,r) in enumerate(source):
   mu=torch.tensor(all_mu[episode]);prediction=head(mu).numpy();parent=head.base(mu).numpy();assert np.array_equal(prediction[:,1:],parent[:,1:]);saved[f'prediction_{episode}']=prediction;d=items[episode][0];metrics.append(dict(episode=episode,metrics={role:dict(original=command_metrics(baseline[episode],d['labels'],mask),new=command_metrics(prediction,d['labels'],mask)) for role,mask in [('all',d['fit_mask']),('pre',pre[episode])]}))
   with np.load(path/'probes.npz') as z:
    for j,row in enumerate(r['reports']):
     task=row['task'];neutral=head(torch.tensor(z[f'case{j}_neutral_muscles'])).numpy();report=dict(task=task,commands={})
     for field,offset in plan['offsets'].items():
      mu=z[f'case{j}_{field}_muscles'];prediction=head(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,32,3);saved[f'episode{episode}_case{j}_{field}_prediction']=prediction;value=prediction[:,:,0] if field=='yaw' else prediction[:,:,1]-prediction[:,:,2];base=neutral[:,0] if field=='yaw' else neutral[:,1]-neutral[:,2];sign=-1 if field=='yaw' else 1;minus=value[0]-base;plus=value[1]-base;correct=bool(sign*minus.mean()<0 and sign*plus.mean()>0);report['commands'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_minus_slope=float(-sign*minus.mean()/offset),mean_plus_slope=float(sign*plus.mean()/offset),correct_minus_fraction=float((sign*minus< -1e-7).mean()),correct_plus_fraction=float((sign*plus>1e-7).mean()),wrong_minus_fraction=float((sign*minus>1e-7).mean()),wrong_plus_fraction=float((sign*plus< -1e-7).mean()),mean_both_directions_correct=correct);counts[task['role']][field]+=int(correct)
     counts[task['role']]['total']+=1;reports.append(report)
  vr=read(ROOT/'reserved-response-01/result.json');assert sha256(ROOT/'reserved-response-01/responses.npz')==vr['responses_sha256'];directions={k:0 for k in ['fine','calibration_off_after_prefix','motion_off_after_prefix']};target=0
  with np.load(ROOT/'reserved-response-01/responses.npz') as z:
   for g in range(vr['groups']):
    for arm in directions:
     mu=z[f'group{g}_{arm}_muscles'];p=head(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,-1,3);saved[f'group{g}_{arm}_prediction']=p;margin=float(-np.mean(p[1,vr['late_index']:,0]-p[0,vr['late_index']:,0])/2);directions[arm]+=int(margin>.0001);target+=int(arm=='fine' and margin>=.002)
 out=begin_run(ROOT/f'{a.variant}-evaluation-01',dict(program_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'protocol-01.json'),receipt=receipt,seed=None,optimizer_updates=0,scope='Stateless new decoder on frozen parent neural trajectories, not a new full recurrence. All development/late windows evaluated after fit, no selection or vehicle control.',actor_qualified=False,vehicle_control_sent=False));np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_multicorner_decoder_evaluation',actual_recordings=metrics,command_reports=reports,command_counts=counts,reserved_direction_counts=directions,reserved_target_pairs=target,predictions_sha256=sha256(out/'predictions.npz'),gas_brake_predictions_exact_from_parent=True,actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False,termination_reason='All original three recordings, 45 command windows and seed781191 same-body image pairs decoded from fixed source muscles.'))
 print(dict(counts=counts,directions=directions,pre_steering_reduction_percent=[100*(1-v['metrics']['pre']['new']['rmse'][0]/v['metrics']['pre']['original']['rmse'][0]) for v in metrics]))
if __name__=='__main__':main()
