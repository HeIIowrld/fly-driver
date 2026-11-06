"""Independent readout/metric/command/image arithmetic on candidate-owned trajectories."""
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data
from scripts.improve_sac_anticipation import command_metrics

ROOT=Path('runs/fixture-adjoint-learning-20261104')
def main():
 plan=read(ROOT/'plan-01.json');data_plan=read('runs/multicorner-command-20261104/protocol-01.json')['data_plan'];_,items,_,_,pre=load_data(read(data_plan));cp=torch.load(plan['fixed_decoder'],map_location='cpu',weights_only=True);s={k:v.numpy().astype(float) for k,v in cp['readout'].items()};assert cp['degree']==1;models={}
 def decode(mu):
  x=(mu.astype(float)-s['base.training_reference'])/s['base.training_scale'];steer=np.tanh(x@s['steering.weight'][0]+s['steering.bias'][0]);pedal=expit(x@s['base.projection.weight'][1:].T+s['base.projection.bias'][1:]);return np.column_stack([steer,pedal])
 for arm in ['stop_all','input_adjoint']:
  path=ROOT/f'{arm}-evaluation-01';r=read(path/'result.json');assert r['status']=='completed_fixture_adjoint_speed_evaluation' and sha256(path/'predictions.npz')==r['predictions_sha256'] and sha256(path/'reserved-groups.json')==r['reserved_groups_sha256'] and sha256(path/'real-image-groups.json')==r['real_image_groups_sha256'];errors=[];metric_errors=[];counts={k:dict(total=0,yaw=0,speed=0) for k in ['TRAIN','development']}
  with np.load(path/'predictions.npz') as z:
   for episode in range(3):
    p=decode(z[f'muscles_{episode}']);errors.append(float(abs(p-z[f'prediction_{episode}']).max()));d=items[episode][0]
    for role,mask in [('all',d['fit_mask']),('pre',pre[episode])]:
     measured=command_metrics(p,d['labels'],mask);reported=r['actual_recordings'][episode]['metrics'][role]['new'];metric_errors.append(float(abs(np.array(measured['rmse'])-reported['rmse']).max()))
   for j,row in enumerate(r['command_reports']):
    task=row['task'];role='TRAIN' if task['episode']<2 else 'development';counts[role]['total']+=1;neutral=z[f"prediction_{task['episode']}"][task['begin']:task['end']]
    for field in ['yaw','speed']:
     p=z[f'case{j}_{field}_prediction'];v=p[:,:,0] if field=='yaw' else p[:,:,1]-p[:,:,2];base=neutral[:,0] if field=='yaw' else neutral[:,1]-neutral[:,2];sign=-1 if field=='yaw' else 1;minus=v[0]-base;plus=v[1]-base;correct=bool(sign*minus.mean()<0 and sign*plus.mean()>0);counts[role][field]+=int(correct);assert correct==row['commands'][field]['mean_both_correct'];assert abs(float(minus.mean())-row['commands'][field]['mean_minus_change'])<1e-12 and abs(float(plus.mean())-row['commands'][field]['mean_plus_change'])<1e-12
   directions={k:0 for k in r['reserved_direction_counts']};target=0;native_control=0.;reserved=read(path/'reserved-groups.json')
   for row in reserved:
    g=row['group']
    for key in directions:
     p=z[f'group{g}_{key}_prediction'];margin=float(-np.mean(p[1,14:,0]-p[0,14:,0])/2);assert abs(margin-row['arms'][key]['signed_margin'])<1e-12;directions[key]+=int(margin>.0001);target+=int(key=='fine' and margin>=.002)
    native_control=max(native_control,float(abs(z[f'group{g}_fine_prediction']-z[f'group{g}_native_change_off_prediction']).max()))
   image_counts={k:0 for k in r['real_image_direction_counts']};image_targets={k:0 for k in image_counts};image_groups=read(path/'real-image-groups.json');native_image_control=0.
   for row in image_groups:
    g=row['group']
    for key in image_counts:
     p=z[f'image{g}_{key}_prediction'];margin=float(-np.mean(p[0,:,0]-p[1,:,0])/2);assert abs(margin-row['arms'][key]['signed_teacher_bend_margin'])<1e-12;image_counts[key]+=int(margin>0);image_targets[key]+=int(margin>=.002)
    native_image_control=max(native_image_control,float(abs(z[f'image{g}_fine_prediction']-z[f'image{g}_native_change_off_prediction']).max()))
  assert max(errors)<2e-6 and max(metric_errors)<2e-6 and counts==r['command_counts'] and directions==r['reserved_direction_counts'] and target==r['reserved_target_pairs'] and image_counts==r['real_image_direction_counts']
  models[arm]=dict(factual_speed_error_ranges=r['factual_speed_error_ranges'],factual_power_ranges=r['factual_power_ranges'],speed_policy_control_maximum_difference=r['speed_policy_control_max_difference'],maximum_independent_prediction_errors=errors,maximum_rmse_difference=max(metric_errors),command_counts=counts,reserved_direction_counts=directions,reserved_target_pairs=target,real_image_direction_counts=image_counts,real_image_target_counts=image_targets,native_synthetic_control_difference=native_control,native_real_image_control_difference=native_image_control,steering_pre_rmse_reduction_percent=[100*(1-v['metrics']['pre']['new']['rmse'][0]/v['metrics']['pre']['original']['rmse'][0]) for v in r['actual_recordings']],all_rmse_reduction_percent=[[100*(1-new/old) for new,old in zip(v['metrics']['all']['new']['rmse'],v['metrics']['all']['original']['rmse'])] for v in r['actual_recordings']],pre_rmse_reduction_percent=[[100*(1-new/old) for new,old in zip(v['metrics']['pre']['new']['rmse'],v['metrics']['pre']['original']['rmse'])] for v in r['actual_recordings']])
 write_json(ROOT/'evaluation-audit-01.json',dict(status='completed_independent_flight_speed_policy_evaluation_audit',models=models,actor_qualified=False,actual_driving_improvement_verified=False));print(models)
if __name__=='__main__':main()
