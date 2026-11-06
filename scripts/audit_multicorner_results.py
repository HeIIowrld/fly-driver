"""Independent NumPy steering, constraint and body/command coverage verification."""
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.multicorner_command_data import ROOT,collections
from scripts.paired_decoder_data import load_data
from scripts.improve_sac_anticipation import command_metrics

def main():
 protocol=read(ROOT/'protocol-01.json');source=collections(protocol);_,items,_,_,pre=load_data(read(protocol['data_plan']));source_counts={role:dict(total=0,yaw=0,speed=0) for role in ['TRAIN','TRAIN_diagnostic','development']};body_reports=[];histories=[]
 for path,r in source:
  assert r['full_prediction_parity']==0 and r['full_muscle_parity']==0 and all(v['neutral_branch_parity']==0 for v in r['reports']);episode=r['episode']
  with np.load(path/'neutral-full.npz') as z:histories.append(z['muscles'].copy())
  with np.load(path/'probes.npz') as z:
   body=np.concatenate([z[f'case{j}_body'] for j in range(15)]);native=np.concatenate([z[f'case{j}_native'] for j in range(15)]);valid=np.concatenate([z[f'case{j}_native_valid'] for j in range(15)]);ranges=[]
   for i in range(19):
    values=native[:,i][valid[:,i]];ranges.append(None if len(values)==0 else [float(values.min()),float(values.max())])
   body_reports.append(dict(episode=episode,body_speed_range_m_s=[float(body[:,16].min()),float(body[:,16].max())],body_yaw_range_rad_s=[float(body[:,4].min()),float(body[:,4].max())],pitch_roll_range_rad=[[float(body[:,i].min()),float(body[:,i].max())] for i in range(2)],native_valid_fraction=valid.mean(0).tolist(),native_valid_ranges=ranges))
  for row in r['reports']:
   bucket=source_counts[row['task']['role']];bucket['total']+=1
   for field in ['yaw','speed']:bucket[field]+=int(row['commands'][field]['mean_both_directions_correct'])
 variants=['affine','quadratic'];models={}
 for variant in variants:
  fit=ROOT/f'{variant}-fit-01';evaluation=ROOT/f'{variant}-evaluation-01';fr=read(fit/'result.json');er=read(evaluation/'result.json');assert sha256(fit/'decoder.pt')==fr['checkpoint_sha256'] and sha256(evaluation/'predictions.npz')==er['predictions_sha256'];cp=torch.load(fit/'decoder.pt',map_location='cpu',weights_only=True);s={k:v.numpy().astype(float) for k,v in cp['readout'].items()};degree=cp['degree']
  def features(mu):
   z=(mu.astype(float)-s['base.training_reference'])/s['base.training_scale']
   if degree==1:return z
   q=np.stack([z[...,i]*z[...,j] for i in range(6) for j in range(i,6)],-1);return np.concatenate([z,(q-s['quadratic_reference'])/s['quadratic_scale']],-1)
  def decode(mu):return np.tanh(features(mu)@s['steering.weight'][0]+s['steering.bias'][0]).astype(np.float32)
  errors=[];metric_errors=[];buckets={k:dict(total=0,yaw=0,speed=0) for k in source_counts};max_yaw_violation=-np.inf
  with np.load(evaluation/'predictions.npz') as saved:
   for e,mu in enumerate(histories):
    steer=decode(mu);errors.append(float(abs(steer-saved[f'prediction_{e}'][:,0]).max()));d=items[e][0]
    for role,mask in [('all',d['fit_mask']),('pre',pre[e])]:
     rmse=float(np.sqrt(np.mean((steer[mask].astype(float)-d['labels'][mask,0])**2)));metric_errors.append(abs(rmse-er['actual_recordings'][e]['metrics'][role]['new']['rmse'][0]))
   for e,(path,r) in enumerate(source):
    with np.load(path/'probes.npz') as z:
     for j,row in enumerate(r['reports']):
      role=row['task']['role'];bucket=buckets[role];bucket['total']+=1;neutral=decode(z[f'case{j}_neutral_muscles']);p=decode(z[f'case{j}_yaw_muscles']);bucket['yaw']+=int(np.mean(p[0]-neutral)>0 and np.mean(p[1]-neutral)<0);bucket['speed']+=int(row['commands']['speed']['mean_both_directions_correct']);assert abs(p-saved[f'episode{e}_case{j}_yaw_prediction'][:,:,0]).max()<2e-6
      if role=='TRAIN':
       phi=features(z[f'case{j}_yaw_muscles'])-features(z[f'case{j}_neutral_muscles']);delta=phi@s['steering.weight'][0]/.01;max_yaw_violation=max(max_yaw_violation,float((-delta[0]).max()),float(delta[1].max()),float(-delta[0].mean()+.005),float(delta[1].mean()+.005))
   directions={k:0 for k in er['reserved_direction_counts']};targets=0;vr=read(ROOT/'reserved-response-01/result.json')
   for g in range(vr['groups']):
    for arm in directions:
     p=saved[f'group{g}_{arm}_prediction'];margin=float(-np.mean(p[1,vr['late_index']:,0]-p[0,vr['late_index']:,0])/2);directions[arm]+=int(margin>.0001);targets+=int(arm=='fine' and margin>=.002)
  assert max(errors)<2e-6 and max(metric_errors)<2e-6 and max_yaw_violation<1e-7;assert directions==er['reserved_direction_counts'] and targets==er['reserved_target_pairs'];assert all(buckets[k]['yaw']==er['command_counts'][k]['yaw'] and buckets[k]['speed']==er['command_counts'][k]['speed'] for k in buckets);models[variant]=dict(maximum_numpy_errors=errors,maximum_rmse_error=max(metric_errors),maximum_train_yaw_constraint_violation=max_yaw_violation,command_counts=buckets,reserved_direction_counts=directions,reserved_target_pairs=targets,steering_pre_rmse_reduction_percent=[100*(1-v['metrics']['pre']['new']['rmse'][0]/v['metrics']['pre']['original']['rmse'][0]) for v in er['actual_recordings']])
 write_json(ROOT/'independent-audit-01.json',dict(status='completed_independent_multicorner_audit',models=models,parent_command_counts=source_counts,body_coverage=body_reports,total_full_neural_events=sum(r['full_events'] for path,r in source),total_command_windows=sum(len(r['reports']) for path,r in source),actor_qualified=False,actual_driving_improvement_verified=False));print(models)
if __name__=='__main__':main()
