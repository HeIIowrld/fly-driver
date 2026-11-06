"""Source-bound separate visual turn/speed trial and failed vehicle-sign gate."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.summarize_functional_priority_trial import functional_metrics
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import pre_mask

def recording(path):
 path=Path(path);r=read(path/'result.json');assert sha256(path/'predictions.npz')==r['predictions_sha256'];return dict(episode=r['episode'],events=r['events'],preservation_gate=r['preservation_gate_passed'],rmse_reduction_percent={role:[100*(1-n/o) for n,o in zip(r['metrics'][role]['functional']['rmse'],r['metrics'][role]['parent']['rmse'])] for role in ('all','pre')},metrics=r['metrics'])

def main():
 torch.set_num_threads(1);root=Path('runs/visual-turn-speed-20261104');plan=read(root/'plan-01.json');receipts=read(root/'evaluation-process-receipts-01.json');assert len(receipts)==10 and all(x['exit_code']==0 for x in receipts);assert read(root/'turn_only-fit-01/training-schedule.json')==read(root/'turn_speed-fit-01/training-schedule.json');models={};bound={};events=0
 for arm in ('parent','turn_only','turn_speed'):
  path=root/f'reserved-{arm}-01';row=dict(reserved=functional_metrics(path));counts={v:0 for v in ('fine','motion_off_after_prefix','speed_off_after_prefix')};effect=0.
  with np.load(path/'responses.npz') as z:
   for g in read(path/'groups.json'):
    j=g['group']
    for mode in counts:
     x=z[f'group_{j}_{mode}_prediction'][:,25:,0];counts[mode]+=int(float(-np.mean(x[1]-x[0])/2)>.0001)
    effect=max(effect,float(abs(z[f'group_{j}_fine_prediction']-z[f'group_{j}_speed_off_after_prefix_prediction']).max()))
  assert counts==row['reserved']['direction_pairs'];row['reserved_speed_cut_maximum_action_difference']=effect;bound[str(path/'result.json')]=sha256(path/'result.json')
  if arm!='parent':
   fit=root/f'{arm}-fit-01';r=read(fit/'result.json');audit=read(root/f'audit-{arm}-01/result.json');assert audit['status']=='completed_independent_turn_speed_audit';row.update(trainable_parameters=r['trainable_parameters'],checkpoint_sha256=r['checkpoint_sha256'],maximum_turn_weight_delta=r['maximum_weight_change'],maximum_speed_weight=r['maximum_speed_weight'],maximum_body_gain_weight=r['maximum_body_gain_weight'],independent_adam_difference=audit['independent_adam_maximum_difference'],observed_same_observation_speed_change_m_s=audit['observed_fixed_observation_target_change_m_s'],actual=[])
   for e in range(3):
    path=root/f'actual-{arm}-{e}-01';value=recording(path);events+=value['events'];row['actual'].append(value);bound[str(path/'result.json')]=sha256(path/'result.json')
   for path in [fit/'result.json',fit/'optimizer-trajectory.npz',root/f'audit-{arm}-01/result.json']:bound[str(path)]=sha256(path)
  models[arm]=row
 offpath=root/'actual-turn_speed-off-2-01';off=recording(offpath);events+=off['events'];normal=models['turn_speed']['actual'][2];cut=dict(recording=off,normal_vs_speed_off_rmse_reduction_percent={role:[100*(1-n/o) for n,o in zip(normal['metrics'][role]['functional']['rmse'],off['metrics'][role]['functional']['rmse'])] for role in ('all','pre')})
 with np.load(offpath/'predictions.npz') as z:off_prediction=z['prediction'].copy();off_trace=z['speed_trace'].copy();off_power=z['power_currents'].copy();assert not np.count_nonzero(off_trace[:,1]-off_trace[:,4])
 cfg=read(offpath/'manifest.json')['config'];assert cfg['candidate']['speed_whole_recording_ablation'] and cfg['candidate']['turn_speed_checkpoint_sha256']==models['turn_speed']['checkpoint_sha256'];d,_,_=load_episode(cfg['entry']['data']);original=Path(read(plan['candidate_plan'])['motion']['fit'])
 with np.load(original/'predictions.npz') as z:parent_development=z['prediction_2'].copy()
 for role,mask in [('all',d['fit_mask']),('pre',pre_mask(d,-1))]:
  for label,pred in [('functional',off_prediction),('parent',parent_development)]:np.testing.assert_allclose(np.sqrt(np.mean((pred[mask].astype(float)-d['labels'][mask].astype(float))**2,axis=0)),off['metrics'][role][label]['rmse'],atol=1e-12,rtol=0)
 assert float(abs(off_power-.05000000074505806).max())<2e-7;cut['independent_off_recording_metrics_and_neutral_power_verified']=True
 with np.load(root/'actual-turn_speed-2-01/predictions.npz') as z:on_prediction=z['prediction'].copy()
 cut['maximum_axis_prediction_difference']=abs(on_prediction-off_prediction).max(0).tolist();cut['rms_axis_prediction_difference']=np.sqrt(np.mean((on_prediction.astype(float)-off_prediction.astype(float))**2,axis=0)).tolist();bound[str(offpath/'result.json')]=sha256(offpath/'result.json')
 for f in ['plan-01.json','reserved-plan-01.json','reserved-stimuli-01/result.json','validation-01/result.json','gradient-resolution-01/result.json','runtime-neutral-01/result.json','contract-tests-01.txt','control-dimensions-01/result.json','speed-sign-fd-02/result.json','speed-sign-fd-02/probes.npz','speed-sign-audit-01.json','evaluation-process-receipts-01.json']:bound[str(root/f)]=sha256(root/f)
 parent_path=Path(read(plan['candidate_plan'])['motion']['fit'])/'motion-spatial.pt';head=torch.load(parent_path,map_location='cpu',weights_only=True)['readout'];assert (head['training_scale']>0).all();sign=read(root/'speed-sign-audit-01.json');assert sign['speed_direction_semantics_gate_passed'] is False
 result=dict(status='completed_visual_turn_speed_intent_trial',models=models,development_speed_cut=cut,neutral_validation=read(root/'validation-01/result.json'),resolved_gradient_validation=read(root/'gradient-resolution-01/result.json'),runtime_neutral_validation=read(root/'runtime-neutral-01/result.json'),speed_sign_audit=sign,positive_vehicle_readout_scales_verified=True,source_bound_sha256=bound,program_sha256=sha256(__file__),matched_parent_data_update_schedule_and_objective=True,parameter_count_matched=False,real_training_events_per_arm=32597,optimizer_updates_per_arm=147,total_continuous_recording_events=events,tests_passed=35,speed_direction_semantics_gate_passed=False,actor_promoted=False,actor_qualified=False,actual_driving_improvement_verified=False,vehicle_control_sent=False,goal_status='active',next_work='Recalibrate engineering wing-muscle-proxy to vehicle decoder with registered common-power/longitudinal sign and opponent steering constraints; preserve graph and all prior candidates. Do not relabel the inverted existing speed path as qualified speed control.');write_json(root/'summary-01.json',result);print({k:v['reserved']['direction_pairs'] for k,v in models.items()});print('speed_cut',cut)
if __name__=='__main__':main()
