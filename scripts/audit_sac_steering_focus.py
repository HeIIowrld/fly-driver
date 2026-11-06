"""Independent saved-data and TRAIN-only readout audit, never a live actor."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.anticipation_targets import ridge_fit
from flybrain_drive.sac_unified_student import read
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment,paired_sustained_onsets
from scripts.sac_steering_experiment import dataset,SOURCE,SOURCE_SHA,onsets,paired_onsets
from scripts.improve_sac_anticipation import command_metrics

def check_metrics(actual,want):
 for key in ['weighted_mse','mae','rmse']:np.testing.assert_allclose(actual[key],want[key],atol=1e-8,rtol=1e-6)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
 root=Path(a.root);entries,items,extras=dataset();d=items[2][0]
 if sha256(SOURCE/'spatial-port.pt')!=SOURCE_SHA:raise ValueError('Source changed')
 source=torch.load(SOURCE/'spatial-port.pt',map_location='cpu',weights_only=True);source_metrics=read(SOURCE/'result.json')['dev']
 run=begin_run(a.output,dict(arguments=vars(a),seed=780503,program_sha256=sha256(__file__),helper_sha256=sha256('scripts/sac_steering_experiment.py'),source_checkpoint_sha256=SOURCE_SHA,entries=entries,scope='Independent stored prediction, calibration and train-only ridge audit, not neural rerun or live qualification',actor_deployed=False,vehicle_control_sent=False))
 result=dict(status='failed',actor_deployed=False,vehicle_control_sent=False)
 try:
  reports={};predictions={}
  alignment,sign_evidence=steering_yaw_alignment([item[0] for item in items[:2]])
  stable_windows=paired_sustained_onsets(d,alignment);stable_mask=np.zeros(len(d['labels']),dtype=bool)
  for w in stable_windows:stable_mask|=(d['decision_time']>=w['time']-.8)&(d['decision_time']<=w['time']-.08)&(abs(d['body'][:,4])<.07)
  stable_mask&=d['fit_mask']
  with np.load(SOURCE/'predictions.npz') as src:source_pre=command_metrics(src['dev'],d['labels'],stable_mask)
  for arm in ['control','descending','optical']:
   p=root/(arm+'-fit-01');r=read(p/'result.json')
   if r['status']!='completed_fixed_flight_steering_focus' or sha256(p/'steering.pt')!=r['checkpoint_sha256'] or sha256(p/'predictions.npz')!=r['predictions_sha256']:raise ValueError('Exact completed fit required: '+arm)
   cp=torch.load(p/'steering.pt',map_location='cpu',weights_only=True)
   if cp['source_checkpoint_sha256']!=SOURCE_SHA or cp['actor_qualified']:raise ValueError('Bad lineage/qualification')
   for k in ['training_reference','training_scale']:assert torch.equal(cp['readout'][k],source['readout'][k])
   for k in ['projection.weight','projection.bias']:assert torch.equal(cp['readout'][k][1:],source['readout'][k][1:])
   for k,v in cp['adapter'].items():
    if arm!='descending' or k!='projection.weight':assert torch.equal(v,source['adapter'][k])
   if arm=='optical':
    raw=np.concatenate([x[dd['fit_mask'],64:] for x,(dd,c,e) in zip(extras[:2],items[:2])])
    np.testing.assert_allclose(cp['optical']['reference'].numpy(),raw.mean(0),atol=1e-8,rtol=1e-6)
    np.testing.assert_allclose(cp['optical']['scale'].numpy(),np.maximum(raw.std(0),1e-4),atol=1e-8,rtol=1e-6)
   else:assert cp['optical'] is None
   for group in ['adapter','readout','optical']:
    if cp.get(group):assert all(torch.isfinite(v).all() for v in cp[group].values())
   with np.load(p/'predictions.npz',allow_pickle=False) as z:
    dev=z['dev'].copy();x=[];y=[]
    for i in [0,1]:
     dd=items[i][0];check_metrics(command_metrics(z[f'train_{i}'],dd['labels'],dd['fit_mask']),r['train'][i])
     mu=torch.tensor(z[f'train_{i}_muscles'])
     value=((mu-cp['readout']['training_reference'])/cp['readout']['training_scale']).numpy()
     x.append(value[dd['fit_mask']]);y.append(dd['labels'][dd['fit_mask'],:1])
    fit=ridge_fit(np.concatenate(x),np.arctanh(np.concatenate(y)),.001)
    beta=fit['weights'][:,0]/fit['scale'];bias=fit['target_mean'][0]-fit['mean']@beta
    np.testing.assert_allclose(cp['readout']['projection.weight'][0].numpy(),beta,atol=1e-5,rtol=1e-6)
    np.testing.assert_allclose(cp['readout']['projection.bias'][0].item(),bias,atol=1e-5,rtol=1e-6)
   check_metrics(command_metrics(dev,d['labels'],d['fit_mask']),r['dev'])
   pre=np.zeros(len(dev),dtype=bool)
   for w in onsets(d):pre|=(d['decision_time']>=w['time']-.8)&(d['decision_time']<=w['time']-.08)&(abs(d['body'][:,4])<.07)
   pre&=d['fit_mask']
   reports[arm]=dict(metrics=r['dev'],pre_transition_metrics=command_metrics(dev,d['labels'],pre),pre_transition_labels=int(pre.sum()),sustained_pre_metrics=command_metrics(dev,d['labels'],stable_mask),sustained_pre_labels=int(stable_mask.sum()),updates=r['optimizer_updates'],checkpoint_sha256=r['checkpoint_sha256'])
   predictions[arm]=dev
  control=reports['control']['metrics'];comparisons={}
  for arm in ['descending','optical']:
   m=reports[arm]['metrics'];gate=m['rmse'][0]<=.98*min(source_metrics['rmse'][0],control['rmse'][0]) and np.all(np.asarray(m['rmse'][1:])<=1.02*np.asarray(source_metrics['rmse'][1:]))
   comparisons[arm]=dict(sustained_pre_rmse_change_vs_source_percent=100*(reports[arm]['sustained_pre_metrics']['rmse'][0]/source_pre['rmse'][0]-1),sustained_pre_rmse_change_vs_control_percent=100*(reports[arm]['sustained_pre_metrics']['rmse'][0]/reports['control']['sustained_pre_metrics']['rmse'][0]-1),steering_rmse_change_vs_source_percent=100*(m['rmse'][0]/source_metrics['rmse'][0]-1),steering_rmse_change_vs_control_percent=100*(m['rmse'][0]/control['rmse'][0]-1),development_steering_gate_passed=bool(gate))
  exchanges={}
  for path in sorted(root.glob('*exchange*/result.json')):
   if path.parent.name not in ['source-exchange-dev-01','source-stable-exchange-dev-01','control-stable-exchange-dev-01','descending-stable-exchange-dev-01','optical-stable-exchange-dev-01']:continue
   r=read(path)
   if r['status']!='completed_same_body_precorner_visual_exchange':raise ValueError('Incomplete exchange')
   file=path.parent/'predictions.npz'
   if sha256(file)!=r['predictions_sha256']:raise ValueError('Changed exchange arrays')
   manifest=read(path.parent/'manifest.json');fit=manifest['config']['arguments'].get('fit');arm=Path(fit).name.split('-')[0] if fit else None
   with np.load(file,allow_pickle=False) as z:
    if arm:np.testing.assert_array_equal(z['normal'],predictions[arm])
    else:
     with np.load(SOURCE/'predictions.npz',allow_pickle=False) as original:np.testing.assert_array_equal(z['normal'],original['dev'])
    windows=[]
    for i,w in enumerate(r['windows']):
     ids=z[f'window_{i}_indices'];mask=z[f'window_{i}_pre_mask'];donor=z[f'window_{i}_donor_indices']
     expected=np.searchsorted(d['decision_time'],d['decision_time'][ids]-w['time']+w['donor_time'],side='right')-1
     np.testing.assert_array_equal(donor,expected)
     np.testing.assert_array_equal(z[f'window_{i}_normal'],z['normal'][ids])
     delta=(z[f'window_{i}_normal'][:,0]-z[f'window_{i}_opposite_video'][:,0])*w['direction']
     np.testing.assert_allclose(float(delta[mask].mean()),w['mean_signed_normal_minus_opposite'],atol=1e-7)
     eligible=np.flatnonzero(mask);lead=None
     for position in range(len(eligible)-2):
      group=eligible[position:position+3]
      if np.all(delta[group]>.005):lead=float(w['time']-d['decision_time'][ids[group[0]]]);break
     window=dict(event=w['event'],pre_labels=int(mask.sum()),signed_delta=float(delta[mask].mean()),fraction_signed_positive=float(np.mean(delta[mask]>0)),first_three_label_effect_lead_seconds=lead)
     if f'window_{i}_same_direction_video' in z.files:
      same=(z[f'window_{i}_same_direction_video'][:,0]-z[f'window_{i}_opposite_video'][:,0])*w['direction']
      window['same_vs_opposite_signed_delta']=float(same[mask].mean())
      window['independent_same_direction_donor']=w.get('same_direction_donor_event') is not None
     window['normal_pre_steering_mse']=float(np.mean((z[f'window_{i}_normal'][mask,0]-d['labels'][ids[mask],0])**2))
     if f'window_{i}_source_path' in z.files:
      window['source_path_pre_steering_mse']=float(np.mean((z[f'window_{i}_source_path'][mask,0]-d['labels'][ids[mask],0])**2))
     windows.append(window)
   exchanges[path.parent.name]=dict(windows=windows,right_direction_windows=sum(w['signed_delta']>0 for w in windows),total_windows=len(windows),independent_same_direction_control_windows=sum(w.get('independent_same_direction_donor',False) for w in windows))
  result.update(status='completed_steering_focus_evidence_audit',fits=reports,comparisons=comparisons,exchanges=exchanges,source_metrics=source_metrics,source_sustained_pre_metrics=source_pre,sign_evidence=sign_evidence,termination_reason='Stored metrics, TRAIN-only analytical steering fit, calibration, unchanged pedal rows/port provenance and counterfactual joins verified; no generalization/live/physiology claim')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit audit failure')
 write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
