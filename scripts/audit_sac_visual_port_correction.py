"""Independent evidence audit for the frozen-head four-port visual correction."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.sustained_turn_diagnostic import steering_yaw_alignment,paired_sustained_onsets
from scripts.sac_steering_experiment import dataset,onsets,SOURCE,SOURCE_SHA
from scripts.improve_sac_anticipation import command_metrics

def verify_metric(p,y,m,want):
 got=command_metrics(p,y,m)
 for key in ['weighted_mse','mae','rmse']:np.testing.assert_allclose(got[key],want[key],atol=1e-8,rtol=1e-6)
 return got

def weighted_head(x,y,w):
 mean=(x*w[:,None]).sum(0)/w.sum();ym=(y*w).sum()/w.sum();scale=x.std(0);scale=np.where(scale>1e-8,scale,1.);z=(x-mean)/scale
 beta=np.linalg.solve(z.T@(w[:,None]*z)+.001*w.sum()*np.eye(6),z.T@(w*(y-ym)))/scale
 return beta,ym-mean@beta

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
 root=Path(a.root);entries,items,extras=dataset();d=items[2][0]
 run=begin_run(a.output,dict(arguments=vars(a),seed=780507,program_sha256=sha256(__file__),helper_sha256=sha256('scripts/sac_steering_experiment.py'),entries=entries,scope='Independent stored metrics, weighted head, all frozen source state and same-body intervention audit; not another neural run',actor_deployed=False,vehicle_control_sent=False))
 out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False)
 try:
  assert sha256(SOURCE/'spatial-port.pt')==SOURCE_SHA
  for dd,c,e in items:assert dd['body_valid'][:,4].all()
  alignment,sign_evidence=steering_yaw_alignment([x[0] for x in items[:2]])
  stable=paired_sustained_onsets(d,alignment);mask=np.zeros(len(d['labels']),bool)
  for w in stable:mask|=(d['decision_time']>=w['time']-.8)&(d['decision_time']<=w['time']-.08)&(abs(d['body'][:,4])<.07)
  mask&=d['fit_mask'];train_pre=[]
  for dd,c,e in items[:2]:
   m=np.zeros(len(dd['labels']),bool)
   for w in onsets(dd):m|=(dd['decision_time']>=w['time']-.8)&(dd['decision_time']<=w['time']-.08)&(abs(dd['body'][:,4])<.07)
   train_pre.append(m[dd['fit_mask']])
  parent=root/'optical-fit-01';baseline=root/'optical-pre16-fit-01';candidate=root/'current-correction-fit-01'
  pr,br,cr=[read(p/'result.json') for p in [parent,baseline,candidate]]
  for p,r in [(parent,pr),(baseline,br),(candidate,cr)]:
   assert r['status'].startswith('completed_')
   assert sha256(p/'steering.pt')==r['checkpoint_sha256'] and sha256(p/'predictions.npz')==r['predictions_sha256']
  pc,bc,cc=[torch.load(p/'steering.pt',map_location='cpu',weights_only=True) for p in [parent,baseline,candidate]]
  for group in ['adapter','optical']:
   for k,v in bc[group].items():assert torch.equal(v,pc[group][k])
  for k in ['training_reference','training_scale']:assert torch.equal(bc['readout'][k],pc['readout'][k])
  for k in ['projection.weight','projection.bias']:assert torch.equal(bc['readout'][k][1:],pc['readout'][k][1:])
  for group in ['adapter','optical','readout']:
   for k,v in cc[group].items():assert torch.equal(v,bc[group][k])
  assert cc['port_correction_parent_sha256']==br['checkpoint_sha256']
  assert cc['source_checkpoint_sha256']==SOURCE_SHA and not cc['actor_qualified']
  for k in ['reference','scale']:assert torch.equal(cc['port_correction'][k],bc['adapter'][k])
  assert all(torch.isfinite(v).all() for v in cc['port_correction'].values())
  with np.load(parent/'predictions.npz') as pz,np.load(baseline/'predictions.npz') as bz,np.load(candidate/'predictions.npz') as cz:
   x=[];y=[]
   for i in [0,1]:
    dd=items[i][0];key=f'train_{i}_muscles';np.testing.assert_array_equal(pz[key],bz[key])
    z=((torch.tensor(pz[key])-bc['readout']['training_reference'])/bc['readout']['training_scale']).numpy().astype(float)
    x.append(z[dd['fit_mask']]);y.append(np.arctanh(dd['labels'][dd['fit_mask'],0]))
    np.testing.assert_array_equal(pz[f'train_{i}'][:,1:],bz[f'train_{i}'][:,1:])
    verify_metric(bz[f'train_{i}'],dd['labels'],dd['fit_mask'],br['train'][i])
   beta,bias=weighted_head(np.concatenate(x),np.concatenate(y),np.where(np.concatenate(train_pre),16.,1.))
   np.testing.assert_allclose(bc['readout']['projection.weight'][0].numpy(),beta,atol=1e-5,rtol=1e-6)
   np.testing.assert_allclose(bc['readout']['projection.bias'][0].item(),bias,atol=1e-5,rtol=1e-6)
   np.testing.assert_array_equal(pz['dev_muscles'],bz['dev_muscles']);np.testing.assert_array_equal(pz['dev'][:,1:],bz['dev'][:,1:])
   bp,cp=bz['dev'].copy(),cz['dev'].copy();verify_metric(bp,d['labels'],d['fit_mask'],br['dev']);verify_metric(cp,d['labels'],d['fit_mask'],cr['dev'])
  probe=read(root/'readout-weight-probe-01.json');verified_probe_rows=0
  for arm in ['control','descending','optical']:
   folder=root/(arm+'-fit-01');checkpoint=torch.load(folder/'steering.pt',map_location='cpu',weights_only=True)
   with np.load(folder/'predictions.npz') as z:
    xx=[]
    for i in [0,1]:
     value=((torch.tensor(z[f'train_{i}_muscles'])-checkpoint['readout']['training_reference'])/checkpoint['readout']['training_scale']).numpy().astype(float)
     xx.append(value[items[i][0]['fit_mask']])
    dev=((torch.tensor(z['dev_muscles'])-checkpoint['readout']['training_reference'])/checkpoint['readout']['training_scale']).numpy().astype(float)
    for weight in [1,4,16]:
     beta,bias=weighted_head(np.concatenate(xx),np.concatenate(y),np.where(np.concatenate(train_pre),float(weight),1.));prediction=np.tanh(dev@beta+bias)
     row=next(row for row in probe['reports'] if row['arm']==arm and row['preweight']==weight)
     np.testing.assert_allclose(float(np.sqrt(np.mean((prediction[d['fit_mask']]-d['labels'][d['fit_mask'],0])**2))),row['rmse'],atol=1e-7)
     np.testing.assert_allclose(float(np.sqrt(np.mean((prediction[mask]-d['labels'][mask,0])**2))),row['pre_rmse'],atol=1e-7);verified_probe_rows+=1
  before=command_metrics(bp,d['labels'],mask);after=command_metrics(cp,d['labels'],mask)
  passed=cr['dev']['rmse'][0]<=.98*br['dev']['rmse'][0] and after['rmse'][0]<=.98*before['rmse'][0] and np.all(np.asarray(cr['dev']['rmse'][1:])<=1.02*np.asarray(br['dev']['rmse'][1:]))
  exchanges={}
  for name,normal in [('optical-pre16-stable-exchange-dev-01',bp),('current-correction-stable-exchange-dev-01',cp)]:
   r=read(root/name/'result.json');assert r['status']=='completed_same_body_precorner_visual_exchange';file=root/name/'predictions.npz';assert sha256(file)==r['predictions_sha256']
   with np.load(file) as z:
    np.testing.assert_allclose(z['normal'],normal,atol=2e-6,rtol=0)
    windows=[];normal_sq=[];source_sq=[]
    for i,w in enumerate(r['windows']):
     ids=z[f'window_{i}_indices'];m=z[f'window_{i}_pre_mask'];np.testing.assert_array_equal(z[f'window_{i}_normal'],z['normal'][ids])
     donor=np.searchsorted(d['decision_time'],d['decision_time'][ids]-w['time']+w['donor_time'],side='right')-1
     np.testing.assert_array_equal(donor,z[f'window_{i}_donor_indices'])
     delta=(z[f'window_{i}_normal'][:,0]-z[f'window_{i}_opposite_video'][:,0])*w['direction']
     nerr=(z[f'window_{i}_normal'][m,0]-d['labels'][ids[m],0])**2;serr=(z[f'window_{i}_source_path'][m,0]-d['labels'][ids[m],0])**2
     normal_sq.extend(nerr.tolist());source_sq.extend(serr.tolist())
     eligible=np.flatnonzero(m);effect=z[f'window_{i}_normal'][:,0]-z[f'window_{i}_source_path'][:,0];lead=None
     for j in range(len(eligible)-2):
      group=eligible[j:j+3]
      if np.all(abs(effect[group])>.005):lead=float(w['time']-d['decision_time'][ids[group[0]]]);break
     windows.append(dict(event=w['event'],pre_count=int(m.sum()),signed_normal_minus_opposite=float(delta[m].mean()),normal_pre_rmse=float(np.sqrt(nerr.mean())),source_path_pre_rmse=float(np.sqrt(serr.mean())),first_3_label_correction_effect_lead_seconds=lead,independent_same_direction_donor=w.get('same_direction_donor_event') is not None))
    exchanges[name]=dict(windows=windows,normal_pre_rmse=float(np.sqrt(np.mean(normal_sq))),source_path_pre_rmse=float(np.sqrt(np.mean(source_sq))),expected_order_windows=sum(w['signed_normal_minus_opposite']>0 for w in windows),total_windows=len(windows),normal_replay_max_error=float(np.max(abs(z['normal']-normal))))
  out.update(status='completed_visual_port_correction_evidence_audit',baseline=br['dev'],candidate=cr['dev'],baseline_pre=before,candidate_pre=after,pre_labels=int(mask.sum()),sign_evidence=sign_evidence,
      global_steering_rmse_change_percent=100*(cr['dev']['rmse'][0]/br['dev']['rmse'][0]-1),pre_steering_rmse_change_percent=100*(after['rmse'][0]/before['rmse'][0]-1),development_gate_passed=bool(passed),weighted_head_train_only_verified=True,exploratory_probe_rows_independently_recomputed=verified_probe_rows,all_source_adapters_and_readout_byte_exact=True,exchanges=exchanges,
      termination_reason='Frozen baseline states/heads, TRAIN weighted fit, original masks/metrics and same-body visual/correction interventions verified; actor and generalization remain unqualified')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit correction audit failure')
 write_json(run/'result.json',out);print(json.dumps(out))
 if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
