"""Independent optimizer/state-journal and paired/continuous metric audit."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_controlled_motion_function import digest
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import pre_mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=['continuous','parent_prefix'],required=True);p.add_argument('--fit-only',action='store_true');a=p.parse_args();torch.set_num_threads(1);root=Path(a.root);fit=root/f'{a.arm}-fit-01';r=read(fit/'result.json');m=read(fit/'manifest.json')['config'];plan=m['plan'];cp=torch.load(fit/'descending-intent.pt',map_location='cpu',weights_only=True);components,motion,parent=load_readonly_motion_candidate(read(plan['candidate_plan']));assert r['status']=='completed_descending_intent_fit' and sha256(fit/'descending-intent.pt')==r['checkpoint_sha256'] and cp['schema']=='flybrain.descending_intent.v1' and cp['parent_candidate']==m['parent_candidate']==parent and cp['actor_qualified'] is False
 hashes={}
 for key,index in [('model',0),('readout',1),('adapter',2),('optical',3),('body_port',4),('motion_port',None)]:
  before=(components[index] if index is not None else motion).state_dict();after=cp[key]
  if key=='adapter':
   assert set(before)==set(after);initial=before['projection.weight'].numpy().copy();weight=after['projection.weight'].numpy().copy();before={k:v for k,v in before.items() if k!='projection.weight'};after={k:v for k,v in after.items() if k!='projection.weight'}
  assert digest(before)==digest(after);hashes[key]=digest(before)
 assert weight.shape==(1,320) and np.isfinite(weight).all() and abs(weight-initial).max()<=plan['maximum_weight_delta']+1e-7
 for f,h in plan['source_sha256'].items():assert sha256(fit/'executed-source'/f)==h
 schedule=read(fit/'training-schedule.json');records=read(fit/'training-records.json');journal=read(fit/'state-journal.json');assert len(schedule)==len(records)==cp['optimizer_updates']==r['optimizer_updates']==147 and sha256(fit/'state-journal.json')==r['state_journal_sha256'];states=torch.load(Path(plan['prefix_run'])/'parent-prefix-states.pt',map_location='cpu',weights_only=True);assert sha256(Path(plan['prefix_run'])/'parent-prefix-states.pt')==r['parent_prefix_states_sha256'];episodes={e:load_episode(m['entries'][e]['data'])[0] for e in (0,1)};expected=[];expected_blocks=[]
 for e,d in episodes.items():
  blocks=(len(d['scene'])+plan['window']-1)//plan['window'];selected=np.union1d(np.unique(np.flatnonzero(pre_mask(d,-1))//plan['window']),np.linspace(0,blocks-1,plan['anchor_blocks_per_episode'],dtype=int))
  expected.extend((e,int(b*plan['window']),min(int((b+1)*plan['window']),len(d['scene']))) for b in selected);expected_blocks.extend((e,b,min(b+plan['window'],len(d['scene']))) for b in range(0,len(d['scene']),plan['window']))
 assert [(x['episode'],x['begin'],x['end']) for x in journal]==expected_blocks;actual=[(t['episode'],t['begin'],t['end']) for kind,t in schedule if kind=='real'];assert actual==expected and len(actual)==93;synthetic={(t['source_slot'],t['pair'],t['temperature']) for kind,t in schedule if kind=='synthetic'};assert synthetic=={(s,k,t) for s in range(3) for k in range(0,12,2) for t in (40.,80.,110.)}
 last={};injections=0;resumes=0
 for row in journal:
  e,b,end=row['episode'],row['begin'],row['end'];d=episodes[e];assert row['previous_time_before']==(float(d['decision_time'][b-1]) if b else None) and row['previous_time_after']==float(d['decision_time'][end-1])
  if b:assert row['incoming_numeric_sha256']==last[e]
  else:assert row['incoming_numeric_sha256']==state_digest(states[e,0])
  injected=a.arm=='parent_prefix' and row['selected'];assert row['parent_state_injected']==injected;injections+=int(injected)
  if injected:assert row['start_numeric_sha256']==state_digest(states[e,b])
  else:assert row['start_numeric_sha256']==row['incoming_numeric_sha256']
  if 'synthetic_resume_sha256' in row:assert row['synthetic_resume_sha256']==row['detach_numeric_sha256'];resumes+=1
  last[e]=row['detach_numeric_sha256']
 assert resumes==54 and injections==r['parent_state_injections']==(93 if a.arm=='parent_prefix' else 0);assert sum(x['end']-x['begin'] for x in journal)==r['real_events']==32597
 for (kind,task),rec in zip(schedule,records):assert rec['kind']==kind and rec['task']==task and abs(sum(rec['parts'].values())-rec['loss'])<2e-6
 with np.load(fit/'optimizer-trajectory.npz') as z:before=z['weight_before'].astype(float);after=z['weight_after'].astype(float);grads=z['applied_gradients'].astype(float)
 assert sha256(fit/'optimizer-trajectory.npz')==r['optimizer_trajectory_sha256'] and before.shape==after.shape==grads.shape==(147,1,320);np.testing.assert_array_equal(before[0],initial);np.testing.assert_array_equal(before[1:],after[:-1]);np.testing.assert_array_equal(after[-1],weight);assert np.isfinite(grads).all() and np.linalg.norm(grads.reshape(147,-1),axis=1).max()<=1+2e-6
 first=np.zeros_like(initial,dtype=float);second=first.copy();value=initial.astype(float).copy();maximum=0.
 for index,g in enumerate(grads,1):
  first=.9*first+.1*g;second=.999*second+.001*g*g;value-=plan['learning_rate']*(first/(1-.9**index))/(np.sqrt(second/(1-.999**index))+1e-8);value=np.clip(value,initial-plan['maximum_weight_delta'],initial+plan['maximum_weight_delta']);maximum=max(maximum,float(abs(value-after[index-1]).max()))
 assert maximum<2e-5
 report=dict(status='completed_independent_continuous_intent_fit_audit',independent_adam_maximum_difference=maximum,fixed_components_sha256=hashes,projection_maximum_weight_delta=float(abs(weight-initial).max()),real_events_verified=32597,real_windows_verified=93,synthetic_slots_verified=54,continuous_state_journal_blocks_verified=len(journal),parent_state_injections=injections,synthetic_exact_resumes=resumes,full_neural_replay_by_audit=False,vehicle_control_sent=False,actor_qualified=False)
 if not a.fit_only:
  path=root/f'reserved-{a.arm}-01';rr=read(path/'result.json');cfg=read(path/'manifest.json')['config'];source=Path(cfg['arguments']['stimuli']);sm=read(source/'manifest.json')['config'];assert sm['plan_sha256']==plan['reserved_test_plan_sha256'] and sha256(path/'responses.npz')==rr['responses_sha256'] and sha256(path/'groups.json')==rr['groups_sha256'] and cfg['candidate']['descending_intent_checkpoint_sha256']==r['checkpoint_sha256'];readout={k:v.numpy().astype(float) for k,v in cp['readout'].items()};counts={k:0 for k in ('fine','motion_off_after_prefix')};muscle_max=0.;target=0
  with np.load(path/'responses.npz') as z:
   for g in read(path/'groups.json'):
    j=g['group'];assert hashlib.sha256(b''.join(z[f'group_{j}_{k}'].tobytes() for k in ('body21','body21_valid','native19','native19_valid'))).hexdigest()==g['fixed_body_sha256']
    for arm in counts:
     mu=z[f'group_{j}_{arm}_muscles'];prediction=z[f'group_{j}_{arm}_prediction'];logits=((mu-readout['training_reference'])/readout['training_scale'])@readout['projection.weight'].T+readout['projection.bias'];formula=np.concatenate([np.tanh(logits[...,:1]),1/(1+np.exp(-logits[...,1:]))],axis=-1);muscle_max=max(muscle_max,float(abs(formula-prediction).max()));np.testing.assert_array_equal(prediction[:,:11],z[f'group_{j}_fine_prediction'][:,:11]);late=prediction[:,sm['plan']['evaluation_first_frame']:];margin=float(sm['plan']['expected_canonical_steering_sign_per_left_bend']*np.mean(late[1,:,0]-late[0,:,0])/2);assert margin==g['branches'][arm]['hypothesized_bend_margin'];counts[arm]+=int(margin>sm['plan']['bend_ordering_minimum_margin']);target+=int(arm=='fine' and margin>=plan['target_margin'])
  assert counts==rr['hypothesized_positive_bend_pairs'] and muscle_max<2e-6;recorded=[];original=Path(read(plan['candidate_plan'])['motion']['fit']);bound={str(path/'result.json'):sha256(path/'result.json')}
  for e in range(3):
   path=root/f'actual-{a.arm}-{e}-01';rr=read(path/'result.json');cfg=read(path/'manifest.json')['config'];assert cfg['candidate']['descending_intent_checkpoint_sha256']==r['checkpoint_sha256'] and sha256(path/'predictions.npz')==rr['predictions_sha256'];d,_,_=load_episode(cfg['entry']['data']);pre=pre_mask(d,-1)
   with np.load(path/'predictions.npz') as z:prediction=z['prediction'].copy()
   with np.load(original/'predictions.npz') as z:baseline=z[f'prediction_{e}'].copy()
   assert len(prediction)==rr['events']==len(d['scene'])
   for role,mask in [('all',d['fit_mask']),('pre',pre)]:
    for key,x in [('parent',baseline),('functional',prediction)]:np.testing.assert_allclose(np.sqrt(np.mean((x[mask].astype(float)-d['labels'][mask].astype(float))**2,axis=0)),rr['metrics'][role][key]['rmse'],atol=1e-12,rtol=0)
   ratio=plan['actual_recording_preservation_maximum_ratio'];passed=all(rr['metrics']['all']['functional']['rmse'][k]<=ratio*rr['metrics']['all']['parent']['rmse'][k] for k in range(3)) and rr['metrics']['pre']['functional']['rmse'][0]<=ratio*rr['metrics']['pre']['parent']['rmse'][0];assert bool(passed)==rr['preservation_gate_passed'];recorded.append(dict(episode=e,events=rr['events'],preservation_gate=bool(passed)));bound[str(path/'result.json')]=sha256(path/'result.json')
  report.update(status='completed_independent_continuous_intent_audit',reserved_direction_counts=counts,reserved_target_pairs=target,independent_muscle_readout_difference=muscle_max,recordings=recorded,source_bound_result_sha256=bound)
 suffix='fit-audit' if a.fit_only else 'audit';out=begin_run(root/f'{suffix}-{a.arm}-01',dict(program_sha256=sha256(__file__),fit_checkpoint_sha256=r['checkpoint_sha256'],fit_result_sha256=sha256(fit/'result.json'),actor_qualified=False,vehicle_control_sent=False));write_json(out/'result.json',report)
if __name__=='__main__':main()
