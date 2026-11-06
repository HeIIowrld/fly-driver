"""Fixed-schedule two-row pedal continuation on genuine unmodified low-speed labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.learned_flight_pedal_readout import LearnedFlightPedalReadout
ROOT=Path('runs/student-launch-learning-20261105')
SPLIT=Path('runs/thermal-control-improvement-20261104/newhot-cohort-01.json')

def datasets(evaluation):
    er=read(evaluation/'result.json');fn=evaluation/'predictions.npz'
    if sha256(fn)!=er['predictions_sha256']:raise ValueError('Audited original trajectory hash differs')
    items=[]
    with np.load(fn,allow_pickle=False) as pred:
      for i,ep in enumerate(read(SPLIT)['episodes']):
        root=Path(ep['data']);datafile=root/'actor-only.npz'
        if sha256(datafile)!=ep['arrays_sha256']:raise ValueError('Original role/actor arrays differ')
        with np.load(datafile,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        meta=read(root/'manifest.json')['config'];recording=Path(meta['arguments']['recording']);file=recording/'teacher/events.jsonl';events=[json.loads(v) for v in file.read_text(encoding='utf8').splitlines()];requests=[v for v in events if v['kind']=='teacher_request'];unchanged=np.zeros(len(d['labels']),dtype=bool)
        for j in np.flatnonzero(d['label_valid']):
            request=requests[int(d['teacher_request_index'][j])];u=request['request']['requested_control'];label=np.array([u['steering']/.6,u['throttle'],u['brake']]);np.testing.assert_allclose(label,d['labels'][j],rtol=0,atol=1e-8)
            unchanged[j]=request['unlimited']==request['limited'] and not request.get('automatic_shift_coast',False)
        low=d['label_valid']&unchanged&d['body_valid'][:,16]&(d['body'][:,16]<13)&~d['fit_mask'];high=d['fit_mask']&unchanged
        mu=pred[f'muscles_{i}'].copy();original=pred[f'prediction_{i}'].copy()
        if mu.shape!=(len(d['labels']),6) or not low.any() or not high.any():raise ValueError('Continuous current-cohort features and genuine low/high labels required')
        items.append(dict(data=d,muscles=mu,original=original,low=low,high=high,receipt=dict(role=ep['role'],data=str(root),arrays_sha256=sha256(datafile),teacher_events_sha256=sha256(file),low_events=np.flatnonzero(low).tolist(),high_labels=int(high.sum()),low_labels=int(low.sum()),low_speed_range_m_s=[float(d['body'][low,16].min()),float(d['body'][low,16].max())])))
    return items

def metrics(head,items):
    rows=[];saved={}
    with torch.no_grad():
      for i,v in enumerate(items):
        prediction=head(torch.tensor(v['muscles'])).numpy();saved[f'prediction_{i}']=prediction
        row=dict(episode=i,role=v['receipt']['role'],steering_max_difference=float(abs(prediction[:,0]-v['original'][:,0]).max()),groups={})
        for name,m in [('low',v['low']),('high',v['high'])]:
          labels=v['data']['labels'][m];old=v['original'][m];new=prediction[m];row['groups'][name]=dict(labels=int(m.sum()),baseline_rmse=np.sqrt(((old-labels)**2).mean(0)).tolist(),learned_rmse=np.sqrt(((new-labels)**2).mean(0)).tolist(),learned_mean=new.mean(0).tolist(),baseline_mean=old.mean(0).tolist(),label_mean=labels.mean(0).tolist(),coactivation_fraction=float(np.mean((new[:,1]>.02)&(new[:,2]>.02))))
        rows.append(row)
    return rows,saved

def main():
 p=argparse.ArgumentParser();p.add_argument('--actor-kind',choices=['parent','dual_guard'],required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(783105);plan=read('runs/dual-reference-learning-20261105/plan-01.json')
 checkpoint=Path('runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt') if a.actor_kind=='parent' else Path('runs/dual-reference-learning-20261105/dual_guard-fit-01/ports.pt');evaluation=Path(plan['direct_parent_evaluation']) if a.actor_kind=='parent' else Path('runs/dual-reference-learning-20261105/dual_guard-evaluation-01')
 items=datasets(evaluation);parent,_=load_muscle_interaction_head(plan['fixed_decoder']);head=LearnedFlightPedalReadout(parent);head.project();before={k:v.clone() for k,v in head.state_dict().items()};train=items[:2];assert all(v['receipt']['role']=='train' for v in train)
 config=dict(actor_kind=a.actor_kind,parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=sha256(checkpoint),fixed_decoder=plan['fixed_decoder'],fixed_decoder_sha256=plan['fixed_decoder_sha256'],evaluation=str(evaluation),evaluation_predictions_sha256=sha256(evaluation/'predictions.npz'),split=str(SPLIT),split_sha256=sha256(SPLIT),seed=783105,steps=3000,learning_rate=.003,low_loss_weight=5.,high_loss_weight=1.,high_anchor_weight=2.,coefficient_regularization=.001,actor_inputs='Existing six muscle proxies only; original normalization, steering and recurrent circuit frozen',source_sha256={f:sha256(f) for f in [__file__,'src/flybrain_drive/learned_flight_pedal_readout.py']},receipts=[v['receipt'] for v in items],development_used_for_fit=False,optimizer='New Adam for newly unfrozen14 pedal coefficients; frozen original circuit Adam history preserved in parent artifacts',selection='Fixed steps, no heldout early stopping, no action postprocessing')
 config.update(training_high_mse_limit_ratio=1.02,projection='Backtrack each learned coefficient step against each TRAIN episode and pedal MSE <= original*1.02; no development/student selection')
 out=begin_run(a.output,config);x=torch.tensor(np.concatenate([v['muscles'] for v in train]));target=torch.tensor(np.concatenate([v['data']['labels'] for v in train]),dtype=torch.float32);anchor=torch.tensor(np.concatenate([v['original'] for v in train]));low=torch.tensor(np.concatenate([v['low'] for v in train]));high=torch.tensor(np.concatenate([v['high'] for v in train]));optimizer=torch.optim.Adam(head.pedal_projection.parameters(),lr=config['learning_rate']);journal=[];trajectory={k:[] for k in ['before','gradient','proposal','after']};start=time.perf_counter()
 for source in config['source_sha256']: (out/('executed-'+Path(source).name)).write_bytes(Path(source).read_bytes())
 def vector():return np.concatenate([head.pedal_projection.weight.detach().numpy().reshape(-1),head.pedal_projection.bias.detach().numpy()]).copy()
 groups=[];offset=0
 for v in train:
  indices=torch.tensor(np.flatnonzero(v['high'])+offset);groups.append(indices);offset+=len(v['muscles'])
 high_limits=torch.stack([(anchor[idx,1:]-target[idx,1:]).square().mean(0)*1.02+1e-8 for idx in groups]);feasible=[]
 # Full trajectory states were independently audited already. Readout does not feed
 # recurrence; no neural state reset, synthetic low-speed state or teacher input.
 z=(x-head.parent.base.training_reference)/head.parent.base.training_scale
 for step in range(config['steps']):
  trajectory['before'].append(vector());old_w=head.pedal_projection.weight.detach().clone();old_b=head.pedal_projection.bias.detach().clone();prediction=head.pedal_projection(z)[:,1:].sigmoid();lo=(prediction[low]-target[low,1:]).square().mean();hi=(prediction[high]-target[high,1:]).square().mean();an=(prediction[high]-anchor[high,1:]).square().mean();reg=((head.pedal_projection.weight-before['pedal_projection.weight']).square().mean()+(head.pedal_projection.bias-before['pedal_projection.bias']).square().mean());loss=5*lo+hi+2*an+.001*reg;optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.pedal_projection.parameters(),1.);trajectory['gradient'].append(np.concatenate([head.pedal_projection.weight.grad.numpy().reshape(-1),head.pedal_projection.bias.grad.numpy()]).copy());optimizer.step();trajectory['proposal'].append(vector());head.project()
  proposed_w=head.pedal_projection.weight.detach().clone();proposed_b=head.pedal_projection.bias.detach().clone();accepted=False
  with torch.no_grad():
   for scale in [2.**(-k) for k in range(21)]:
    head.pedal_projection.weight.copy_(old_w+scale*(proposed_w-old_w));head.pedal_projection.bias.copy_(old_b+scale*(proposed_b-old_b))
    errors=torch.stack([(head.pedal_projection(z[idx])[:,1:].sigmoid()-target[idx,1:]).square().mean(0) for idx in groups])
    if (errors<=high_limits).all():accepted=True;break
   if not accepted:head.pedal_projection.weight.copy_(old_w);head.pedal_projection.bias.copy_(old_b);scale=0.
  feasible.append(scale);trajectory['after'].append(vector())
  if step%100==0 or step==config['steps']-1:journal.append(dict(step=step+1,loss=float(loss.detach()),low=float(lo.detach()),high=float(hi.detach()),anchor=float(an.detach())));write_json(out/'progress.json',dict(step=step+1,total=config['steps'],elapsed=time.perf_counter()-start))
 head.verify();rows,preds=metrics(head,items);assert all(v['steering_max_difference']==0 for v in rows)
 cp=out/'pedals.pt';torch.save(dict(schema='flybrain.learned_flight_pedals.experimental.v1',parent_checkpoint_sha256=sha256(checkpoint),fixed_decoder_sha256=plan['fixed_decoder_sha256'],actor_kind=a.actor_kind,pedal_projection=head.pedal_projection.state_dict(),actor_qualified=False),cp);torch.save(optimizer.state_dict(),out/'optimizer-state.pt');np.savez_compressed(out/'predictions.npz',**preds);write_json(out/'training-journal.json',journal);np.savez_compressed(out/'optimizer-trajectory.npz',**{k:np.asarray(v) for k,v in trajectory.items()},feasible_step_scale=np.asarray(feasible))
 result=dict(status='completed_fixed_schedule_flight_pedal_learning',actor_qualified=False,vehicle_control_sent=False,ui_actions_sent=0,optimizer_updates=config['steps'],checkpoint_sha256=sha256(cp),optimizer_sha256=sha256(out/'optimizer-state.pt'),optimizer_trajectory_sha256=sha256(out/'optimizer-trajectory.npz'),predictions_sha256=sha256(out/'predictions.npz'),metrics=rows,elapsed_seconds=time.perf_counter()-start,steering_and_recurrence_changed=False,low_labels_train=int(low.sum()),full_lap_labels_train=int(high.sum()),training_high_mse_limit_ratio=1.02,feasible_nonzero_steps=int(np.count_nonzero(feasible)),coefficient_change_max=float((head.pedal_projection.weight-before['pedal_projection.weight']).abs().max()),termination_reason='Fixed3000 supervised steps on unmodified TRAIN labels completed; no student/heldout-based selection')
 write_json(out/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
