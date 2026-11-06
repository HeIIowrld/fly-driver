"""Independent balanced-source fixed-graph/training-cut and synthetic/actual metric audit."""
import argparse,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_controlled_motion_function import digest
from scripts.balanced_visual_sources import load_balanced_sources
from scripts.fit_bounded_sac_steering import pre_mask
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','reserved-parent','reserved-candidate','train-candidate','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--recordings',nargs='+',required=True);a=p.parse_args();torch.set_num_threads(1);fit=Path(a.fit);r=read(fit/'result.json');m=read(fit/'manifest.json')['config'];plan=m['plan'];cp=torch.load(fit/'route-protected.pt',map_location='cpu',weights_only=True);components,motion,parent=load_readonly_motion_candidate(read(plan['candidate_plan']));assert r['status']=='completed_route_protected_visual_fit' and sha256(fit/'route-protected.pt')==r['checkpoint_sha256'] and cp['parent_candidate']==parent and cp['actor_qualified'] is False
    current=cp['model'];original=components[0].state_dict();fixed_hash=digest({k:v for k,v in original.items() if k!='visual.weight'});assert fixed_hash==digest({k:v for k,v in current.items() if k!='visual.weight'});np.testing.assert_array_equal(current['visual.weight'][:,:12].numpy(),original['visual.weight'][:,:12].numpy());pooled_delta=float(abs(current['visual.weight']-original['visual.weight']).max());assert pooled_delta<=plan['maximum_pooled_weight_change']+1e-7
    hashes={}
    for key,index in [('readout',1),('adapter',2),('optical',3),('body_port',4)]:hashes[key]=digest(cp[key]);assert hashes[key]==digest(components[index].state_dict())
    for key in ('reference','scale'):np.testing.assert_array_equal(cp['motion_port'][key].numpy(),motion.state_dict()[key].numpy())
    motion_delta=float(abs(cp['motion_port']['projection.weight']-motion.projection.weight).max());assert motion_delta<=plan['maximum_motion_weight_change']+1e-7
    for path,expected in plan['source_sha256'].items():assert sha256(fit/'executed-source'/path)==expected
    schedule=read(fit/'training-schedule.json');records=read(fit/'training-records.json');prefix=Path(plan['prefix_run'])/'parent-prefix-states.pt';states=torch.load(prefix,map_location='cpu',weights_only=True);assert len(schedule)==len(records)==r['optimizer_updates']==cp['optimizer_updates'] and sha256(prefix)==r['parent_prefix_states_sha256'];expected_tasks=set()
    for e in (0,1):
        d,c,_=load_episode(m['entries'][e]['data']);blocks=(len(d['scene'])+plan['window']-1)//plan['window'];selected=np.union1d(np.unique(np.flatnonzero(pre_mask(d,-1))//plan['window']),np.linspace(0,blocks-1,plan['anchor_blocks_per_episode'],dtype=int))
        for b in selected:
            begin=int(b*plan['window']);end=min(begin+plan['window'],len(d['scene']));expected_tasks.add((e,begin,end));state=states[(e,begin)]
            if begin==0:assert state['previous_time'] is None and state['previous_body_receipt'] is None
            else:
                assert state['previous_time']==float(d['decision_time'][begin-1]) and state['previous_body_receipt']==float(d['body_receipt'][begin-1]);np.testing.assert_array_equal(state['previous_scene'][0].numpy(),d['scene'][begin-1])
    real=[task for kind,task in schedule if kind=='real'];actual_tasks={(z['episode'],z['begin'],z['end']) for z in real};assert actual_tasks==expected_tasks and len(real)==len(actual_tasks)==r['real_updates'];synthetic=[z for kind,z in schedule if kind=='synthetic'];assert len(synthetic)==r['synthetic_updates']==54 and {(z['source_slot'],z['pair'],z['temperature']) for z in synthetic}=={(s,k,t) for s in range(3) for k in range(0,12,2) for t in (40.,80.,110.)};assert len(real)==93 and len(schedule)==147;assert r['parent_prefix_reused'] and sha256(Path(plan['prefix_run'])/'result.json')==plan['prefix_run_result_sha256'];assert len(plan['synthetic_sources'])==3 and all(s['current_role']=='train_this_trial' for s in plan['synthetic_sources']);assert plan['reserved_test_plan'] not in [s['stimuli'] for s in plan['synthetic_sources']]
    assert all(rec['kind']==kind and rec['task']==task and np.isfinite(rec['loss']) and np.isfinite(rec['gradient_norm']) for (kind,task),rec in zip(schedule,records));assert all(v<=2e-6 for v in r['parent_full_train_max_differences']) and sha256(plan['reserved_test_plan'])==plan['reserved_test_plan_sha256']
    sources=load_balanced_sources(plan,parent);reserved_plan=read(plan['reserved_test_plan']);assert all(source['plan']['texture_seed']!=reserved_plan['texture_seed'] for source in sources);assert r['arm'] in ('diverse','repeated');source_slots={slot:sum(z['source_slot']==slot for z in synthetic) for slot in range(3)};assert source_slots=={0:18,1:18,2:18}
    readout={k:v.numpy().astype(float) for k,v in cp['readout'].items()};reports=[];maximum=0.
    for role,path in [('train_parent',Path(plan['training_response'])),('train_candidate',Path(a.train_candidate)),('reserved_parent',Path(a.reserved_parent)),('reserved_candidate',Path(a.reserved_candidate))]:
        result=read(path/'result.json');cfg=read(path/'manifest.json')['config'];source=Path(cfg['arguments']['stimuli']);sm=read(source/'manifest.json')['config'];sr=read(source/'result.json');assert result['status']=='completed_controlled_motion_factorial_response' and sha256(path/'responses.npz')==result['responses_sha256'] and sha256(path/'groups.json')==result['groups_sha256'] and sha256(source/'features.npz')==sr['features_sha256']
        if role.startswith('reserved'):assert sm['plan_sha256']==plan['reserved_test_plan_sha256']
        if role.endswith('candidate'):assert cfg['candidate']['route_protected_checkpoint_sha256']==r['checkpoint_sha256']
        else:assert cfg['candidate']==parent
        with np.load(path/'responses.npz') as z:arrays={k:z[k].copy() for k in z.files}
        groups=read(path/'groups.json');counts={arm:0 for arm in ('fine','motion_off_after_prefix')};target_count=0
        for g in groups:
            j=g['group'];values=[arrays[f'group_{j}_{key}'] for key in ('body21','body21_valid','native19','native19_valid')];assert hashlib.sha256(b''.join(v.tobytes() for v in values)).hexdigest()==g['fixed_body_sha256']
            for arm in counts:
                prediction=arrays[f'group_{j}_{arm}_prediction'];mu=arrays[f'group_{j}_{arm}_muscles'];logits=((mu-readout['training_reference'])/readout['training_scale'])@readout['projection.weight'].T+readout['projection.bias'];expected=np.concatenate([np.tanh(logits[...,:1]),1/(1+np.exp(-logits[...,1:]))],axis=-1);maximum=max(maximum,float(abs(expected-prediction).max()));assert maximum<=2e-6;np.testing.assert_array_equal(prediction[:,:11],arrays[f'group_{j}_fine_prediction'][:,:11]);late=prediction[:,sm['plan']['evaluation_first_frame']:];margin=float(sm['plan']['expected_canonical_steering_sign_per_left_bend']*np.mean(late[1,:,0]-late[0,:,0])/2);assert margin==g['branches'][arm]['hypothesized_bend_margin'];counts[arm]+=int(margin>sm['plan']['bend_ordering_minimum_margin'])
                if arm=='fine':target_count+=int(margin>=plan['target_margin'])
        assert counts==result['hypothesized_positive_bend_pairs'];reports.append(dict(role=role,positive_pairs=counts,target_margin_pairs=target_count,groups=len(groups),result_sha256=sha256(path/'result.json')))
    recorded=[];original_fit=Path(read(plan['candidate_plan'])['motion']['fit'])
    for name in a.recordings:
        root=Path(name);rr=read(root/'result.json');rm=read(root/'manifest.json')['config'];assert rr['status']=='completed_functional_motion_recording_preservation' and rm['candidate']['route_protected_checkpoint_sha256']==r['checkpoint_sha256'] and sha256(root/'predictions.npz')==rr['predictions_sha256'];d,_,_=load_episode(rm['entry']['data']);pre=pre_mask(d,-1)
        with np.load(root/'predictions.npz') as z:new=z['prediction'].copy()
        with np.load(original_fit/'predictions.npz') as z:old=z[f"prediction_{rr['episode']}"].copy()
        assert len(new)==rr['events']==len(d['labels']) and rr['output_body_age_max_seconds']<=.05
        for label,mask in [('all',d['fit_mask']),('pre',pre)]:
            for key,pred in [('parent',old),('functional',new)]:np.testing.assert_allclose(np.sqrt(np.mean((pred[mask].astype(float)-d['labels'][mask].astype(float))**2,axis=0)),rr['metrics'][label][key]['rmse'],atol=1e-12,rtol=0)
        ratio=plan['actual_recording_preservation_maximum_ratio'];preserved=all(rr['metrics']['all']['functional']['rmse'][k]<=ratio*rr['metrics']['all']['parent']['rmse'][k] for k in range(3)) and rr['metrics']['pre']['functional']['rmse'][0]<=ratio*rr['metrics']['pre']['parent']['rmse'][0];assert bool(preserved)==rr['preservation_gate_passed'];recorded.append(dict(episode=rr['episode'],events=len(new),preservation_gate=bool(preserved),result_sha256=sha256(root/'result.json')))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256=sha256(fit/'result.json'),vehicle_control_sent=False,optimizer_updates=0));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_balanced_visual_audit',arm=r['arm'],prefix_reused=True,synthetic_source_slot_counts=source_slots,source_seed_separation_verified=True,fixed_model_except_pooled_state_sha256=fixed_hash,frozen_component_sha256=hashes,pooled_weight_change=pooled_delta,motion_weight_change=motion_delta,real_training_windows_verified=len(real),synthetic_updates_verified=len(synthetic),parent_training_prefix_clock_bindings_verified=True,synthetic_reports=reports,independent_muscle_readout_max_difference=maximum,actual_recordings=recorded,actual_events_verified=sum(v['events'] for v in recorded),full_neural_replay_by_audit=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent unchanged chemical/body/readout states, allowed visual changes, exact TRAIN-only task coverage and prefix clocks, declared reserved source and readout/metric checks. Updated continuous neural trajectories are source-bound evaluations, not rerun by this audit; no actual autonomous or physiology qualification.'))


if __name__=='__main__':main()
