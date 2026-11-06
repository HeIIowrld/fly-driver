"""Independent frozen-state, functional readout and real-recording metric audit."""
import argparse,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fit_bounded_sac_steering import pre_mask
from scripts.train_qualified_sac_unified import load_episode


def digest(state):
    h=hashlib.sha256()
    for key in sorted(state):
        v=state[key];h.update(str((key,str(v.dtype),tuple(v.shape),str(v.layout))).encode())
        arrays=[v] if v.layout==torch.strided else [v.to_sparse_coo().coalesce().indices(),v.to_sparse_coo().coalesce().values()]
        for value in arrays:h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','train-response','test-parent','test-response','actual-preservation','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();torch.set_num_threads(1);fit=Path(a.fit);fr=read(fit/'result.json');fm=read(fit/'manifest.json')['config'];plan=fm['plan'];cp=torch.load(fit/'functional-motion.pt',map_location='cpu',weights_only=True);stimulus_plan=read(Path(plan['training_stimuli'])/'manifest.json')['config']['plan'];components,motion,parent=load_readonly_motion_candidate(read(stimulus_plan['candidate_plan']))
    assert fr['status']=='completed_controlled_motion_function_fit' and sha256(fit/'functional-motion.pt')==fr['checkpoint_sha256'] and cp['parent_candidate']==parent and cp['actor_qualified'] is False
    states={key:digest(cp[key]) for key in ('model','readout','adapter','optical','body_port')}
    for key,index in [('model',0),('readout',1),('adapter',2),('optical',3),('body_port',4)]:assert states[key]==digest(components[index].state_dict())
    for key in ('reference','scale'):np.testing.assert_array_equal(cp['motion_port'][key].numpy(),motion.state_dict()[key].numpy())
    change=float(abs(cp['motion_port']['projection.weight']-motion.projection.weight).max());assert 0<change<=plan['maximum_weight_change']+1e-7 and cp['optimizer_updates']==fr['optimizer_updates']==18
    records=read(fit/'training-records.json');assert len(records)==18 and max(v['gradient_norm'] for v in records)>0 and sha256(fit/'training-records.json')==fr['training_records_sha256']
    for name,expected in plan['source_sha256'].items():assert sha256(fit/'executed-source'/name)==expected
    reports=[];maximum_muscle_difference=0.;readout={k:v.numpy().astype(float) for k,v in cp['readout'].items()}
    for role,path in [('train_parent',Path(plan['training_response'])),('train_functional',Path(a.train_response)),('test_parent',Path(a.test_parent)),('test_functional',Path(a.test_response))]:
        r=read(path/'result.json');m=read(path/'manifest.json')['config'];source=Path(m['arguments']['stimuli']);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];sp=sm['plan'];assert r['status']=='completed_controlled_motion_factorial_response' and sha256(path/'responses.npz')==r['responses_sha256'] and sha256(path/'groups.json')==r['groups_sha256'] and sha256(source/'result.json')==m['stimuli_result_sha256'] and sha256(source/'features.npz')==sr['features_sha256'];assert len(sm['cases'])==12 and sr['opposite_pair_prefix_max_difference']<=2e-6
        if role.startswith('test'):assert sm['plan_sha256']==plan['reserved_test_plan_sha256'] and sp['texture_seed']!=stimulus_plan['texture_seed']
        else:assert sha256(source/'result.json')==plan['training_stimuli_result_sha256']
        functional=role.endswith('functional')
        if functional:assert m['candidate']['functional_checkpoint_sha256']==fr['checkpoint_sha256']
        else:assert m['candidate']==parent
        with np.load(path/'responses.npz') as z:arrays={k:z[k].copy() for k in z.files}
        groups=read(path/'groups.json');counts={arm:0 for arm in ('fine','motion_off_after_prefix')}
        for group in groups:
            g=group['group'];body=arrays[f'group_{g}_body21'];mask=arrays[f'group_{g}_body21_valid'];native=arrays[f'group_{g}_native19'];valid=arrays[f'group_{g}_native19_valid'];assert body.shape==(21,) and native.shape==(19,) and np.all(body[12:16]==group['temperature_c']) and np.all(native[11:15]==group['temperature_c']) and body[16]==group['speed_mps'];assert hashlib.sha256(body.tobytes()+mask.tobytes()+native.tobytes()+valid.tobytes()).hexdigest()==group['fixed_body_sha256'];assert len(group['initial_neural_state_sha256'])==64
            for arm in counts:
                prediction=arrays[f'group_{g}_{arm}_prediction'];muscles=arrays[f'group_{g}_{arm}_muscles'];assert prediction.shape==(2,41,3) and muscles.shape==(2,41,6);logits=((muscles-readout['training_reference'])/readout['training_scale'])@readout['projection.weight'].T+readout['projection.bias'];expected=np.concatenate([np.tanh(logits[...,:1]),1/(1+np.exp(-logits[...,1:]))],axis=-1);error=float(abs(expected-prediction).max());maximum_muscle_difference=max(maximum_muscle_difference,error);assert error<=2e-6
                np.testing.assert_array_equal(prediction[0,:sp['branch_frame']],prediction[1,:sp['branch_frame']]);np.testing.assert_array_equal(prediction[:,:sp['branch_frame']],arrays[f'group_{g}_fine_prediction'][:,:sp['branch_frame']]);late=prediction[:,sp['evaluation_first_frame']:];margin=float(sp['expected_canonical_steering_sign_per_left_bend']*np.mean(late[1,:,0]-late[0,:,0])/2);assert abs(margin-group['branches'][arm]['hypothesized_bend_margin'])<1e-12;positive=margin>sp['bend_ordering_minimum_margin'];assert bool(positive)==group['branches'][arm]['hypothesized_bend_direction_positive'];counts[arm]+=int(positive)
            assert sm['cases'][group['right_case']]['bend']==-1 and sm['cases'][group['left_case']]['bend']==1
        assert counts==r['hypothesized_positive_bend_pairs']
        with np.load(source/'features.npz') as z:normalized=(z['motion']-cp['motion_port']['reference'].numpy())/cp['motion_port']['scale'].numpy();clip_fraction=float(np.mean(abs(normalized)>5))
        reports.append(dict(role=role,texture_seed=sp['texture_seed'],groups=len(groups),hypothesized_positive_bend_pairs=counts,raw_motion_feature_outside_normalization_clip_fraction=clip_fraction,result_sha256=sha256(path/'result.json')))
    actual=Path(a.actual_preservation);ar=read(actual/'result.json');am=read(actual/'manifest.json')['config'];assert ar['status']=='completed_functional_motion_recording_preservation' and am['candidate']['functional_checkpoint_sha256']==fr['checkpoint_sha256'] and sha256(actual/'predictions.npz')==ar['predictions_sha256'];d,c,_=load_episode(am['entry']['data']);mask_pre=pre_mask(d,-1);original=Path(read(stimulus_plan['candidate_plan'])['motion']['fit'])
    with np.load(original/'predictions.npz') as z:old=z[f"prediction_{ar['episode']}"].copy()
    with np.load(actual/'predictions.npz') as z:new=z['prediction'].copy()
    assert len(new)==ar['events']==len(d['labels']) and ar['output_body_age_max_seconds']<=.05
    for name,mask in [('all',d['fit_mask']),('pre',mask_pre)]:
        for tag,value in [('parent',old),('functional',new)]:
            error=value[mask].astype(float)-d['labels'][mask].astype(float);np.testing.assert_allclose(np.sqrt(np.mean(error**2,axis=0)),ar['metrics'][name][tag]['rmse'],atol=1e-12,rtol=0)
    limit=plan['actual_recording_preservation_maximum_ratio'];preserved=all(ar['metrics']['all']['functional']['rmse'][k]<=limit*ar['metrics']['all']['parent']['rmse'][k] for k in range(3)) and ar['metrics']['pre']['functional']['rmse'][0]<=limit*ar['metrics']['pre']['parent']['rmse'][0];assert bool(preserved)==ar['preservation_gate_passed']
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256=sha256(fit/'result.json'),actual_result_sha256=sha256(actual/'result.json'),actor_qualified=False,optimizer_updates=0,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_controlled_motion_function_audit',frozen_component_state_sha256=states,maximum_weight_change=change,optimizer_updates_verified=18,maximum_independent_muscle_readout_difference=maximum_muscle_difference,functional_reports=reports,actual_events_verified=len(new),actual_preservation_gate_verified=bool(preserved),same_prefix_output_arrays_verified=True,body_temperature_arrays_verified=True,reserved_test_texture_separation_verified=True,full_neural_replay_by_this_audit=False,physical_counterfactual_action_correctness_verified=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent frozen sparse/dense component hashes, TRAIN/test separation, muscle formulas, fixed-body/prefix arrays, factorial ordering counts and complete consumed-recording RMSE/gate checked; source-bound neural branches were not replayed by this audit, and synthetic response is not actual driving qualification'))


if __name__=='__main__':main()
