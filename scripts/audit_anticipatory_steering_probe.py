"""Independent arithmetic/label-join audit; no fitting or actor access."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.fit);r=read(root/'result.json');manifest=read(root/'manifest.json');plan=manifest['config']['plan']
    assert r['status']=='completed_protected_anticipatory_probe'
    assert sha256(root/'executed-entrypoint.py')==plan['program_sha256']
    for name,key in [('native-steering.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256')]:assert sha256(root/name)==r[key]
    parent=Path(plan['parent']);assert sha256(parent/'native-steering.pt')==plan['checkpoint_sha256']
    assert sha256(parent/'frozen-trajectories.npz')==plan['frozen_trajectories_sha256']
    entries=three_episode_cohort(plan['split']);assert entries==manifest['config']['entries']
    items=[load_episode(e['data']) for e in entries]
    cp=torch.load(root/'native-steering.pt',weights_only=True,map_location='cpu')
    old=torch.load(parent/'native-steering.pt',weights_only=True,map_location='cpu')
    for key in ('body_port',):
        assert set(cp[key])==set(old[key])
        for k,v in old[key].items():assert torch.equal(v,cp[key][k])
    for key in ('training_reference','training_scale'):assert torch.equal(cp['readout'][key],old['readout'][key])
    for key in ('projection.weight','projection.bias'):assert torch.equal(cp['readout'][key][1:],old['readout'][key][1:])
    group_ratios=[];reports=[];max_error=0.
    with np.load(root/'predictions.npz',allow_pickle=False) as arrays,np.load(parent/'frozen-trajectories.npz',allow_pickle=False) as frozen:
        for e,(d,_,_) in enumerate(items):
            t=d['decision_time'];target_ids=np.flatnonzero(d['fit_mask']&d['label_valid'])
            expected=np.full(len(t),-1,dtype=np.int64)
            # Independently walk qualified labels rather than call the fit join.
            cursor=-1
            for i,clock in enumerate(t):
                wanted=clock+plan['horizon_seconds']
                while cursor+1<len(target_ids) and t[target_ids[cursor+1]]<=wanted:cursor+=1
                if cursor>=0 and wanted<=t[-1]:
                    j=target_ids[cursor]
                    if t[j]>clock and wanted-t[j]<=plan['max_label_age_seconds']:expected[i]=j
            np.testing.assert_array_equal(expected,arrays[f'future_indices_{e}'])
            np.testing.assert_array_equal(expected>=0,arrays[f'future_valid_{e}'])
            pre=frozen[f'pre_{e}'];future=pre&(expected>=0)
            predictions=[]
            for name,state in [('baseline',old['readout']),('candidate',cp['readout'])]:
                x=((torch.tensor(frozen[f'muscles_{e}'])-state['training_reference'])/state['training_scale']).numpy().astype(float)
                z=x@state['projection.weight'][0].numpy().astype(float)+float(state['projection.bias'][0])
                steer=np.tanh(z);stored=arrays[f'{name}_{e}'][:,0]
                error=float(np.max(abs(steer-stored)));max_error=max(max_error,error)
                assert error<2e-6
                predictions.append(steer)
            baseline,candidate=predictions;report=r['comparisons'][e]
            masks=[d['fit_mask'],pre,future];ys=[d['labels'][:,0],d['labels'][:,0],d['labels'][np.maximum(expected,0),0]]
            for mask,y in zip(masks,ys):
                old_mse=np.mean((baseline[mask]-y[mask])**2);new_mse=np.mean((candidate[mask]-y[mask])**2)
                if e<2:group_ratios.append(float(new_mse/old_mse))
            for name,pred in zip(('baseline','candidate'),predictions):
                for k,mask in [('all_rmse',d['fit_mask']),('pre_rmse',pre)]:
                    actual=float(np.sqrt(np.mean((pred[mask]-d['labels'][mask,0])**2)))
                    assert abs(actual-report[name][k])<2e-6
                actual=float(np.sqrt(np.mean((pred[future]-d['labels'][expected[future],0])**2)))
                assert abs(actual-report[f'future_pre_{name}_rmse'])<2e-6
            assert np.array_equal(arrays[f'baseline_{e}'][:,1:],arrays[f'candidate_{e}'][:,1:])
            reports.append(dict(episode=e,future_pre_count=int(future.sum()),join_exact=True,pedals_exact=True))
    recorded=read(root/'train-fit.json')['group_mse_ratios']
    np.testing.assert_allclose(group_ratios,recorded,rtol=2e-4,atol=2e-6)
    assert max(group_ratios)<=1.00001
    dev=r['comparisons'][2]
    gate=all(dev['candidate'][k]<=.98*dev['baseline'][k] for k in ('all_rmse','pre_rmse')) and dev['future_pre_candidate_rmse']<=.98*dev['future_pre_baseline_rmse']
    assert gate==r['development_present_and_future_two_percent_gate_passed']
    run=begin_run(a.output,dict(fit=str(root),fit_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),optimizer_updates=0,vehicle_control_sent=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_anticipatory_probe_audit',comparisons=reports,arithmetic_max_error=max_error,train_group_mse_ratios=group_ratios,development_gate_passed=gate,actor_qualified=False,termination_reason='Independent future-label joins and current/future errors, TRAIN group protection and fixed body/pedal boundaries verified'))


if __name__=='__main__':main()
