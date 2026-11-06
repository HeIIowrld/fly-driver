"""Independent array/head and frozen-component audit of native steering refit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


def predict(muscles,head):
    x=(muscles.astype(float)-head['training_reference'].numpy())/head['training_scale'].numpy()
    logits=x@head['projection.weight'].numpy().astype(float).T+head['projection.bias'].numpy()
    return np.column_stack([np.tanh(logits[:,0]),1/(1+np.exp(-logits[:,1:]))])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.fit);result=read(root/'result.json');config=read(root/'manifest.json')['config'];plan=config['plan']
    assert result['status']=='completed_native_flight_steering_refit'
    assert sha256(root/'executed-entrypoint.py')==plan['program_sha256']
    for filename,key in [('native-steering.pt','checkpoint_sha256'),('predictions.npz','predictions_sha256'),('frozen-trajectories.npz','frozen_trajectories_sha256')]:
        assert sha256(root/filename)==result[key]
    parent_root=Path(plan['native_fit']);parent_result=read(parent_root/'result.json')
    assert sha256(parent_root/'body-extension.pt')==parent_result['checkpoint_sha256']==plan['native_checkpoint_sha256']
    assert sha256(parent_root/'predictions.npz')==parent_result['predictions_sha256']
    parent=torch.load(parent_root/'body-extension.pt',map_location='cpu',weights_only=True)
    candidate=torch.load(root/'native-steering.pt',map_location='cpu',weights_only=True)
    assert candidate['native_checkpoint_sha256']==plan['native_checkpoint_sha256']
    old=torch.load(Path(parent['source_fit_directory'])/'steering.pt',map_location='cpu',weights_only=True)
    assert sha256(Path(parent['source_fit_directory'])/'steering.pt')==parent['source_checkpoint_sha256']
    for k,v in parent['body_port'].items(): assert torch.equal(v,candidate['body_port'][k])
    for k in ('training_reference','training_scale'): assert torch.equal(old['readout'][k],candidate['readout'][k])
    for k in ('projection.weight','projection.bias'): assert torch.equal(old['readout'][k][1:],candidate['readout'][k][1:])
    cfg=read(parent_root/'manifest.json')['config'];entries=three_episode_cohort(cfg['arguments']['split'])
    items=[load_episode(e['data']) for e in entries]; reports=[]; train_x=[];train_y=[];train_w=[]
    run=begin_run(a.output,dict(fit=str(root),fit_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),seed=None,actor_qualified=False,vehicle_control_sent=False))
    with np.load(root/'predictions.npz',allow_pickle=False) as z, np.load(root/'frozen-trajectories.npz',allow_pickle=False) as frozen, np.load(parent_root/'predictions.npz',allow_pickle=False) as reference:
        for e,((d,c,provenance),report) in enumerate(zip(items,result['comparisons'],strict=True)):
            assert provenance['arrays_sha256']==entries[e]['arrays_sha256']
            baseline=z[f'baseline_{e}'];values=z[f'candidate_{e}'];muscles=z[f'muscles_{e}'];pre=z[f'pre_{e}']
            for key in (f'baseline_{e}',f'muscles_{e}',f'pre_{e}'): np.testing.assert_array_equal(z[key],frozen[key])
            np.testing.assert_allclose(baseline,reference[f'candidate_{e}'],rtol=0,atol=2e-6)
            assert np.array_equal(values[:,1:],baseline[:,1:])
            errors={}
            for name,head,prediction in [('baseline',old['readout'],baseline),('candidate',candidate['readout'],values)]:
                error=float(abs(predict(muscles,head)-prediction).max());assert error<2e-6;errors[name]=error
                for key,mask in [('all_rmse',d['fit_mask']),('pre_rmse',pre)]:
                    assert mask.any();value=float(np.sqrt(np.mean((prediction[mask,0]-d['labels'][mask,0])**2)))
                    assert abs(value-report[name][key])<1e-12
            if e<2:
                h=old['readout'];x=((torch.tensor(muscles)-h['training_reference'])/h['training_scale']).numpy().astype(float)
                mask=d['fit_mask'];train_x.append(x[mask]);train_y.append(d['labels'][mask,0]);train_w.append(np.where(pre[mask],16.,1.))
            reports.append(dict(episode=e,model_events=len(muscles),comparison=report,independent_head_max_error=errors))
    x=np.concatenate(train_x);y=np.concatenate(train_y);w=np.concatenate(train_w)
    scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    objective={}
    for name,h in [('initial',old['readout']),('final',candidate['readout'])]:
        beta=h['projection.weight'][0].numpy().astype(float);bias=float(h['projection.bias'][0])
        pr=np.tanh(x@beta+bias)
        objective[name]=float(np.average((pr-y)**2,weights=w)+.0001*np.sum((beta*scale)**2))
    protected_ratios=None
    if result.get('method')=='train_group_constraints':
        sizes=[len(v) for v in train_y];groups=[]
        for e in (0,1):
            m=np.zeros(len(x),dtype=bool);start=0 if e==0 else sizes[0];m[start:start+sizes[e]]=True
            groups.extend([m,m&(w>1)])
        predictions={}
        for name,h in [('baseline',old['readout']),('candidate',candidate['readout'])]:
            predictions[name]=np.tanh(x@h['projection.weight'][0].numpy().astype(float)+float(h['projection.bias'][0]))
        protected_ratios=[float(np.mean((predictions['candidate'][m]-y[m])**2)/np.mean((predictions['baseline'][m]-y[m])**2)) for m in groups]
        assert max(protected_ratios)<=1+1e-6
        np.testing.assert_allclose(protected_ratios,result['group_mse_ratios'],rtol=0,atol=1e-6)
        assert result['minimum_training_mse_improvement']<=min(1-v for v in protected_ratios)+1e-6
    else:
        assert objective['final']<=objective['initial']
        assert abs(objective['initial']-result['train_initial_loss'])<2e-8
        assert abs(objective['final']-result['train_final_loss'])<2e-8
    dev=result['comparisons'][2]
    gate=all(dev['candidate'][k]<=.98*dev['baseline'][k] for k in ('all_rmse','pre_rmse'))
    assert gate==result['development_two_percent_gate_passed']
    write_json(run/'result.json',dict(status='completed_independent_native_flight_steering_refit_audit',comparisons=reports,
        train_objective=objective,protected_train_group_mse_ratios=protected_ratios,train_fit_rows=len(x),development_two_percent_gate_passed=gate,
        native_port_and_pedal_rows_exact=True,original_muscle_trajectories_preserved=True,
        actor_qualified=False,vehicle_control_sent=False,
        termination_reason='Protected native checkpoint/trajectories, independent head formula/RMSE/TRAIN objective, unchanged pedals and declared gate verified; no new-road or live claim'))


if __name__=='__main__':main()
