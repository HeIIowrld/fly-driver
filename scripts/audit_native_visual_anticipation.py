"""Independent command/forecast arithmetic and fixed sensory component audit."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.native_visual_experiment import load_experiment,tensor_state_equal


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.fit);r=read(root/'result.json');cfg=read(root/'manifest.json')['config'];plan=cfg['plan']
    if r['status']!='completed_native_visual_anticipation_fit':raise ValueError('Complete fixed evaluation required')
    for path,expected in plan['source_sha256'].items():
        assert sha256(root/'executed-source'/path)==expected
    assert sha256(root/'visual-anticipation.pt')==r['checkpoint_sha256']
    assert sha256(root/'predictions.npz')==r['predictions_sha256']
    parts=load_experiment(plan['reference_plan']);model,head,dn,optical,port,_=parts[0]
    assert cfg['entries']==parts[2] and cfg['native_receipt']==parts[1]
    cp=torch.load(root/'visual-anticipation.pt',weights_only=True,map_location='cpu')
    assert cp['schema']=='flybrain.native_visual_anticipatory.v1' and cp['actor_qualified'] is False
    assert cp['auxiliary_is_actor_output'] is False
    for state,module in [('readout',head),('adapter',dn),('body_port',port)]:assert tensor_state_equal(cp[state],module.state_dict())
    for key in ('reference','scale'):assert torch.equal(cp['optical'][key],optical.state_dict()[key])
    change=float((cp['optical']['projection.weight']-optical.projection.weight.detach()).abs().max())
    assert 0<change<=plan['maximum_weight_delta']+1e-7
    assert abs(change-r['maximum_visual_weight_delta'])<1e-10
    assert r['optimizer_updates']==cp['optimizer_updates']>0
    expected_future=151 if cp['arm']=='future200' else 0
    assert sum(x['current_rows'] for x in r['training_rows'])==5657
    assert sum(x['auxiliary_future_rows'] for x in r['training_rows'])==expected_future
    reports=[];max_error=0.
    with np.load(root/'predictions.npz',allow_pickle=False) as z:
        for e,(d,_,_) in enumerate(parts[3]):
            norm=((torch.tensor(z[f'muscles_{e}'])-head.training_reference)/head.training_scale).numpy().astype(float)
            state=cp['readout'];raw=norm@state['projection.weight'].numpy().astype(float).T+state['projection.bias'].numpy().astype(float)
            pred=np.column_stack([np.tanh(raw[:,0]),1/(1+np.exp(-raw[:,1:]))]);stored=z[f'prediction_{e}']
            err=float(abs(pred-stored).max());max_error=max(max_error,err);assert err<2e-6
            aux=np.tanh(norm@cp['auxiliary_beta'].numpy().astype(float)+float(cp['auxiliary_bias']))
            assert float(abs(aux-z[f'forecast_{e}']).max())<2e-6
            # Independent future target join, restricted to the rows actually used.
            times=d['decision_time'];label_ids=np.flatnonzero(d['fit_mask']&d['label_valid'])
            expected=np.full(len(times),-1,dtype=np.int64);cursor=-1
            for i,t in enumerate(times):
                wanted=t+.2
                while cursor+1<len(label_ids) and times[label_ids[cursor+1]]<=wanted:cursor+=1
                if cursor>=0 and wanted<=times[-1]:
                    j=label_ids[cursor]
                    if times[j]>t and wanted-times[j]<=.06:expected[i]=j
            np.testing.assert_array_equal(z[f'future_indices_{e}'],expected)
            np.testing.assert_array_equal(z[f'future_valid_{e}'],expected>=0)
            pre=parts[7][e];mask=pre&(expected>=0)
            for field,valid in [('all_rmse',d['fit_mask']),('pre_rmse',pre)]:
                actual=float(np.sqrt(np.mean((pred[valid,0]-d['labels'][valid,0])**2)))
                assert abs(actual-r['comparisons'][e]['candidate'][field])<2e-6
            axes=np.sqrt(np.mean((pred[d['fit_mask']]-d['labels'][d['fit_mask']])**2,axis=0))
            np.testing.assert_allclose(axes,r['comparisons'][e]['candidate_axes']['rmse'],rtol=1e-5,atol=2e-6)
            future_error=float(np.sqrt(np.mean((aux[mask]-d['labels'][expected[mask],0])**2)))
            assert abs(future_error-r['comparisons'][e]['auxiliary_future_pre_rmse'])<2e-6
            oldnorm=((torch.tensor(parts[9][e])-head.training_reference)/head.training_scale).numpy().astype(float)
            oldaux=np.tanh(oldnorm@cp['auxiliary_beta'].numpy().astype(float)+float(cp['auxiliary_bias']))
            olderr=float(np.sqrt(np.mean((oldaux[mask]-d['labels'][expected[mask],0])**2)))
            reports.append(dict(episode=e,current_axes_rmse=axes.tolist(),auxiliary_future_pre_baseline_rmse=olderr,auxiliary_future_pre_candidate_rmse=future_error,pre_count=int(mask.sum())))
    dev=r['comparisons'][2]
    gate=all(dev['candidate'][k]<=.98*dev['baseline'][k] for k in ('all_rmse','pre_rmse')) and all(dev['candidate_axes']['rmse'][k]<=1.02*dev['baseline_axes']['rmse'][k] for k in (1,2))
    assert gate==r['development_two_percent_gate_passed']
    run=begin_run(a.output,dict(fit=str(root),fit_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),optimizer_updates=0,vehicle_control_sent=False))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_native_visual_anticipation_arithmetic_audit',arithmetic_max_error=max_error,
        original_native_dn_and_all_readout_rows_exact=True,visual_normalization_exact=True,comparisons=reports,development_gate_passed=gate,
        actor_qualified=False,scope='Independent saved-muscle arithmetic; full neural trajectory replay is separate evidence',termination_reason='Frozen component lineage, target joins, current/forecast metrics and visual delta bound recomputed'))


if __name__=='__main__':main()
