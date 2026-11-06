"""Independent metrics and six-muscle prediction audit for frozen head transfer."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


def steering_metrics(prediction, data, pre):
    error = prediction.astype(np.float64)-data['labels'][:,0].astype(np.float64)
    return dict(all_rmse=float(np.sqrt(np.mean(error[data['fit_mask']]**2))),
        pre_rmse=float(np.sqrt(np.mean(error[pre]**2))),
        all_count=int(data['fit_mask'].sum()),pre_count=int(pre.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.evaluation);result=read(root/'result.json');manifest=read(root/'manifest.json')['config'];plan=manifest['plan']
    assert result['status']=='completed_frozen_steering_head_transfer'
    assert result['optimizer_updates']==0 and result['actor_qualified'] is False
    assert sha256(root/'predictions.npz')==result['predictions_sha256']
    assert sha256(root/'executed-entrypoint.py')==plan['program_sha256']
    assert sha256(plan['split'])==plan['split_sha256']
    entries=three_episode_cohort(plan['split']);assert entries==manifest['entries']
    checkpoints={}
    for name in ['source','candidate']:
        path=Path(plan[name])/'steering.pt';assert sha256(path)==plan[name+'_sha256']
        checkpoints[name]=torch.load(path,map_location='cpu',weights_only=True)
    for section in ['adapter','optical']:
        for key,value in checkpoints['source'][section].items():
            assert torch.equal(value,checkpoints['candidate'][section][key])
    heads={k:{n:t.numpy().astype(np.float64) for n,t in c['readout'].items()} for k,c in checkpoints.items()}
    rows=[]
    with np.load(root/'predictions.npz') as z, np.load(Path(plan['baseline_fit'])/'predictions.npz') as baseline:
        for i,entry in enumerate(entries):
            data,_,provenance=load_episode(entry['data']);assert provenance['arrays_sha256']==entry['arrays_sha256']
            pre=z[f'pre_mask_{i}'];assert pre.dtype==bool and pre.any() and not (pre&~data['fit_mask']).any()
            row={};max_error={}
            for case,head in heads.items():
                features=(z[f'muscles_{i}'].astype(np.float64)-head['training_reference'])/head['training_scale']
                logits=features@head['projection.weight'].T+head['projection.bias']
                predicted=np.concatenate([np.tanh(logits[:,:1]),1/(1+np.exp(-logits[:,1:]))],axis=1)
                saved=z[f'{case}_{i}']
                assert np.isfinite(saved).all()
                max_error[case]=float(abs(predicted-saved).max())
                np.testing.assert_allclose(predicted,saved,rtol=0,atol=2e-6)
                row[case]=steering_metrics(saved[:,0],data,pre)
                for key in ['all_rmse','pre_rmse','all_count','pre_count']:
                    np.testing.assert_allclose(row[case][key],result['comparisons'][i][case][key],rtol=1e-10,atol=1e-12)
            assert np.array_equal(z[f'source_{i}'][:,1:],z[f'candidate_{i}'][:,1:])
            error=float(abs(z[f'source_{i}']-baseline[f'baseline_{i}']).max())
            assert error==result['comparisons'][i]['original_replay_max_difference'] and error<=2e-6
            row.update(episode=i,independent_projection_max_difference=max_error,source_reproduction_max_difference=error)
            rows.append(row)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),evaluation_result_sha256=sha256(root/'result.json')))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_frozen_steering_transfer_audit',comparisons=rows,
        optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,
        pre_window_scope='Saved diagnostic mask under original sustained-turn definition, not track geometry truth',
        termination_reason='Independent float64 muscle-to-control formula, RMSE, full original replay reproduction, frozen adapters and byte-exact pedals checked'))


if __name__=='__main__':main()
