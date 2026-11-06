"""Recompute stored diagnostic body-only predictions independently, without fit."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.diagnostic);r=read(root/'result.json');cfg=read(root/'manifest.json')['config']
    assert r['status']=='completed_body_only_forecast_diagnostic'
    assert sha256(root/'executed-entrypoint.py')==cfg['program_sha256']
    assert sha256(root/'predictors.npz')==r['predictors_sha256']
    items=[load_episode(e['data']) for e in cfg['entries']]
    probe=root.parent/'fit-01';pm=read(probe/'manifest.json')['config']
    assert sha256(probe/'result.json')==cfg['probe_result_sha256']
    parent=Path(pm['plan']['parent']);checks=[]
    with np.load(root/'predictors.npz',allow_pickle=False) as z,np.load(probe/'predictions.npz',allow_pickle=False) as joins,np.load(parent/'frozen-trajectories.npz',allow_pickle=False) as frozen:
        for report in r['reports']:
            name=report['name'];history=name.startswith('history');future=name.endswith('future200')
            train_y=[]
            for e,(d,_,_) in enumerate(items):
                valid=joins[f'future_valid_{e}'] if future else np.ones(len(d['labels']),dtype=bool)
                ids=np.maximum(joins[f'future_indices_{e}'],0) if future else np.arange(len(d['labels']))
                y=d['labels'][ids,0];mask=d['fit_mask']&valid
                if e<2:train_y.append(y[mask])
            constant=float(np.concatenate(train_y).mean());assert abs(constant-float(z[name+'_bias']))<1e-12
            for e,(d,_,_) in enumerate(items):
                b=d['body'].copy();b[~d['body_valid']]=0
                pieces=[b,d['body_valid'].astype(float)]
                if history:
                    past=d['view_body'].copy();past[~d['view_body_valid']]=0
                    pieces.extend([past.reshape(len(b),-1),d['view_body_valid'].reshape(len(b),-1).astype(float)])
                x=np.concatenate(pieces,axis=1)
                pred=np.clip(((x-z[name+'_mean'])/z[name+'_scale'])@z[name+'_beta']+z[name+'_bias'],-1,1)
                np.testing.assert_array_equal(pred,z[name+f'_prediction_{e}'])
                valid=joins[f'future_valid_{e}'] if future else np.ones(len(b),dtype=bool)
                ids=np.maximum(joins[f'future_indices_{e}'],0) if future else np.arange(len(b))
                y=d['labels'][ids,0];whole=d['fit_mask']&valid;pre=frozen[f'pre_{e}']&whole
                for k,m in [('all',whole),('pre',pre)]:
                    error=float(np.sqrt(np.mean((pred[m]-y[m])**2)));baseline=float(np.sqrt(np.mean((constant-y[m])**2)))
                    assert abs(error-report['comparisons'][e][k+'_rmse'])<1e-12
                    assert abs(baseline-report['comparisons'][e]['constant_'+k+'_rmse'])<1e-12
                checks.append(dict(name=name,episode=e,predictions_exact=True))
    run=begin_run(a.output,dict(diagnostic=str(root),result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),optimizer_updates=0))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_body_forecast_audit',checks=checks,vehicle_control_sent=False,termination_reason='All twelve predictions and present/future whole/pre errors independently reconstructed; TRAIN target mean verified'))


if __name__=='__main__':main()
