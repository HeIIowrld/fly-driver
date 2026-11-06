"""Independently recompute fixed native-port trial metrics and checkpoint scope."""
import argparse
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


def metric(pred,data):
    error=(pred-data['labels'])[data['fit_mask']]
    if not np.isfinite(pred).all() or np.any(abs(pred[:,0])>1) or np.any(pred[:,1:]<0) or np.any(pred[:,1:]>1):
        raise ValueError('Finite bounded muscle readout predictions required')
    return dict(weighted_mse=float(np.mean(error**2*np.array([4.,1.,1.]))),
                rmse=np.sqrt(np.mean(error**2,axis=0)).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fit','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();fit=Path(a.fit);result=read(fit/'result.json');manifest=read(fit/'manifest.json')['config']
    if result['status']!='completed_thermal_native_body_port_fit':raise ValueError('Completed fit required')
    entries=three_episode_cohort(manifest['arguments']['split'])
    if entries!=manifest['entries'] or sha256(manifest['arguments']['split'])!=manifest['cohort_sha256']:
        raise ValueError('Sealed train/development cohort changed')
    if sha256(fit/'body-extension.pt')!=result['checkpoint_sha256'] or sha256(fit/'predictions.npz')!=result['predictions_sha256']:
        raise ValueError('Fit outputs changed')
    checkpoint=torch.load(fit/'body-extension.pt',map_location='cpu',weights_only=True)
    if (checkpoint['trained'] is not True or checkpoint['optimizer_updates']<=0
            or checkpoint['actor_qualified'] is not False
            or checkpoint['source_checkpoint_sha256']!=manifest['source_checkpoint_sha256']
            or sha256(Path(checkpoint['source_fit_directory'])/'steering.pt')!=checkpoint['source_checkpoint_sha256']):
        raise ValueError('Native-only trained checkpoint/source scope differs')
    state=checkpoint['body_port'];weights=state['projection.weight']
    if weights.shape!=(201,46) or not torch.isfinite(weights).all() or not (weights.abs()>0).any():
        raise ValueError('Finite learned existing201-port boundary required')
    with np.load(Path(manifest['arguments']['extension'])/'training-normalization.npz') as z:
        assert np.array_equal(state['reference'].numpy(),z['reference'])
        assert np.array_equal(state['scale'].numpy(),z['scale'])
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),
        fit_result_sha256=sha256(fit/'result.json'),source_checkpoint_sha256=checkpoint['source_checkpoint_sha256'],entries=entries))
    reports=[]
    with np.load(fit/'predictions.npz') as predictions:
        for i,entry in enumerate(entries):
            d,_,provenance=load_episode(entry['data']);assert provenance['arrays_sha256']==entry['arrays_sha256']
            row={}
            for case in ['baseline','candidate']:
                values=metric(predictions[f'{case}_{i}'],d);row[case]=values
                for key in ['weighted_mse','rmse']:
                    np.testing.assert_allclose(values[key],result[case][i][key],atol=1e-12,rtol=1e-10)
            reports.append(row)
        d=load_episode(entries[2]['data'])[0]
        temperature=metric(predictions['development_temperature_neutral'],d)
        for key in ['weighted_mse','rmse']:
            np.testing.assert_allclose(temperature[key],result['development_temperature_neutral'][key],atol=1e-12,rtol=1e-10)
        error=float(abs(predictions['development_native_neutral']-predictions['baseline_2']).max())
        assert error==result['neutral_source_max_difference'] and error<=2e-6
    smoke=read(fit/'nonzero-port-smoke.json')
    assert smoke['optimizer_updates']==0 and smoke['restored_zero_weights'] is True
    assert smoke['source_prediction_max_difference']==result['nonzero_body_port_readout_response']>0
    write_json(run/'result.json',dict(status='completed_independent_thermal_native_fit_audit',comparisons=reports,
        development_temperature_neutral=temperature,neutral_source_max_difference=error,
        trained_projection_shape=list(weights.shape),nonzero_projection_weights=int((weights!=0).sum()),
        checkpoint_contains_no_replacement_motor_or_visual_state=True,actor_qualified=False,
        physiological_registration_verified=False,actual_student_lap_verified=False,
        termination_reason='Independent command-space metrics, neutral replay identity, source-bound native-only checkpoint, TRAIN calibration and nonzero engineering stimulus verified; no physical/generalization qualification'))


if __name__=='__main__':main()
