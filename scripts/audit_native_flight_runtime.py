"""Independently audit native flight runtime replay parity and output-age accounting."""
import argparse,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.replay);r=read(root/'result.json');m=read(root/'manifest.json')['config']
    assert r['status']=='completed_native_flight_runtime_replay' and r['actor_qualified'] is False
    assert r['optimizer_updates']==0 and r['vehicle_control_sent'] is False
    assert sha256(root/'executed-runtime.py')==m['runtime_sha256']
    assert sha256(root/'executed-entrypoint.py')==m['program_sha256']
    assert sha256(root/'predictions.npz')==r['predictions_sha256']
    source=Path(m['arguments']['fit']);sr=read(source/'result.json');assert sha256(source/'result.json')==m['fit_result_sha256']
    assert sha256(source/'predictions.npz')==sr['predictions_sha256']
    d,_,provenance=load_episode(m['entry']['data']);assert provenance['arrays_sha256']==m['entry']['arrays_sha256']
    with np.load(root/'predictions.npz') as z,np.load(source/'predictions.npz') as base:
        prediction=z['prediction'];times=z['inference_seconds'];ages=z['output_body_age_seconds'];n=len(prediction)
        assert n==r['events'] and times.shape==ages.shape==(n,)
        assert np.isfinite(prediction).all() and np.isfinite(times).all() and (times>=0).all()
        assert (ages<=.05).all() and (ages>=0).all()
        expected=d['decision_time'][:n]-d['body_receipt'][:n]+times
        np.testing.assert_allclose(ages,expected,atol=1e-12,rtol=0)
        error=float(abs(prediction-base[f"candidate_{m['arguments']['episode']}"][:n]).max())
        assert error==r['source_prediction_max_difference'] and error<=2e-6
        np.testing.assert_allclose(np.percentile(times,[50,95,99,100]),r['inference_seconds_percentiles'],atol=1e-12,rtol=0)
        assert max(ages)==r['output_body_age_max_seconds']
        assert r['full_episode']==(n==len(d['scene']))
    warm=read(root/'backend-initialization.json')
    assert warm['model_recurrent_steps']==warm['optimizer_updates']==0 and warm['constant_zero_steps']==3
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),replay_result_sha256=sha256(root/'result.json'),runtime_sha256=m['runtime_sha256']))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_native_flight_runtime_audit',events=n,full_episode=r['full_episode'],
        original_prediction_max_difference=error,output_body_age_max_seconds=float(max(ages)),
        inference_seconds_percentiles=np.percentile(times,[50,95,99,100]).tolist(),
        actor_qualified=False,live_camera_verified=False,actual_student_lap_verified=False,vehicle_control_sent=False,
        termination_reason='Recomputed parity, original receipt plus measured computation50ms accounting, timing percentiles and frozen recorded-source bindings; live end-to-end latency remains unverified'))


if __name__=='__main__':main()
