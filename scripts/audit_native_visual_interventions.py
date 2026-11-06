"""Independently recompute candidate visual intervention joins and metrics."""
import argparse,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interventions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.interventions);r=read(root/'result.json');cfg=read(root/'manifest.json')['config']
    assert r['status']=='completed_native_visual_anticipation_interventions'
    assert sha256(root/'executed-entrypoint.py')==cfg['program_sha256']
    fit=Path(cfg['fit']);assert sha256(fit/'result.json')==cfg['fit_result_sha256']
    fr=read(fit/'result.json');assert sha256(fit/'predictions.npz')==fr['predictions_sha256']
    old=Path('runs/native-visual-precorners-20261104/replay-01');assert sha256(old/'result.json')==cfg['prior_visual_result_sha256']
    prior=read(old/'result.json');reports=[];signed=[];model_events=0
    with np.load(fit/'predictions.npz',allow_pickle=False) as fixed:
        for e,entry in enumerate(cfg['entries']):
            d,_,provenance=load_episode(entry['data']);assert provenance['arrays_sha256']==entry['arrays_sha256']
            file=root/f'episode-{e}.npz';assert sha256(file)==r['comparisons'][e]['predictions_sha256']
            model_events+=len(d['labels']);cases={name:[] for name in ('normal','inherited_optical','spatial_neutral')};targets=[];indices=[]
            old_file=old/f'episode-{e}.npz';assert sha256(old_file)==prior['comparisons'][e]['predictions_sha256']
            with np.load(file,allow_pickle=False) as z,np.load(old_file,allow_pickle=False) as old_arrays:
                np.testing.assert_allclose(z['normal'],fixed[f'prediction_{e}'],rtol=0,atol=2e-6)
                times=d['decision_time']
                for wi,w in enumerate(prior['comparisons'][e]['windows']):
                    ids=np.arange(w['start'],w['event']+1);mask=z[f'window_{wi}_pre_mask'];stored=r['comparisons'][e]['windows'][wi]
                    np.testing.assert_array_equal(z[f'window_{wi}_receiver_indices'],ids)
                    assert mask.dtype==np.bool_ and mask.shape==ids.shape and mask.any()
                    np.testing.assert_array_equal(mask,old_arrays[f'window_{wi}_pre_mask'])
                    assert np.all(d['fit_mask'][ids[mask]]) and np.all(abs(d['body'][ids[mask],4])<.07)
                    assert np.all(times[ids[mask]]>=w['time']-.8) and np.all(times[ids[mask]]<=w['time']-.08)
                    np.testing.assert_allclose(z[f'window_{wi}_normal'],z['normal'][ids],rtol=0,atol=2e-6)
                    for name in cases:
                        np.testing.assert_array_equal(z[f'window_{wi}_{name}_indices'],ids)
                        prediction=z[f'window_{wi}_{name}'];assert prediction.shape==(len(ids),3)
                        cases[name].append(prediction[mask,0])
                        actual=float(np.sqrt(np.mean((prediction[mask,0]-d['labels'][ids[mask],0])**2)))
                        assert abs(actual-stored['original_label_rmse'][name])<1e-10
                    targets.append(d['labels'][ids[mask],0]);indices.extend(ids[mask].tolist())
                    if w['donor_time'] is not None:
                        wanted=times[ids]-w['time']+w['donor_time'];expected=np.searchsorted(times,wanted,side='right')-1
                        assert np.all(expected>=0) and np.all(wanted<=times[-1]) and np.all(times[expected]<=wanted)
                        np.testing.assert_array_equal(expected,z[f'window_{wi}_opposite_indices'])
                        effect=float(np.mean((z[f'window_{wi}_normal'][mask,0]-z[f'window_{wi}_opposite'][mask,0])*w['direction']))
                        assert abs(effect-stored['signed_normal_minus_opposite'])<1e-10
                        signed.append(effect)
                assert len(set(indices))==len(indices)
                target=np.concatenate(targets)
                rmses={name:float(np.sqrt(np.mean((np.concatenate(pred)-target)**2))) for name,pred in cases.items()}
                reports.append(dict(episode=e,pre_count=len(target),pre_rmse=rmses,
                    inherited_optical_error_change_percent=100*(rmses['inherited_optical']/rmses['normal']-1),
                    spatial_neutral_error_change_percent=100*(rmses['spatial_neutral']/rmses['normal']-1)))
    assert sum(v>0 for v in signed)==r['expected_direction_windows'] and len(signed)==r['paired_windows']
    run=begin_run(a.output,dict(interventions=str(root),interventions_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),optimizer_updates=0))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_native_visual_intervention_audit',verified_model_events=model_events,
        verified_pre_labels=sum(x['pre_count'] for x in reports),expected_direction_windows=sum(v>0 for v in signed),paired_windows=len(signed),comparisons=reports,
        actor_qualified=False,vehicle_control_sent=False,counterfactual_action_truth=False,
        termination_reason='Saved full fixed neural replay and restored normal branches agree; raw array hashes, relative donor floor joins and intervention RMSE/direction statistics independently reconstructed'))


if __name__=='__main__':main()
