"""Independent receiver provenance and saved same-body intervention statistic audit."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.sustained_turn_diagnostic import paired_sustained_onsets
from scripts.motion_spatial_experiment import load_motion_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interventions',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.interventions);r=read(root/'result.json');m=read(root/'manifest.json')['config'];fit=Path(m['arguments']['fit']);fit_result=read(fit/'result.json');fit_config=read(fit/'manifest.json')['config'];e=m['arguments']['episode']
    assert r['status']=='completed_motion_spatial_same_body_interventions' and sha256(root/'branches.npz')==r['branches_sha256'] and sha256(root/'executed-entrypoint.py')==m['program_sha256'] and sha256(fit/'result.json')==m['fit_result_sha256']
    (_,receipt,entries,items,extras,native,valid,baseline,pre,features,maps,coarse,reference,scale)=load_motion_experiment(fit_config['plan'],'fine_motion');d,c,_=items[e];windows=[dict(w,end=int(np.searchsorted(d['decision_time'],w['time']+.25,side='right'))) for w in paired_sustained_onsets(d,-1)];assert windows==m['windows'];positive=0;indices=[];normal_error=[];neutral_error=[];opposite_error=[];maximum_change=0.
    with np.load(root/'branches.npz') as z,np.load(fit/'predictions.npz') as f:
        for k,w in enumerate(windows):
            row=r['windows'][k];begin,end=w['start'],w['end'];mask=pre[e][begin:end];ids=np.arange(begin,end)[mask];np.testing.assert_array_equal(ids,row['pre_indices']);indices.extend(map(int,ids));values={name:np.asarray(d[name][begin:end]) for name in ['body','body_valid','view_body','view_body_valid','body_packet','body_receipt','decision_time','render_lowers','camera_receipt','scene']};values.update(native19=native[e][begin:end].numpy(),native19_valid=valid[e][begin:end].numpy(),old_static=extras[e][begin:end].numpy());assert {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()}==row['receiver_input_sha256'];assert len(row['initial_state_sha256'])==64
            branches={case:z[f'window_{k}_{case}'] for case in ['normal','mean_motion','opposite_motion']}
            for value in branches.values():assert value.shape==(end-begin,3) and np.isfinite(value).all()
            assert float(abs(branches['normal']-f[f'prediction_{e}'][begin:end]).max())<=2e-6
            target=d['labels'][ids,0];normal=branches['normal'][mask,0];mean=branches['mean_motion'][mask,0];opposite=branches['opposite_motion'][mask,0]
            for name,prediction in [('normal_pre_rmse',normal),('mean_motion_pre_rmse',mean),('opposite_motion_pre_rmse',opposite)]:assert abs(float(np.sqrt(np.mean((prediction-target)**2)))-row[name])<1e-12
            margin=float(w['direction']*np.mean(normal.astype(float)-opposite.astype(float)));assert abs(margin-row['hypothesized_directional_ordering_margin'])<1e-7;assert bool(margin>1e-6)==row['hypothesized_directional_ordering_positive'];positive+=int(margin>1e-6);maximum_change=max(maximum_change,float(abs(normal-opposite).max()));normal_error.extend((normal-target).tolist());neutral_error.extend((mean-target).tolist());opposite_error.extend((opposite-target).tolist())
    assert len(indices)==len(set(indices)) and positive==r['hypothesized_direction_windows'] and len(windows)==r['paired_windows']
    rms=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),interventions_result_sha256=sha256(root/'result.json'),seed=None,optimizer_updates=0,vehicle_control_sent=False))
    write_json(out/'result.json',dict(status='completed_independent_motion_same_body_statistics_audit',paired_windows=len(windows),pre_labels=len(indices),normal_pre_rmse=rms(normal_error),mean_motion_pre_rmse=rms(neutral_error),opposite_motion_pre_rmse=rms(opposite_error),maximum_opposite_steering_change=maximum_change,hypothesized_direction_windows=positive,receiver_body_native_old_scene_and_clocks_verified=True,original_normal_branches_match_fitted_trajectory=True,independent_full_counterfactual_neural_replay=False,physical_counterfactual_action_correctness_verified=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Receiver byte hashes, paired stimulus/mask selection, original normal trajectory and all saved branch statistics verified. Common prefix-state equality is asserted by source-bound intervention run, not independently reconstructed here. Donor road pose/action truth remain unverified.'))

if __name__=='__main__':main()
