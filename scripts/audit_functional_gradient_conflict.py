"""Independent recorded gradient vectors and finite-difference arithmetic audit."""
import argparse,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read

def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.diagnostic);r=read(root/'result.json');m=read(root/'manifest.json')['config'];validation=Path(a.validation);v=read(validation/'result.json');assert r['status']=='completed_train_functional_gradient_conflict' and sha256(root/'gradients.npz')==r['gradients_sha256'] and read(validation/'manifest.json')['config']['diagnostic_result_sha256']==sha256(root/'result.json');maximum=0.;reports=[]
    with np.load(root/'gradients.npz') as z:
        for row in r['reports']:
            key=row['point'];actual=z[f'{key}_real_gradients'].astype(float);full=z[f'{key}_episode_full_synthetic_gradients'].astype(float);legacy=z[f'{key}_legacy_synthetic_gradients'].astype(float);assert actual.shape==(12,3,42) and full.shape==legacy.shape==(10,2,42);scores=[cos(aa,bb) for aa in actual[:,0] for bb in full[:,0] if np.linalg.norm(aa)*np.linalg.norm(bb)>1e-24];value=cos(actual[:,0].mean(0),full[:,0].mean(0));maximum=max(maximum,abs(value-row['supervised_vs_full_functional_cosine']));assert maximum<1e-5 and sum(x<0 for x in scores)==row['real_functional_negative_pair_count'];np.testing.assert_array_equal(z[f'{key}_legacy_predictions'],z[f'{key}_episode_full_predictions']);reports.append(dict(point=key,mean_gradient_cosine=value,negative_pair_count=sum(x<0 for x in scores),pairs=len(scores)))
    for row in v['reports']:
        for check in row['checks']:assert abs(abs(check['numerical']-check['analytic'])-check['absolute_error'])<1e-15
    assert v['all_largest_step_relative_errors_below_05'] and all(row['checks'][0]['relative_error']<.05 for row in v['reports']);assert all(t['episode'] in (0,1) for t in m['selected_real_tasks']) and all(t['source_slot'] in (0,1,2) and t['temperature']==80. for t in m['selected_synthetic_tasks'])
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),diagnostic_result_sha256=sha256(root/'result.json'),validation_result_sha256=sha256(validation/'result.json'),optimizer_updates=0,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_independent_functional_gradient_conflict_audit',reports=reports,maximum_recomputed_cosine_difference=maximum,identical_numeric_trajectories_verified=True,subsets_and_source_roles_verified=True,largest_step_arithmetic_and_threshold_verified=True,neural_gradients_rerun_by_audit=False,full_population_gradient_claimed=False,actor_qualified=False,vehicle_control_sent=False,termination_reason='Independent stored gradient-vector cosine/counts and finite-difference arithmetic plus TRAIN subset indices. Not a rerun of the neural gradients or global derivative correctness.'))

if __name__=='__main__':main()
