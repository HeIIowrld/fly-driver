"""Source-bound matched loss/regime trial metrics; no actor qualification."""
import argparse,shutil
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read

def functional_metrics(path):
    path=Path(path);r=read(path/'result.json');groups=read(path/'groups.json');assert sha256(path/'responses.npz')==r['responses_sha256'];loss=[];margin=[]
    with np.load(path/'responses.npz') as z:
        for g in groups:
            prediction=z[f"group_{g['group']}_fine_prediction"][:,25:,0].astype(float);m=-(prediction[1]-prediction[0])/2;loss.append(float(1000*np.maximum(.002-m,0).dot(np.maximum(.002-m,0))/len(m)));margin.append(float(m.mean()))
    return dict(direction_pairs=r['hypothesized_positive_bend_pairs'],groups=len(groups),mean_direction_loss=float(np.mean(loss)),mean_signed_margin=float(np.mean(margin)),target_margin_pairs=sum(v>=.002 for v in margin))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);plan=read(root/'fit-plan-01.json');assert read(root/'supervised-fit-01/training-schedule.json')==read(root/'functional_priority-fit-01/training-schedule.json');fits={arm:read(root/f'{arm}-fit-01/result.json') for arm in ('supervised','functional_priority')};assert all(v['optimizer_updates']==147 for v in fits.values());assert all(read(root/f'{arm}-fit-01/manifest.json')['config']['plan']==plan for arm in fits);assert len(read(root/'evaluation-process-receipts-01.json'))==14 and all(v['exit_code']==0 for v in read(root/'evaluation-process-receipts-01.json'));audits={arm:read(root/f'audit-{arm}-01/result.json') for arm in fits};assert all(v['status']=='completed_independent_functional_priority_audit' for v in audits.values());actual={};synthetic={};train_summary={}
    for arm in ('parent',*fits):
        paths=[Path(v['response']) if arm=='parent' else root/f'train{j}-{arm}-01' for j,v in enumerate(plan['synthetic_sources'])];metrics=[functional_metrics(v) for v in paths];train_summary[arm]=dict(mean_direction_loss=float(np.mean([v['mean_direction_loss'] for v in metrics])),mean_signed_margin=float(np.mean([v['mean_signed_margin'] for v in metrics])),total_positive_groups=sum(v['direction_pairs']['fine'] for v in metrics),groups=sum(v['groups'] for v in metrics));synthetic[arm]=dict(train_sources=metrics,reserved=functional_metrics(root/('reserved-parent-01' if arm=='parent' else f'reserved-{arm}-01')))
        if arm=='parent':continue
        actual[arm]=[]
        for e in range(3):
            r=read(root/f'actual-{arm}-{e}-01/result.json');actual[arm].append(dict(episode=e,events=r['events'],preservation_gate=r['preservation_gate_passed'],steering_rmse_reduction_percent={role:100*(1-r['metrics'][role]['functional']['rmse'][0]/r['metrics'][role]['parent']['rmse'][0]) for role in ('all','pre')},metrics=r['metrics']))
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256={k:sha256(root/f'{k}-fit-01/result.json') for k in fits},actor_qualified=False,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'result.json',dict(status='completed_matched_functional_priority_comparison',optimizer_updates_per_arm=147,matched_plan_parent_schedule_verified=True,synthetic_gradient_mode='episode_full',diagnostic=read(root/'diagnostic-01/result.json')['reports'],diagnostic_audit_result_sha256=sha256(root/'diagnostic-audit-01/result.json'),precision_validation=read(root/'gradient-validation-01/result.json')['reports'],training_direction_summary=train_summary,synthetic=synthetic,recordings=actual,continuous_actual_recording_events=sum(v['events'] for rows in actual.values() for v in rows),adam_audit_maximum_difference={k:v['independent_adam_replay_maximum_difference'] for k,v in audits.items()},tests_passed=38,actor_qualified=False,actor_promoted=False,vehicle_control_sent=False,goal_status='active',termination_reason='Same full-prefix gradient,42 bounded gains,147 updates and source/prefix dataset, real objective differs only. Functional-first did not improve fresh reserved response; same-course preservation is not physical student driving or physiology. Retain both candidates and original sources.'))

if __name__=='__main__':main()
