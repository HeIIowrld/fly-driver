"""Matched-budget research comparison; no autonomous qualification."""
import argparse,json,shutil
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root)
    fits={arm:read(root/f'{arm}-fit-01/result.json') for arm in ('diverse','repeated')}
    assert read(root/'diverse-fit-01/training-schedule.json')==read(root/'repeated-fit-01/training-schedule.json')
    assert all(r['optimizer_updates']==147 and r['real_updates']==93 and r['synthetic_updates']==54 for r in fits.values())
    configs={arm:read(root/f'{arm}-fit-01/manifest.json')['config'] for arm in fits}
    assert configs['diverse']['plan']==configs['repeated']['plan'] and configs['diverse']['parent_candidate']==configs['repeated']['parent_candidate']
    synthetics={};recorded={}
    for arm in fits:
        synthetics[arm]={role:read(root/f'{role}-{arm}-01/result.json')['hypothesized_positive_bend_pairs'] for role in ('train0','reserved')}
        recorded[arm]=[]
        for e in range(3):
            r=read(root/f'actual-{arm}-{e}-01/result.json');assert r['status']=='completed_functional_motion_recording_preservation';recorded[arm].append(dict(episode=e,events=r['events'],preserved=r['preservation_gate_passed'],steering_rmse_reduction_percent={role:100*(1-r['metrics'][role]['functional']['rmse'][0]/r['metrics'][role]['parent']['rmse'][0]) for role in ('all','pre')},all_axis_rmse={role:r['metrics'][role] for role in ('all','pre')}))
    receipts=read(root/'evaluation-process-receipts-01.json');assert len(receipts)==10 and all(r['exit_code']==0 for r in receipts)
    audits={arm:read(root/f'audit-{arm}-01/result.json') for arm in fits};assert all(r['status']=='completed_independent_balanced_visual_audit' for r in audits.values())
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256={arm:sha256(root/f'{arm}-fit-01/result.json') for arm in fits},actor_qualified=False,vehicle_control_sent=False));shutil.copyfile(__file__,out/'executed-entrypoint.py')
    write_json(out/'result.json',dict(status='completed_balanced_visual_comparison',optimizer_updates_per_arm=147,matched_schedule_verified=True,matched_plan_parent_verified=True,parent_reserved=read(root/'reserved-parent-01/result.json')['hypothesized_positive_bend_pairs'],synthetic=synthetics,recordings=recorded,continuous_events=sum(r['events'] for values in recorded.values() for r in values),full_neural_audit_replay=False,actor_qualified=False,vehicle_control_sent=False,goal_status='active',termination_reason='Two fixed-budget same-parent sensory-boundary trials completed with frozen test synthetic responses and consumed actual recording preservation; no physical student driving or unseen actual corner qualification.'))

if __name__=='__main__':main()
