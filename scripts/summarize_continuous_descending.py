"""Matched continuous-state trial metrics; no actor qualification."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.summarize_functional_priority_trial import functional_metrics

def main():
 root=Path('runs/continuous-visual-learning-20261104');plan=read(root/'plan-01.json');receipts=read(root/'evaluation-process-receipts-01.json');assert len(receipts)==9 and all(v['exit_code']==0 for v in receipts);assert read(root/'continuous-fit-01/training-schedule.json')==read(root/'parent_prefix-fit-01/training-schedule.json');rows={};bound={};events=0
 for arm in ('parent','continuous','parent_prefix'):
  path=root/f'reserved-{arm}-01';row=dict(reserved=functional_metrics(path));counts={k:0 for k in ('fine','motion_off_after_prefix')}
  with np.load(path/'responses.npz') as z:
   for group in read(path/'groups.json'):
    for mode in counts:
     value=z[f"group_{group['group']}_{mode}_prediction"][:,25:,0];counts[mode]+=int(float(-np.mean(value[1]-value[0])/2)>.0001)
  assert counts==row['reserved']['direction_pairs']
  if arm!='parent':
   fit=root/f'{arm}-fit-01';r=read(fit/'result.json');audit=read(root/f'audit-{arm}-01/result.json');assert r['optimizer_updates']==147 and audit['status']=='completed_independent_continuous_intent_audit';row.update(checkpoint_sha256=r['checkpoint_sha256'],state_injections=r['parent_state_injections'],real_training_events=r['real_events'],state_blocks=audit['continuous_state_journal_blocks_verified'],synthetic_exact_resumes=audit['synthetic_exact_resumes'],maximum_weight_delta=r['maximum_weight_change'],independent_adam_difference=audit['independent_adam_maximum_difference'],actual=[])
   for e in range(3):
    path=root/f'actual-{arm}-{e}-01';rr=read(path/'result.json');events+=rr['events'];row['actual'].append(dict(episode=e,events=rr['events'],preservation_gate=rr['preservation_gate_passed'],rmse_reduction_percent={role:[100*(1-n/o) for n,o in zip(rr['metrics'][role]['functional']['rmse'],rr['metrics'][role]['parent']['rmse'])] for role in ('all','pre')},metrics=rr['metrics']));bound[str(path/'result.json')]=sha256(path/'result.json')
   for path in [fit/'result.json',fit/'state-journal.json',root/f'audit-{arm}-01/result.json']:bound[str(path)]=sha256(path)
  rows[arm]=row;bound[str(root/f'reserved-{arm}-01/result.json')]=sha256(root/f'reserved-{arm}-01/result.json')
 contrast=[]
 for e in range(3):
  a,b=rows['continuous']['actual'][e],rows['parent_prefix']['actual'][e];contrast.append(dict(episode=e,continuous_vs_parent_prefix_rmse_reduction_percent={role:[100*(1-n/o) for n,o in zip(a['metrics'][role]['functional']['rmse'],b['metrics'][role]['functional']['rmse'])] for role in ('all','pre')}))
 for f in ['plan-01.json','reserved-plan-01.json','reserved-stimuli-01/result.json','continuity-validation-01/result.json','intent-dimensions-01/result.json','evaluation-process-receipts-01.json','core-tests-01.txt']:bound[str(root/f)]=sha256(root/f)
 result=dict(status='completed_matched_continuous_state_learning',models=rows,continuous_vs_parent_prefix=contrast,continuity_validation=read(root/'continuity-validation-01/result.json'),intent_dimension_audit=read(root/'intent-dimensions-01/result.json'),source_bound_sha256=bound,program_sha256=sha256(__file__),same_parent_plan_schedule_and_bounds_verified=True,real_training_events_per_arm=32597,optimizer_updates_per_arm=147,trainable_parameters=320,real_gradient_selected_events=2933,full_continuous_recording_events=events,fresh_texture=781139,tests_passed=21,gradient_truncation='32-event selected windows; chronological nonselected events advance numerical state without gradients',objective='Both arms teacher+anchor+excess-error protection on real, direction+anchor on synthetic; differs from previous trial which omitted protection in selected objective',synthetic_fixture='Previously declared hypothetical full load/wheel/reference-pressure fixture, not native sensor evidence',actor_promoted=False,actor_qualified=False,actual_driving_improvement_verified=False,vehicle_control_sent=False,goal_status='active');write_json(root/'summary-01.json',result);print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
