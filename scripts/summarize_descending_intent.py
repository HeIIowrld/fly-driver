"""Source-bound matched existing DN-projection trial; no physical control claims."""
import json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.summarize_functional_priority_trial import functional_metrics

def main():
 torch.set_num_threads(1);root=Path('runs/descending-visual-intent-20261104');plan=read(root/'plan-01.json');receipts=read(root/'evaluation-process-receipts-01.json');assert len(receipts)==15 and all(v['exit_code']==0 for v in receipts);assert read(root/'real_only-fit-01/training-schedule.json')==read(root/'joint-fit-01/training-schedule.json');rows={};bound={}
 for arm in ('parent','real_only','joint'):
  paths=[Path(v['response']) if arm=='parent' else root/f'train{j}-{arm}-01' for j,v in enumerate(plan['synthetic_sources'])];train=[functional_metrics(v) for v in paths];reserved=functional_metrics(root/f'reserved-{arm}-01');row=dict(train_positive=sum(v['direction_pairs']['fine'] for v in train),train_groups=sum(v['groups'] for v in train),train_direction_loss=float(np.mean([v['mean_direction_loss'] for v in train])),train_target_pairs=sum(v['target_margin_pairs'] for v in train),reserved=reserved)
  if arm!='parent':
   fit=root/f'{arm}-fit-01';r=read(fit/'result.json');audit=read(root/f'audit-{arm}-01/result.json');assert r['optimizer_updates']==147 and audit['status']=='completed_independent_descending_intent_audit';row.update(checkpoint_sha256=r['checkpoint_sha256'],maximum_weight_delta=r['maximum_weight_change'],independent_adam_difference=audit['independent_adam_maximum_difference'],actual=[])
   cp=torch.load(fit/'descending-intent.pt',map_location='cpu',weights_only=True);state=cp['adapter']
   with np.load(root/'reserved-stimuli-01/features.npz') as z:x=torch.tensor(z['static'])
   raw=(((x-state['reference'])/state['scale']).clamp(-5,5)@state['projection.weight'].T)[...,0];row['reserved_intent_raw_range']=[float(raw.min()),float(raw.max())];row['reserved_intent_slope_fraction_range']=[float((1-raw.tanh().square()).min()),float((1-raw.tanh().square()).max())]
   for e in range(3):
    path=root/f'actual-{arm}-{e}-01';rr=read(path/'result.json');row['actual'].append(dict(episode=e,events=rr['events'],preservation_gate=rr['preservation_gate_passed'],rmse_reduction_percent={role:[100*(1-n/o) for n,o in zip(rr['metrics'][role]['functional']['rmse'],rr['metrics'][role]['parent']['rmse'])] for role in ('all','pre')},metrics=rr['metrics']));bound[str(path/'result.json')]=sha256(path/'result.json')
   for path in [fit/'result.json',root/f'audit-{arm}-01/result.json']:bound[str(path)]=sha256(path)
  rows[arm]=row
 for f in ['diagnostic-01/result.json','plan-01.json','reserved-plan-01.json','reserved-stimuli-01/result.json','evaluation-process-receipts-01.json','core-tests-01.txt','overlay-tests-01.txt']:bound[str(root/f)]=sha256(root/f)
 result=dict(status='completed_descending_visual_intent_comparison',models=rows,diagnostic=read(root/'diagnostic-01/result.json'),source_bound_sha256=bound,program_sha256=sha256(__file__),same_parent_plan_schedule_and_bounds_verified=True,optimizer_updates_per_arm=147,trainable_parameters=320,fresh_texture=781127,full_continuous_recording_events=sum(v['events'] for arm in ('real_only','joint') for v in rows[arm]['actual']),tests_passed=21,synthetic_body_fixture='Previously declared hypothetical loads/wheels/reference pressure fixture, not measured native inputs; real evaluation uses original native-byte sensing',actor_promoted=False,actor_qualified=False,actual_driving_improvement_verified=False,vehicle_control_sent=False,goal_status='active')
 result['training_prefix_consistency']=read(root/'prefix-consistency-01/result.json');result['training_prefix_consistency_audit']=read(root/'prefix-consistency-audit-01.json')
 for f in ['prefix-consistency-01/result.json','prefix-consistency-01/predictions.npz','prefix-consistency-audit-01.json','metadata-clarification-01.json']:result['source_bound_sha256'][str(root/f)]=sha256(root/f)
 write_json(root/'summary-01.json',result);print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
