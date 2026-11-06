"""Audit old-fine versus current-parent preservation on fixed TRAIN trajectories."""

from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run,write_json

from flybrain_drive.graph import sha256

from flybrain_drive.sac_unified_student import read

from scripts.paired_decoder_data import load_data

ROOT=Path('runs/dual-reference-learning-20261105')



def main():

 plan=read(ROOT/'plan-01.json');protocol=read('runs/multicorner-command-20261104/protocol-01.json');_,items,_,_,pre=load_data(read(protocol['data_plan']));parent_plan=read(plan['parent_fit_plan']);ratio=parent_plan['actual_recording_preservation_maximum_ratio'];paths=dict(original='runs/motion-spatial-learning-20261104/fine_motion-fit-01/predictions.npz',parent=str(Path(plan['direct_parent_evaluation'])/'predictions.npz'),original_guard=str(Path(plan['control_evaluation'])/'predictions.npz'),dual_guard=str(ROOT/'dual_guard-evaluation-01/predictions.npz'));arrays={}

 for name,path in paths.items():

  with np.load(path) as z:arrays[name]={i:z[f'prediction_{i}'].copy() for i in [0,1]}

 out=begin_run(ROOT/'anchor-protection-diagnostic-01',dict(program_sha256=sha256(__file__),source_sha256={name:sha256(path) for name,path in paths.items()},ratio=ratio,TRAIN_only=True,optimizer_updates=0,vehicle_control_sent=False));reports={}

 for arm in ['original_guard','dual_guard']:

  rows=[]

  for task in plan['selected_real_windows']:

   ep,b,e=task['episode'],task['begin'],task['end'];d=items[ep][0];mask=pre[ep][b:e]&d['fit_mask'][b:e]

   if not mask.any():continue

   losses={name:float(np.mean((arrays[name][ep][b:e,0][mask]-d['labels'][b:e,0][mask])**2)) for name in ['original','parent',arm]};old_penalty=max(0.,(losses[arm]-ratio**2*losses['original'])/max(losses['original'],1e-5));parent_penalty=max(0.,(losses[arm]-ratio**2*losses['parent'])/max(losses['parent'],1e-5));rows.append(dict(task=task,events=int(mask.sum()),mse=losses,original_fine_relative_violation=old_penalty,current_parent_relative_violation=parent_penalty,regression_not_detected_by_old_anchor=old_penalty==0 and parent_penalty>0))

  reports[arm]=dict(windows=len(rows),regression_windows_not_detected_by_old_anchor=sum(x['regression_not_detected_by_old_anchor'] for x in rows),regression_events_not_detected_by_old_anchor=sum(x['events'] for x in rows if x['regression_not_detected_by_old_anchor']),windows_with_original_violation=sum(x['original_fine_relative_violation']>0 for x in rows),windows_with_parent_violation=sum(x['current_parent_relative_violation']>0 for x in rows),rows=rows)

 write_json(out/'result.json',dict(status='completed_train_anchor_protection_diagnostic',models=reports,scope='Fixed final-weight TRAIN trajectories projected onto the exact declared recording-loss masks and ratio. Original preservation anchor remains original fine. This identifies a protection gap relative to the immediate parent, not a proof of sole training-error causation; online training states/weights were different.',actor_qualified=False,vehicle_control_sent=False))

 print({k:{n:v for n,v in d.items() if n!='rows'} for k,d in reports.items()})



if __name__=='__main__':main()
