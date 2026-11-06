"""Validate the untrained dual-reference loss prototype on an audited TRAIN gap."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.dual_reference_recording_objective import dual_reference_recording_losses
from flybrain_drive.functional_priority_objective import recording_losses
from flybrain_drive.sac_unified_student import read
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.paired_decoder_data import load_data
ROOT=Path('runs/direct-scene-gain-learning-20261105')

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');diagnostic=read(ROOT/'anchor-protection-diagnostic-01/result.json');rows=[r for r in diagnostic['models']['learn']['rows'] if r['regression_not_detected_by_old_anchor']];row=max(rows,key=lambda x:x['current_parent_relative_violation']);task=row['task'];ep,b,e=task['episode'],task['begin'],task['end'];protocol=read('runs/multicorner-command-20261104/protocol-01.json');_,items,_,_,pre=load_data(read(protocol['data_plan']));d=items[ep][0];paths=dict(original='runs/motion-spatial-learning-20261104/fine_motion-fit-01/predictions.npz',parent=str(Path(plan['direct_parent_evaluation'])/'predictions.npz'),candidate=str(ROOT/'learn_direct-evaluation-01/predictions.npz'));values={}
 for name,path in paths.items():
  with np.load(path) as z:values[name]=torch.tensor(z[f'prediction_{ep}'][b:e].copy())
 options=dict(anchor_weight=1.,maximum_ratio=read(plan['parent_fit_plan'])['actual_recording_preservation_maximum_ratio'],protection_weight=1.);target=torch.tensor(d['labels'][b:e]);fit=torch.tensor(d['fit_mask'][b:e]);pm=torch.tensor(pre[ep][b:e]);p=values['candidate'].clone().requires_grad_();old=recording_losses(p,values['original'],target,fit,pm,**options);new=dual_reference_recording_losses(p,values['original'],values['parent'],target,fit,pm,**options);assert torch.equal(old[0],new[0]) and torch.equal(old[1],new[1]) and new[2]>old[2];gradient=torch.autograd.grad(new[2],p)[0];norm=float(gradient.norm());direction=gradient/norm;finite=[]
 for step in [.001,.0002,.0001]:
  losses=[float(dual_reference_recording_losses(p.detach()+sign*step*direction,values['original'],values['parent'],target,fit,pm,**options)[2]) for sign in [-1,1]];fd=(losses[1]-losses[0])/(2*step);finite.append(dict(step=step,analytic=norm,finite_difference=fd,relative_error=abs(fd-norm)/norm))
 assert min(x['relative_error'] for x in finite)<.01;out=begin_run(ROOT/'dual-reference-prototype-validation-01',dict(program_sha256=sha256(__file__),objective_sha256=sha256('src/flybrain_drive/dual_reference_recording_objective.py'),source_sha256={k:sha256(v) for k,v in paths.items()},TRAIN_only=True,optimizer_updates=0,vehicle_control_sent=False));write_json(out/'result.json',dict(status='completed_untrained_dual_reference_loss_validation',task=task,supervision_and_original_anchor_terms_exact=True,original_protection=float(old[2].detach()),dual_protection=float(new[2].detach()),new_loss_gradient_norm=norm,finite_differences=finite,source_pre_steering_gap=row,trained_with_new_objective=False,actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False))

if __name__=='__main__':main()
