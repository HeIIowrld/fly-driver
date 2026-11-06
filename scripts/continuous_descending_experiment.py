"""Continuous numerical TRAIN state; detach gradients without injecting parent state."""
import torch
from flybrain_drive.functional_priority_objective import recording_losses
from scripts.sac_steering_experiment import snapshot,restore,DYNAMIC
from scripts.intervene_motion_spatial_same_body import state_digest

def detach_exact(model):
 before=state_digest(snapshot(model));model.detach_training_state();after=state_digest(snapshot(model))
 if before!=after:raise ValueError('Gradient cut changed numerical state')
 for name in DYNAMIC:
  value=getattr(model,name)
  for tensor in value if isinstance(value,tuple) else (value,):
   if isinstance(tensor,torch.Tensor) and (tensor.requires_grad or tensor.grad_fn is not None):raise ValueError('Gradient history was not detached')
 return before

def real_window(e,episode,begin,end,*,learn,parent_prefix=False):
 e.holder.update(mode='real',episode=episode)
 if parent_prefix:
  if not learn:raise ValueError('Parent counterfactual cut only at selected learning windows')
  restore(e.model,e.states[episode,begin])
 prediction=torch.stack([e.step(i,learn) for i in range(begin,end)])
 if not learn:return None,prediction
 d=e.items[episode][0];parts=recording_losses(prediction,torch.tensor(e.anchors[episode][begin:end]),torch.tensor(d['labels'][begin:end]),torch.tensor(d['fit_mask'][begin:end]),torch.tensor(e.pre[episode][begin:end]),anchor_weight=e.plan['real_anchor_weight'],maximum_ratio=e.plan['actual_recording_preservation_maximum_ratio'],protection_weight=e.plan['protection_weight'])
 return parts,prediction

def isolated_synthetic(e,task,update):
 # This is a separate synthetic episode. Resume the exact current real trajectory,
 # never a stored parent trajectory, after its independent gradient update.
 saved=snapshot(e.model);holder=e.holder.copy();expected=state_digest(saved)
 parts,prediction=e.synthetic(task,'episode_full');record=update('synthetic',task,parts,prediction)
 restore(e.model,saved);e.holder.clear();e.holder.update(holder)
 if state_digest(snapshot(e.model))!=expected:raise ValueError('Synthetic episode changed resumed real numerical state')
 return record,expected
