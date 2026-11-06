"""Offline functional-first loss and explicit numerical-state gradient handling."""
import copy
import torch

DYNAMIC=('state','traces','previous_drive','previous_scene','previous_raw_scene','previous_time','previous_body_receipt','previous_render_lowers')

def _clone(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,tuple):return tuple(_clone(v) for v in value)
    return copy.deepcopy(value)

def gradient_snapshot(model):return {key:_clone(getattr(model,key)) for key in DYNAMIC}

def gradient_restore(model,state):
    if set(state)!=set(DYNAMIC):raise ValueError('Complete original numerical state required')
    for key,value in state.items():setattr(model,key,_clone(value))

def split_functional_loss(right,left,anchor,*,left_sign,target_margin,direction_weight,anchor_weight):
    if right.shape!=left.shape or right.ndim!=2 or right.shape[1]!=3 or anchor.shape!=(2,*right.shape) or left_sign not in (-1,1) or not 0<target_margin<=.01 or not 0<direction_weight<=1000 or not 0<anchor_weight<=10 or any(not torch.isfinite(v).all() for v in (right,left,anchor)):raise ValueError('Aligned finite functional training outputs required')
    margin=left_sign*(left[:,0]-right[:,0])/2
    direction=direction_weight*torch.relu(target_margin-margin).square().mean()
    weights=right.new_tensor([4.,1.,1.]);preservation=anchor_weight*.5*(((right-anchor[0].detach()).square()*weights).sum(1).mean()+((left-anchor[1].detach()).square()*weights).sum(1).mean())
    return direction,preservation

def recording_losses(prediction,anchor,target,fit_mask,pre_mask,*,anchor_weight,maximum_ratio,protection_weight):
    if prediction.shape!=anchor.shape or prediction.shape!=target.shape or prediction.ndim!=2 or prediction.shape[1]!=3 or fit_mask.shape!=prediction.shape[:1] or pre_mask.shape!=fit_mask.shape or fit_mask.dtype!=torch.bool or pre_mask.dtype!=torch.bool or not 0<anchor_weight<=10 or not 1<=maximum_ratio<=1.1 or not 0<protection_weight<=100 or any(not torch.isfinite(v).all() for v in (prediction,anchor,target)):raise ValueError('Explicit aligned recording preservation contract required')
    zero=prediction.sum()*0;preservation=anchor_weight*((prediction-anchor.detach()).square()*prediction.new_tensor([4.,1.,1.])).sum(1).mean();supervision=zero;protection=zero;violations=[]
    if fit_mask.any():
        weight=prediction.new_tensor([4.,1.,1.]).expand_as(prediction).clone();weight[pre_mask]=prediction.new_tensor([20.,1.,4.]);supervision=(((prediction-target.detach()).square()*weight).sum(1)[fit_mask]).mean()
        for mask,axes in ((fit_mask,[0,1,2]),(pre_mask & fit_mask,[0])):
            if not mask.any():continue
            now=(prediction[mask][:,axes]-target.detach()[mask][:,axes]).square().mean(0);old=(anchor.detach()[mask][:,axes]-target.detach()[mask][:,axes]).square().mean(0);violations.append(torch.relu((now-maximum_ratio**2*old)/old.clamp_min(1e-5)))
        protection=protection_weight*torch.cat(violations).square().mean()
    return supervision,preservation,protection
