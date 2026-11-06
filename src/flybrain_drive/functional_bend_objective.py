"""Synthetic bend-order training target only; no actor/vehicle geometry input."""
import torch


def functional_bend_loss(right,left,right_anchor,left_anchor,*,left_sign,target_margin,direction_weight,anchor_weight):
    if right.shape!=left.shape or right.ndim!=2 or right.shape[1]!=3 or right.shape!=right_anchor.shape or left.shape!=left_anchor.shape or left_sign not in (-1,1) or not 0<target_margin<=.01 or not 0<direction_weight<=1000 or not 0<anchor_weight<=10 or any(not torch.isfinite(x).all() for x in (right,left,right_anchor,left_anchor)):raise ValueError('Aligned finite three-axis functional targets and explicit bounded weights required')
    margin=left_sign*(left[:,0]-right[:,0])/2
    preservation=right.new_tensor([4.,1.,1.]);anchor=((right-right_anchor.detach()).square()*preservation).sum(1).mean()+((left-left_anchor.detach()).square()*preservation).sum(1).mean()
    return direction_weight*torch.relu(target_margin-margin).square().mean()+anchor_weight*anchor/2
