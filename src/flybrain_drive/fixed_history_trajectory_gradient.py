"""Full vehicle credit for fixed synthetic visual/body histories, offline only."""
import torch
from .differentiable_planar_vehicle import differentiable_bicycle_step


def fixed_history_trajectory_gradient(initial_state, intent_turn, acceleration, *, bend,
                                      dt_s=.04, heading_weight=.25):
    """Return full-horizon d(mean trajectory cost)/d(turn intent).

    Source positions and synthetic centerline are loss-only; they never reach the
    actor. The detached turn vector is replayed exactly in a second graph pass.
    """
    if (not isinstance(initial_state,torch.Tensor) or initial_state.shape!=(1,4)
            or not isinstance(intent_turn,torch.Tensor) or intent_turn.ndim!=1
            or not isinstance(acceleration,torch.Tensor) or acceleration.shape!=intent_turn.shape
            or intent_turn.numel()<1 or bend not in (-1,0,1)
            or initial_state.device!=intent_turn.device or acceleration.device!=intent_turn.device
            or initial_state.dtype!=intent_turn.dtype or acceleration.dtype!=intent_turn.dtype
            or not isinstance(heading_weight,(int,float)) or not 0<=heading_weight<=10):
        raise ValueError('Explicit compatible fixed-history inputs required')
    control=intent_turn.detach().clone().requires_grad_(True);state=initial_state.detach().clone();losses=[];states=[]
    for i in range(len(control)):
        state=differentiable_bicycle_step(state,control[i:i+1]*.35,acceleration[i:i+1],dt_s=dt_s)
        offset=(state[:,0]-12.).clamp_min(0.);center=float(bend)*.018*offset.square();heading=torch.atan(float(bend)*.036*offset)
        losses.append((((state[:,1]-center)/.3).square()+float(heading_weight)*((state[:,2]-heading)/.15).square()).mean());states.append(state.detach()[0])
    loss=torch.stack(losses).mean();gradient,=torch.autograd.grad(loss,control)
    return loss.detach(),gradient.detach(),torch.stack(states)
