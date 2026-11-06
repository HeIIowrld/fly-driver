"""Bounded joint learned sensory-port objective; teacher/anchor are target-only."""
import torch


def joint_sensory_loss(prediction,target,anchor,*,pre):
    if prediction.shape!=(1,3) or target.shape!=(3,) or anchor.shape!=(3,) or type(pre) is not bool:
        raise ValueError('One three-axis output, fixed targets and explicit pre-corner flag required')
    if not all(torch.isfinite(x).all() for x in [prediction,target,anchor]):raise ValueError('Finite sensory supervision required')
    weights=prediction.new_tensor([20. if pre else 4.,1.,4. if pre else 1.])
    preservation=prediction.new_tensor([4.,1.,1.])
    return ((prediction[0]-target.detach()).square()*weights).sum()+.5*((prediction[0]-anchor.detach()).square()*preservation).sum()


def project_sensory_weight(weight,reference,bound):
    if (weight.shape!=reference.shape or weight.dtype!=reference.dtype or weight.device!=reference.device
            or isinstance(bound,bool) or not isinstance(bound,(float,int)) or not 0<bound<=.02
            or not torch.isfinite(weight).all() or not torch.isfinite(reference).all()):raise ValueError('Finite compatible sensory weights and explicit bounded change required')
    with torch.no_grad():weight.copy_(reference+(weight-reference).clamp(-bound,bound))
