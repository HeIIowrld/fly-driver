"""Learned modeled-motion spatial residual at existing flight visual ports.

Four views x eight modeled T4/T5 types x8x8 remain explicit. This is an
engineering adapter, not a measured receptive field or vehicle controller.
"""
import torch
from torch import nn

MOTION_CELL_TYPES=tuple(f'{kind}{direction}' for kind in ('T4','T5') for direction in 'abcd')
MOTION_FEATURES=4*8*8*8


class MotionSpatialResidualPort(nn.Module):
    def __init__(self,reference,scale,ports,*,maximum_logit_delta=.25):
        super().__init__()
        if (reference.shape!=(MOTION_FEATURES,) or scale.shape!=reference.shape
                or reference.dtype!=torch.float32 or scale.dtype!=reference.dtype or reference.device!=scale.device
                or not torch.isfinite(reference).all() or not torch.isfinite(scale).all() or (scale<=0).any()
                or type(ports) is not int or ports<1 or isinstance(maximum_logit_delta,bool)
                or not 0<maximum_logit_delta<=.5):raise ValueError('Finite TRAIN-only2048 motion normalization and explicit visual port count required')
        self.register_buffer('reference',reference.detach().clone());self.register_buffer('scale',scale.detach().clone())
        self.maximum_logit_delta=float(maximum_logit_delta)
        self.projection=nn.Linear(MOTION_FEATURES,ports,bias=False,device=reference.device);nn.init.zeros_(self.projection.weight)

    def forward(self,features):
        if (not isinstance(features,torch.Tensor) or features.ndim!=2 or not len(features) or features.shape[1]!=MOTION_FEATURES
                or features.dtype!=self.reference.dtype or features.device!=self.reference.device or not torch.isfinite(features).all()):raise ValueError('Finite current four-view/direction/spatial motion features required')
        z=((features-self.reference)/self.scale).clamp(-5,5)
        return self.maximum_logit_delta*torch.tanh(self.projection(z))
