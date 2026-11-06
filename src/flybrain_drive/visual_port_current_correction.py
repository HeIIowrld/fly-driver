"""Zero-neutral engineering correction of four existing flight DN input currents."""
import torch
from torch import nn

class VisualPortCurrentCorrection(nn.Module):
    def __init__(self,reference,scale,maximum=.2):
        super().__init__()
        if reference.shape!=(320,) or scale.shape!=(320,) or not torch.isfinite(reference).all() or not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Explicit finite TRAIN-only retinal calibration required')
        self.register_buffer('reference',reference.detach().clone());self.register_buffer('scale',scale.detach().clone())
        self.maximum=float(maximum);self.projection=nn.Linear(320,4,bias=False);nn.init.zeros_(self.projection.weight)
    def forward(self,features,base):
        if features.ndim!=2 or features.shape[1]!=320 or base.shape!=(len(features),4) or not torch.isfinite(features).all() or not torch.isfinite(base).all() or (base<0).any() or (base>self.maximum).any():
            raise ValueError('Current original retinal features and four bounded flight currents required')
        x=((features-self.reference)/self.scale).clamp(-5,5);delta=2*torch.tanh(self.projection(x))
        ratio=(base/self.maximum).clamp(1e-7,1-1e-7);logit=torch.logit(ratio)
        # Subtract the same numerical baseline, so zero weights preserve bytes.
        return (base+self.maximum*((logit+delta).sigmoid()-logit.sigmoid())).clamp(0,self.maximum)
