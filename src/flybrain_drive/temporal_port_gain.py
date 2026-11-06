"""Research-only positive gain on existing temporal port outputs.

The fixed opponent projection and numerical traces are preserved. These gains
are engineering calibration parameters, not measured synaptic/physiological gains.
"""
from pathlib import Path
import torch
from torch import nn
from .audited_motion_spatial_candidate import require_frozen_motion_parent
from .functional_motion_overlay import exact_model
from .graph import sha256
from .sac_unified_student import read

class TemporalPortGain(nn.Module):
    def __init__(self,ports,maximum_change=.75):
        super().__init__()
        if type(ports) is not int or ports<1 or not 0<maximum_change<1:raise ValueError('Positive bounded port gains required')
        self.maximum_change=maximum_change;self.value=nn.Parameter(torch.ones(ports))
    def forward(self,delta):
        if delta.shape!=(1,len(self.value)) or delta.dtype!=self.value.dtype or delta.device!=self.value.device or not torch.isfinite(delta).all() or not torch.isfinite(self.value).all() or torch.any(abs(self.value-1)>self.maximum_change+1e-7):raise ValueError('Exact bounded finite temporal gain contract required')
        return delta*self.value
    @torch.no_grad()
    def project(self):self.value.clamp_(1-self.maximum_change,1+self.maximum_change)

class _GainAttachment:
    def __init__(self,temporal,gain):
        if getattr(temporal,'_research_gain_attachment',None) is not None:raise ValueError('Temporal gain already attached')
        self.temporal=temporal;self.original=temporal.step;self.active=True;temporal._research_gain_attachment=self
        def step(*args,**kwargs):
            delta,traces=self.original(*args,**kwargs);return gain(delta),traces
        temporal.step=step
    def remove(self):
        if self.active:self.temporal.step=self.original;self.temporal._research_gain_attachment=None;self.active=False

def attach_temporal_gain(temporal,gain):return _GainAttachment(temporal,gain)

def load_temporal_gain(path,components,motion,receipt):
    path=Path(path);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];cp=torch.load(path,map_location='cpu',weights_only=True)
    if r['status']!='completed_temporal_gain_fit' or sha256(path)!=r['checkpoint_sha256'] or cp['schema']!='flybrain.temporal_gain.v1' or cp['actor_qualified'] is not False or cp['parent_candidate']!=receipt or m['parent_candidate']!=receipt or cp['optimizer_updates']!=r['optimizer_updates']:raise ValueError('Exact unqualified temporal gain fit required')
    require_frozen_motion_parent(cp,components)
    if not exact_model(cp['model'],components[0].state_dict()) or not exact_model(cp['motion_port'],motion.state_dict()):raise ValueError('Temporal gain altered fixed original graph or sensory ports')
    gain=TemporalPortGain(len(components[0].motor.backbone.vision_ports),m['plan']['maximum_gain_change']);gain.load_state_dict(cp['temporal_gain'],strict=True);gain.eval().requires_grad_(False);gain(torch.zeros(1,len(gain.value)));handle=attach_temporal_gain(components[0].temporal,gain)
    return dict(receipt,temporal_gain_checkpoint_sha256=sha256(path),temporal_gain_scope='Research positive temporal output calibration only; no live/actuator authority',actor_qualified=False,vehicle_control_sent=False),handle
