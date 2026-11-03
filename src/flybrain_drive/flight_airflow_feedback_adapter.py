"""Experimental relative-airflow input to existing flight body sensory ports.

This learned boundary is an engineering hypothesis, not measured receptor wiring.
Unverified/absent sensors contribute exactly zero and are not made valid here.
"""
import math
import numpy as np
import torch
from torch import nn
from .relative_airflow_sensor import AirflowObservation


class AirflowFeedbackAdapter(nn.Module):
    def __init__(self, sensory_ports):
        super().__init__()
        if type(sensory_ports) is not int or sensory_ports<1:raise ValueError('Original body sensory ports required')
        self.register_buffer('scale',torch.tensor([30.,30.,30.]))
        self.projection=nn.Linear(3,sensory_ports,bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, observation, *, decision_time, like):
        if (not isinstance(observation,AirflowObservation) or type(observation.source_verified) is not bool
                or isinstance(decision_time,bool) or not isinstance(decision_time,(int,float)) or not math.isfinite(decision_time)
                or observation.decision_time!=decision_time or not isinstance(like,torch.Tensor)
                or like.shape!=(1,1) or not torch.is_floating_point(like)
                or like.device!=self.scale.device or like.dtype!=self.scale.dtype):
            raise ValueError('Explicit same-decision airflow observation required')
        values,mask=observation.velocity_body_m_s,observation.valid
        if (not isinstance(values,np.ndarray) or values.shape!=(3,) or not np.isfinite(values).all()
                or not isinstance(mask,np.ndarray) or mask.shape!=(3,) or mask.dtype!=np.bool_
                or (mask.any() and not observation.source_verified)):
            raise ValueError('Finite airflow and qualified explicit validity required')
        tensor=torch.as_tensor(values.copy(),dtype=like.dtype,device=like.device).unsqueeze(0)
        valid=torch.as_tensor(mask.copy(),dtype=torch.bool,device=like.device).unsqueeze(0)
        features=torch.where(valid,torch.tanh(tensor/self.scale),torch.zeros_like(tensor))
        return .5*torch.tanh(self.projection(features))
