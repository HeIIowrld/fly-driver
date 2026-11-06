"""Explicit same-camera binding for the experimental motion residual runtime."""
from dataclasses import dataclass
import numpy as np
import torch
from .native_flight_decision_runtime import NativeFlightDecisionRuntime
from .motion_spatial_residual_port import MotionSpatialResidualPort,MOTION_FEATURES
from .sensor_contract import readonly_array


@dataclass(frozen=True)
class MotionSpatialObservation:
    episode_id:str
    camera_receipt_time:float
    render_start_lowers:tuple
    coarse_motion_reference:np.ndarray
    features:np.ndarray

    def __post_init__(self):
        if (not isinstance(self.episode_id,str) or not self.episode_id or not np.isfinite(self.camera_receipt_time)
                or len(self.render_start_lowers)!=4 or not np.isfinite(self.render_start_lowers).all()
                or self.camera_receipt_time<0 or min(self.render_start_lowers)<0
                or self.camera_receipt_time<max(self.render_start_lowers)
                or np.asarray(self.coarse_motion_reference).shape!=(512,) or np.asarray(self.features).shape!=(MOTION_FEATURES,)):
            raise ValueError('Explicit same-camera motion observation and original coarse reference required')
        object.__setattr__(self,'render_start_lowers',tuple(map(float,self.render_start_lowers)))
        object.__setattr__(self,'features',readonly_array(self.features,np.float32));object.__setattr__(self,'coarse_motion_reference',readonly_array(self.coarse_motion_reference,np.float32))


def validate_motion_binding(observation,*,episode_id,scene_features,body_context):
    if (not isinstance(observation,MotionSpatialObservation) or observation.episode_id!=episode_id
            or observation.camera_receipt_time!=body_context.camera_receipt_time
            or tuple(observation.render_start_lowers)!=tuple(body_context.render_start_lowers)):
        raise ValueError('Motion residual belongs to another episode/capture clock')
    expected=np.asarray(scene_features).reshape(4,9,16)[:,1:].reshape(-1)
    if not np.allclose(observation.coarse_motion_reference,expected,atol=2e-6,rtol=0):raise ValueError('Motion residual coarse reference differs from original scene')


class MotionSpatialDecisionRuntime(NativeFlightDecisionRuntime):
    def __init__(self,*components,motion_port,**kwargs):
        if type(motion_port) is not MotionSpatialResidualPort or any(t.device.type!='cpu' for t in [*motion_port.parameters(),*motion_port.buffers()]):raise ValueError('Explicit CPU learned motion port required')
        if motion_port.projection.out_features!=components[3].projection.out_features:raise ValueError('Motion and existing visual port dimensions differ')
        self.motion_port=motion_port.eval().requires_grad_(False);self.motion_active=None
        super().__init__(*components,**kwargs)

    def _optical(self,module,args,value):
        result=super()._optical(module,args,value)
        if self.motion_active is None:raise ValueError('Same-camera motion input unavailable')
        return result+self.motion_port(self.motion_active)

    def decide(self,*,motion_observation,**kwargs):
        if self.closed or self.episode is None or self.poisoned:raise RuntimeError('Explicit episode restart required')
        started=self.monotonic()
        try:
            validate_motion_binding(motion_observation,episode_id=self.episode,scene_features=kwargs['scene_features'],body_context=kwargs['body_context'])
            self.motion_active=torch.tensor(motion_observation.features.copy()[None],dtype=torch.float32,device='cpu')
            result=super().decide(**kwargs)
            elapsed=self.monotonic()-started
            age=kwargs['decision_time']-kwargs['body_context'].decision_body.original_received_monotonic_seconds
            self.last_computation=dict(input_body_age_seconds=age,inference_seconds=elapsed,output_body_age_seconds=age+elapsed,includes_motion_binding=True)
            if elapsed<0 or age+elapsed>.05:raise ValueError('Motion binding/inference output exceeds original body50ms age')
            result.update(inference_seconds=elapsed,estimated_output_body_age_seconds=age+elapsed)
            return result
        except Exception as error:
            self.last_rejection=dict(error=type(error).__name__,reason=str(error),computation=getattr(self,'last_computation',None))
            self.poisoned=True
            raise
        finally:self.motion_active=None
