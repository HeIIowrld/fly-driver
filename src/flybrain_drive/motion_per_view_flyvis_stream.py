"""Opt-in motion residual from the same authoritative author visual state."""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from .per_view_flyvis_stream import PerViewFlyvisStream,VisualFlightObservation
from .spatial_pool_residual import discarded_retinal_component
from .sensor_contract import readonly_array


@dataclass(frozen=True)
class MotionVisualFlightObservation(VisualFlightObservation):
    motion_features:np.ndarray=None
    coarse_motion_reference:np.ndarray=None
    motion_features_sha256:str=''

    def __post_init__(self):
        if np.asarray(self.motion_features).shape!=(2048,) or np.asarray(self.coarse_motion_reference).shape!=(512,):raise ValueError('Complete same-state motion2048/coarse512 required')
        features=readonly_array(self.motion_features,np.float32);coarse=readonly_array(self.coarse_motion_reference,np.float32)
        digest=hashlib.sha256(features.tobytes()).hexdigest()
        if self.motion_features_sha256 and self.motion_features_sha256!=digest:raise ValueError('Motion feature transport hash differs')
        object.__setattr__(self,'motion_features',features);object.__setattr__(self,'coarse_motion_reference',coarse);object.__setattr__(self,'motion_features_sha256',digest)


class MotionPerViewFlyvisStream(PerViewFlyvisStream):
    @torch.no_grad()
    def process(self,**kwargs):
        observation=super().process(**kwargs)
        try:
            # Read the already-committed state. No additional neural integration,
            # reset, invented frame or change to the original scene/static paths.
            motion=torch.tanh(self.state.nodes.activity[:,self.indices]).cpu().numpy().astype(float)
            residual=discarded_retinal_component(motion,self.geometry['old'],self.geometry['inverse'])
            coarse=(motion@self.geometry['old'].T).reshape(-1).astype(np.float32)
            expected=observation.scene.reshape(4,9,16)[:,1:].reshape(-1)
            if not np.allclose(coarse,expected,atol=2e-6,rtol=0):raise ValueError('Same-state motion reference differs from original scene')
            features=(residual@self.geometry['grid8_narrow'].T).reshape(-1).astype(np.float32)
            return MotionVisualFlightObservation(**vars(observation),motion_features=features,coarse_motion_reference=coarse)
        except Exception:
            self.poisoned=True
            raise
