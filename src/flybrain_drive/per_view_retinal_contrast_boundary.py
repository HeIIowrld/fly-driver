"""Research-only per-view normalization of existing pooled retinal contrast.

This is an engineering intervention, not a measured fly retinal circuit.
"""
import torch
from .retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary


class PerViewRetinalSpatialContrastBoundary(RetinalSpatialContrastBoundary):
    """Keep each view's bounded gain local; preserve original projection topology."""
    def bounded_features(self, scene, view_valid):
        if (not isinstance(scene,torch.Tensor) or scene.ndim!=2 or scene.shape[1]!=576
                or not torch.isfinite(scene).all() or not isinstance(view_valid,torch.Tensor)
                or view_valid.shape!=(len(scene),4) or view_valid.dtype!=torch.bool
                or not view_valid.all()):
            raise ValueError('Four finite valid pooled views required')
        retina=scene.reshape(-1,4,9,16)[:,:,0,:]
        centered=retina-retina.mean(dim=-1,keepdim=True)-self.reference
        rms=(centered.square().mean(dim=-1,keepdim=True)+1e-24).sqrt()
        gain=torch.where(self.max_contrast_rms>0,(self.max_contrast_rms/rms).clamp(max=1.),torch.ones_like(rms))
        return centered*gain

    def forward(self,scene,view_valid):
        features=self.bounded_features(scene,view_valid)
        return .5*torch.tanh(self.projection(features.reshape(-1,64)))
