import pytest
import torch
from flybrain_drive.per_view_retinal_contrast_boundary import PerViewRetinalSpatialContrastBoundary
from flybrain_drive.retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary


def test_other_view_features_do_not_change_when_front_contrast_changes():
    per=PerViewRetinalSpatialContrastBoundary(2,max_contrast_rms=.1)
    valid=torch.ones(1,4,dtype=torch.bool)
    scene=torch.zeros(1,4,9,16);scene[:,1:,0,:]=torch.linspace(-.02,.02,16)
    changed=scene.clone();changed[:,0,0,:]=torch.linspace(-10,10,16)
    before=per.bounded_features(scene.reshape(1,576),valid);after=per.bounded_features(changed.reshape(1,576),valid)
    torch.testing.assert_close(before[:,1:],after[:,1:],atol=0,rtol=0)
    assert torch.sqrt((after[:,0]**2).mean())<=.100001


def test_unlimited_matches_original_and_rejects_missing_view():
    per=PerViewRetinalSpatialContrastBoundary(2);original=RetinalSpatialContrastBoundary(2)
    with torch.no_grad():per.projection.weight.normal_();original.load_state_dict(per.state_dict())
    x=torch.randn(1,576);valid=torch.ones(1,4,dtype=torch.bool)
    torch.testing.assert_close(per(x,valid),original(x,valid))
    valid[0,1]=False
    with pytest.raises(ValueError):per(x,valid)
