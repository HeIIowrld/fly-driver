import pytest
import torch
from flybrain_drive.flyvis_parameter_integrity import assert_nonnegative_synapse_strength, project_nonnegative_bounded


def test_target_clamps_to_nonnegative_and_bound_without_touching_others():
    initial = torch.tensor([0., .01, .5, .1])
    candidate = torch.tensor([-.03, .2, -.4, 9.])
    mask = torch.tensor([True, True, True, False])
    result = project_nonnegative_bounded(candidate, initial, mask, .05)
    assert torch.allclose(result, torch.tensor([0., .06, .45, .1]))
    assert torch.equal(result[~mask], initial[~mask])


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf')])
def test_invalid_strength_is_rejected(bad):
    with pytest.raises(ValueError, match='nonnegative'):
        assert_nonnegative_synapse_strength(torch.tensor([0., bad]))


def test_invalid_mask_and_limit_are_rejected():
    p=torch.tensor([0., .2]);m=torch.tensor([True, False])
    with pytest.raises(ValueError):
        project_nonnegative_bounded(p,p,m,0.)
    with pytest.raises(ValueError):
        project_nonnegative_bounded(p,p,torch.ones(2),.1)
