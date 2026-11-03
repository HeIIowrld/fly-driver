import numpy as np
import pytest
import torch

from flybrain_drive.flight_visual_body_temporal_context import (
    PerViewVisualBodyContextBoundary,
    VisualBodyContextBoundary,
)
from flybrain_drive.temporal_body_context import TemporalBodyContext
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def traces():
    return (torch.arange(512, dtype=torch.float32).reshape(1, 512)/512,
            torch.zeros(1, 512), torch.zeros(1, 512))


def test_same_yaw_per_view_reproduces_single_yaw_boundary():
    old, new = VisualBodyContextBoundary(3), PerViewVisualBodyContextBoundary(3)
    with torch.no_grad():
        old.projection.weight.fill_(.002)
        new.projection.weight.copy_(old.projection.weight)
    values = np.zeros(21, np.float32)
    values[4] = .25
    mask = np.ones(21, bool)
    obs = NavigationBodyObservation('run', 1, 1., 2., values, mask, 'contract')
    context = TemporalBodyContext(obs, obs, 2., 1.5, 2.)
    body = torch.from_numpy(np.tile(values, (1, 4, 1)))
    valid = torch.ones_like(body, dtype=torch.bool)
    torch.testing.assert_close(new(traces(), body, valid), old(traces(), context))


def test_missing_yaw_masks_only_its_view_and_zero_weights_are_neutral():
    boundary = PerViewVisualBodyContextBoundary(1)
    body = torch.zeros(1, 4, 21)
    body[:, :, 4] = .5
    valid = torch.ones_like(body, dtype=torch.bool)
    assert torch.equal(boundary(traces(), body, valid), torch.zeros(1, 1))
    with torch.no_grad():
        boundary.projection.weight.zero_()
        boundary.projection.weight[:, :16] = .01
    full = boundary(traces(), body, valid)
    missing = valid.clone()
    missing[:, 0, 4] = False
    body2 = body.clone()
    body2[:, 0, 4] = -999.
    assert not torch.equal(full, boundary(traces(), body2, missing))
    body2[:, 0, 4] = 999.
    torch.testing.assert_close(boundary(traces(), body2, missing),
                               boundary(traces(), body, missing))
    with pytest.raises(ValueError, match='masks'):
        boundary(traces(), body, valid.float())
