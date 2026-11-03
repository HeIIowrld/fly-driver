import numpy as np
import torch

from flybrain_drive.flight_visual_body_temporal_context import (PerViewVisualBodyContextBoundary,
                                                               VisualBodyContextBoundary)
from flybrain_drive.temporal_body_context import temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def test_transferred_projection_matches_common_yaw_and_respects_missing_mask():
    torch.manual_seed(11)
    common = VisualBodyContextBoundary(5)
    with torch.no_grad():
        common.projection.weight.normal_(0, .02)
    views = PerViewVisualBodyContextBoundary(5)
    views.load_state_dict(common.state_dict())
    values = np.zeros(21, dtype=np.float64)
    values[4] = .27
    valid = np.ones(21, dtype=bool)
    prior = NavigationBodyObservation('e', 3, 1., 2., values, valid, 'c')
    current = NavigationBodyObservation('e', 4, 2.1, 2.2, values, valid, 'c')
    context = temporal_body_context(prior,current,camera_receipt_time=2.,
                                    render_start_lower=1.5,decision_time=2.2)
    traces = tuple(torch.randn(1,512) for _ in range(3))
    body = torch.tensor(np.tile(values,(1,4,1)),dtype=torch.float32)
    masks = torch.ones(1,4,21,dtype=torch.bool)
    torch.testing.assert_close(common(traces,context),views(traces,body,masks),rtol=0,atol=1e-7)
    masks[:,:,4] = False
    assert torch.count_nonzero(views(traces,body,masks)) == 0
