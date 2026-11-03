import pytest
import torch
from torch import nn

from scripts.replay_unified_flight_candidate import FeedbackValuesIntervention, reverse_view_motion


class Capture(nn.Module):
    def forward(self, body, valid):
        self.seen_body = body.clone()
        self.seen_valid = valid.clone()
        return body.sum(1,keepdim=True)


@pytest.mark.parametrize('condition,indices', FeedbackValuesIntervention.GROUPS.items())
def test_value_intervention_only_changes_declared_values(condition, indices):
    body = torch.arange(1,22,dtype=torch.float32)[None,:]
    valid = torch.ones_like(body,dtype=torch.bool)
    valid[0,4] = False
    original = body.clone()
    adapter = Capture()
    wrapper = FeedbackValuesIntervention(adapter)
    wrapper.condition = condition
    wrapper(body,valid)
    expected = body.clone()
    expected[:,indices] = 0
    assert torch.equal(adapter.seen_body,expected)
    assert torch.equal(adapter.seen_valid,valid)
    assert torch.equal(body,original)


@pytest.mark.parametrize('view', range(4))
def test_motion_reverse_preserves_retina_and_other_views(view):
    source = torch.arange(576,dtype=torch.float32).reshape(1,4,9,16)
    changed = reverse_view_motion(source.reshape(1,576),view).reshape(1,4,9,16)
    assert torch.equal(changed[:,view,0],source[:,view,0])
    assert torch.equal(changed[:,view,1:9],source[:,view,[2,1,4,3,6,5,8,7]])
    assert all(torch.equal(changed[:,v],source[:,v]) for v in range(4) if v != view)
    assert torch.equal(reverse_view_motion(changed.reshape(1,576),view),source.reshape(1,576))
