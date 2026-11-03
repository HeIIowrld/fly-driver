import torch
from torch import nn

from scripts.train_unified_geometry_yaw_synthetic import GeometryVisualPort


class Base(nn.Module):
    def forward(self,traces,body,valid):
        return torch.tensor([[.2,-.1]])


def test_geometry_port_starts_neutral_and_responds_after_learning():
    port=GeometryVisualPort(Base(),2)
    port.current=torch.tensor([[1.,0.,0.,0.,0.,0.]])
    initial=port(None,None,None)
    assert torch.equal(initial,torch.tensor([[.2,-.1]]))
    with torch.no_grad():
        port.projection.weight[0,0]=1.
    changed=port(None,None,None)
    assert changed[0,0] > initial[0,0] and changed[0,1] == initial[0,1]
