import torch
from test_learned_flight_pedals import parent
from flybrain_drive.joint_balance_muscle_readout import JointBalanceMusclePedalReadout

def test_zero_joint_registration_preserves_original_head_exactly():
    p=parent();h=JointBalanceMusclePedalReadout(p,torch.zeros(4,4),torch.tensor(1.));x=torch.rand(7,6)
    assert torch.equal(h(x),p(x))

def test_context_uses_balance_muscles_only_and_preserves_signed_power_for_any_context():
    h=JointBalanceMusclePedalReadout(parent(),torch.randn(4,4),torch.tensor(1.));x=torch.ones(8,6)*.4
    with torch.no_grad():h.offset.weight.uniform_(-2,2)
    old=h.features(x)
    for j in [4,5]:
        y=x.clone();y[:,j]+=.05
        assert torch.equal(h.features(y),old)
        assert (h(y)[:,1]>=h(x)[:,1]).all() and (h(y)[:,2]<=h(x)[:,2]).all()

def test_joint_registration_cannot_change_steering_or_suppress_simultaneous_pedals():
    h=JointBalanceMusclePedalReadout(parent(),torch.zeros(4,4),torch.tensor(1.));x=torch.ones(1,6)*.5;old=h(x)[:,0]
    with torch.no_grad():h.pedal_projection.bias[1:]=0;h.offset.weight.zero_()
    assert torch.equal(h(x)[:,0],old) and torch.equal(h(x)[:,1:],torch.tensor([[.5,.5]]))
