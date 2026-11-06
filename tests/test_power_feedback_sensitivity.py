import pytest
import torch
from flybrain_drive.power_feedback_sensitivity import project_power_sensitivity,power_sensitivity
from flybrain_drive.symmetric_normalized_flight_speed import SymmetricNormalizedFlightSpeedCommand

def scale():return torch.tensor([.0007,.0004,.00001,.0002,.00013,.00004])
def project(w):project_power_sensitivity(w,scale(),max_delta_m_s=10.,maximum=2000.)
def test_weighted_projection_preserves_other_channels_and_step_bound():
 rng=torch.Generator().manual_seed(781535)
 for _ in range(100):
  old=torch.randn(1,7,generator=rng)*.04;project(old);delta=torch.randn(1,7,generator=rng);delta*=.001/delta.norm();w=old+delta;untouched=w[0,[0,1,3,4,6]].clone();project(w)
  assert float(power_sensitivity(w,scale(),10.))<=2000.001 and (w-old).norm()<=delta.norm()+1e-7
  assert torch.equal(w[0,[0,1,3,4,6]],untouched)

def test_failed_candidate_power_coefficients_are_actually_reduced():
 w=torch.tensor([[.0073,-.0229,-.005612,-.0061,.001,-.03429,-.0016]])
 assert power_sensitivity(w,scale(),10.)>9000
 project(w);assert power_sensitivity(w,scale(),10.)<=2000.001

def test_raw_power_mn_partials_obey_the_bound_after_normalization():
 policy=SymmetricNormalizedFlightSpeedCommand(torch.zeros(6),scale())
 with torch.no_grad():policy.projection.weight.copy_(torch.tensor([[.01,-.02,.03,-.01,.01,-.04,-.01]]));project(policy.projection.weight)
 for left,right in [(.0001,.0002),(.0002,.0001),(.00001,.00004)]:
  raw=torch.tensor([[.0002,.0003,.0001,.0002,left,right]],requires_grad=True);target=policy(raw,torch.tensor([[30.]]),torch.tensor([[0.,3.]]));target[0,1].backward();assert raw.grad[0,4:].abs().max()<=2000.001

@pytest.mark.parametrize('weight',[torch.zeros(1,6),torch.zeros(1,7,dtype=torch.float64),torch.full((1,7),float('nan'))])
def test_invalid_coefficients_are_rejected(weight):
 with pytest.raises(ValueError):project(weight)
