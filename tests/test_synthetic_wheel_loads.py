import numpy as np
import pytest
from flybrain_drive.synthetic_wheel_loads import hypothetical_wheel_loads

def test_normalized_and_mirror_signs():
    q=lambda s,a:hypothetical_wheel_loads(speed_m_s=5.,steer_rad=s,acceleration_m_s2=a)
    a=q(.1,0.);b=q(-.1,0.);assert np.allclose(a,b[[1,0,3,2]]) and np.isclose(a.sum(),1)
    assert q(0.,-2.)[:2].sum()>.5 and q(0.,2.)[:2].sum()<.5
    assert np.allclose(q(0.,0.),.25)

def test_invalid_not_silently_repaired():
    with pytest.raises(ValueError):hypothetical_wheel_loads(speed_m_s=-1,steer_rad=0,acceleration_m_s2=0)
    with pytest.raises(ValueError):hypothetical_wheel_loads(speed_m_s=1,steer_rad=float('nan'),acceleration_m_s2=0)
