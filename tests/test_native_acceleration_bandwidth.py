import numpy as np
import pytest
from scripts.audit_native_acceleration_bandwidth import integrated_slope_weights


def test_derivative_equivalent_kernel_matches_integrated_irregular_acceleration():
 t=np.array([-.075,-.061,-.04,-.008,.009,.038,.075]);a=np.array([0.,10.,-10.,0.,20.,-10.,0.]);v=np.r_[2.,2.+np.cumsum(.5*np.diff(t)*(a[:-1]+a[1:]))]
 d=t-t.mean();slope=(d*(v-v.mean())).sum()/(d@d);weights=integrated_slope_weights(t)
 np.testing.assert_allclose(weights@a,slope,atol=1e-12);np.testing.assert_allclose(weights@np.ones(len(t)),1.,atol=1e-12)
 assert (weights>=0).all()


def test_nonadvancing_or_nonfinite_reference_times_are_rejected():
 for times in [[0.,0.,1.],[0.,2.,1.],[0.,float('nan'),1.]]:
  with pytest.raises(ValueError):integrated_slope_weights(times)
