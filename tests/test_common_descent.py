import numpy as np
import pytest
from flybrain_drive.common_descent import minimum_norm_common_gradient

def test_shared_descent_for_conflicting_gradients():
    g=np.array([[1.,0.],[-.4,1.],[0.,1.]])*1e-5;d,w,info=minimum_norm_common_gradient(g)
    assert info['status']=='common_descent' and np.all(g@d>0) and np.isclose(w.sum(),1)

def test_exact_opposites_report_stationary():
    d,w,info=minimum_norm_common_gradient([[1.,0.],[-1.,0.]])
    assert info['status']=='stationary' and np.linalg.norm(d)<1e-9

def test_rejects_nonfinite():
    with pytest.raises(ValueError):minimum_norm_common_gradient([[1.,float('nan')],[0.,1.]])
