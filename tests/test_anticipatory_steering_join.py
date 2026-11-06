import numpy as np
import pytest
from scripts.probe_protected_anticipatory_steering import future_label_join


def test_future_targets_are_label_only_floor_join_without_boundary_clamp():
    t=np.array([0.,.03,.10,.21,.24,.30,.41])
    valid=np.array([True,False,True,True,False,True,True])
    j,ok=future_label_join(t,valid,.2,.06)
    np.testing.assert_array_equal(j,[-1,3,5,-1,-1,-1,-1])
    assert np.all(t[j[ok]]>t[ok])
    assert np.all(t[j[ok]]<=t[ok]+.2)
    assert np.all(valid[j[ok]])


def test_no_late_target_or_unqualified_label_is_used():
    j,ok=future_label_join([0.,.1,.2,.3],np.array([True,False,False,True]),.15,.04)
    assert not ok.any()
    assert np.all(j==-1)


@pytest.mark.parametrize('t,v',[
    ([0.,0.],np.array([True,True])),
    ([0.,float('nan')],np.array([True,True])),
    ([0.,1.],np.array([1,1])),
    ([0.,1.],np.array([False,False])),
])
def test_bad_or_missing_teacher_evidence_is_explicit(t,v):
    with pytest.raises(ValueError):future_label_join(t,v)


def test_exact_representable_episode_end_is_allowed():
    j,ok=future_label_join([0.,.125,.25,.375,.5],np.ones(5,dtype=bool),.25,.01)
    np.testing.assert_array_equal(j,[2,3,4,-1,-1])


def test_body_diagnostic_uses_values_and_explicit_masks_without_labels():
    from scripts.diagnose_body_only_steering_forecast import body_features
    d=dict(body=np.array([[2.,999.]]),body_valid=np.array([[True,False]]),
        view_body=np.array([[[3.,888.]]]),view_body_valid=np.array([[[True,False]]]))
    np.testing.assert_array_equal(body_features(d,False),[[2.,0.,1.,0.]])
    np.testing.assert_array_equal(body_features(d,True),[[2.,0.,1.,0.,3.,0.,1.,0.]])


def test_body_forecast_transform_uses_training_calibration_only():
    from scripts.diagnose_body_only_steering_forecast import fit_predict
    x=np.arange(30,dtype=float).reshape(10,3);y=np.linspace(-.2,.2,10)
    first,state=fit_predict(x,y,[x])
    repeated,again=fit_predict(x,y,[x,np.full_like(x,1e6)])
    np.testing.assert_array_equal(first[0],repeated[0])
    for key in state:np.testing.assert_array_equal(state[key],again[key])
