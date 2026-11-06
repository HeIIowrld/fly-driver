import numpy as np
import pytest
from flybrain_drive.synthetic_yaw_body import planar_pose,yaw_profile,synthetic_turn_body

P=dict(onset=.44,duration=1.2)

def test_profile_rate_is_derivative_of_angle_across_duration():
    for t in [.2,.5,.7,1.2]:
        rate,angle=yaw_profile(t,.3,**P);h=1e-5;derivative=(yaw_profile(t+h,.3,**P)[1]-yaw_profile(t-h,.3,**P)[1])/(2*h);assert derivative==pytest.approx(float(rate),abs=1e-10)

def test_motion_velocity_and_lateral_acceleration_match_body():
    for t in [.2,.6,1.2]:
        h=1e-5;pose=planar_pose(t,6.,.3,**P);plus=planar_pose(t+h,6.,.3,**P);minus=planar_pose(t-h,6.,.3,**P);velocity=(plus[:2]-minus[:2])/(2*h);np.testing.assert_allclose(velocity,6*np.array([np.cos(pose[2]),np.sin(pose[2])]),atol=1e-8);body,mask,extension,known=synthetic_turn_body(t,6.,.3,80.,**P);assert body[4]==pytest.approx(pose[3]) and extension[1]==pytest.approx(6*pose[3])

def test_zero_rate_exact_straight_and_rate_mirrors():
    np.testing.assert_allclose(planar_pose(1.5,6.,0.,**P),[9.,0.,0.,0.],atol=1e-12);a=planar_pose(1.5,6.,.3,**P);b=planar_pose(1.5,6.,-.3,**P);np.testing.assert_allclose(a,[b[0],-b[1],-b[2],-b[3]],atol=1e-12)

def test_unknown_tyre_and_wheel_channels_are_not_filled_with_reference():
    body,mask,extra,valid=synthetic_turn_body(1.,6.,.3,40.,**P);assert not mask[8:12].any() and not mask[17:].any() and not valid[2:11].any() and not valid[15:].any();assert np.all(body[~mask]==0) and np.all(extra[~valid]==0);assert mask[12:17].all() and valid[11:15].all()

def test_prior_sensor_samples_differ_from_future_current_in_duration():
    lower=.5;prior=synthetic_turn_body(lower-.001,6.,.3,80.,**P)[0][4];at=yaw_profile(lower,.3,**P)[0];current=synthetic_turn_body(lower+.011,6.,.3,80.,**P)[0][4];assert prior<at<current

@pytest.mark.parametrize('target,onset,duration',[(float('nan'),.4,.4),(.6,.4,.4),(.2,-1,.4),(.2,.4,0)])
def test_invalid_profile_rejected(target,onset,duration):
    with pytest.raises(ValueError):yaw_profile(.5,target,onset=onset,duration=duration)
