import json
from pathlib import Path
import numpy as np
import pytest
from flybrain_drive.synthetic_roll_road_visual import SyntheticRollRoadVisual
from flybrain_drive.synthetic_yaw_road_visual import SyntheticYawRoadVisual


def test_zero_roll_equals_existing_zero_yaw_and_roll_changes_each_view():
    camera=json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    roll=SyntheticRollRoadVisual(camera);old=SyntheticYawRoadVisual(camera)
    args=dict(bend=0,forward_m=1.,texture_seed=863101)
    zero=roll.render(**args,roll_rad=0.);positive=roll.render(**args,roll_rad=.05);negative=roll.render(**args,roll_rad=-.05)
    assert np.array_equal(zero,old.render(**args,yaw_rad=0.))
    assert all(np.count_nonzero(positive[i]!=negative[i])>0 for i in range(4))
    assert np.array_equal(zero,roll.render(**args,roll_rad=0.))
    with pytest.raises(ValueError):roll.render(**args,roll_rad=float('nan'))


def test_roll_basis_is_right_handed_and_matches_declared_left_up_sign():
    angle=.05;c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    assert np.allclose(rotation@rotation.T,np.eye(3))
    assert np.linalg.det(rotation)==pytest.approx(1.)
    assert (rotation@np.array([0.,1.,0.]))[2]>0
