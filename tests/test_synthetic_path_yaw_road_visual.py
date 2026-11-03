import json
from pathlib import Path
import numpy as np
import pytest
from flybrain_drive.synthetic_path_yaw_road_visual import SyntheticPathYawRoadVisual
from flybrain_drive.synthetic_yaw_road_visual import SyntheticYawRoadVisual


def test_zero_lateral_preserves_original_and_shifts_four_views():
    camera=json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    model=SyntheticPathYawRoadVisual(camera);original=SyntheticYawRoadVisual(camera)
    args=dict(bend=0,forward_m=2.,texture_seed=882001,yaw_rad=.05)
    assert np.array_equal(model.render(**args,lateral_m=0.),original.render(**args))
    left=model.render(**args,lateral_m=.1);right=model.render(**args,lateral_m=-.1)
    assert all(np.any(left[i]!=right[i]) for i in range(4))
    with pytest.raises(ValueError):model.render(**args,lateral_m=float('nan'))


def test_circular_kinematics_have_heading_aligned_velocity():
    speed=3.;omega=.05;t=1.2;h=1e-5
    def xy(t):return np.array([speed*np.sin(omega*t)/omega,speed*(1-np.cos(omega*t))/omega])
    velocity=(xy(t+h)-xy(t-h))/(2*h)
    assert np.allclose(velocity,speed*np.array([np.cos(omega*t),np.sin(omega*t)]),atol=1e-8)
