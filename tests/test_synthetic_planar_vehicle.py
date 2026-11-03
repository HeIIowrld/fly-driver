import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.synthetic_planar_road_visual import SyntheticPlanarRoadVisual
from flybrain_drive.synthetic_planar_vehicle import PlanarVehicleState, advance_planar_vehicle
from flybrain_drive.synthetic_yaw_road_visual import SyntheticYawRoadVisual


def test_opposite_steering_and_braking_stop():
    state = PlanarVehicleState(speed_m_s=2.)
    left = advance_planar_vehicle(state, steer_rad=.2, acceleration_m_s2=0., dt_s=.1)
    right = advance_planar_vehicle(state, steer_rad=-.2, acceleration_m_s2=0., dt_s=.1)
    assert left.x_m == pytest.approx(right.x_m)
    assert left.y_left_m == pytest.approx(-right.y_left_m)
    assert left.yaw_left_rad == pytest.approx(-right.yaw_left_rad)
    stopped = advance_planar_vehicle(PlanarVehicleState(speed_m_s=.5), steer_rad=0., acceleration_m_s2=-10., dt_s=.2)
    assert stopped.speed_m_s == 0. and stopped.x_m == pytest.approx(.0125)


def test_scene_matches_existing_renderer_then_responds_to_lateral_motion():
    camera = json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    old, new = SyntheticYawRoadVisual(camera), SyntheticPlanarRoadVisual(camera)
    args = dict(bend=1, forward_m=3., yaw_rad=.1, texture_seed=42)
    reference = old.render(**args)
    assert np.array_equal(reference, new.render(**args, lateral_m=0.))
    shifted = new.render(**args, lateral_m=1.)
    assert shifted.shape == (4, 180, 320, 3) and not np.array_equal(reference, shifted)


def test_invalid_synthetic_vehicle_controls_rejected():
    with pytest.raises(ValueError):
        advance_planar_vehicle(PlanarVehicleState(), steer_rad=1., acceleration_m_s2=0., dt_s=.1)
