import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual
from flybrain_drive.synthetic_yaw_road_visual import SyntheticYawRoadVisual


def test_rotation_matches_independently_rotated_camera_geometry():
    config = json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    road = SyntheticYawRoadVisual(config)
    for yaw in (0., .18, -.18):
        rotated = copy.deepcopy(config)
        c, s = math.cos(yaw), math.sin(yaw)
        matrix = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        for view in rotated['views'].values():
            for name in ('forward_body', 'right_body', 'up_body', 'origin_body_m'):
                view[name] = (matrix @ np.asarray(view[name])).tolist()
        expected = SyntheticRoadVisual(rotated).render(bend=1, forward_m=2., texture_seed=17)
        actual = road.render(bend=1, forward_m=2., texture_seed=17, yaw_rad=yaw)
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(road.render(bend=1, forward_m=2., texture_seed=17, yaw_rad=.18),
                              road.render(bend=1, forward_m=2., texture_seed=17, yaw_rad=-.18))


def test_rejects_invalid_rotation():
    config = json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    road = SyntheticYawRoadVisual(config)
    for yaw in (True, float('nan'), float('inf'), math.pi + .1):
        with pytest.raises(ValueError):
            road.render(bend=0, forward_m=0., texture_seed=1, yaw_rad=yaw)
