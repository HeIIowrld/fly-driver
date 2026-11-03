import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.synthetic_road_visual import SyntheticRoadVisual


def test_road_is_reproducible_four_view_and_bend_changes_front_pixels():
    config = json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text())
    road = SyntheticRoadVisual(config)
    left = road.render(bend=1, forward_m=1., texture_seed=17)
    right = road.render(bend=-1, forward_m=1., texture_seed=17)
    assert left.shape == (4, 180, 320, 3) and left.dtype == np.uint8
    assert np.array_equal(left, road.render(bend=1, forward_m=1., texture_seed=17))
    assert np.count_nonzero(left[0] != right[0]) > 100
    assert all(np.unique(view.reshape(-1, 3), axis=0).shape[0] > 16 for view in left)
    with pytest.raises(ValueError):
        road.render(bend=2, forward_m=1., texture_seed=17)
