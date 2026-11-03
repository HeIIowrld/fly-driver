import json
from pathlib import Path
import numpy as np
import pytest
from flybrain_drive.synthetic_planar_road_visual import SyntheticPlanarRoadVisual
from flybrain_drive.synthetic_antialiased_planar_road_visual import AntialiasedSyntheticPlanarRoadVisual


def test_single_sample_preserves_original_and_supersampling_is_bounded():
    camera=json.loads(Path('configs/ac-csp-exterior-front-v3-retina.json').read_text(encoding='utf8'))
    args=dict(bend=1,forward_m=13.,lateral_m=.2,yaw_rad=.02,texture_seed=882617)
    original=SyntheticPlanarRoadVisual(camera).render(**args)
    assert np.array_equal(original,AntialiasedSyntheticPlanarRoadVisual(camera,samples_per_axis=1).render(**args))
    result=AntialiasedSyntheticPlanarRoadVisual(camera,samples_per_axis=2).render(**args)
    assert result.shape==(4,180,320,3) and result.dtype==np.uint8
    assert not np.array_equal(original,result)
    with pytest.raises(ValueError):AntialiasedSyntheticPlanarRoadVisual(camera,samples_per_axis=3)
