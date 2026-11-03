import numpy as np
import pytest

from flybrain_drive.retinotopic_feature_scaling import ChannelMoments


def test_development_cannot_update_channel_calibration_and_constant_floor_is_explicit():
    moments = ChannelMoments()
    activity = np.ones((2, 4, 14, 721), np.float32)*3
    moments.add(activity, role='train')
    before = moments.finish(1e-5)
    with pytest.raises(ValueError, match='TRAIN'):
        moments.add(activity*1e10, role='development')
    after = moments.finish(1e-5)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after[0], np.full(14, 3.))
    np.testing.assert_array_equal(after[1], np.full(14, 1e-5))
