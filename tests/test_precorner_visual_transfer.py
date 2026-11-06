import numpy as np
import pytest
from scripts.evaluate_precorner_visual_transfer import floor_indices


def test_delayed_and_relative_donor_selection_never_uses_future_feature():
    times = np.array([0., .03, .071, .12, .2])
    wanted = np.array([.03, .119, .2])
    ids = floor_indices(times,wanted)
    np.testing.assert_array_equal(ids,[1,2,4])
    assert np.all(times[ids] <= wanted)


@pytest.mark.parametrize('wanted',[-.001,.201,float('nan')])
def test_missing_source_not_clamped_to_boundary(wanted):
    with pytest.raises(ValueError): floor_indices([0.,.1,.2],wanted)


@pytest.mark.parametrize('times',[[0.,.1,.1],[0.,.2,.1],[0.,float('nan'),.2]])
def test_ambiguous_donor_time_rejected(times):
    with pytest.raises(ValueError): floor_indices(times,.05)
