import numpy as np
import pytest

from scripts.diagnose_registered_anticipation import anticipation_groups


def test_all_rows_partition_including_threshold_ties_and_unknown_states():
    current = np.array([.001, .003, -.002, .002, np.nan, np.nan])
    target = np.array([-.003, -.001, .003, .002, .004, np.nan])
    usable = np.array([True, True, True, True, False, False])
    valid = np.array([True, True, True, True, True, False])
    groups = anticipation_groups(current, target, usable, valid)
    assert [int(groups[k].sum()) for k in ('entering', 'exiting', 'opposite', 'other_valid',
                                          'unavailable_body', 'invalid_target')] == [1]*6
    assert np.all(np.stack(list(groups.values())).sum(0) == 1)


def test_unknown_body_cannot_be_relabelled_as_steady_turn():
    groups = anticipation_groups([np.nan], [.004], np.array([False]), np.array([True]))
    assert groups['unavailable_body'][0] and not groups['other_valid'][0]
    with pytest.raises(ValueError):
        anticipation_groups([np.nan], [.004], np.array([True]), np.array([True]))
