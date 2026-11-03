import numpy as np
import pytest

from flybrain_drive.retinotopic_readout import column_indices


def test_cell_order_and_original_column_order_are_both_preserved():
    indices = column_indices(['L1', 'Mi1', 'L1', 'Mi1'], [1, 0, 0, 1], [0]*4,
                             [0, 1], [0, 0], ['Mi1', 'L1'])
    np.testing.assert_array_equal(indices, [[1, 3], [2, 0]])


def test_missing_or_duplicate_columns_cannot_be_filled_or_averaged():
    with pytest.raises(ValueError, match='missing'):
        column_indices(['L1'], [0], [0], [0, 1], [0, 0], ['L1'])
    with pytest.raises(ValueError, match='duplicate'):
        column_indices(['L1', 'L1'], [0, 0], [0, 0], [0], [0], ['L1'])
