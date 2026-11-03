import pytest
import torch

from scripts.probe_flyvis_visual import movie_with_channel_axis, valid_stimulus_prefix


def test_author_flash_axis_adaptation_keeps_time_and_receptor_order():
    original = torch.arange(15).reshape(3, 5)
    adapted = movie_with_channel_axis(original, 5)
    assert adapted.shape == (3, 1, 5)
    assert torch.equal(adapted[:, 0], original)
    assert adapted.data_ptr() == original.data_ptr()
    assert movie_with_channel_axis(adapted, 5) is adapted


@pytest.mark.parametrize('shape', [(3, 2, 5), (3, 4), (1, 3, 1, 5)])
def test_input_geometry_mismatch_is_not_flattened_or_silently_resampled(shape):
    with pytest.raises(ValueError):
        movie_with_channel_axis(torch.zeros(shape), 5)


def test_declared_author_padding_does_not_hide_missing_temporal_history():
    padded = torch.zeros(5, 1, 3)
    padded[3:] = torch.nan
    assert valid_stimulus_prefix(padded, allow_author_padding=True) == 3
    with pytest.raises(ValueError):
        valid_stimulus_prefix(padded)
    padded[1] = torch.nan
    with pytest.raises(ValueError):
        valid_stimulus_prefix(padded, allow_author_padding=True)


def test_mixed_nonfinite_tail_cannot_be_called_author_storage_padding():
    padded = torch.zeros(5, 1, 3)
    padded[3:, 0, 1] = torch.nan
    with pytest.raises(ValueError):
        valid_stimulus_prefix(padded, allow_author_padding=True)
