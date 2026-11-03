import pytest

from scripts.train_flyvis_temporal_folds import temporal_window_layout


@pytest.mark.parametrize('detach', [2, 8])
def test_equal_optimizer_windows_preserve_all738_original_intervals_and_tail(detach):
    windows = list(temporal_window_layout(738, detach))
    assert len(windows) == 93
    assert [(start, end) for start, end, _ in windows] == (
        [(i, i+8) for i in range(0, 736, 8)] + [(736, 738)])
    observed = [index for _, _, chunks in windows
                for start, end in chunks for index in range(start, end)]
    assert observed == list(range(738))
    assert windows[-1][2] == ((736, 738),)
    assert sum(len(chunks) for _, _, chunks in windows) == (369 if detach == 2 else 93)


@pytest.mark.parametrize('intervals,detach', [(0, 2), (True, 2), (738, 3), (738, True)])
def test_invalid_temporal_layout_rejected(intervals, detach):
    with pytest.raises(ValueError):
        list(temporal_window_layout(intervals, detach))
