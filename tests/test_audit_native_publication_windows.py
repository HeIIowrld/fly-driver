import pytest

from scripts.audit_native_publication_windows import conditional_window


def test_conditional_publication_bounds_keep_previous_receipt_uncertainty():
    bounds = conditional_window(10.0, 10.03, 10.04, 10.07)
    assert bounds['publication_window_width_ms'] == pytest.approx(30)
    assert bounds['decision_minus_receipt_ms'] == pytest.approx(40)
    assert bounds['conditional_decision_since_publication_upper_ms'] == pytest.approx(70)


@pytest.mark.parametrize('times', [(1, 1, 2, 3), (1, 2, 2, 3), (1, 2, 4, 3),
                                   (1, 2, 3, float('nan'))])
def test_invalid_or_ambiguous_receipt_order_is_rejected(times):
    with pytest.raises(ValueError, match='ordering'):
        conditional_window(*times)
