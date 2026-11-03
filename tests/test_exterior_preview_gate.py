import pytest

from scripts.audit_exterior_preview_learning import auxiliary_gate


def rows(error):
    return [dict(count=100, rmse=error, train_mean_rmse=2., zero_rmse=3.) for _ in range(6)]


def test_auxiliary_gate_requires_both_generalization_and_visual_causal_contribution():
    assert auxiliary_gate(rows(1.), rows(1.5), [1.]*6)['auxiliary_gate_passed']
    assert not auxiliary_gate(rows(1.), rows(.9), [1.]*6)['auxiliary_gate_passed']
    intact = rows(1.)
    intact[4]['zero_rmse'] = .5
    assert not auxiliary_gate(intact, rows(1.5), [1.]*6)['auxiliary_gate_passed']


def test_empty_invalid_or_unmatched_development_cannot_pass():
    empty = rows(0.)
    empty[0]['count'] = 0
    with pytest.raises(ValueError):
        auxiliary_gate(empty, rows(1.), [1.]*6)
    with pytest.raises(ValueError):
        auxiliary_gate(rows(1.), rows(1.5), [float('nan')]*6)
