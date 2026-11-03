import numpy as np

from scripts.audit_native_student_replay import braking_metrics


def test_braking_subset_excludes_invalid_and_zero_labels():
    target = np.array([[0, 0, .5], [0, 0, .8], [0, 0, 0.]])
    prediction = np.array([[0, 0, 0.], [0, 0, .8], [0, 0, .9]])
    valid = np.ones_like(target, dtype=bool)
    valid[1, 2] = False
    result = braking_metrics(prediction, target, valid)
    assert result['supervised_braking_samples'] == 1
    assert result['brake_rmse'] == .5
    assert result['predicted_brake_above_threshold'] == 0
    valid[0, 2] = False
    result = braking_metrics(prediction, target, valid)
    assert result['supervised_braking_samples'] == 0
    assert result['brake_rmse'] is None
