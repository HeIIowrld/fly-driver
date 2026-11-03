import numpy as np

from scripts.audit_training_episode_fit import brake_fit_metrics


def test_brake_fit_excludes_missing_and_separates_false_requests():
    labels = np.zeros((5, 3))
    labels[:, 2] = [.5, .5, 0, .05, 1]
    prediction = labels.copy()
    prediction[:, 2] = [.4, 0, .2, .05, 1]
    valid = np.ones_like(labels)
    valid[4, 2] = 0
    result = brake_fit_metrics(prediction, labels, valid)
    assert result['positive']['samples'] == 2
    assert result['positive']['requests_above_threshold'] == 1
    assert result['negative']['samples'] == 2
    assert result['negative']['requests_above_threshold'] == 1
    assert np.isclose(result['positive']['rmse'], np.sqrt(.13))


def test_absent_brake_class_is_explicit():
    values = np.zeros((2, 3))
    result = brake_fit_metrics(values, values, np.ones_like(values))
    assert result['positive']['samples'] == 0
    assert result['positive']['rmse'] is None
    assert result['positive']['mean_request'] is None


def test_coast_and_drive_metrics_are_selected_by_valid_labels_not_policy_predictions():
    from scripts.audit_training_episode_fit import throttle_fit_metrics

    labels = np.zeros((4, 3))
    labels[:, 1] = [0, .2, 0, .03]
    prediction = labels.copy()
    prediction[:, 1] = [.1, 0, 1, .9]
    valid = np.ones_like(labels)
    valid[2, 1] = 0
    result = throttle_fit_metrics(prediction, labels, valid)
    assert result['coast']['samples'] == 1
    assert result['coast']['mean_request'] == .1
    assert result['positive_drive']['samples'] == 1
    assert result['positive_drive']['rmse'] == .2
