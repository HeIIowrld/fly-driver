import numpy as np
import pytest
import torch

from scripts.prepare_flyvis_preview_folds import (
    TRAIN_IDS,
    fit_feature_moments,
    fold_statistics,
    registered_cpu,
    selected_histories,
)


def histories():
    rows = []
    for i, name in enumerate(TRAIN_IDS):
        rows.append(dict(id=name, role='train', activity=np.full((2, 4, 14, 721), i+.1,
            np.float32), body=np.full((2, 21), i+.1), body_valid=np.ones((2, 21), bool),
            y=np.full((2, 6), i+.2), valid=np.ones((2, 6), bool)))
    return rows


def test_all_fitted_moments_ignore_extreme_heldout_values_and_invalid_targets():
    rows = histories()
    fit = TRAIN_IDS[1:]
    before = fold_statistics(rows, fit, 1e-5, [.01]*6)
    rows[0]['activity'][:] = 1e6
    rows[0]['body'][:] = -1e5
    rows[0]['y'][:] = 1e9
    rows[0]['valid'][:] = False
    after = fold_statistics(rows, fit, 1e-5, [.01]*6)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    features = {key: {'intact': np.full((2, 1024), i+.1)}
                for i, key in enumerate(TRAIN_IDS)}
    mean, scale = fit_feature_moments(features, fit)
    features[TRAIN_IDS[0]]['intact'][:] = -1e9
    changed_mean, changed_scale = fit_feature_moments(features, fit)
    np.testing.assert_array_equal(mean, changed_mean)
    np.testing.assert_array_equal(scale, changed_scale)
    np.testing.assert_array_equal(mean[896:], 0)
    np.testing.assert_array_equal(scale[896:], 1)


def test_fold_rejects_duplicate_full_or_nontrain_fit_histories():
    rows = histories()
    for ids in ([TRAIN_IDS[0]]*2, TRAIN_IDS, [TRAIN_IDS[0], 'development']):
        with pytest.raises(ValueError):
            selected_histories(rows, ids)
    rows[0]['role'] = 'development'
    with pytest.raises(ValueError):
        selected_histories(rows, TRAIN_IDS[:2])


def test_cpu_projection_view_removal_preserves_geometry_support_bits():
    torch.set_num_threads(1)
    rays = np.zeros((4, 721, 3))
    rays[..., 2] = -1
    origins = np.zeros((4, 3))
    origins[:, 2] = 1
    e = dict(clock=np.array([0.]), activity=np.ones((1, 4, 14, 721), np.float32),
             body=np.zeros((1, 21)), body_valid=np.ones((1, 21), bool),
             rays=rays, origins=origins)
    result = registered_cpu(e, np.zeros(14), np.ones(14), batch_size=1)
    assert result['intact'].shape == (1, 1024)
    assert result['intact'][:, 896:].any()
    for view in ('front', 'left', 'right', 'rear'):
        np.testing.assert_array_equal(result['intact'][:, 896:], result[view+'_mean'][:, 896:])
        assert not np.array_equal(result['intact'][:, :896], result[view+'_mean'][:, :896])
