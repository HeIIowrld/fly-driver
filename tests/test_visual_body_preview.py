import json

import numpy as np
import pytest

from flybrain_drive.graph import sha256
from scripts.train_visual_body_preview import body_features, body_moments, direct_body_reference


def test_body_moments_exclude_development_and_invalid_values():
    train = np.ones((3, 21))
    train[:, 0] = [2, 4, 1e10]
    valid = np.ones((3, 21), bool)
    valid[2, 0] = False
    valid[:, 1] = False
    episodes = [dict(role='train', body=train, body_valid=valid),
                dict(role='development', body=np.full((1, 21), -1e12),
                     body_valid=np.ones((1, 21), bool))]
    mean, scale, count = body_moments(episodes)
    assert (mean[0], scale[0], count[0]) == (3, 1, 2)
    assert (mean[1], scale[1], count[1]) == (0, 1, 0)
    assert train[2, 0] == 1e10


def test_route_limits_head_inputs_without_destroying_sensor_data_or_validity():
    values = np.arange(63, dtype=float).reshape(3, 21)
    valid = np.ones_like(values, bool)
    valid[:, 4] = False
    before, masks = values.copy(), valid.copy()
    full = body_features(values, valid, np.zeros(21), np.ones(21))
    expected = np.concatenate([np.where(valid, values.clip(-5, 5), 0), valid], axis=1)
    np.testing.assert_array_equal(full, expected.astype(np.float32))
    routed = body_features(values, valid, np.zeros(21), np.ones(21), [0, 1, 4, 16])
    assert not routed[:, 12:16].any() and not routed[:, 33:37].any()
    np.testing.assert_array_equal(routed[:, [0, 1, 16]], full[:, [0, 1, 16]])
    assert not routed[:, 4].any() and not routed[:, 25].any()
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(valid, masks)


@pytest.mark.parametrize('indices', [[], [0, 0], [21], [-1], [True]])
def test_invalid_body_routes_rejected(indices):
    with pytest.raises(ValueError):
        body_features(np.zeros((2, 21)), np.ones((2, 21), bool), np.zeros(21),
                      np.ones(21), indices)


def body_reference(tmp_path):
    path = tmp_path/'virtual-body.npz'
    values, valid = np.ones((3, 21)), np.ones((3, 21), bool)
    values[:, 2], valid[:, 2] = 123., False
    np.savez(path, values=values, valid=valid, decision_monotonic_seconds=np.array([1., 2., 3.]))
    proof = dict(status='completed', legacy16_whole_replay_exact=True,
                 contract_sha256='expected', virtual_body_sha256=sha256(path))
    (tmp_path/'result.json').write_text(json.dumps(proof), encoding='utf8')
    return dict(role='development', body_replay=str(tmp_path),
                body_result_sha256=sha256(tmp_path/'result.json'))


def test_direct_body_reference_preserves_development_clock_and_invalid_channels(tmp_path):
    entry = body_reference(tmp_path)
    values, valid, _ = direct_body_reference(entry, [1., 2., 3.], 'expected')
    assert (values[:, 2] == 123.).all() and not valid[:, 2].any()


@pytest.mark.parametrize('change', ['role', 'clock', 'contract', 'payload', 'proof'])
def test_body_reference_rejects_role_clock_contract_or_digest_changes(tmp_path, change):
    entry = body_reference(tmp_path)
    clock, contract = [1., 2., 3.], 'expected'
    if change == 'role':
        entry['role'] = 'evaluation'
    elif change == 'clock':
        clock = [1., 2., 4.]
    elif change == 'contract':
        contract = 'different'
    elif change == 'payload':
        with (tmp_path/'virtual-body.npz').open('ab') as f:
            f.write(b'changed')
    elif change == 'proof':
        (tmp_path/'result.json').write_text('{}', encoding='utf8')
    with pytest.raises((ValueError, AssertionError, KeyError)):
        direct_body_reference(entry, clock, contract)
