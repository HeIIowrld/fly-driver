"""Counterfactual label changes may never alter measured actor data or add route truth."""

import json
import sys

import numpy as np
import pytest

from flybrain_drive.ac_native_relabel import relabel_episode
from flybrain_drive.multiview_sensory import MultiViewPacker


def fixture():
    return dict(**{'obs_'+key: np.arange(6, dtype=np.float32).reshape(2, 3)
                   for key in MultiViewPacker.fields},
                action=np.zeros((2, 3), np.float32),
                action_valid=np.ones((2, 3), bool), observation_time=np.array([1., 2.]))


def test_labels_change_without_modifying_or_aliasing_original_actor_arrays():
    original = fixture()
    changed = relabel_episode(original, [[-.1, .2, .5], [.1, .25, 0.]])
    assert not original['action'].any()
    for key in original:
        if key != 'action':
            np.testing.assert_array_equal(changed[key], original[key])
            assert not np.shares_memory(changed[key], original[key])
    assert changed['action'][0, 2] == .5


def test_privileged_array_or_invalid_action_is_rejected():
    with pytest.raises(ValueError, match='pose-free'):
        relabel_episode(dict(fixture(), world_position=np.zeros((2, 3))), np.zeros((2, 3)))
    for labels in [np.zeros((1, 3)), [[0, -1, 0], [0, 0, 0]],
                   [[0, 0, float('nan')], [0, 0, 0]]]:
        with pytest.raises(ValueError, match='bounded'):
            relabel_episode(fixture(), labels)


def test_merge_rejects_different_teacher_tasks_before_writing(tmp_path, monkeypatch):
    from scripts.merge_multiview_teacher import main

    roots = [tmp_path / 'a', tmp_path / 'b']
    for root, speed in zip(roots, [20., 70.], strict=True):
        root.mkdir()
        (root / 'dataset.json').write_text(json.dumps(dict(
            teacher_label_contract=dict(maximum_speed_kmh=speed))), encoding='utf8')
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['merge', '--input', str(roots[0]),
                                     '--input', str(roots[1]), '--output', str(output)])
    with pytest.raises(ValueError, match='task contracts differ'):
        main()
    assert not output.exists()
