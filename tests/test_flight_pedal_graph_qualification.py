"""CPU diagnostic policy helpers only; never load the original graph/checkpoint."""

import hashlib
import json

import numpy as np
import pytest
import torch

from scripts.qualify_flight_pedal_graph_v2 import load_plan, rank_report, tensor_digest, write


def test_rank_uses_original_float32_precision_and_retains_zero_or_nonfinite():
    assert rank_report([[0., 0.], [0., 0.]])['rank'] == 0
    weak = rank_report([[1., 1.], [1., 1.+1e-8]])
    assert weak['rank'] == 1
    assert weak['threshold'] > weak['singular_values'][1]
    assert rank_report([[1., 0.], [0., 1.]])['rank'] == 2
    assert rank_report([[float('nan'), 0.]])['finite'] is False
    with pytest.raises(ValueError):
        rank_report([[1., 2., 3.]])


def test_fingerprint_matches_original_quiet_hasher_and_failure_metadata_keeps_nonfinite(tmp_path):
    array = np.array([[.1, .2]], dtype=np.float32)
    expected = hashlib.sha256(array.dtype.str.encode()+str(array.shape).encode()
                              +array.tobytes()).hexdigest()
    assert tensor_digest(torch.from_numpy(array)) == expected
    path = tmp_path/'failure.json'
    write(path, {'all_cases': [float('nan'), float('inf'), 0.], 'qualified': False})
    assert json.loads(path.read_bytes()) == {
        'all_cases': ['nan', 'inf', 0.], 'qualified': False}


def test_registration_check_rejects_condition_selection_before_model_loading(tmp_path):
    path = tmp_path/'plan.json'
    original = json.loads(open('configs/flight-pedal-original-graph-cpu-20261030.json',
                              encoding='utf8').read())
    original['conditions_goal_body_kmh'] = original['conditions_goal_body_kmh'][1:]
    path.write_text(json.dumps(original), encoding='utf8')
    with pytest.raises(ValueError, match='Exact preregistered'):
        load_plan(path)
