import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_perturbation import perturb_request, yaw_metrics
from flybrain_drive.flight_scratch_warmup import prime_flight_operators
from flybrain_drive.flight_yaw_trial import apply_yaw_mode


def test_yaw_intervention_preserves_raw_sensor_tensor_mask_and_all_other_channels():
    seen = []
    stream = SimpleNamespace(flight=SimpleNamespace(
        _drive_tensor=lambda *args, **kwargs: seen.append(args) or args[2]))
    apply_yaw_mode(stream, 'zero_yaw')
    raw = torch.arange(16., dtype=torch.float32)[None]
    valid = torch.ones_like(raw, dtype=torch.bool)
    response = stream.flight._drive_tensor(None, None, raw, valid, None, None)
    assert response[0, 4] == 0 and raw[0, 4] == 4 and seen[0][3] is valid
    np.testing.assert_array_equal(response[0, [0, 1, 5]], raw[0, [0, 1, 5]])
    with pytest.raises(ValueError):
        apply_yaw_mode(stream, 'full')


def test_disturbance_is_recorded_separately_and_cannot_bypass_axis_limit():
    base = dict(steering=.08, throttle=.03, brake=0.)
    response = perturb_request(base, phase='perturbation', sign=1)
    assert response['requested_control']['steering'] == .1 and response['clipped']
    assert response['disturbance'] == .05 and base['steering'] == .08


def test_unrecovered_or_missing_native_response_cannot_receive_invented_recovery_time():
    spec = json.loads(Path('configs/flight-physical-yaw-perturbation-v1-20261029.json').read_text(
        encoding='utf8'))['evaluation']
    t = np.arange(0, 3., .01)
    y = np.where(t >= 1., .05, 0.)
    result = yaw_metrics(t, y, pulse_on=1., pulse_off=1.28, spec=spec)
    assert result['observed_recovery_seconds_upper'] is None and not result['recovery_observed']
    with pytest.raises(ValueError):
        yaw_metrics(t[:200], y[:200], pulse_on=1., pulse_off=1.28, spec=spec)


def test_scratch_preparation_cannot_be_used_to_repair_an_already_started_episode():
    with pytest.raises(ValueError, match='before the first actual observation'):
        prime_flight_operators(SimpleNamespace(steps=1))
