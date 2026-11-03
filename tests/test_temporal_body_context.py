from dataclasses import replace

import numpy as np
import pytest

from flybrain_drive.temporal_body_context import temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def observation(packet, received, evaluated, *, episode='one', contract='hash'):
    return NavigationBodyObservation(episode, packet, received, evaluated,
                                     np.zeros(21), np.ones(21, dtype=bool), contract)


def test_context_preserves_original_distinct_times_and_validity():
    prior = observation(3, 1., 1.03)
    current = observation(4, 1.035, 1.04)
    result = temporal_body_context(prior, current, render_start_lower=1.01,
                                   camera_receipt_time=1.03, decision_time=1.04)
    assert result.visual_context is prior and result.decision_body is current
    assert result.source_synchronization_verified is False


@pytest.mark.parametrize('change', [
    {'render_start_lower': 1.0}, {'camera_receipt_time': 1.05},
    {'decision_time': 1.03}, {'render_start_lower': float('nan')},
])
def test_reject_unproven_order_or_redated_evaluation(change):
    kwargs = dict(render_start_lower=1.01, camera_receipt_time=1.03, decision_time=1.04)
    kwargs.update(change)
    with pytest.raises(ValueError):
        temporal_body_context(observation(3, 1., 1.03), observation(4, 1.035, 1.04), **kwargs)


def test_reject_different_episode_or_sensor_contract():
    prior, current = observation(3, 1., 1.03), observation(4, 1.035, 1.04)
    for mismatched in (replace(current, episode_id='two'),
                       replace(current, contract_sha256='other')):
        with pytest.raises(ValueError):
            temporal_body_context(prior, mismatched, render_start_lower=1.01,
                                  camera_receipt_time=1.03, decision_time=1.04)
