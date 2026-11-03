import copy

import pytest

from scripts.observe_wind_lifecycle_candidate import validate_observation


def receipt():
    obs = dict(generation='g', actor_admissible=False, source_verified=False,
        wind_velocity_candidate_m_s=[1, 0, 0], weather_wind_velocity_candidate_kmh=[0, 0],
        vehicle_velocity_candidate_m_s=[0, 0, 0], body_forward_candidate=[1, 0, 0],
        body_left_candidate=[0, 1, 0], body_up_candidate=[0, 0, 1],
        weather_wind_speed_candidate_kmh=0, car_physics_timestamp_ms=1,
        started_source_clock=1, finished_source_clock=1.01)
    request = dict(id='test', command='wind-observe-diagnostic', generation='g')
    result = dict(id='test', command=request['command'], status='wind_observed_diagnostic',
        actor_admissible=False, ui_actions_sent=0, actuator_messages_sent=0, observation=obs)
    return result, request


def test_valid_wind_receipt_stays_unqualified():
    result, request = receipt()
    assert validate_observation(result, request)['source_verified'] is False


@pytest.mark.parametrize('change', ['source', 'actor', 'nan', 'wrong_generation', 'clock'])
def test_reject_wind_receipt_that_loses_provenance_or_finiteness(change):
    result, request = receipt()
    result = copy.deepcopy(result)
    obs = result['observation']
    if change == 'source':
        obs['source_verified'] = True
    elif change == 'actor':
        obs['actor_admissible'] = True
    elif change == 'nan':
        obs['wind_velocity_candidate_m_s'][0] = float('nan')
    elif change == 'wrong_generation':
        obs['generation'] = 'other'
    else:
        obs['finished_source_clock'] = .5
    with pytest.raises(ValueError):
        validate_observation(result, request)
