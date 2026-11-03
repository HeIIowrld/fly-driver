import pytest

from flybrain_drive.owned_camera_postflight import OwnedCameraPostflightTracker, classify_owned_camera_postflight


def sample():
    return dict(state=dict(generation='g', paused=True, ai_controlled=False, is_online=False,
                           in_menu=False, window_handle='w', car_id='bmw_z4_gt3', track_id='monza',
                           camera_active=True, camera_request_id='owned'),
                age=.1, generation='g', request_id='owned', window_handle='w', device_status=1,
                native_speed=.001, native_gas=0., native_brake=1.)


def test_wait_only_for_owned_request_then_accept_idle():
    args = sample()
    assert classify_owned_camera_postflight(**args) == 'wait'
    args['state']['camera_active'] = False
    assert classify_owned_camera_postflight(**args) == 'idle'


@pytest.mark.parametrize('key,value', [('generation', 'other'), ('paused', False),
                                       ('ai_controlled', True), ('is_online', True),
                                       ('in_menu', None), ('window_handle', 'other'),
                                       ('camera_active', None), ('camera_request_id', 'foreign')])
def test_reject_ambiguous_or_foreign_state(key, value):
    args = sample()
    args['state'][key] = value
    with pytest.raises(ValueError):
        classify_owned_camera_postflight(**args)


@pytest.mark.parametrize('key,value', [('age', 2.1), ('age', float('nan')),
                                       ('native_speed', .1), ('native_gas', .02),
                                       ('native_brake', .98), ('device_status', True)])
def test_reject_stale_or_unsafe_physical_state(key, value):
    args = sample()
    args[key] = value
    with pytest.raises(ValueError):
        classify_owned_camera_postflight(**args)


def test_idle_requires_a_newer_heartbeat_after_owned_active():
    args = sample()
    args['state']['script_clock'] = 10.
    tracker = OwnedCameraPostflightTracker()
    args['state']['camera_active'] = False
    with pytest.raises(ValueError, match='Newer idle'):
        tracker.observe(**args)
    args['state']['camera_active'] = True
    assert tracker.observe(**args) == 'wait'
    args['state']['camera_active'] = False
    with pytest.raises(ValueError, match='Newer idle'):
        tracker.observe(**args)
    args['state']['script_clock'] = 10.1
    assert tracker.observe(**args) == 'idle'
