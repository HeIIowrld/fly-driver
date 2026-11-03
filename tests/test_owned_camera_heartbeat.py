import time

import pytest

from scripts.record_ac_engine_ai_probe import wait_owned_camera_heartbeat


def state(**changes):
    return dict(generation='g', paused=True, ai_controlled=False, camera_active=False,
                camera_request_id=None, **changes)


def test_waits_for_owned_camera_without_launching(monkeypatch):
    import flybrain_drive.ac_published_state as published
    ready = state()
    ready.update(camera_active=True, camera_request_id='owned')
    sequence = iter([state(), ready])
    monkeypatch.setattr(published, 'read_published_json', lambda _: (next(sequence), time.time()))
    assert wait_owned_camera_heartbeat('unused', 'g', 'owned')['request_id'] == 'owned'


def test_rejects_other_camera_and_other_generation(monkeypatch):
    import flybrain_drive.ac_published_state as published
    other = state()
    other.update(camera_active=True, camera_request_id='other')
    monkeypatch.setattr(published, 'read_published_json', lambda _: (other, time.time()))
    with pytest.raises(ValueError, match='Another camera'):
        wait_owned_camera_heartbeat('unused', 'g', 'owned')
    other['generation'] = 'other'
    with pytest.raises(ValueError, match='same-generation'):
        wait_owned_camera_heartbeat('unused', 'g', 'owned')
