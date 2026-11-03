from types import SimpleNamespace

import pytest

from flybrain_drive.wind_stimulus_config import isolated_wind_copy
from scripts.restart_ac_engine_wind_candidate import (
    validate_new_engine,
    validate_restart_intent,
    validate_transaction,
)


def test_restart_receipt_must_match_original_request():
    request = dict(id='123', command='restart-engine-wind-diagnostic', old_engine_pid=12,
                   wind_direction_deg=90, wind_speed_kmh=36, original_race_sha256='a'*64)
    intent = dict(request, status='engine_restart_intent_committed', ui_actions_sent=0,
                  actuator_messages_sent=0)
    validate_restart_intent(intent, request)
    for key, value in [('id', 'other'), ('wind_speed_kmh', 0), ('old_engine_pid', 13),
                       ('status', 'accepted'), ('ui_actions_sent', 1)]:
        with pytest.raises(ValueError, match='receipt'):
            validate_restart_intent(dict(intent, **{key: value}), request)


def test_new_engine_requires_signaled_old_process_and_new_matching_sources():
    old = {'pid': 12, 'active': False, 'termination_signaled': True}
    before = {'generation': 'old'}
    window = SimpleNamespace(pid=13, hwnd=99)
    state = dict(generation='new', window_handle='99ULL', car_id='bmw_z4_gt3',
                 track_id='monza', is_online=False, camera_active=False, ai_controlled=False)
    validate_new_engine(old, before, 'ac-file-12-abc', window, state, 'ac-file-13-xyz')
    with pytest.raises(ValueError):
        validate_new_engine(dict(old, termination_signaled=False), before, 'ac-file-12-abc',
                            window, state, 'ac-file-13-xyz')
    with pytest.raises(ValueError):
        validate_new_engine(old, before, 'ac-file-12-abc', window,
                            dict(state, generation='old'), 'ac-file-13-xyz')
    with pytest.raises(ValueError):
        validate_new_engine(old, before, 'ac-file-12-abc', window, state, 'ac-file-12-abc')
    with pytest.raises(ValueError):
        validate_new_engine(old, before, 'ac-file-12-abc', window,
                            dict(state, window_handle='98ULL'), 'ac-file-13-xyz')


def test_stimulus_must_match_candidate_contract_and_saved_bytes(tmp_path):
    original = b'[WIND]\nDIRECTION_DEG=-1\nSPEED_KMH_MIN=0\nSPEED_KMH_MAX=0\n'
    (tmp_path/'original.bin').write_bytes(original)
    (tmp_path/'stimulus.bin').write_bytes(
        isolated_wind_copy(original, direction_deg=90, speed_kmh=36))
    validate_transaction({'direction_deg': 90, 'speed_kmh': 36}, tmp_path)
    with pytest.raises(ValueError, match='contract'):
        validate_transaction({'direction_deg': 90, 'speed_kmh': 0}, tmp_path)
    with pytest.raises(ValueError, match='differs'):
        validate_transaction({'direction_deg': 180, 'speed_kmh': 36}, tmp_path)
