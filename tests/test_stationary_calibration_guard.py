import pytest

from scripts.calibrate_ac_sdk_steering import validate_stationary_calibration_state


def safe():
    return (dict(generation='new', paused=False, in_menu=False, is_online=False,
                 ai_controlled=False, camera_active=False, expert_probe_active=False),
            dict(builtin_ai_controlled=False), dict(speedKmh=0., gas=0., brake=1.))


def test_fresh_stationary_live_ai_off_only():
    status, flag, fields = safe()
    validate_stationary_calibration_state(status, .1, flag, fields, 'new')
    for field in ('paused', 'in_menu', 'is_online', 'ai_controlled', 'camera_active',
                  'expert_probe_active'):
        with pytest.raises(ValueError):
            validate_stationary_calibration_state(dict(status, **{field: True}), .1,
                                                  flag, fields, 'new')


@pytest.mark.parametrize('fault', ['stale', 'new_engine', 'native_ai', 'moving', 'gas', 'brake'])
def test_calibration_rejects_ambiguous_or_unsafe_transition(fault):
    status, flag, fields = safe()
    age = 3 if fault == 'stale' else .1
    if fault == 'new_engine':
        status['generation'] = 'different'
    elif fault == 'native_ai':
        flag['builtin_ai_controlled'] = True
    elif fault == 'moving':
        fields['speedKmh'] = .1
    elif fault == 'gas':
        fields['gas'] = .02
    elif fault == 'brake':
        fields['brake'] = .98
    with pytest.raises(ValueError):
        validate_stationary_calibration_state(status, age, flag, fields, 'new')
