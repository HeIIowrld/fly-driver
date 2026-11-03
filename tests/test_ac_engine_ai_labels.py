import pytest

from flybrain_drive.ac_engine_ai_labels import match_future_applied_controls, observation_segments


def row(stamp, packet, steer=.3):
    return dict(received_monotonic_seconds=stamp, packet_id=packet,
                fields=dict(steerAngle=steer, gas=.2, brake=0))


def test_targets_use_future_new_packets_not_current_body_or_redated_duplicates():
    events = [dict(decision_monotonic_seconds=1., native=dict(packet_id=1))]
    native = [row(1., 1), row(1.03, 2), row(1.05, 2), row(1.06, 3)]
    matched = match_future_applied_controls(events, native, native_gain=1, ai_window=(.9, 2))
    assert matched['label_packet_id'].tolist() == [3]
    assert matched['label_receipt_time'].tolist() == [1.06]
    assert matched['action_valid'].all()


def test_unrepresentable_executed_steer_is_preserved_and_masked_not_clipped():
    events = [dict(decision_monotonic_seconds=1., native=dict(packet_id=1))]
    matched = match_future_applied_controls(events, [row(1.05, 2, steer=.9)],
                                            native_gain=1, ai_window=(.9, 2))
    assert matched['raw_canonical'][0, 0] == 1.5
    assert not matched['action_valid'][0, 0]
    assert matched['action'][0, 0] == 0
    assert matched['action_valid'][0, 1:].all()


@pytest.mark.parametrize('stamp,window', [(1.08, (.9, 2)), (1.05, (1.01, 2)),
                                        (1.05, (.9, 1.04))])
def test_late_or_unverified_ai_phase_cannot_supervise(stamp, window):
    events = [dict(decision_monotonic_seconds=1., native=dict(packet_id=1))]
    matched = match_future_applied_controls(events, [row(stamp, 2)],
                                            native_gain=1, ai_window=window)
    assert not matched['action_valid'].any()


def test_native_sequence_reset_is_not_repaired():
    with pytest.raises(ValueError, match='same-session'):
        match_future_applied_controls([], [row(1., 2), row(1.1, 1)],
                                      native_gain=1, ai_window=(.9, 2))


def test_observation_gap_stays_an_explicit_boundary():
    assert observation_segments([1., 1.04, 2., 2.04]) == [(0, 2), (2, 4)]
    with pytest.raises(ValueError):
        observation_segments([1., 1.])


def test_speed_task_is_not_silently_mixed_or_assigned_old_hint():
    from flybrain_drive.ac_engine_ai_labels import common_expert_settings

    low = dict(speed_limit_kmh=15, ai_steer_multiplier=.9)
    fast = dict(speed_limit_kmh=30, ai_steer_multiplier=.9)
    assert common_expert_settings([fast, fast])['ai_speed_hint_kmh'] == 30
    assert common_expert_settings([low])['ai_speed_hint_kmh'] == 15
    with pytest.raises(ValueError):
        common_expert_settings([low, fast])


def test_lateral_recovery_labels_require_exact_preset_and_keep15kmh_task():
    from flybrain_drive.ac_engine_ai_labels import common_expert_settings

    request = dict(command='expert-lateral-recovery15', speed_limit_kmh=15,
                   ai_steer_multiplier=.9, hold_seconds=30, lateral_recovery=True)
    assert common_expert_settings([request]) == dict(ai_speed_hint_kmh=15,
                                                    ai_steer_multiplier=.9)
    for field, value in [('lateral_recovery', False), ('hold_seconds', 120),
                         ('speed_limit_kmh', 30)]:
        with pytest.raises(ValueError):
            common_expert_settings([dict(request, **{field: value})])


def test_speed20_warmup_cannot_be_mixed_into15_targets():
    from flybrain_drive.ac_engine_ai_labels import common_expert_settings, fixed_hint_window

    request = dict(command='expert-speed20-recovery15', speed_limit_kmh=20,
                   recovery_target_kmh=15, warmup_seconds=3, hold_seconds=30,
                   ai_steer_multiplier=.9)
    assert common_expert_settings([request])['ai_speed_hint_kmh'] == 15
    with pytest.raises(ValueError):
        common_expert_settings([dict(request, speed_limit_kmh=30)])
    telemetry = [dict(ai_controlled=True, paused=False, phase='driving',
                      script_clock=t, teacher_speed_hint_kmh=s)
                 for t, s in [(1, 20), (4, 20), (4.1, 15), (5, 15)]]
    assert fixed_hint_window(telemetry, 15, .1, .2) == (4.3, 5.1)


def test_recovery_target_is15_and_warmup_is_excluded_from_supervised_window():
    from flybrain_drive.ac_engine_ai_labels import common_expert_settings, fixed_hint_window

    request = dict(command='expert-speed-recovery15', speed_limit_kmh=30,
        recovery_target_kmh=15, warmup_seconds=3, hold_seconds=30, ai_steer_multiplier=.9)
    assert common_expert_settings([request])['ai_speed_hint_kmh'] == 15
    with pytest.raises(ValueError):
        common_expert_settings([dict(request, recovery_target_kmh=0)])
    rows = [dict(script_clock=i, phase='driving', ai_controlled=True, paused=False,
                 teacher_speed_hint_kmh=hint) for i, hint in [(1, 30), (2, 15), (3, 15)]]
    assert fixed_hint_window(rows, 15, 10, 10.01) == (12.01, 13)
    rows[1]['teacher_speed_hint_kmh'] = 30
    rows.append(dict(rows[-1], script_clock=4, teacher_speed_hint_kmh=30))
    rows.append(dict(rows[-1], script_clock=5, teacher_speed_hint_kmh=15))
    with pytest.raises(ValueError, match='Discontinuous'):
        fixed_hint_window(rows, 15, 10, 10.01)


def test_partial_recovery_admission_cannot_hide_unknown_api_or_bad_stop():
    import copy

    from flybrain_drive.ac_engine_ai_labels import validate_incomplete_recovery

    request = dict(command='expert-speed-recovery15', speed_limit_kmh=30,
        recovery_target_kmh=15, warmup_seconds=3, hold_seconds=30, ai_steer_multiplier=.9)
    names = ['enable_ai', 'set_ai_steer_multiplier', 'set_ai_speed_limit', 'resume_owned_ai',
             'set_recovery15_speed_limit', 'disable_ai', 'pause_after_brake_hold',
             'restore_default_ai_speed_limit', 'restore_default_ai_steer_multiplier']
    receipt = dict(command=request['command'], status='api_returned_false', recovery15_passed=False,
        stop_reason='owned motion probe deadline',
        api_calls=[dict(name=n, returned=True) for n in names],
        before=dict(generation='one', epoch=1), after=dict(generation='one', epoch=1,
            car_id='bmw_z4_gt3', track_id='monza', paused=True, ai_controlled=False,
            speed_kmh=.01, gas=0, brake=1), maximum_speed_kmh=31, maximum_position_delta_m=180,
        telemetry=[dict(teacher_speed_hint_kmh=15, ai_controlled=True,
                        phase='driving', paused=False, speed_kmh=25)])
    validate_incomplete_recovery(receipt, request)
    for key, value in [('status', 'api_outcome_unknown'), ('maximum_speed_kmh', 41),
                       ('stop_reason', 'owned camera ended or became stale')]:
        with pytest.raises(ValueError):
            validate_incomplete_recovery(dict(receipt, **{key: value}), request)
    unknown = copy.deepcopy(receipt)
    unknown['api_calls'][4]['returned'] = False
    with pytest.raises(ValueError):
        validate_incomplete_recovery(unknown, request)
    with pytest.raises(ValueError):
        validate_incomplete_recovery(dict(receipt, after=dict(receipt['after'], brake=0)), request)
