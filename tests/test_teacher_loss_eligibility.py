"""Pure eligibility tests; fixtures only, no data admission, clocks, maps or control."""
import copy
import dataclasses
import hashlib
import json
import math
import struct

import pytest

from flybrain_drive.ac_teacher_physics import TeacherPhysicsSample, decode_teacher_physics
from flybrain_drive.teacher_loss_eligibility import (
    NativeControlCalibration,
    ProjectedHostReceipt,
    SourceInterval,
    TaskPhase,
    TeacherActivity,
    diagnostic_control_loss_eligibility,
    project_host_receipt_interval,
)


def sample(*, packet=8, flag=1, steering=.05, gas=0., brake=1., receipt=10.):
    raw = bytearray(420)
    for offset, value in [(4, gas), (8, brake), (24, steering)]:
        struct.pack_into('<f', raw, offset, value)
    struct.pack_into('<i', raw, 0, packet)
    struct.pack_into('<i', raw, 416, flag)
    return decode_teacher_physics(bytes(raw), receipt)


def interval(lower, upper, generation='g'):
    return SourceInterval(generation, lower, upper)


def fixture(**updates):
    values = dict(sample=sample(), camera_intervals={
        view: interval(1.4, 1.5) for view in ('front', 'left', 'right', 'rear')},
        native_host_receipt=ProjectedHostReceipt(10., interval(1.6, 1.61), 'a'*64),
        activity=TeacherActivity(interval(1., 1.1), interval(2.9, 3.)),
        task_phases=(TaskPhase('0:accelerate60', interval(1., 3.)),),
        expected_phase_id='0:accelerate60',
        calibration=NativeControlCalibration('synthetic-explicit-domain', 'g', 1.4, .1, .02))
    values.update(updates)
    return values


def assess(**updates):
    return diagnostic_control_loss_eligibility(**fixture(**updates))


def test_fullbrake1_without_coactivation_is_retained_but_grants_no_fit_or_yaw_policy():
    result = assess()
    assert result.channel_mask == (True, True, True)
    assert not result.fit_allowed and result.yaw_reference_policy == 'separate_unresolved'
    assert not result.exact_native_physical_clock_verified
    assert not result.exclusive_teacher_control_verified
    assert result.scope == 'diagnostic_native_control_loss_only'
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.brake = False


@pytest.mark.parametrize('activity', [
    None,
    TeacherActivity(interval(1., 1.1), None),
    TeacherActivity(None, interval(2.9, 3.)),
    TeacherActivity(interval(1., 1.), interval(3., 3.)),
    TeacherActivity(interval(1., 1.1), interval(2.9, 2.9)),
    TeacherActivity(interval(1., 2.), interval(1.9, 3.)),
])
def test_missing_or_postcall_point_action_bounds_are_unknown_all_false(activity):
    result = assess(activity=activity)
    assert result.channel_mask == (False, False, False)
    assert any('bounds' in reason for reason in result.temporal_reasons)
    assert not result.fit_allowed


@pytest.mark.parametrize('view', ['front', 'left', 'right', 'rear'])
@pytest.mark.parametrize('bounds', [(1.05, 1.15), (1.1, 1.2), (2.8, 2.9), (2.85, 2.95)])
def test_whole_camera_interval_must_be_strictly_inside_enable_finish_disable_start(view, bounds):
    values = fixture()
    values['camera_intervals'][view] = interval(*bounds)
    result = diagnostic_control_loss_eligibility(**values)
    assert result.channel_mask == (False, False, False)
    assert 'whole_camera_or_HOST_receipt_outside_teacher_active_window' in result.temporal_reasons


@pytest.mark.parametrize('bounds', [(1.05, 1.15), (1.1, 1.2), (2.85, 2.95), (2.9, 2.9)])
def test_projected_HOST_receipt_entire_bound_must_fit_not_just_midpoint(bounds):
    result = assess(native_host_receipt=ProjectedHostReceipt(10., interval(*bounds), 'a'*64))
    assert result.channel_mask == (False, False, False)


def test_projected_receipt_cannot_be_attached_to_a_different_native_clock():
    result = assess(native_host_receipt=ProjectedHostReceipt(10.01, interval(1.6, 1.61), 'a'*64))
    assert result.channel_mask == (False, False, False)
    assert 'native_receipt_attachment_mismatch' in result.temporal_reasons


@pytest.mark.parametrize('updates', [
    {'camera_intervals': None}, {'camera_intervals': {'front': interval(1.4, 1.5)}},
    {'native_host_receipt': None}, {'sample': sample(flag=0)},
    {'task_phases': ()}, {'expected_phase_id': None},
])
def test_missing_view_clock_phase_and_false_flag_are_explicitly_ineligible(updates):
    result = assess(**updates)
    assert result.channel_mask == (False, False, False)
    assert result.temporal_reasons


@pytest.mark.parametrize('updates', [
    {'camera_intervals': {v: interval(1.4, 1.5, 'other')
                          for v in ('front', 'left', 'right', 'rear')}},
    {'native_host_receipt': ProjectedHostReceipt(10., interval(1.6, 1.61, 'other'), 'a'*64)},
    {'activity': TeacherActivity(interval(1., 1.1), interval(2.9, 3., 'other'))},
    {'task_phases': (TaskPhase('0:accelerate60', interval(1., 3., 'other')),)},
])
def test_generation_mismatch_never_passes_on_matching_numeric_times(updates):
    assert assess(**updates).channel_mask == (False, False, False)


@pytest.mark.parametrize('phases', [
    (TaskPhase('0:accelerate60', interval(1., 2.)),
     TaskPhase('0:accelerate60', interval(2., 3.))),
    (TaskPhase('0:accelerate60', interval(1., 2.)),
     TaskPhase('1:accelerate100', interval(1.9, 3.))),
    (TaskPhase('1:accelerate100', interval(2., 3.)),
     TaskPhase('0:accelerate60', interval(1., 2.))),
])
def test_duplicate_overlapping_or_reversed_phase_declarations_are_ambiguous(phases):
    result = assess(task_phases=phases)
    assert result.channel_mask == (False, False, False)
    assert 'phase_declaration_ambiguous' in result.temporal_reasons


def test_repeated_semantic60_phases_keep_distinct_ids_and_do_not_merge():
    phases = (TaskPhase('0:accelerate60', interval(1., 2.)),
              TaskPhase('3:accelerate60', interval(2., 3.)))
    assert assess(task_phases=phases).channel_mask == (True, True, True)
    result = assess(task_phases=phases, expected_phase_id='3:accelerate60')
    assert result.channel_mask == (False, False, False)
    assert 'whole_camera_or_HOST_receipt_phase_ambiguous' in result.temporal_reasons


def test_single_camera_crossing_task_phase_is_not_replaced_by_receipt_phase():
    values = fixture(task_phases=(TaskPhase('0:accelerate60', interval(1., 1.55)),
                                  TaskPhase('1:accelerate100', interval(1.55, 3.))),
                     expected_phase_id='1:accelerate100')
    assert diagnostic_control_loss_eligibility(**values).channel_mask == (False, False, False)


@pytest.mark.parametrize('gas', [.8, .192])
def test_realistic_coactivation_masks_only_both_pedals_steering_stays_independent(gas):
    evidence = sample(gas=gas, brake=1.)
    raw_before = evidence.authority_prefix_bytes
    result = assess(sample=evidence)
    assert result.channel_mask == (True, False, False)
    assert result.steering_reasons == ()
    assert result.throttle_reasons == result.brake_reasons == (
        'pedal_coactivation_unrepresentable',)
    assert evidence.authority_prefix_bytes == raw_before and evidence.physics.fields['brake'] == 1.


@pytest.mark.parametrize('brake', [0., .03125, .5, 1.])
def test_noncoactive_full_and_other_measured_positive_brakes_not_blanket_masked(brake):
    assert assess(sample=sample(brake=brake)).channel_mask == (True, True, True)


def test_outside_small_steering_domain_masks_only_steering():
    result = assess(sample=sample(steering=.2, gas=0., brake=1.))
    assert result.channel_mask == (False, True, True)
    assert result.steering_reasons == ('steering_outside_reviewed_native_calibration_domain',)


@pytest.mark.parametrize('gas,brake,expected', [
    (.01, 0., (True, False, True)), (0., .01, (True, True, False)),
    (-.1, 0., (True, False, True)), (0., 1.1, (True, True, False)),
])
def test_pedal_calibration_floor_and_range_are_per_channel_without_projection(gas, brake, expected):
    assert assess(sample=sample(gas=gas, brake=brake)).channel_mask == expected


def test_exact_declared_native_domain_endpoint_is_inclusive_without_hidden_tolerance():
    calibration = NativeControlCalibration('synthetic-binary-domain', 'g', 1., .125, .03125)
    assert assess(sample=sample(steering=.125, brake=.03125),
                  calibration=calibration).channel_mask == (True, True, True)
    # Float32(.02) is below a strict float64 .02 floor. No silent rounding/projection.
    result = assess(sample=sample(brake=.02))
    assert result.brake is False


@pytest.mark.parametrize('calibration', [None,
    NativeControlCalibration('synthetic', 'other', 1.4, .1, .02)])
def test_missing_or_foreign_calibration_adds_no_permission(calibration):
    assert assess(calibration=calibration).channel_mask == (False, False, False)


@pytest.mark.parametrize('field,value', [
    ('steering_native_per_canonical', 0.), ('steering_native_per_canonical', math.nan),
    ('max_abs_canonical_steering', 0.), ('max_abs_canonical_steering', 2.),
    ('pedal_min_positive', 0.), ('pedal_min_positive', math.inf),
    ('reference', ''), ('generation', ''),
])
def test_calibration_requires_explicit_finite_domain(field, value):
    values = dict(reference='synthetic', generation='g', steering_native_per_canonical=1.4,
                  max_abs_canonical_steering=.1, pedal_min_positive=.02)
    values[field] = value
    with pytest.raises(ValueError):
        NativeControlCalibration(**values)


@pytest.mark.parametrize('lower,upper', [(math.nan, 1.), (1., math.inf), (2., 1.), (True, 2.)])
def test_nonfinite_reversed_or_boolean_bounds_reject(lower, upper):
    with pytest.raises(ValueError):
        interval(lower, upper)


def clock_record():
    quantum = 1e-7
    pings = [dict(sent=9., received=9.002, producer_clock=1.),
             dict(sent=11., received=11.002, producer_clock=3.)]
    return dict(generation='g', exchanges=pings, host_clock_quantization_seconds=quantum,
                offset_lower_seconds=max(x['sent']-x['producer_clock']-quantum for x in pings),
                offset_upper_seconds=min(x['received']-x['producer_clock']+quantum for x in pings))


def test_projector_recomputes_original_pings_padding_fingerprint_and_receipt_attachment():
    record = clock_record()
    original = copy.deepcopy(record)
    projected = project_host_receipt_interval(10., record, generation='g')
    assert projected.original_host_receipt_seconds == 10.
    assert projected.source_interval.lower == 10.-record['offset_upper_seconds']-1e-7
    assert projected.source_interval.upper == 10.-record['offset_lower_seconds']+1e-7
    assert projected.clock_record_sha256 == hashlib.sha256(json.dumps(
        record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert record == original
    assert not hasattr(projected, 'physical_sample_time')


@pytest.mark.parametrize('mutation', [
    'generation', 'nonfinite', 'reversed', 'intersection', 'short'])
def test_projector_refuses_invalid_clock_evidence(mutation):
    record = clock_record()
    if mutation == 'generation':
        record['generation'] = 'other'
    elif mutation == 'nonfinite':
        record['exchanges'][0]['producer_clock'] = math.nan
    elif mutation == 'reversed':
        record['exchanges'][0]['received'] = 8.
    elif mutation == 'intersection':
        record['offset_lower_seconds'] += .1
    else:
        record['exchanges'] = record['exchanges'][:1]
    with pytest.raises(ValueError):
        project_host_receipt_interval(10., record, generation='g')


@pytest.mark.parametrize('receipt', [8., 12., math.inf, math.nan, True])
def test_projector_never_extrapolates_or_redates_unknown_receipts(receipt):
    with pytest.raises(ValueError):
        project_host_receipt_interval(receipt, clock_record(), generation='g')


@pytest.mark.parametrize('evidence', [
    lambda: sample(flag=2),
    lambda: TeacherPhysicsSample(sample().physics, False, sample().authority_prefix_bytes),
    lambda: TeacherPhysicsSample(sample().physics, True, sample(packet=9).authority_prefix_bytes),
])
def test_malformed_or_separately_sampled_native_flag_cannot_create_loss_permission(evidence):
    with pytest.raises(ValueError):
        assess(sample=evidence())


def test_unknown_metadata_view_or_wrong_input_type_is_not_silently_accepted():
    values = fixture()
    values['camera_intervals']['oracle'] = interval(1.4, 1.5)
    with pytest.raises(ValueError):
        diagnostic_control_loss_eligibility(**values)
    with pytest.raises(ValueError):
        assess(sample=sample().physics)


def test_input_history_records_clocks_and_states_are_never_filtered_or_changed():
    history = [sample(packet=i, receipt=float(i)) for i in (7, 8, 9)]
    before = [(x.physics.packet_id, x.physics.received_monotonic_seconds,
               x.authority_prefix_bytes) for x in history]
    masks = []
    for i, evidence in enumerate(history):
        cameras = fixture()['camera_intervals']
        if i != 1:
            cameras['front'] = interval(.9, 1.) if i == 0 else interval(3., 3.1)
        receipt = ProjectedHostReceipt(evidence.physics.received_monotonic_seconds,
                                       interval(1.6, 1.61), 'a'*64)
        masks.append(assess(sample=evidence, camera_intervals=cameras,
                            native_host_receipt=receipt).channel_mask)
    assert masks == [(False, False, False), (True, True, True), (False, False, False)]
    assert len(history) == 3
    assert before == [(x.physics.packet_id, x.physics.received_monotonic_seconds,
                       x.authority_prefix_bytes) for x in history]


def test_four_coactivations_do_not_mask_other_positive_fullbrakes_in_synthetic_history():
    controls = [sample(packet=i+7, gas=g, brake=1.)
                for i, g in enumerate([.8, .8, .8, .192, 0., 0., 0.])]
    original = [x.authority_prefix_bytes for x in controls]
    masks = [assess(sample=x).channel_mask for x in controls]
    assert masks[:4] == [(True, False, False)]*4
    assert masks[4:] == [(True, True, True)]*3
    assert original == [x.authority_prefix_bytes for x in controls]


def test_foreign_generation_in_other_phase_cannot_make_declared_program_qualified():
    phases = (TaskPhase('0:accelerate60', interval(1., 2.)),
              TaskPhase('1:accelerate100', interval(2., 3., 'other')))
    assert assess(task_phases=phases).channel_mask == (False, False, False)


@pytest.mark.parametrize('field,value', [('packet_id', True), ('received_monotonic_seconds', True)])
def test_boolean_native_packet_or_receipt_cannot_masquerade_as_original_scalar(field, value):
    original = sample(packet=1, receipt=1.)
    body = dataclasses.replace(original.physics, **{field: value})
    invalid = TeacherPhysicsSample(body, True, original.authority_prefix_bytes)
    with pytest.raises(ValueError):
        assess(sample=invalid)
