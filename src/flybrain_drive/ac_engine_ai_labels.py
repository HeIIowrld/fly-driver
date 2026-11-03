"""Future applied-control receipt targets; never claims access to internal AI commands."""

import bisect

import numpy as np


def common_expert_settings(requests):
    settings = set()
    for request in requests:
        if request.get('command') == 'expert-brake-pulse30':
            raise ValueError('Forced braking needs separate labels, not natural AI labels')
        hint = request['speed_limit_kmh']
        if request.get('command') == 'expert-lateral-recovery15' and (
                hint != 15 or request.get('hold_seconds') != 30
                or request.get('lateral_recovery') is not True):
            raise ValueError('Exact lateral recovery preset required')
        if request.get('command') in ('expert-speed-recovery15', 'expert-speed20-recovery15'):
            warmup = 20 if request['command'] == 'expert-speed20-recovery15' else 30
            if (hint != warmup or request.get('recovery_target_kmh') != 15
                    or request.get('warmup_seconds') != 3 or request.get('hold_seconds') != 30):
                raise ValueError('Exact recovery warmup and target task required')
            hint = 15  # only post-retarget observations may be supervised by this task
        settings.add((hint, request.get('ai_steer_multiplier')))
    if len(settings) != 1 or not settings <= {(15, .9), (30, .9)}:
        raise ValueError('One explicitly matched expert speed/steering task required')
    speed, steer = settings.pop()
    return dict(ai_speed_hint_kmh=speed, ai_steer_multiplier=steer)


def fixed_hint_window(telemetry, label_speed, offset_lower, offset_upper):
    driving = [row for row in telemetry
               if row['ai_controlled'] and not row['paused'] and row['phase'] == 'driving']
    selected = [i for i, row in enumerate(driving)
                if row.get('teacher_speed_hint_kmh', label_speed) == label_speed]
    if not selected:
        raise ValueError('No observed AI phase with the requested label speed')
    interval = driving[selected[0]:selected[-1]+1]
    if any(row.get('teacher_speed_hint_kmh', label_speed) != label_speed for row in interval):
        raise ValueError('Discontinuous teacher targets cannot share one supervised window')
    return (interval[0]['script_clock']+offset_upper, interval[-1]['script_clock']+offset_lower)


def validate_incomplete_recovery(receipt, request):
    """Admit only a known, cleanly stopped partial phase; never reclassify task success."""
    common_expert_settings([request])
    calls = receipt.get('api_calls', [])
    names = [call.get('name') for call in calls]
    after, before = receipt.get('after', {}), receipt.get('before', {})
    expected = ['enable_ai', 'set_ai_steer_multiplier', 'set_ai_speed_limit', 'resume_owned_ai',
                'set_recovery15_speed_limit', 'disable_ai', 'pause_after_brake_hold',
                'restore_default_ai_speed_limit', 'restore_default_ai_steer_multiplier']
    if (request.get('command') != 'expert-speed-recovery15'
            or receipt.get('command') != request['command']
            or receipt.get('status') != 'api_returned_false'
            or receipt.get('recovery15_passed') is not False
            or receipt.get('stop_reason') != 'owned motion probe deadline'
            or names != expected or not all(call.get('returned') is True for call in calls)
            or after.get('generation') != before.get('generation')
            or after.get('epoch') != before.get('epoch')
            or after.get('car_id') != 'bmw_z4_gt3' or after.get('track_id') != 'monza'
            or after.get('paused') is not True or after.get('ai_controlled') is not False
            or not 0 <= after.get('speed_kmh', float('inf')) < .1
            or not 0 <= after.get('gas', float('inf')) < .02
            or not .98 < after.get('brake', -1) <= 1
            or not 0 <= receipt.get('maximum_speed_kmh', float('inf')) <= 40
            or not 0 < receipt.get('maximum_position_delta_m', float('inf')) <= 360):
        raise ValueError('Only fully known, bounded, stationary partial recovery may be used')
    phase = [row for row in receipt.get('telemetry', [])
             if row.get('teacher_speed_hint_kmh') == 15 and row.get('ai_controlled') is True
             and row.get('phase') == 'driving' and row.get('paused') is False]
    if not phase or not any(row.get('speed_kmh', 0) > 20 for row in phase):
        raise ValueError('No observed post-retarget overspeed phase')


def match_future_applied_controls(events, native, *, native_gain, ai_window,
                                  device_scale=.6, horizon=.04, maximum_lateness=.025):
    if (not np.isfinite([native_gain, device_scale, horizon, maximum_lateness, *ai_window]).all()
            or native_gain <= 0 or device_scale != .6 or horizon != .04
            or not 0 < maximum_lateness <= .025 or len(ai_window) != 2
            or not 0 <= ai_window[0] < ai_window[1]):
        raise ValueError('Explicit calibrated codec and bounded future receipt contract required')
    unique, last_packet, previous = [], -1, -1.
    for row in native:
        stamp, packet = row['received_monotonic_seconds'], row['packet_id']
        if not np.isfinite(stamp) or stamp <= previous or packet < last_packet:
            raise ValueError('Ordered same-session native receipt stream required')
        previous = stamp
        if packet > last_packet:
            unique.append(row)
        last_packet = packet
    times = [row['received_monotonic_seconds'] for row in unique]
    raw = np.full((len(events), 3), np.nan)
    mask = np.zeros((len(events), 3), bool)
    matched = np.full(len(events), -1, dtype=np.int64)
    receipts = np.full(len(events), np.nan)
    previous = -1.
    for i, event in enumerate(events):
        decision = event['decision_monotonic_seconds']
        if not np.isfinite(decision) or decision <= previous:
            raise ValueError('Increasing actual observation times required')
        previous = decision
        target = decision+horizon
        index = bisect.bisect_left(times, target)
        if index == len(times):
            continue
        row = unique[index]
        stamp = times[index]
        if (stamp-target > maximum_lateness or not ai_window[0] <= decision <= stamp <= ai_window[1]
                or row['packet_id'] <= event['native']['packet_id']):
            continue
        fields = row['fields']
        raw[i] = [fields['steerAngle']/(native_gain*device_scale), fields['gas'], fields['brake']]
        matched[i], receipts[i] = row['packet_id'], stamp
        finite = np.isfinite(raw[i])
        mask[i] = finite & (raw[i] >= [-1, 0, 0]) & (raw[i] <= [1, 1, 1])
    # Invalid heads have explicit placeholders, never clipped executed targets.
    action = np.where(mask, raw, 0).astype(np.float32)
    return dict(action=action, action_valid=mask, raw_canonical=raw,
                label_packet_id=matched, label_receipt_time=receipts)


def observation_segments(times, maximum_gap=.08):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not 0 < maximum_gap <= .08):
        raise ValueError('Ordered observation times and explicit gap bound required')
    bounds = [0, *(np.flatnonzero(np.diff(times) > maximum_gap)+1).tolist(), len(times)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
