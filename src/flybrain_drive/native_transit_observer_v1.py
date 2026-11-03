"""Derived GET-only native observer; no camera/worker/model/control activation.

The functions are copied byte-exact from the validated1829 source, except the
old807-specific error text now says registered Lua. Neither validate_plan nor
coordinate_link nor the camera-based orchestration is imported or copied.
"""

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

DERIVED_SOURCE_SHA256 = '16707542c18a07b82e59c64e6449564dc92bfda72c257575f755e90f25dfa429'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_idle(value, age_seconds, plan, window_handle, *, after_source=None):
    """Normal CSP status has no route; route is never read from this dictionary."""
    if (not finite(age_seconds) or not 0 <= age_seconds <= 2
            or value.get('generation') != plan['generation']
            or value.get('epoch') != plan['epoch']
            or value.get('window_handle') != str(window_handle)+'ULL'
            or value.get('car_id') != 'bmw_z4_gt3' or value.get('track_id') != 'monza'
            or value.get('paused') is not True or value.get('ai_controlled') is not False
            or value.get('is_online') is not False
            or any(value.get(k) for k in ('camera_active', 'expert_probe_active',
                'brake_settle_active', 'pending_receipt_id', 'error', 'in_menu'))
            or any(not finite(value.get(k)) for k in ('speed_kmh', 'gas', 'brake', 'script_clock'))
            or abs(value['speed_kmh']) >= .1 or not 0 <= value['gas'] < .02
            or not .98 < value['brake'] <= 1):
        raise ValueError('Fresh same-plant paused AIoff/fullbrake idle status required')
    return after_source is None or value['script_clock'] > after_source


def decode_observer(raw, received):
    from flybrain_drive.ac_native_clock import GRAPHICS_PREFIX, decode_graphics_timing
    from flybrain_drive.ac_native_position import decode_diagnostic_position

    position = decode_diagnostic_position(raw, received)
    timing = decode_graphics_timing(raw[:GRAPHICS_PREFIX.size], received)
    point = struct.unpack_from('<f', raw, 248)[0]
    if (position.status != 'pause' or timing.completed_laps != 0
            or not finite(point) or not 0 <= point < 1):
        raise ValueError('Paused lap0 native normalizedCarPosition observer required')
    return dict(spline_position=point, lap_count=timing.completed_laps,
        graphics_packet=position.packet_id, graphics_status=position.status,
        received_perf_counter_seconds=position.received_perf_counter_seconds,
        coordinates_native_xyz_m=list(position.coordinates_native_xyz_m),
        source='native_graphics.normalizedCarPosition@248', actor_input=False,
        exact_physical_sample_time_verified=False, repeated_copy_is_engine_seqlock=False)


def validate_pair(first, second):
    if (second['observer']['graphics_packet'] <= first['observer']['graphics_packet']
            or second['physics420_sha256'] != first['physics420_sha256']
            or second['native_packet'] != first['native_packet']
            or second['observer']['received_perf_counter_seconds'] <= (
                first['observer']['received_perf_counter_seconds'])
            or second['status']['script_clock'] < first['status']['script_clock']):
        raise ValueError('Two advancing graphics packets with unchanged paused physics required')


def stable_native_copy(mapping):
    for attempt in range(8):
        first, second = mapping.read(), mapping.read()
        if first == second:
            return second, attempt+1
    raise RuntimeError('Native bytes changed during every bounded repeated-copy attempt')


def observe_paused(plan, directory, *, after_source=None):
    """GET-only: existing mappings, status file and vJoy status; no activation/pings."""
    from flybrain_drive.ac_capture import WindowsACWindow
    from flybrain_drive.ac_image_buffer import ExistingNamedMapping
    from flybrain_drive.ac_native_physics import decode_static_identity_prefix
    from flybrain_drive.ac_published_state import read_published_json
    from flybrain_drive.ac_teacher_physics import decode_teacher_physics
    from flybrain_drive.ac_vjoy import load_vjoy_sdk
    from flybrain_drive.evidence import write_json
    from flybrain_drive.session_heartbeat_admission import refuse_leased_transport

    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    window_api = WindowsACWindow()
    windows = window_api.discover()
    if len(windows) != 1 or windows[0].pid != plan['engine_pid']:
        raise ValueError('One exact registered engine PID required')
    window = windows[0]
    if window_api.user.GetForegroundWindow() == window.hwnd:
        raise ValueError('Background engine required; never automate the window')
    installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha(installed) != plan['lua_sha256']:
        raise ValueError('Exact registered installed Lua required; never deploy from staging')
    axis = json.loads(Path('configs/ac-axis-probe.json').read_bytes())
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    if sdk.GetVJDStatus(axis['device_id']) != 1:
        raise ValueError('FREE controller required; staging never acquires a driver')

    def idle():
        refuse_leased_transport('GET-only staging observer', status_path=status_path)
        value, modified = read_published_json(status_path)
        ready = check_idle(value, time.time()-modified, plan, window.hwnd,
                           after_source=after_source)
        return value, modified, ready

    deadline = time.perf_counter()+plan['observation_timeout_seconds']
    while True:
        _, _, ready = idle()
        if ready:
            break
        if time.perf_counter() >= deadline:
            raise TimeoutError('No advancing fresh SYNC after frozen handoff; no new owner')
        time.sleep(.01)
    with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
        static, _ = stable_native_copy(mapping)
    (directory/'static200.bin').write_bytes(static)
    if decode_static_identity_prefix(static) != {
            'shared_memory_version':'1.7','ac_version':'1.16.4',
            'car_model':'bmw_z4_gt3','track':'monza'}:
        raise ValueError('Exact repeated native static identity required')
    samples = []
    with (ExistingNamedMapping('Local\\acpmf_physics', 420) as physics,
          ExistingNamedMapping('Local\\acpmf_graphics', 264) as graphics):
        for index in range(2):
            physics_raw, physics_attempts = stable_native_copy(physics)
            physics_received = time.perf_counter()
            graphics_raw, graphics_attempts = stable_native_copy(graphics)
            graphics_copied = time.perf_counter()
            graphics_file = directory/('graphics-%d-264.bin' % index)
            physics_file = directory/('physics-AI-%d-420.bin' % index)
            graphics_file.write_bytes(graphics_raw)
            physics_file.write_bytes(physics_raw)
            # Original offending bytes and read receipts survive even decoder/guard failure.
            write_json(directory/('raw-read-%d.json' % index), dict(
                graphics_file=str(graphics_file), graphics_sha256=sha(graphics_file),
                graphics_copy_host_perf_counter_seconds=graphics_copied,
                graphics_repeated_copy_attempts=graphics_attempts,
                physics420_file=str(physics_file), physics420_sha256=sha(physics_file),
                physics_copy_host_perf_counter_seconds=physics_received,
                physics_repeated_copy_attempts=physics_attempts, actor_input=False))
            graphics_received = graphics_copied
            if samples and struct.unpack_from('<i', graphics_raw)[0] == (
                    samples[0]['observer']['graphics_packet']):
                if graphics_raw != Path(samples[0]['graphics_file']).read_bytes():
                    raise ValueError('Duplicate graphics packet changed bytes')
                graphics_received = samples[0]['observer']['received_perf_counter_seconds']
            observer = decode_observer(graphics_raw, graphics_received)
            coupled = decode_teacher_physics(physics_raw, physics_received)
            state, modified, ready = idle()
            if not ready:
                raise ValueError('SYNC clock did not retain advancement after handoff')
            f = coupled.physics.fields
            if abs(f['speedKmh']) >= .1 or not 0 <= f['gas'] < .02 or not .98 < f['brake'] <= 1:
                raise ValueError('Original native paused stationary fullbrake required')
            sample = dict(observer=observer, status=state, status_file_mtime=modified,
                status_original_bytes_claimed=False,
                graphics_file=str(graphics_file), graphics_sha256=sha(graphics_file),
                physics420_file=str(physics_file), physics420_sha256=sha(physics_file),
                native_packet=coupled.physics.packet_id,
                native_read_host_receipt=coupled.physics.received_monotonic_seconds,
                native_AI_frozen_baseline=coupled.builtin_ai_controlled,
                native_gear=struct.unpack_from('<i', coupled.physics.prefix_bytes, 16)[0],
                native_speed_kmh=f['speedKmh'], native_gas=f['gas'], native_brake=f['brake'],
                fresh_motion_or_AIownership_claimed=False, actor_input=False)
            samples.append(sample)
            write_json(directory/('sample-%d.json' % index), sample)
            if index == 0:
                time.sleep(plan['native_graphics_pair_minimum_seconds'])
    validate_pair(*samples)
    if (sdk.GetVJDStatus(axis['device_id']) != 1
            or window_api.user.GetForegroundWindow() == window.hwnd):
        raise ValueError('FREE/background must remain true through observer read')
    if not idle()[2]:
        raise ValueError('Advancing no-lease SYNC must remain valid through observer end')
    return dict(observer=samples[-1]['observer'], samples=samples,
        FREE_background=True, same_frozen_physics=True, graphics_advance_observed=True,
        no_fixed_gear_admission=True, driver_acquisitions=0, vehicle_API_calls=0,
        CSP_clock_pings=0, actor_input=False)
