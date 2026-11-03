"""Mocked integration only: no actual mappings, processes, control or GPU."""
import base64
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_teacher_physics import (
    NativeTeacherPhysicsReader,
    TeacherPhysicsSample,
    decode_teacher_physics,
)
from scripts import record_ac_engine_ai_probe as recorder


def raw_packet(packet=8, active=1, *, baseline=False):
    value = bytearray(420)
    struct.pack_into('<i', value, 0, packet)
    struct.pack_into('<f', value, 4, 0. if baseline else .4)
    struct.pack_into('<f', value, 8, 1. if baseline else .2)
    struct.pack_into('<f', value, 28, 0. if baseline else 42.)
    struct.pack_into('<i', value, 416, active)
    return bytes(value)


def coupled(active=1, clock=10.):
    return decode_teacher_physics(raw_packet(active=active), clock)


def metadata(sample=None, **overrides):
    values = dict(baseline_packet=7, decision_time=10.02, csp_running=True,
                  maximum_body_age=.12, teacher_kind='engine_ai')
    values.update(overrides)
    return recorder.native_authority_metadata(sample or coupled(), **values)


@pytest.mark.parametrize('active,eligible', [(0, False), (1, True)])
def test_eligibility_is_explicit_native_flag_evidence_only_and_raw_is_unchanged(active, eligible):
    sample = coupled(active)
    value = metadata(sample)
    assert value['teacher_authority_eligible'] is eligible
    assert value['packet_id'] == sample.physics.packet_id
    assert value['received_monotonic_seconds'] == sample.physics.received_monotonic_seconds
    assert base64.b64decode(value['prefix_base64']) == sample.authority_prefix_bytes
    assert value['body_prefix_sha256'] == hashlib.sha256(sample.physics.prefix_bytes).hexdigest()
    assert value['eligibility_scope'] == 'same_copy_native_AI_flag_at_body_packet_only'
    assert not value['actor_input'] and not value['fit_permission_granted']
    assert not value['camera_teacher_authority_synchronized']
    assert not value['exclusive_teacher_control_verified']
    assert not value['simulation_clock_verified'] and not value['atomicity_verified']
    assert not value['optimal_teacher_control_verified'] and value['source_time_seconds'] is None
    assert 'builtin_ai_controlled' not in sample.physics.fields


@pytest.mark.parametrize('changes,reason', [
    ({'baseline_packet': 8}, 'packet_advanced_since_baseline'),
    ({'csp_running': False}, 'CSP_AI_running_at_loop_start'),
    ({'decision_time': 9.9}, 'receipt_age_within_existing_retina_bound'),
    ({'decision_time': 10.2}, 'receipt_age_within_existing_retina_bound'),
])
def test_baseline_inactive_future_and_stale_receipts_remain_explicitly_ineligible(changes, reason):
    value = metadata(**changes)
    assert value['teacher_authority_eligible'] is False
    assert reason in value['ineligible_reasons']
    assert value['received_monotonic_seconds'] == 10.


@pytest.mark.parametrize('changes', [
    {'decision_time': math.nan}, {'decision_time': math.inf}, {'decision_time': -1.},
    {'decision_time': True}, {'maximum_body_age': 0.}, {'maximum_body_age': math.nan},
    {'csp_running': 1},
])
def test_nonfinite_or_untyped_eligibility_clock_fails_closed(changes):
    with pytest.raises(ValueError, match='original decision clock'):
        metadata(**changes)


@pytest.mark.parametrize('enabled', [False, True])
def test_reader_factory_preserves_original_timeout_poll_clock(monkeypatch, enabled):
    calls = []
    sentinel = object()
    monkeypatch.setattr(recorder, 'NativePhysicsReader',
                        lambda **kw: calls.append(('legacy', kw)) or sentinel)
    monkeypatch.setattr(recorder, 'NativeTeacherPhysicsReader',
                        lambda **kw: calls.append(('coupled', kw)) or sentinel)
    def clock():
        return 12.
    assert recorder.native_recording_reader(enabled, clock=clock) is sentinel
    expected = ('coupled', dict(timeout=.35, poll_seconds=.005, clock=clock)) if enabled else (
        'legacy', dict(timeout=.35, clock=clock))
    assert calls == [expected]


def test_split_uses_one_receive_and_reuses_original_body_object():
    sample = coupled()
    calls = []
    reader = SimpleNamespace(receive=lambda: calls.append(1) or sample)
    body, authority = recorder.receive_recording_native(reader, same_copy_native_authority=True)
    assert calls == [1] and body is sample.physics and authority is sample
    legacy = SimpleNamespace(receive=lambda: sample.physics)
    body, authority = recorder.receive_recording_native(legacy, same_copy_native_authority=False)
    assert body is sample.physics and authority is None


@pytest.mark.parametrize('bad', [
    lambda: SimpleNamespace(physics=coupled().physics),
    lambda: TeacherPhysicsSample(coupled().physics, False, raw_packet(active=1)),
    lambda: TeacherPhysicsSample(coupled().physics, True, raw_packet(packet=9)),
    lambda: TeacherPhysicsSample(coupled().physics, True, raw_packet(active=2)),
])
def test_separate_or_mismatched_flag_cannot_masquerade_as_same_copy(bad):
    reader = SimpleNamespace(receive=bad)
    with pytest.raises(ValueError):
        recorder.receive_recording_native(reader, same_copy_native_authority=True)


class MemoryMap:
    def __init__(self, values):
        self.values = iter(values)
        self.last = None
        self.closed = False

    def read(self):
        self.last = next(self.values, self.last)
        return self.last

    def close(self):
        self.closed = True


def test_actual_reader_duplicate_and_reset_semantics_close_mapping_on_exception():
    source = MemoryMap([raw_packet(7), raw_packet(7), raw_packet(7), raw_packet(7),
                        raw_packet(8), raw_packet(8), raw_packet(6), raw_packet(6)])
    with pytest.raises(ValueError, match='reset'):
        with NativeTeacherPhysicsReader(mapping=source, clock=lambda: 10.) as reader:
            first = recorder.receive_recording_native(reader, same_copy_native_authority=True)
            second = recorder.receive_recording_native(reader, same_copy_native_authority=True)
            assert first[0].packet_id == 7 and second[0].packet_id == 8
            recorder.receive_recording_native(reader, same_copy_native_authority=True)
    assert source.closed


@pytest.mark.parametrize('clock', [math.nan, math.inf, -1.])
def test_actual_reader_bad_receipt_clock_closes_without_fabricating_observation(clock):
    source = MemoryMap([raw_packet(), raw_packet()])
    with pytest.raises(ValueError):
        with NativeTeacherPhysicsReader(mapping=source, clock=lambda: clock) as reader:
            recorder.receive_recording_native(reader, same_copy_native_authority=True)
    assert source.closed


def run_mock_recording(tmp_path, monkeypatch, *, enabled, active=1, failure=None, extra_args=()):
    """Exercise real main/serialization/exception cleanup with all live endpoints mocked."""
    out = tmp_path/'recording'
    local = tmp_path/'local'
    (local/'flybrain/csp-session').mkdir(parents=True)
    config = tmp_path/'config.json'
    config.write_text('{}', encoding='utf8')
    argv = ['record', '--config', str(config), '--output', str(out)]
    if enabled:
        argv.append('--same-copy-native-authority')
    argv.extend(extra_args)
    monkeypatch.setattr(recorder.sys, 'argv', argv)
    monkeypatch.setenv('LOCALAPPDATA', str(local))
    calls = []

    def begin(output, data):
        Path(output).mkdir()
        calls.append(('begin', data))
        return Path(output)

    monkeypatch.setattr(recorder, 'begin_run', begin)
    window = SimpleNamespace(hwnd=123, pid=456)
    native = SimpleNamespace(discover=lambda: [window],
                             user=SimpleNamespace(GetForegroundWindow=lambda: 0))
    monkeypatch.setattr(recorder, 'WindowsACWindow', lambda: native)

    class StaticMap:
        def __enter__(self):
            return self

        def read(self):
            return b'identity'

        def __exit__(self, *_):
            calls.append('static_closed')

    monkeypatch.setattr(recorder, 'ExistingNamedMapping', lambda *a: StaticMap())
    monkeypatch.setattr(recorder, 'decode_static_identity_prefix', lambda b: dict(
        car_model='bmw_z4_gt3', track='monza', shared_memory_version='1.7', ac_version='1.16.4'))

    class Retina:
        required_native_fields = ('gas', 'brake', 'speedKmh')
        max_age = .12
        precompute_pixels = None
        calibration = {'fixture': True}

        def pack(self, samples, **kw):
            calls.append(('pack', kw))
            if failure == 'pack':
                raise ValueError('injected pack failure')
            return dict(view_valid=np.ones(4, dtype=np.float32),
                        organs=np.array(list(kw['native_scalars'].values()), dtype=np.float32))

    monkeypatch.setattr(recorder, 'CSPCameraRetina', lambda cfg: Retina())

    class Vision:
        def __init__(self, *args, **kwargs):
            calls.append('vision_constructed')
            calls.append(('vision_period', kwargs['period']))
            assert (out/'executed-program.py').exists() is enabled
            assert (out/'native-authority-source-proof.json').exists() is enabled
            self.request_id = 'camera-test'
            self.clock = SimpleNamespace(generation='g')
            self.triggered = True
            self.directory = tmp_path/'vision'
            self.directory.mkdir()
            self.error = None
            self.host_clock = lambda: 10.02

        def wait_ready(self):
            calls.append('ready')

        def wait_for_candidate(self, **kw):
            return None

        def candidate_before(self, clock):
            calls.append(('candidate_before', clock))
            return dict(views=[])

        def samples_for(self, candidate):
            return {}

        def close(self):
            calls.append('vision_closed')
            return dict(status='closed')

        def archive_to(self, *a, **kw):
            calls.append('vision_archived')
            return dict(status='mock')

    monkeypatch.setattr(recorder, 'CSPVisionRecorder', Vision)
    monkeypatch.setattr(recorder, 'wait_owned_camera_heartbeat',
                        lambda path, generation, request_id: dict(
                            generation=generation, request_id=request_id))
    phases = iter([False, True])
    monkeypatch.setattr(recorder, 'read_ai_phase', lambda *a, **kw: next(phases))
    monkeypatch.setattr(recorder.time, 'sleep', lambda _: None)

    class Reader:
        def __init__(self, same, **kwargs):
            calls.append(('reader', same, kwargs))
            self.index = 0
            self.same = same

        def __enter__(self):
            return self

        def receive(self):
            self.index += 1
            baseline = self.index == 1
            raw = raw_packet(7 if baseline else 8, 0 if baseline else active, baseline=baseline)
            clock = 9. if baseline else math.nan if failure == 'clock' else 10.
            if self.same:
                return decode_teacher_physics(raw, clock)
            return decode_physics_prefix(raw[:308], clock)

        def __exit__(self, *_):
            calls.append('reader_closed')

    monkeypatch.setattr(recorder, 'NativePhysicsReader', lambda **kw: Reader(False, **kw))
    monkeypatch.setattr(recorder, 'NativeTeacherPhysicsReader', lambda **kw: Reader(True, **kw))

    class Child:
        pid = 99
        returncode = None
        polls = 0

        def poll(self):
            self.polls += 1
            if not failure and active in (0, 1) and self.polls > 1:
                self.returncode = 0
            return self.returncode

        def wait(self, timeout):
            calls.append(('child_wait', timeout))
            self.returncode = 0
            return 0

    def launch(*args, **kwargs):
        calls.append(('child', args, kwargs))
        control = out/'control'
        control.mkdir()
        (control/'request.json').write_text('{"id":"owned-test"}', encoding='utf8')
        (control/'api-receipt.json').write_text(json.dumps(dict(
            maximum_speed_kmh=42., maximum_position_delta_m=1.)), encoding='utf8')
        return Child()

    monkeypatch.setattr(recorder.subprocess, 'Popen', launch)
    if failure or active not in (0, 1):
        with pytest.raises(RuntimeError):
            recorder.main()
    else:
        recorder.main()
    return out, local, calls


def test_long_shared_recording_passes_explicit_slower_camera_period(tmp_path, monkeypatch):
    _, _, calls = run_mock_recording(tmp_path, monkeypatch, enabled=False, extra_args=(
        '--shared-transport', '--image-format', 'bmp', '--duration-seconds', '30',
        '--speed-hint-kmh', '60', '--ai-steer-multiplier', '.9',
        '--camera-period-seconds', '.045'))
    assert ('vision_period', .045) in calls


@pytest.mark.parametrize('enabled', [False, True])
def test_main_default_optin_serialization_and_source_proof(tmp_path, monkeypatch, enabled):
    out, _, calls = run_mock_recording(tmp_path, monkeypatch, enabled=enabled)
    event = json.loads((out/'events.jsonl').read_text())
    result = json.loads((out/'result.json').read_text())
    assert result['status'] == 'completed' and result['decisions'] == 1
    assert result['training_admissible'] is result['expert_labels_available'] is False
    assert len(base64.b64decode(event['native']['prefix_base64'])) == 308
    assert event['native']['received_monotonic_seconds'] == 10.
    assert event['decision_monotonic_seconds'] == 10.02
    assert 'reader_closed' in calls and 'vision_closed' in calls
    assert ('native_teacher_authority' in event) is enabled
    with np.load(out/'observations.npz') as arrays:
        assert set(arrays) == {'obs_view_valid', 'obs_organs', 'observation_time'}
        assert np.array_equal(arrays['obs_organs'], np.array([[.4, .2, 42.]], dtype=np.float32))
        assert arrays['observation_time'].tolist() == [10.02]
    if enabled:
        authority = event['native_teacher_authority']
        assert authority['teacher_authority_eligible']
        assert authority['packet_id'] == event['native']['packet_id'] == 8
        assert authority['received_monotonic_seconds'] == (
            event['native']['received_monotonic_seconds'])
        assert base64.b64decode(authority['prefix_base64'])[:308] == (
            base64.b64decode(event['native']['prefix_base64']))
        assert result['same_copy_native_authority']['records'] == 1
        baseline = json.loads((out/'native-before-activation-authority.json').read_text())
        assert not baseline['teacher_authority_eligible'] and baseline['packet_id'] == 7
        proof = json.loads((out/'native-authority-source-proof.json').read_text())
        assert proof['before_camera_activation'] and proof['before_control_activation']
        executed_hash = hashlib.sha256((out/'executed-program.py').read_bytes()).hexdigest()
        assert executed_hash == proof['executed_program_sha256']
        assert set(proof['source_files']) == set(recorder.SAME_COPY_SOURCE_FILES)
    else:
        assert 'same_copy_native_authority' not in result
        assert not (out/'native-before-activation-authority.json').exists()
    assert sum(isinstance(x, tuple) and x[0] == 'reader' for x in calls) == 1


def test_false_native_flag_preserves_observation_without_eligibility(tmp_path, monkeypatch):
    out, _, _ = run_mock_recording(tmp_path, monkeypatch, enabled=True, active=0)
    event = json.loads((out/'events.jsonl').read_text())
    result = json.loads((out/'result.json').read_text())
    assert result['decisions'] == 1 and result['status'] == 'completed'
    assert not event['native_teacher_authority']['teacher_authority_eligible']
    assert result['same_copy_native_authority']['eligible_records'] == 0
    assert result['same_copy_native_authority']['ineligible_records'] == 1
    assert not result['training_admissible'] and not result['expert_labels_available']


@pytest.mark.parametrize('failure,active', [('pack', 1), ('clock', 1), (None, 2)])
def test_main_exception_closes_and_cancels_only_owned_child(
        tmp_path, monkeypatch, failure, active):
    out, local, calls = run_mock_recording(tmp_path, monkeypatch, enabled=True,
                                          failure=failure, active=active)
    result = json.loads((out/'result.json').read_text())
    assert result['status'] == 'failed' and result['decisions'] == 0
    assert result['owned_camera_stop_requested'] and result['owned_cancel_requested']
    assert (local/'flybrain/csp-session/owned-test.stop').exists()
    assert 'reader_closed' in calls and 'vision_closed' in calls
    assert ('child_wait', 10) in calls
    assert sum(isinstance(x, tuple) and x[0] == 'child' for x in calls) == 1
    assert result['ui_actions_sent'] == result['vjoy_messages_sent'] == 0


def test_default_and_optin_actor_arrays_are_exactly_identical(tmp_path, monkeypatch):
    legacy, _, _ = run_mock_recording(tmp_path/'legacy', monkeypatch, enabled=False)
    coupled_run, _, _ = run_mock_recording(tmp_path/'coupled', monkeypatch, enabled=True)
    with np.load(legacy/'observations.npz') as old, np.load(coupled_run/'observations.npz') as new:
        assert set(old) == set(new)
        for key in old:
            assert old[key].dtype == new[key].dtype
            assert np.array_equal(old[key], new[key])
