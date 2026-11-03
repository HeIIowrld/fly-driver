"""Synthetic slots/clocks/files only; no live MMF, worker, or vehicle calls."""

import copy
import ctypes
import json
import struct
import time

import pytest

from flybrain_drive import session_heartbeat_candidate as candidate
from flybrain_drive.ac_csp_clock import CSPClockBounds

OWNER = 'a'*32
GENERATION = 'synthetic-generation'


def snapshot(sequence=1, source_clock=10.1, *, owner=OWNER, generation=GENERATION):
    content = json.dumps(dict(schema=candidate.STATUS_SCHEMA, generation=generation,
        script_clock=source_clock, paused=True, ai_controlled=False,
        expert_probe_active=False, source_observation='keep-original',
        external_task=dict(phase='stop0', desired_speed_m_s=0))).encode()
    return candidate.HeartbeatSnapshot(owner, generation, sequence, source_clock, content)


def calibration_record():
    clock = CSPClockBounds(GENERATION)
    clock.observe(100., 100.002, 10.001, GENERATION)
    clock.observe(100.010, 100.012, 10.011, GENERATION)
    return clock.record()


def calibration():
    return candidate.CachedClockLowerBound.from_record(calibration_record(), generation=GENERATION)


class Clocks:
    def __init__(self):
        self.now = 100_200_000_000
        self.offset = time.time_ns()-self.now

    def bracket(self):
        return candidate.WallQPCBracket(self.now-1000, self.now+self.offset, self.now+1000)

    def qpc(self):
        return self.now+2000


def writer(tmp_path, **options):
    target = tmp_path/'synthetic-status.json'
    lease = candidate.OwnedWriterLease(target, owner_id=OWNER, generation=GENERATION).acquire()
    clocks = Clocks()
    instance = candidate.OwnedLatestHeartbeatWriter(
        lease, calibration(), qpc_clock=clocks.qpc, bracket_clock=clocks.bracket, **options)
    return instance, clocks, target


def test_codec_keeps_exact_json_bytes_and_uses_newest_of_two_stable_slots():
    one, two = snapshot(), snapshot(2, 10.2)
    first, second = candidate.encode_slot(one), candidate.encode_slot(two)
    result = candidate.latest_snapshot([(second, second), (first, first)],
                                      owner_id=OWNER, generation=GENERATION)
    assert result == two and result.snapshot_json == two.snapshot_json
    assert candidate.channel_name(OWNER, 1) == f'Local\\FlyBrainSessionHeartbeat_{OWNER}_1'
    assert candidate.HEADER.size == 20 and len(first) == candidate.SLOT_SIZE == 8212


def test_torn_or_odd_slot_cannot_replace_a_valid_previous_snapshot():
    first, second = candidate.encode_slot(snapshot()), candidate.encode_slot(snapshot(2, 10.2))
    odd = bytearray(second)
    struct.pack_into('<I', odd, 8, 3)
    selected = candidate.latest_snapshot([(bytes(odd), bytes(odd)), (first, first)],
                                         owner_id=OWNER, generation=GENERATION)
    assert selected.sequence == 1
    assert candidate.decode_slot(first, second, owner_id=OWNER, generation=GENERATION) is None


@pytest.mark.parametrize('field,offset', [('magic', 0), ('length', 12), ('crc', 16)])
def test_stable_corruption_never_falls_back_to_other_slot(field, offset):
    good = candidate.encode_slot(snapshot())
    bad = bytearray(candidate.encode_slot(snapshot(2, 10.2)))
    struct.pack_into('<I', bad, offset, 0)
    with pytest.raises(ValueError, match='header|CRC'):
        candidate.latest_snapshot([(bytes(bad), bytes(bad)), (good, good)],
                                  owner_id=OWNER, generation=GENERATION)


@pytest.mark.parametrize('owner,generation', [('b'*32, GENERATION), (OWNER, 'other')])
def test_foreign_owner_or_generation_never_qualifies(owner, generation):
    data = candidate.encode_slot(snapshot())
    with pytest.raises(ValueError, match='Foreign'):
        candidate.decode_slot(data, data, owner_id=owner, generation=generation)


def test_new_sequence_with_old_clock_is_rejected_without_redating():
    data = candidate.encode_slot(snapshot(2, 10.1))
    with pytest.raises(ValueError, match='source clock'):
        candidate.latest_snapshot([(data, data), (bytes(candidate.SLOT_SIZE),)*2],
            owner_id=OWNER, generation=GENERATION, last_sequence=1, last_source_clock=10.1)


def test_slot_identity_is_not_inferred_from_a_valid_crc_alone():
    data = candidate.encode_slot(snapshot())
    with pytest.raises(ValueError, match='wrong owned slot'):
        candidate.latest_snapshot([(data, data), (bytes(candidate.SLOT_SIZE),)*2],
                                  owner_id=OWNER, generation=GENERATION)


@pytest.mark.parametrize('payload', [b'[]', b'{"bad":NaN}', b'x'*6145])
def test_unbounded_or_nonstandard_json_cannot_enter_snapshot_queue(payload):
    with pytest.raises(ValueError):
        candidate.HeartbeatSnapshot(OWNER, GENERATION, 1, 10.1, payload)


@pytest.mark.parametrize('mutation', [dict(generation='other'), dict(exchanges=[]),
    dict(offset_lower_seconds=500), dict(offset_upper_seconds=float('nan'))])
def test_invalid_cached_calibration_cannot_start_source_mapping(mutation):
    record = calibration_record()
    record.update(mutation)
    with pytest.raises(ValueError, match='calibration|intersection'):
        candidate.CachedClockLowerBound.from_record(record, generation=GENERATION)


def test_cached_lower_bound_recomputes_intersection_without_pings_and_expires():
    record = calibration_record()
    original = copy.deepcopy(record)
    bound = candidate.CachedClockLowerBound.from_record(record, generation=GENERATION)
    lower = bound.source_host_lower_ns(10.1, GENERATION, 100_200_000_000)
    assert lower <= 100_100_000_000 and record == original
    for generation, clock in [('other', 100_200_000_000), (GENERATION, 161_000_000_000)]:
        with pytest.raises(ValueError, match='Expired/foreign'):
            bound.source_host_lower_ns(10.1, generation, clock)


def test_direct_invalid_cached_bound_cannot_bypass_basic_constructor_checks():
    with pytest.raises(ValueError, match='validated cached'):
        candidate.CachedClockLowerBound(GENERATION, 500., -1., 0., 100, 10, '0'*64)


def test_wall_bracket_formula_is_exact_and_conservative_without_source_resample():
    bracket = candidate.WallQPCBracket(1000, 1_700_000_000_000_000_000, 1100)
    result = candidate.conservative_unix_lower_ns(500, bracket,
                                                rounding_margin_ns=10, wall_guard_ns=20)
    assert result == 500+bracket.unix_ns-1100-10-20
    assert result < bracket.unix_ns
    with pytest.raises(ValueError, match='bracket'):
        candidate.WallQPCBracket(1000, bracket.unix_ns, 2_000_000)


def test_worker_stamps_private_file_before_replace_and_preserves_payload(tmp_path):
    instance, clocks, target = writer(tmp_path)
    original = snapshot()
    instance.submit(original)
    record = instance.pump_one()
    assert record['status'] == 'published'
    assert target.read_bytes() == original.snapshot_json
    assert target.stat().st_mtime_ns == record['actual_mtime_ns']
    assert target.stat().st_mtime_ns <= record['source_unix_lower_ns']
    # A newly written target still exposes original snapshot age, not delivery time.
    assert target.stat().st_mtime_ns < clocks.now+clocks.offset-90_000_000
    assert record['source_clock'] == original.source_clock
    instance.close()


def test_blocked_old_write_fails_before_publication_without_refreshing_old_file(tmp_path):
    instance, clocks, target = writer(tmp_path)
    target.write_bytes(b'previous-original')
    previous_mtime = target.stat().st_mtime_ns

    def slow(path, payload):
        path.write_bytes(payload)
        clocks.now += 10_000_000_000

    instance.write_private = slow
    instance.submit(snapshot())
    with pytest.raises(ValueError, match='too old'):
        instance.pump_one()
    assert target.read_bytes() == b'previous-original'
    assert target.stat().st_mtime_ns == previous_mtime
    assert instance.records[-1]['status'] == 'failed'
    instance.close()


def test_one_flight_latest_pending_and_no_regressing_callback(tmp_path):
    instance, clocks, target = writer(tmp_path)

    def while_in_flight(path, payload):
        path.write_bytes(payload)
        instance.submit(snapshot(2, 10.11))
        instance.submit(snapshot(3, 10.12))
        with pytest.raises(RuntimeError, match='one heartbeat'):
            instance.pump_one()

    instance.write_private = while_in_flight
    instance.submit(snapshot())
    instance.pump_one()
    assert instance.queued_replaced == 1 and instance.pending.sequence == 3
    instance.write_private = instance._write_private
    clocks.now += 10_000_000
    instance.pump_one()
    assert instance.last_committed_sequence == 3
    assert json.loads(target.read_bytes())['script_clock'] == 10.12
    with pytest.raises(ValueError, match='regressing'):
        instance.submit(snapshot(2, 10.11))
    instance.close()


def test_retire_fences_old_generation_before_publication_and_lease_takeover(tmp_path):
    instance, _, target = writer(tmp_path)

    def retire(path, payload):
        path.write_bytes(payload)
        instance.retire()

    instance.write_private = retire
    instance.submit(snapshot())
    with pytest.raises(RuntimeError, match='Retired'):
        instance.pump_one()
    assert not target.exists()
    second = candidate.OwnedWriterLease(target, owner_id='b'*32, generation='new-generation')
    with pytest.raises(FileExistsError):
        second.acquire()
    instance.close()
    second.acquire()
    second.release()


@pytest.mark.parametrize('failure', ['stamp', 'replace'])
def test_stamp_or_rename_failure_never_forges_success_or_resamples(tmp_path, failure):
    instance, _, target = writer(tmp_path)
    target.write_bytes(b'previous')

    def fail(*args):
        raise OSError('synthetic '+failure+' failure')

    if failure == 'stamp':
        instance.stamp_private = fail
    else:
        instance.replace = fail
    instance.submit(snapshot())
    with pytest.raises(OSError, match='synthetic'):
        instance.pump_one()
    assert target.read_bytes() == b'previous'
    assert instance.records[-1]['source_clock'] == 10.1
    assert instance.retired
    instance.close()


def test_filesystem_rounding_into_future_never_publishes(tmp_path):
    instance, _, target = writer(tmp_path)
    instance.stamp_private = lambda path, lower: None  # leaves new-delivery mtime, unsafe
    instance.submit(snapshot())
    with pytest.raises(ValueError, match='rounded later'):
        instance.pump_one()
    assert not target.exists() and instance.retired
    instance.close()


def test_old_snapshot_cannot_be_refreshed_by_queue_delivery_time(tmp_path):
    instance, _, target = writer(tmp_path)
    instance.submit(snapshot(source_clock=1.0))
    with pytest.raises(ValueError, match='too old'):
        instance.pump_one()
    assert not target.exists()
    assert instance.records[-1]['source_clock'] == 1.0
    instance.close()


def test_inflight_close_does_not_release_owned_lease_early(tmp_path):
    instance, _, target = writer(tmp_path)

    def writing(path, content):
        path.write_bytes(content)
        with pytest.raises(RuntimeError, match='in-flight'):
            instance.close()
        competitor = candidate.OwnedWriterLease(target, owner_id='b'*32, generation='new')
        with pytest.raises(FileExistsError):
            competitor.acquire()

    instance.write_private = writing
    instance.submit(snapshot())
    instance.pump_one()
    instance.close()


def test_finite_loop_has_no_thread_creation_and_retires_same_owned_writer(tmp_path, monkeypatch):
    instance, clocks, _ = writer(tmp_path)

    def step(seconds):
        clocks.now += round(seconds*1e9)

    monkeypatch.setattr(candidate.time, 'sleep', step)
    instance.run_finite(lambda: None, seconds=.03)
    assert instance.retired and not instance.records
    instance.close()


def test_wall_clock_step_invalidates_publication_and_keeps_previous_snapshot(tmp_path):
    instance, clocks, target = writer(tmp_path)
    instance.submit(snapshot())
    instance.pump_one()
    original, stamp = target.read_bytes(), target.stat().st_mtime_ns
    clocks.now += 100_000_000
    clocks.offset += 5_000_000_000
    instance.submit(snapshot(2, 10.2))
    with pytest.raises(ValueError, match='clock step'):
        instance.pump_one()
    assert target.read_bytes() == original and target.stat().st_mtime_ns == stamp
    instance.close()


def test_source_expiry_future_clock_and_finite_attempt_budget_fail_closed(tmp_path):
    instance, _, target = writer(tmp_path, maximum_attempts=1)
    instance.submit(snapshot())
    instance.pump_one()
    instance.submit(snapshot(2, 10.11))
    with pytest.raises(RuntimeError, match='budget'):
        instance.pump_one()
    assert json.loads(target.read_bytes())['script_clock'] == 10.1
    instance.close()
    with pytest.raises(ValueError, match='future'):
        calibration().source_host_lower_ns(99., GENERATION, 100_200_000_000)


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class FakeKernel:
    def __init__(self, *, exists_at=None):
        self.error, self.created, self.closed, self.unmapped = 0, [], [], []
        self.exists_at, self.mapping_flags = exists_at, []
        self.CreateFileMappingW = Function(self.create)
        self.MapViewOfFile = Function(self.map)
        self.UnmapViewOfFile = Function(lambda address: self.unmapped.append(address) or 1)
        self.CloseHandle = Function(lambda handle: self.closed.append(handle) or 1)

    def create(self, *args):
        self.error = 183 if len(self.created) == self.exists_at else 0
        self.created.append(args[-1])
        return len(self.created)

    def map(self, handle, flags, *args):
        self.mapping_flags.append(flags)
        return handle*100


@pytest.mark.parametrize('existing', [0, 1])
def test_exclusive_map_creation_rejects_existing_objects_without_reset(
        tmp_path, monkeypatch, existing):
    instance, _, _ = writer(tmp_path)
    kernel = FakeKernel(exists_at=existing)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: kernel)
    monkeypatch.setattr(ctypes, 'set_last_error', lambda value: None)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: kernel.error)
    monkeypatch.setattr(ctypes, 'string_at', lambda address, size: bytes(size))
    channel = candidate.ExclusiveHeartbeatMaps(instance.lease)
    with pytest.raises(FileExistsError, match='already exists'):
        channel.create()
    assert len(kernel.closed) == existing+1 and not channel.handles
    instance.close()


def test_host_claims_two_zero_maps_and_reads_only_repeated_copies(tmp_path, monkeypatch):
    instance, _, _ = writer(tmp_path)
    kernel = FakeKernel()
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: kernel)
    monkeypatch.setattr(ctypes, 'set_last_error', lambda value: None)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: kernel.error)
    monkeypatch.setattr(ctypes, 'string_at', lambda address, size: bytes(size))
    channel = candidate.ExclusiveHeartbeatMaps(instance.lease)
    receipt = channel.create()
    assert receipt['exclusive_creation'] and receipt['host_created_slots']
    assert receipt['generation'] == GENERATION and kernel.mapping_flags == [4, 4]
    assert channel.repeated_copies() == [(bytes(candidate.SLOT_SIZE),)*2]*2
    channel.close()
    assert kernel.closed == [1, 2] and kernel.unmapped == [100, 200]
    instance.close()
