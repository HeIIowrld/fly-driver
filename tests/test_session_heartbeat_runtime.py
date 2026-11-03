"""Synthetic clocks/files/handles only; never game, service or Windows MMF."""

import hashlib
import json
import zlib

import pytest

from flybrain_drive import session_heartbeat_runtime as runtime
from flybrain_drive.session_heartbeat_candidate import (
    CachedClockLowerBound,
    HeartbeatSnapshot,
    WallQPCBracket,
    encode_slot,
)
from flybrain_drive.session_heartbeat_provenance import capture_lease, read_proven_status

META = dict(domain=runtime.QPC_DOMAIN, callable='time.perf_counter_ns',
    implementation='QueryPerformanceCounter()', monotonic=True,
    adjustable=False, resolution_seconds=1e-7)
OWNER, GEN = 'a'*32, 'fixture-engine'


class Maps:
    def __init__(self, lease):
        self.copies, self.closed = [(bytes(8212), bytes(8212))]*2, False

    def create(self):
        return dict(host_created_slots=True, exclusive_creation=True)

    def repeated_copies(self):
        return self.copies

    def close(self):
        self.closed = True


class Control:
    def __init__(self, lease):
        self.messages, self.ack, self.closed = [], None, False

    def create(self):
        return self

    def send(self, value):
        self.messages.append(value)

    def receive(self):
        return self.ack

    def close(self):
        self.closed = True


def context(tmp_path):
    clock = [100_200_000_000]
    calibration = CachedClockLowerBound(GEN, 90., 90.00001, 0.,
        100_000_000_000, 60_000_000_000, 'b'*64)
    ctx = runtime.HeartbeatSessionContext(tmp_path/'status.json', None,
        owner_id=OWNER, qpc_clock=lambda: clock[0], calibration=calibration,
        clock_metadata=META, map_factory=Maps, control_factory=Control,
        bracket_clock=lambda: WallQPCBracket(clock[0]-100, 1_700_000_000_000_000_000, clock[0]))
    return ctx, clock


def snapshot(sequence=1, source=10.1, **extra):
    body = dict(schema='flybrain.csp.session.status.v1', generation=GEN,
        script_clock=source, epoch=3, ai_controlled=False, paused=True,
        speed_kmh=0., gas=0., brake=1.,
        camera_active=False, expert_probe_active=False, brake_settle_active=False)
    body.update(extra)
    return HeartbeatSnapshot(OWNER, GEN, sequence, source, json.dumps(body).encode())


def final_ack(ctx, snap):
    ctx.last_ack = (2, dict(phase='final', final_sequence=snap.sequence,
                            final_source_clock=snap.source_clock))


def test_construct_has_no_files_maps_clock_handshake_threads_or_workers(tmp_path):
    ctx, _ = context(tmp_path)
    assert ctx.state == 'new' and list(tmp_path.iterdir()) == []
    assert ctx.maps is ctx.control is ctx.poll_thread is ctx.write_thread is None


def test_host_ready_event_follows_actual_initial_control_publication(tmp_path):
    ctx, _ = context(tmp_path)
    ctx.prepare()
    ctx.arm_plan(epoch=3)
    original_send = ctx.control.send

    def observe(value):
        assert not ctx.ready_event.is_set() and ctx.plan['host_ready'] is False
        original_send(value)
        ctx.stop_event.set()  # terminate this private synchronous poll fixture

    ctx.control.send = observe
    ctx._poll()  # no threads or OS mappings; same actual send path
    assert ctx.ready_event.is_set() and len(ctx.control.messages) == 1
    assert ctx.plan['host_ready'] is False
    ctx.lease.release()


def test_prepare_is_finite_and_rejects_extension(tmp_path):
    ctx, clock = context(tmp_path)
    plan = ctx.prepare()
    deadline = plan['deadline_qpc_ns']
    assert deadline == 160_000_000_000
    clock[0] += 16_000_000_000
    with pytest.raises(ValueError, match='never extend'):
        ctx.arm_plan(epoch=3)
    assert ctx.lease.registration['deadline_qpc_ns'] == deadline
    ctx.lease.release()


def test_original_source_proof_precedes_publication_and_commit_ack(tmp_path):
    ctx, _ = context(tmp_path)
    ctx.prepare()
    ctx.arm_plan(epoch=3)
    snap = snapshot()
    raw = encode_slot(snap)
    ctx.maps.copies[1] = (raw, raw)
    assert ctx.poll_once() == snap
    assert ctx._message()['committed_sequence'] == 0
    assert 'committed_source_clock' not in ctx._message()
    result = ctx.writer.pump_one()
    value, _, proof = read_proven_status(ctx.target, qpc_clock=ctx.qpc_clock, clock_metadata=META)
    assert value == json.loads(snap.snapshot_json) and proof['sequence'] == 1
    assert result['status'] == 'published'
    assert ctx._message()['committed_source_clock'] == snap.source_clock
    assert ctx.target.stat().st_mtime_ns <= (result['source_unix_lower_ns']
                                            -ctx.writer.legacy_wall_quantization_ns)
    ctx.lease.release()


def test_no_lease_removal_until_same_writer_fenced_and_fresh_sync_committed(tmp_path):
    ctx, clock = context(tmp_path)
    ctx.prepare()
    ctx.arm_plan(epoch=3)
    old = snapshot()
    ctx.writer.submit(old)
    ctx.writer.pump_one()
    final_ack(ctx, old)
    ctx.fence_writer()
    assert capture_lease(ctx.target) is not None
    with pytest.raises(RuntimeError, match='return acknowledgement'):
        ctx.complete_sync_return()
    assert capture_lease(ctx.target) is not None
    clock[0] += 200_000_000
    sync = snapshot(2, 10.3).snapshot_json
    ctx.target.write_bytes(sync)
    ctx.control.ack = (3, dict(phase='sync_return', sync_crc32=zlib.crc32(sync),
        sync_length=len(sync), sync_source_clock=10.3))
    result = ctx.complete_sync_return()
    assert result['status_sha256'] == hashlib.sha256(sync).hexdigest()
    assert ctx.state == 'closed' and capture_lease(ctx.target) is None
    assert ctx.maps.closed and ctx.control.closed
    # A reader starting after release sees the fresh synchronous bytes.
    assert read_proven_status(ctx.target)[0]['script_clock'] == 10.3


class BlockedThread:
    def join(self, _):
        pass

    def is_alive(self):
        return True


def test_blocked_old_replace_cannot_release_lease_or_resume_sync(tmp_path):
    ctx, _ = context(tmp_path)
    ctx.prepare()
    old = snapshot()
    ctx.writer.submit(old)
    ctx.writer.pump_one()
    final_ack(ctx, old)
    ctx.write_thread = BlockedThread()
    with pytest.raises(TimeoutError, match='not terminal'):
        ctx.fence_writer(join_timeout=.001)
    assert ctx.state == 'failed_fenced' and capture_lease(ctx.target) is not None
    assert all(x['command'] != 'writer_fenced' for x in ctx.control.messages)
    ctx.lease.release()  # private fixture only; no OS writer exists


def test_bad_sync_hash_or_authority_cannot_remove_guard(tmp_path):
    ctx, clock = context(tmp_path)
    ctx.prepare()
    ctx.arm_plan(epoch=3)
    old = snapshot()
    ctx.writer.submit(old)
    ctx.writer.pump_one()
    final_ack(ctx, old)
    ctx.fence_writer()
    clock[0] += 200_000_000
    bad = snapshot(2, 10.3, ai_controlled=True).snapshot_json
    ctx.target.write_bytes(bad)
    ctx.control.ack = (3, dict(phase='sync_return', sync_crc32=zlib.crc32(bad),
        sync_length=len(bad), sync_source_clock=10.3))
    with pytest.raises(ValueError, match='quiescence'):
        ctx.complete_sync_return()
    assert capture_lease(ctx.target) is not None
    ctx.lease.release()


def test_control_codec_preserves_original_commit_clock_and_rejects_torn_foreign_crc():
    message = dict(schema=runtime.CONTROL_SCHEMA, owner_id=OWNER, generation=GEN,
        lease_sha256='c'*64, command='run', committed_sequence=1,
        committed_source_clock=10.1)
    raw = runtime.encode_control(message, 1)
    kwargs = dict(owner_id=OWNER, generation=GEN, lease_sha256='c'*64)
    assert runtime.decode_control(raw, raw, **kwargs) == (1, message)
    assert runtime.decode_control(raw, bytes(len(raw)), **kwargs) is None
    changed = raw[:25]+bytes([raw[25] ^ 1])+raw[26:]
    with pytest.raises(ValueError, match='checksum'):
        runtime.decode_control(changed, changed, **kwargs)
    with pytest.raises(ValueError, match='Foreign'):
        runtime.decode_control(raw, raw, **dict(kwargs, generation='new'))


def test_non_qpc_calibration_cannot_enter_production(monkeypatch):
    monkeypatch.setattr(runtime, 'qpc_clock_metadata', lambda: META)
    with pytest.raises(ValueError, match='explicitly use'):
        runtime.calibration_from_qpc_record(dict(host_clock_callable='time.monotonic'))


def test_poller_failure_during_old_write_cannot_emit_a_late_run_ack(tmp_path):
    ctx, _ = context(tmp_path)
    ctx.prepare()
    ctx.state = 'running'

    def finishing_old_write():
        ctx._fail(RuntimeError('poller detected corruption while write blocked'))
        return dict(status='published')

    ctx.writer.pump_one = finishing_old_write
    ctx._write()
    assert ctx.error and ctx.control.messages == []
    assert capture_lease(ctx.target) is not None
    ctx.lease.release()


def test_transient_windows_reader_denial_retries_only_same_stamped_part(tmp_path):
    import os

    ctx, _ = context(tmp_path)
    ctx.prepare()
    attempts = []

    def commit(part, target):
        attempts.append((part.read_bytes(), part.stat().st_mtime_ns))
        if len(attempts) == 1:
            error = PermissionError(13, 'fixture overlapping open reader')
            error.winerror = 5
            raise error
        os.replace(part, target)

    writer = runtime.ProvenanceLatestWriter(ctx.lease, ctx.calibration,
        qpc_clock=ctx.qpc_clock, bracket_clock=ctx.bracket_clock, replace=commit)
    writer.submit(snapshot())
    assert writer.pump_one()['status'] == 'published'
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert writer.rename_attempts == {1: 2}
    ctx.lease.release()


def test_expired_source_after_denial_cannot_be_retried_or_redated(tmp_path):
    ctx, clock = context(tmp_path)
    ctx.prepare()
    attempts = []

    def denied(part, target):
        attempts.append(1)
        clock[0] += 8_000_000
        error = PermissionError(13, 'fixture delayed old operation')
        error.winerror = 32
        raise error

    writer = runtime.ProvenanceLatestWriter(ctx.lease, ctx.calibration,
        qpc_clock=ctx.qpc_clock, bracket_clock=ctx.bracket_clock, replace=denied)
    lower = ctx.calibration.source_host_lower_ns(10.1, GEN, clock[0])
    clock[0] = lower+writer.max_source_age_ns-4_000_000
    writer.submit(snapshot())
    with pytest.raises(ValueError, match='source expired'):
        writer.pump_one()
    assert attempts == [1] and writer.last_committed_sequence == 0
    assert capture_lease(ctx.target) is not None
    ctx.lease.release()


def test_qpc_under_two_seconds_but_legacy_padding_over_budget_fails_before_overwrite(tmp_path):
    ctx, clock = context(tmp_path)
    ctx.prepare()
    ctx.target.write_bytes(b'original synchronous status must remain')
    lower = ctx.calibration.source_host_lower_ns(10.1, GEN, clock[0])
    clock[0] = lower+1_999_000_000
    assert ctx.writer.max_source_age_ns < 1_999_000_000
    ctx.writer.submit(snapshot())
    with pytest.raises(ValueError, match='too old'):
        ctx.writer.pump_one()
    assert ctx.target.read_bytes() == b'original synchronous status must remain'
    assert ctx.writer.last_committed_sequence == 0 and ctx.control.messages == []
    ctx.lease.release()


@pytest.mark.parametrize('reason', ['deadline', 'fenced'])
def test_held_reader_past_original_retry_budget_or_late_fence_cannot_commit(tmp_path, reason):
    ctx, clock = context(tmp_path)
    ctx.prepare()
    attempts, permitted = [], [True]

    def denied(part, target):
        attempts.append((part.read_bytes(), part.stat().st_mtime_ns))
        if reason == 'deadline':
            clock[0] += 11_000_000
        else:
            permitted[0] = False
        error = PermissionError(13, 'fixture held reader')
        error.winerror = 5
        raise error

    writer = runtime.ProvenanceLatestWriter(ctx.lease, ctx.calibration,
        qpc_clock=ctx.qpc_clock, bracket_clock=ctx.bracket_clock, replace=denied,
        commit_permitted=lambda: permitted[0])
    writer.submit(snapshot())
    with pytest.raises(PermissionError if reason == 'deadline' else RuntimeError):
        writer.pump_one()
    assert len(attempts) == 1 and writer.last_committed_sequence == 0
    assert capture_lease(ctx.target) is not None and not ctx.target.exists()
    ctx.lease.release()


def test_normal_reader_unlink_denial_retries_only_same_owned_lease(tmp_path):
    import os

    ctx, _ = context(tmp_path)
    ctx.prepare()
    contents, calls = ctx.lease.path.read_bytes(), []

    def unlink(path):
        calls.append(path.read_bytes())
        if len(calls) == 1:
            error = PermissionError(13, 'fixture held reader denies delete')
            error.winerror = 5
            raise error
        os.unlink(path)

    result = runtime.release_owned_lease_bounded(ctx.lease,
        qpc_clock=ctx.qpc_clock, unlink=unlink)
    assert result['unlinked'] is True and result['attempts'] == 2
    assert calls == [contents, contents] and not ctx.lease.path.exists()
    assert ctx.lease.fd is None


def test_reader_held_beyond_unlink_budget_retains_exact_owned_marker(tmp_path):
    ctx, clock = context(tmp_path)
    ctx.prepare()
    contents = ctx.lease.path.read_bytes()

    def denied(path):
        clock[0] += 11_000_000
        error = PermissionError(13, 'fixture retained reader')
        error.winerror = 32
        raise error

    with pytest.raises(PermissionError) as failure:
        runtime.release_owned_lease_bounded(ctx.lease, qpc_clock=ctx.qpc_clock, unlink=denied)
    assert failure.value.heartbeat_release_receipt['unlinked'] is False
    assert ctx.lease.path.read_bytes() == contents and ctx.lease.fd is None


def test_foreign_replacement_during_unlink_retry_is_never_removed(tmp_path):
    import os

    ctx, _ = context(tmp_path)
    ctx.prepare()
    foreign = tmp_path/'foreign.part'
    foreign.write_bytes(b'foreign lease must be retained')
    count = []

    def replaced(path):
        count.append(1)
        os.replace(foreign, path)
        error = PermissionError(13, 'fixture foreign owner replaced named file')
        error.winerror = 5
        raise error

    with pytest.raises(RuntimeError, match='Foreign lease'):
        runtime.release_owned_lease_bounded(ctx.lease, qpc_clock=ctx.qpc_clock, unlink=replaced)
    assert count == [1] and ctx.lease.path.read_bytes() == b'foreign lease must be retained'
