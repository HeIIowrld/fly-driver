"""Exact synthetic buffers/private files only; no game/MMF/actor or worker start."""

import ctypes
import hashlib
import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from flybrain_drive import session_heartbeat_evidence_v1 as evidence
from flybrain_drive import session_heartbeat_runtime as core
from flybrain_drive.session_heartbeat_candidate import (
    SLOT_SIZE,
    CachedClockLowerBound,
    HeartbeatSnapshot,
    WallQPCBracket,
    encode_slot,
)

OWNER, GEN = 'a'*32, 'private-fixture'
META = dict(domain=core.QPC_DOMAIN, callable='time.perf_counter_ns',
    implementation='QueryPerformanceCounter()', monotonic=True,
    adjustable=False, resolution_seconds=1e-7)


def sample(sequence=1, source=10.1):
    body = dict(schema='flybrain.csp.session.status.v1', generation=GEN,
        epoch=3, script_clock=source, paused=True, ai_controlled=False,
        speed_kmh=0., gas=0., brake=1., camera_active=False,
        expert_probe_active=False, brake_settle_active=False)
    # Deliberate original whitespace/order must survive without reserialization.
    return HeartbeatSnapshot(OWNER, GEN, sequence, source,
                             json.dumps(body, indent=3).encode())


def test_original_read_pairs_keep_odd_torn_and_duplicates_without_disk(monkeypatch):
    ledger = evidence.HeartbeatEvidence()
    raw = encode_slot(sample())
    odd = raw[:8]+struct.pack('<I', 1)+raw[12:]
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('No evidence disk I/O'))
    ledger.observe_pair('data1', odd, raw)
    ledger.observe_pair('data1', raw, raw)
    ledger.observe_pair('data1', raw, raw)
    assert len(ledger.reads['data1']) == 3 and len(ledger.blobs) == 2
    assert ledger.blobs[ledger.reads['data1'][0][1]] == odd
    assert ledger.blobs[ledger.reads['data1'][0][2]] == raw
    assert ledger.reads['data1'][1][1:] == ledger.reads['data1'][2][1:]


def test_data_factory_calls_old_read_once_and_returns_unchanged_original(monkeypatch):
    raw = encode_slot(sample())
    pairs, calls = [(raw, raw), (raw, raw)], []
    monkeypatch.setattr(evidence.ExclusiveHeartbeatMaps, 'repeated_copies',
                        lambda self: (calls.append(1) or pairs))
    wrapper = evidence.EvidenceHeartbeatMaps(SimpleNamespace(), evidence.HeartbeatEvidence())
    assert wrapper.repeated_copies() is pairs and calls == [1]


def lease_stub():
    return SimpleNamespace(owner_id=OWNER, generation=GEN, payload_sha256='b'*64,
                           verify=lambda: None)


def test_control_reads_exactly_two_original_copies_then_same_decoder(monkeypatch):
    lease = lease_stub()
    message = dict(schema=core.CONTROL_SCHEMA, owner_id=OWNER, generation=GEN,
        lease_sha256=lease.payload_sha256, phase='sync_return')
    raw = core.encode_control(message, 4)
    calls, decodes = [], []
    original = core.decode_control

    def read(view, size):
        calls.append((view, size))
        return raw

    def decode(first, second, **kwargs):
        decodes.append((first, second))
        return original(first, second, **kwargs)

    monkeypatch.setattr(evidence.ctypes, 'string_at', read)
    monkeypatch.setattr(core, 'decode_control', decode)
    wrapper = evidence.EvidenceControlMaps(lease, evidence.HeartbeatEvidence())
    wrapper.views = [0, 'private-buffer']
    assert wrapper.receive() == (4, message)
    assert calls == [('private-buffer', core.CONTROL_SIZE)]*2 and decodes == [(raw, raw)]
    assert wrapper.last_pair == (raw, raw)


def test_host_send_retains_original_encode_buffer_and_identical_core_writes():
    lease, ledger = lease_stub(), evidence.HeartbeatEvidence()
    normal, wrapped = core.ExclusiveControlMaps(lease), evidence.EvidenceControlMaps(lease, ledger)
    buffers = [ctypes.create_string_buffer(core.CONTROL_SIZE) for _ in range(4)]
    normal.views = [ctypes.addressof(b) for b in buffers[:2]]
    wrapped.views = [ctypes.addressof(b) for b in buffers[2:]]
    value = dict(schema=core.CONTROL_SCHEMA, owner_id=OWNER, generation=GEN,
        lease_sha256='b'*64, command='writer_fenced', committed_sequence=1,
        committed_source_clock=10.1)
    normal.send(value)
    wrapped.send(value)
    original, retained = buffers[0].raw, ledger.blobs[ledger.sends[0]['sha256']]
    assert buffers[2].raw == original == retained
    assert normal.sequence == wrapped.sequence == 1 and ledger.sends[0]['completed']


def test_capture_bounds_are_explicit_and_never_change_returned_core_data(monkeypatch):
    ledger = evidence.HeartbeatEvidence()
    monkeypatch.setattr(evidence, 'MAX_READS_PER_CHANNEL', 2)
    monkeypatch.setattr(evidence, 'VARIANT_LIMITS', dict(evidence.VARIANT_LIMITS, data=1))
    one, two = encode_slot(sample()), encode_slot(sample(2, 10.2))
    ledger.observe_pair('data1', one, one)
    ledger.observe_pair('data1', two, two)
    ledger.observe_pair('data1', two, two)
    assert len(ledger.reads['data1']) == 2 and len(ledger.blobs) == 1
    assert ledger.flags['raw_data_budget_exhausted'] and ledger.flags['data1_read_budget_exhausted']


class PrivateData(evidence.EvidenceHeartbeatMaps):
    def create(self):
        self.buffers = [ctypes.create_string_buffer(SLOT_SIZE) for _ in range(2)]
        self.views = [ctypes.addressof(b) for b in self.buffers]
        return dict(exclusive_creation=True, host_created_slots=True)

    def close(self):
        self.views = []


class PrivateControl(evidence.EvidenceControlMaps):
    def create(self):
        self.buffers = [ctypes.create_string_buffer(core.CONTROL_SIZE) for _ in range(2)]
        self.views = [ctypes.addressof(b) for b in self.buffers]
        return self

    def close(self):
        self.views = []


def prepared(tmp_path):
    clock = [100_200_000_000]
    record = dict(generation=GEN, exchanges=[
        dict(sent=100., received=100.00001, producer_clock=10.),
        dict(sent=100.01, received=100.01001, producer_clock=10.01)],
        offset_lower_seconds=90., offset_upper_seconds=90.00001)
    calibration = CachedClockLowerBound.from_record(record, generation=GEN)
    ledger = evidence.HeartbeatEvidence(qpc_clock=lambda: clock[0])
    ledger.capture_calibration(record)
    ctx = core.HeartbeatSessionContext(tmp_path/'status.json', None,
        calibration=calibration, clock_metadata=META, owner_id=OWNER,
        qpc_clock=lambda: clock[0],
        bracket_clock=lambda: WallQPCBracket(clock[0]-100,
            1_700_000_000_000_000_000, clock[0]),
        map_factory=lambda lease: (ledger.capture_lease(lease) or None),
        control_factory=lambda lease: PrivateControl(lease, ledger))

    def maps(lease):
        ledger.capture_lease(lease)
        return PrivateData(lease, ledger)

    ctx.map_factory = maps
    ctx.prepare()
    plan = ctx.arm_plan(epoch=3)
    plan['host_ready'] = True  # synthetic host lifecycle, no threads/OS maps
    arm = tmp_path/'arm.json'
    arm.write_bytes(json.dumps(plan, indent=3).encode())
    ledger.capture_arm(arm)
    first = sample()
    ctypes.memmove(ctx.maps.views[1], encode_slot(first), SLOT_SIZE)
    ctx.poll_once()
    ctx.writer.pump_one()
    final = dict(schema=core.CONTROL_SCHEMA, owner_id=OWNER, generation=GEN,
        lease_sha256=ctx.lease.payload_sha256, phase='final', final_sequence=1,
        final_source_clock=first.source_clock)
    ctypes.memmove(ctx.control.views[1], core.encode_control(final, 2), core.CONTROL_SIZE)
    ctx.poll_once()
    ctx.fence_writer()
    clock[0] += 200_000_000
    sync = sample(2, 10.3).snapshot_json
    ctx.target.write_bytes(sync)
    ack = dict(schema=core.CONTROL_SCHEMA, owner_id=OWNER, generation=GEN,
        lease_sha256=ctx.lease.payload_sha256, phase='sync_return',
        sync_crc32=zlib.crc32(sync), sync_length=len(sync), sync_source_clock=10.3)
    raw_ack = core.encode_control(ack, 4)
    ctypes.memmove(ctx.control.views[1], raw_ack, core.CONTROL_SIZE)
    ctx.control.receive()
    return ledger, ctx, sync, raw_ack


def test_original_sync_and_raw_ack_bind_to_actual_unchanged_core_handoff(tmp_path):
    ledger, context, sync, ack = prepared(tmp_path)
    witness = ledger.capture_sync(context)
    handoff = context.complete_sync_return()  # real frozen core, private files/RAM only
    ledger.bind_handoff(handoff)
    assert witness['status_sha256'] == handoff['status_sha256'] == hashlib.sha256(sync).hexdigest()
    assert ledger.blobs[witness['ack_first_sha256']] == ack
    result = ledger.persist(tmp_path/'evidence', context=context)
    index = json.loads(Path(result['index']).read_bytes())
    assert result['status'] == 'complete' and index['actual_handoff']['lease_removed']
    original_body = sample().snapshot_json
    ref = index['snapshot_json_sha256']['1']
    assert (tmp_path/'evidence'/index['blobs'][ref]['file']).read_bytes() == original_body
    assert index['sync_witness']['ack_value']['sync_crc32'] == zlib.crc32(sync)


def test_changed_frozen_status_or_wrong_actual_hash_cannot_be_accepted(tmp_path):
    ledger, context, sync, _ = prepared(tmp_path)
    context.target.write_bytes(sync+b' ')
    with pytest.raises(ValueError, match='SYNC status bytes'):
        ledger.capture_sync(context)
    context.target.write_bytes(sync)
    ledger.capture_sync(context)
    with pytest.raises(ValueError, match='actual handoff'):
        ledger.bind_handoff(dict(status_sha256='0'*64, same_writer_joined=True, lease_removed=True))
    context.lease.release()  # private fixture only, no producer/worker exists


def test_live_owned_thread_defers_persistence_until_terminal_without_new_worker(tmp_path):
    ledger = evidence.HeartbeatEvidence()
    alive, callbacks = [True], []
    thread = SimpleNamespace(name='same-owned-fixture', is_alive=lambda: alive[0])
    ctx = SimpleNamespace(poll_thread=thread, write_thread=None, state='failed_fenced', writer=None)
    with pytest.raises(evidence.EvidenceDeferred):
        ledger.persist(tmp_path/'trace', context=ctx)
    assert not (tmp_path/'trace').exists()
    report = ledger.defer_persistence(tmp_path/'trace', context=ctx,
        receipt_path=tmp_path/'terminal.json', register=callbacks.append)
    assert report['status'] == 'deferred_owned_threads_live' and len(callbacks) == 1
    callbacks[0]()
    assert not (tmp_path/'trace').exists()
    alive[0] = False
    callbacks[0]()
    assert json.loads((tmp_path/'terminal.json').read_bytes())['status'] == 'partial'


def test_new_runner_is_exact_derivative_and_defaults_to_no_live_work(monkeypatch, capsys):
    import sys

    from scripts import qualify_session_heartbeat_evidence as runner
    from scripts.prepare_session_heartbeat_evidence_runner import BASE, BASE_SHA, OUTPUT, build

    assert build() == OUTPUT.read_bytes()
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    monkeypatch.setattr(sys, 'argv', ['evidence', '--mode', 'camera3', '--expected-engine-pid', '1',
        '--expected-generation', GEN, '--expected-epoch', '3',
        '--expected-installed-sha256', '0'*64,
        '--output', 'not-created'])
    monkeypatch.setattr(runner, 'run', lambda args: pytest.fail('No default live action'))
    runner.main()
    assert json.loads(capsys.readouterr().out)['live_actions'] == 0
