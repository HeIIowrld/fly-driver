"""Three migrated readers use original bytes and fail closed on private leases."""

import json
import time

import pytest

from flybrain_drive.ac_expert_handoff import OwnedExpertHandoff
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.session_heartbeat_candidate import HeartbeatSnapshot
from flybrain_drive.session_heartbeat_provenance import (
    HeartbeatProvenanceError,
    ProvenanceWriterLease,
    qpc_clock_metadata,
    write_immutable_proof,
)
from scripts.record_ac_engine_ai_probe import read_ai_phase


def prepare(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    root = tmp_path/'flybrain/csp-session'
    root.mkdir(parents=True)
    target = root/'status.json'
    now = time.perf_counter_ns()
    lease = ProvenanceWriterLease(target, owner_id='a'*32, generation='synthetic',
        calibration_sha256='b'*64, registered_qpc_ns=now-100_000_000,
        deadline_qpc_ns=now+59_000_000_000, clock_metadata=qpc_clock_metadata()).acquire()
    value = dict(schema='flybrain.csp.session.status.v1', generation='synthetic',
        script_clock=10., epoch=3, ai_controlled=True, paused=False)
    snapshot = HeartbeatSnapshot('a'*32, 'synthetic', 1, 10., json.dumps(value).encode())
    write_immutable_proof(lease, snapshot, now-50_000_000)
    target.write_bytes(snapshot.snapshot_json)
    return target, lease, value


def test_common_controller_and_custom_ai_and_handoff_readers_all_require_same_proof(
        tmp_path, monkeypatch):
    path, lease, original = prepare(tmp_path, monkeypatch)
    assert read_published_json(path)[0] == original
    assert read_ai_phase(path, 'synthetic') is True
    assert OwnedExpertHandoff(None, tmp_path/'out').status() == original
    changed = dict(original, ai_controlled=False)
    path.write_text(json.dumps(changed), encoding='utf8')
    readers = [lambda: read_published_json(path),
               lambda: read_ai_phase(path, 'synthetic', timeout=.01),
               lambda: OwnedExpertHandoff(None, tmp_path/'out').status()]
    for read in readers:
        with pytest.raises(HeartbeatProvenanceError):
            read()
    lease.release()


def test_partial_lease_parse_is_inside_common_bounded_retry(tmp_path, monkeypatch):
    from flybrain_drive import session_heartbeat_provenance as module

    path = tmp_path/'status.json'
    path.write_text('{"paused":true}')
    original, count = module.capture_lease, [0]

    def partial_once(target):
        count[0] += 1
        if count[0] == 1:
            raise HeartbeatProvenanceError('Incomplete ownership JSON')
        return original(target)

    monkeypatch.setattr(module, 'capture_lease', partial_once)
    assert module.read_proven_status(path)[0] == dict(paused=True)
    assert count[0] >= 2
