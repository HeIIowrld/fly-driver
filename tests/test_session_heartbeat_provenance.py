"""Private synthetic files and injected clocks, never the current session service."""

import ctypes
import hashlib
import json
import os

import pytest

from flybrain_drive import session_heartbeat_provenance as provenance
from flybrain_drive.session_heartbeat_candidate import HeartbeatSnapshot

OWNER = 'a'*32
GEN = 'synthetic-generation'
META = dict(domain=provenance.QPC_DOMAIN, callable='time.perf_counter_ns',
            implementation='QueryPerformanceCounter()', monotonic=True,
            adjustable=False, resolution_seconds=1e-7)


def prepared(tmp_path, *, source_lower=100_100_000_000):
    target = tmp_path/'status.json'
    lease = provenance.ProvenanceWriterLease(target, owner_id=OWNER, generation=GEN,
        calibration_sha256='b'*64, registered_qpc_ns=100_000_000_000,
        deadline_qpc_ns=160_000_000_000, clock_metadata=META).acquire()
    content = json.dumps(dict(schema='flybrain.csp.session.status.v1', generation=GEN,
        script_clock=10.1, ai_controlled=False, paused=True)).encode()
    snapshot = HeartbeatSnapshot(OWNER, GEN, 1, 10.1, content)
    provenance.write_immutable_proof(lease, snapshot, source_lower)
    target.write_bytes(content)
    return target, lease, snapshot


def validate(target, snapshot, **kwargs):
    return provenance.validate_opened_status(target, snapshot.snapshot_json,
        json.loads(snapshot.snapshot_json), now_qpc_ns=100_200_000_000,
        clock_metadata=META, **kwargs)


def test_no_lease_preserves_original_default_and_requires_no_windows_api(tmp_path, monkeypatch):
    target = tmp_path/'status.json'
    target.write_text('{"paused":true}')
    monkeypatch.setattr(provenance, 'qpc_clock_metadata',
                        lambda: pytest.fail('default read must not require new clock API'))
    value, modified, proof = provenance.read_proven_status(target)
    assert value == dict(paused=True) and modified == target.stat().st_mtime and proof is None


def test_exact_hash_bound_provenance_uses_qpc_age_not_filemtime(tmp_path):
    target, lease, snap = prepared(tmp_path)
    result = validate(target, snap)
    assert result['source_age_upper_seconds'] == pytest.approx(.1)
    assert result['snapshot_sha256'] == hashlib.sha256(snap.snapshot_json).hexdigest()
    lease.release()


def test_supplied_value_cannot_change_authority_fields_while_reusing_original_hash(tmp_path):
    target, lease, snap = prepared(tmp_path)
    value = json.loads(snap.snapshot_json)
    value['ai_controlled'] = True
    with pytest.raises(provenance.HeartbeatProvenanceError, match='opened bytes'):
        provenance.validate_opened_status(target, snap.snapshot_json, value,
            now_qpc_ns=100_200_000_000, clock_metadata=META)
    lease.release()


def test_backward_wall_jump_cannot_make_stale_snapshot_qpc_fresh(tmp_path):
    target, lease, snap = prepared(tmp_path, source_lower=97_000_000_000)
    os.utime(target, None)  # fresh mtime would fool the old wall-age-only gate
    with pytest.raises(provenance.HeartbeatProvenanceError, match='monotonic age'):
        validate(target, snap)
    lease.release()


def test_old_open_descriptor_keeps_matching_immutable_proof_during_next_publication(tmp_path):
    target, lease, old = prepared(tmp_path)
    captured = provenance.capture_lease(target)
    content = json.dumps(dict(schema='flybrain.csp.session.status.v1', generation=GEN,
                             script_clock=10.15, ai_controlled=True)).encode()
    newer = HeartbeatSnapshot(OWNER, GEN, 2, 10.15, content)
    provenance.write_immutable_proof(lease, newer, 100_150_000_000)
    target.write_bytes(content)
    assert validate(target, old, captured_lease=captured)['sequence'] == 1
    assert validate(target, newer)['sequence'] == 2
    lease.release()


def test_retirement_cannot_remove_validation_of_previously_opened_descriptor(tmp_path):
    target, lease, snap = prepared(tmp_path, source_lower=97_000_000_000)
    captured = provenance.capture_lease(target)
    lease.release()
    with pytest.raises(provenance.HeartbeatProvenanceError, match='monotonic age'):
        validate(target, snap, captured_lease=captured)


@pytest.mark.parametrize('key,value', [
    ('owner_id', 'c'*32), ('generation', 'other'), ('lease_sha256', 'd'*64),
    ('calibration_sha256', 'd'*64), ('clock_domain', 'time.monotonic'),
    ('expires_qpc_ns', 170_000_000_000), ('original_source_clock', 99.),
])
def test_foreign_or_rebased_proof_fails_closed(tmp_path, key, value):
    target, lease, snap = prepared(tmp_path)
    path = provenance.proof_path(target, OWNER, hashlib.sha256(snap.snapshot_json).hexdigest())
    proof = json.loads(path.read_bytes())
    proof[key] = value
    path.write_text(json.dumps(proof))
    with pytest.raises(provenance.HeartbeatProvenanceError):
        validate(target, snap)
    lease.release()


def test_different_serialized_bytes_cannot_use_old_semantic_json_proof(tmp_path):
    target, lease, snap = prepared(tmp_path)
    content = json.dumps(json.loads(snap.snapshot_json), indent=2).encode()
    with pytest.raises(provenance.HeartbeatProvenanceError, match='unavailable'):
        provenance.validate_opened_status(target, content, json.loads(content),
            now_qpc_ns=100_200_000_000, clock_metadata=META)
    lease.release()


def test_conflicting_immutable_provenance_cannot_be_overwritten(tmp_path):
    target, lease, snap = prepared(tmp_path)
    with pytest.raises(provenance.HeartbeatProvenanceError, match='already differs'):
        provenance.write_immutable_proof(lease, snap, 99_000_000_000)
    lease.release()


class Function:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def test_precise_filetime_conversion_and_qpc_bracket_have_no_coarse_fallback():
    from ctypes import wintypes

    desired_unix_ns = 1_700_000_000_000_000_100
    ticks = provenance.FILETIME_UNIX_EPOCH_TICKS+desired_unix_ns//100

    def read(pointer):
        value = ctypes.cast(pointer, ctypes.POINTER(wintypes.FILETIME)).contents
        value.dwLowDateTime, value.dwHighDateTime = ticks & 0xffffffff, ticks >> 32

    kernel = type('Kernel', (), {})()
    kernel.GetSystemTimePreciseAsFileTime = Function(read)
    points = iter([100_000_000_000, 100_000_000_100])
    clock = provenance.PreciseWindowsWallClock(kernel=kernel, qpc_clock=lambda: next(points))
    bracket = clock.bracket()
    assert bracket.unix_ns == desired_unix_ns
    assert bracket.qpc_after_ns-bracket.qpc_before_ns == 100
    assert clock.read_clock.restype is None
