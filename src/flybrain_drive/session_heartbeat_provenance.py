"""QPC-aware heartbeat provenance; new module, no active reader is changed on import.

An active file lease makes a matching immutable, hash-named provenance record
mandatory. Old mtime-only operation remains unchanged when no lease is present.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .session_heartbeat_candidate import OwnedWriterLease, WallQPCBracket

LEASE_SCHEMA = 'flybrain.session_heartbeat.file_lease.v1'
PROOF_SCHEMA = 'flybrain.session_heartbeat.source_provenance.v1'
CALIBRATION_SCHEMA = 'flybrain.session_heartbeat.qpc_calibration.v1'
QPC_DOMAIN = 'host_perf_counter_QueryPerformanceCounter_ns'
FILETIME_UNIX_EPOCH_TICKS = 116444736000000000
MAX_PROOF_BYTES = 16384


class HeartbeatProvenanceError(ValueError):
    """No stale fallback; callers may retry only inside their original read deadline."""


def qpc_clock_metadata():
    info = time.get_clock_info('perf_counter')
    if os.name != 'nt' or 'QueryPerformanceCounter' not in info.implementation:
        raise ValueError('Explicit Windows perf_counter/QueryPerformanceCounter required')
    return dict(domain=QPC_DOMAIN, callable='time.perf_counter_ns',
                implementation=info.implementation, monotonic=info.monotonic,
                adjustable=info.adjustable, resolution_seconds=info.resolution)


class PreciseWindowsWallClock:
    """Explicit Kernel32 UTC API; no coarse time.time_ns fallback.

    The return unit is 100ns since1601, converted exactly to Unix nanoseconds.
    This is a wall clock, not an immunity to system-clock changes.
    """

    def __init__(self, *, kernel=None, qpc_clock=time.perf_counter_ns):
        from ctypes import wintypes

        self.FILETIME = wintypes.FILETIME
        self.kernel = kernel or ctypes.WinDLL('kernel32', use_last_error=True)
        self.read_clock = self.kernel.GetSystemTimePreciseAsFileTime
        self.read_clock.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
        self.read_clock.restype = None
        self.qpc_clock = qpc_clock

    def unix_ns(self):
        value = self.FILETIME()
        self.read_clock(ctypes.byref(value))
        ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
        if ticks <= FILETIME_UNIX_EPOCH_TICKS:
            raise ValueError('Precise Windows FILETIME epoch differs')
        return (ticks-FILETIME_UNIX_EPOCH_TICKS)*100

    def bracket(self):
        before = self.qpc_clock()
        wall = self.unix_ns()
        return WallQPCBracket(before, wall, self.qpc_clock())


def lease_path(target):
    target = Path(target)
    return target.with_name(target.name+'.flybrain-heartbeat-owner')


def proof_path(target, owner_id, snapshot_sha256):
    if not re.fullmatch('[0-9a-f]{32}', owner_id or ''):
        raise HeartbeatProvenanceError('Exact owner ID required for heartbeat provenance')
    if not re.fullmatch('[0-9a-f]{64}', snapshot_sha256 or ''):
        raise HeartbeatProvenanceError('Exact snapshot hash required for heartbeat provenance')
    return Path(target).parent/'.flybrain-heartbeat-provenance'/owner_id/(snapshot_sha256+'.json')


def _read_small(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_PROOF_BYTES:
        raise HeartbeatProvenanceError('Invalid heartbeat ownership/provenance file')
    with path.open('rb') as handle:
        data = handle.read(MAX_PROOF_BYTES+1)
    if len(data) > MAX_PROOF_BYTES:
        raise HeartbeatProvenanceError('Heartbeat proof grew above its bound')
    try:
        value = json.loads(data)
    except ValueError as error:
        raise HeartbeatProvenanceError('Incomplete heartbeat ownership/provenance JSON') from error
    if not isinstance(value, dict):
        raise HeartbeatProvenanceError('Heartbeat ownership/provenance must be an object')
    return data, value


def capture_lease(target):
    """Capture before opening status; retained bytes fence retirement/read-fd races."""
    path = lease_path(target)
    try:
        return _read_small(path)
    except FileNotFoundError:
        return None


def validate_opened_status(target, content, value, *, captured_lease=None,
                           now_qpc_ns=None, clock_metadata=None, maximum_age_seconds=2.):
    """Validate original opened bytes; wall time cannot make old QPC age young.

    A hash-named proof is committed before the legacy file, so old and new file
    descriptors can both retain their own matching provenance during publication.
    """
    if not 0 < maximum_age_seconds <= 2:
        raise ValueError('Original heartbeat freshness budget at most2seconds required')
    active = capture_lease(target)
    lease = captured_lease or active
    if lease is None:
        return None
    if not isinstance(value, dict) or json.loads(content.decode('utf8')) != value:
        raise HeartbeatProvenanceError('Supplied heartbeat value differs from original opened bytes')
    if captured_lease is not None and active is not None and captured_lease[0] != active[0]:
        raise HeartbeatProvenanceError('Heartbeat owner changed during status read')
    raw_lease, owner = lease
    meta = clock_metadata or qpc_clock_metadata()
    now = time.perf_counter_ns() if now_qpc_ns is None else now_qpc_ns
    target = Path(target).resolve()
    if (owner.get('schema') != LEASE_SCHEMA or owner.get('clock_domain') != QPC_DOMAIN
            or owner.get('clock_metadata') != meta
            or owner.get('target') != str(target)
            or type(now) is not int or type(owner.get('deadline_qpc_ns')) is not int
            or type(owner.get('registered_qpc_ns')) is not int
            or not owner['registered_qpc_ns'] <= now <= owner['deadline_qpc_ns']
            or owner.get('generation') != value.get('generation')):
        raise HeartbeatProvenanceError('Heartbeat lease generation/QPC domain/lifetime differs')
    snapshot_hash = hashlib.sha256(content).hexdigest()
    try:
        _, proof = _read_small(proof_path(target, owner.get('owner_id'), snapshot_hash))
    except (OSError, ValueError) as error:
        raise HeartbeatProvenanceError('Matching heartbeat source proof unavailable') from error
    lower = proof.get('source_host_lower_qpc_ns')
    if (proof.get('schema') != PROOF_SCHEMA or proof.get('snapshot_sha256') != snapshot_hash
            or proof.get('owner_id') != owner.get('owner_id')
            or proof.get('generation') != owner.get('generation')
            or proof.get('lease_sha256') != hashlib.sha256(raw_lease).hexdigest()
            or proof.get('clock_domain') != QPC_DOMAIN
            or proof.get('calibration_sha256') != owner.get('calibration_sha256')
            or proof.get('expires_qpc_ns') != owner['deadline_qpc_ns']
            or proof.get('original_source_clock') != value.get('script_clock')
            or type(lower) is not int or not 0 <= lower <= now
            or now-lower > int(maximum_age_seconds*1e9)
            or type(proof.get('sequence')) is not int or not 1 <= proof['sequence'] <= 256):
        raise HeartbeatProvenanceError('Heartbeat source proof/hash/monotonic age differs')
    return dict(owner_id=owner['owner_id'], generation=owner['generation'],
                snapshot_sha256=snapshot_hash, sequence=proof['sequence'],
                source_age_upper_seconds=(now-lower)/1e9,
                deadline_qpc_ns=owner['deadline_qpc_ns'])


def read_proven_status(path, *, maximum_wait=.01, qpc_clock=time.perf_counter_ns,
                      clock_metadata=None):
    """New opt-in reader; migration into existing readers is staged separately."""
    if not 0 < maximum_wait <= .5:
        raise ValueError('Bounded heartbeat proof metadata wait required')
    deadline = qpc_clock()+int(maximum_wait*1e9)
    while True:
        try:
            captured = capture_lease(path)
            with Path(path).open('rb') as stream:
                content = stream.read()
                modified = os.fstat(stream.fileno()).st_mtime
            value = json.loads(content)
            proof = validate_opened_status(path, content, value, captured_lease=captured,
                                           now_qpc_ns=qpc_clock(), clock_metadata=clock_metadata)
            return value, modified, proof
        except (HeartbeatProvenanceError, PermissionError, FileNotFoundError):
            if qpc_clock() >= deadline:
                raise
            time.sleep(.001)


class ProvenanceWriterLease(OwnedWriterLease):
    def __init__(self, target, *, owner_id, generation, calibration_sha256,
                 registered_qpc_ns, deadline_qpc_ns, clock_metadata):
        super().__init__(target, owner_id=owner_id, generation=generation)
        if (type(registered_qpc_ns) is not int or type(deadline_qpc_ns) is not int
                or not 0 < deadline_qpc_ns-registered_qpc_ns <= 60_000_000_000
                or clock_metadata.get('domain') != QPC_DOMAIN
                or clock_metadata.get('callable') != 'time.perf_counter_ns'
                or not re.fullmatch('[0-9a-f]{64}', calibration_sha256 or '')):
            raise ValueError('Explicit finite production heartbeat QPC lease required')
        self.registration = dict(schema=LEASE_SCHEMA, owner_id=owner_id, generation=generation,
            target=str(self.target), host_pid=os.getpid(), clock_domain=QPC_DOMAIN,
            clock_metadata=clock_metadata, registered_qpc_ns=registered_qpc_ns,
            deadline_qpc_ns=deadline_qpc_ns, calibration_sha256=calibration_sha256)
        self._payload = json.dumps(self.registration, sort_keys=True).encode()

    @property
    def payload_sha256(self):
        return hashlib.sha256(self._payload).hexdigest()


def write_immutable_proof(lease, snapshot, source_host_lower_qpc_ns):
    lease.verify()
    digest = hashlib.sha256(snapshot.snapshot_json).hexdigest()
    path = proof_path(lease.target, lease.owner_id, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    proof = dict(schema=PROOF_SCHEMA, owner_id=lease.owner_id, generation=lease.generation,
        snapshot_sha256=digest, lease_sha256=lease.payload_sha256,
        clock_domain=QPC_DOMAIN, calibration_sha256=lease.registration['calibration_sha256'],
        sequence=snapshot.sequence, original_source_clock=snapshot.source_clock,
        source_host_lower_qpc_ns=source_host_lower_qpc_ns,
        expires_qpc_ns=lease.registration['deadline_qpc_ns'])
    content = (json.dumps(proof, sort_keys=True)+'\n').encode()
    if path.exists():
        if path.read_bytes() != content:
            raise HeartbeatProvenanceError('Immutable heartbeat source proof already differs')
        return proof
    part = path.with_suffix('.part')
    with part.open('xb') as stream:
        stream.write(content)
    os.replace(part, path)
    return proof
