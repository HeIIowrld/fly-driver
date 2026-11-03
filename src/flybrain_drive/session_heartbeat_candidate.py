"""Isolated session-heartbeat transport candidate; never starts a worker on import.

No vehicle API or camera module is used. Explicit callers must own the file lease,
exclusive MMF creation and a reviewed cached clock calibration before publication.
This is not connected to the deployed service.
"""

import ctypes
import hashlib
import json
import math
import os
import re
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

MAGIC = 0x46424832
VERSION = 1
CAPACITY = 8192
SNAPSHOT_CAPACITY = 6144
MAX_SNAPSHOTS = 256
HEADER = struct.Struct('<5I')
SLOT_SIZE = HEADER.size + CAPACITY
PACKET_SCHEMA = 'flybrain.session_heartbeat.packet.v1'
STATUS_SCHEMA = 'flybrain.csp.session.status.v1'


def _owner(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('Fresh explicit 128-bit lowercase hexadecimal owner ID required')
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _json_object(content):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON number in heartbeat packet: '+value)

    value = json.loads(content, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError('Original heartbeat JSON object required')
    return value


def channel_name(owner_id, index):
    if type(index) is not int or index not in (0, 1):
        raise ValueError('Exactly two session heartbeat slots')
    return f'Local\\FlyBrainSessionHeartbeat_{_owner(owner_id)}_{index}'


@dataclass(frozen=True)
class HeartbeatSnapshot:
    owner_id: str
    generation: str
    sequence: int
    source_clock: float
    snapshot_json: bytes

    def __post_init__(self):
        _owner(self.owner_id)
        if (not isinstance(self.generation, str) or not 1 <= len(self.generation) <= 128
                or type(self.sequence) is not int or not 1 <= self.sequence <= MAX_SNAPSHOTS
                or not _finite(self.source_clock) or self.source_clock < 0
                or type(self.snapshot_json) is not bytes
                or not 1 <= len(self.snapshot_json) <= SNAPSHOT_CAPACITY):
            raise ValueError('Bounded immutable original heartbeat snapshot required')
        value = _json_object(self.snapshot_json)
        if (value.get('schema') != STATUS_SCHEMA or value.get('generation') != self.generation
                or not _finite(value.get('script_clock'))
                or value['script_clock'] != self.source_clock):
            raise ValueError('Original session JSON clock/generation/schema differs')


def encode_slot(snapshot):
    """Fixture/codec helper. Live producer uses the standalone Lua candidate."""
    payload = json.dumps(dict(schema=PACKET_SCHEMA, owner_id=snapshot.owner_id,
        generation=snapshot.generation, sequence=snapshot.sequence,
        source_clock=snapshot.source_clock,
        snapshot_json=snapshot.snapshot_json.decode('utf8')),
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')
    if len(payload) > CAPACITY:
        raise ValueError('Heartbeat envelope exceeds fixed MMF capacity')
    return (HEADER.pack(MAGIC, VERSION, 2*snapshot.sequence, len(payload), zlib.crc32(payload))
            + payload + bytes(CAPACITY-len(payload)))


def decode_slot(first, second, *, owner_id, generation):
    """A torn/odd slot is unavailable; stable corruption or foreign identity is an error."""
    _owner(owner_id)
    if len(first) != SLOT_SIZE or len(second) != SLOT_SIZE:
        raise ValueError('Exact session heartbeat slot length required')
    if first != second:
        return None
    magic, version, seq, length, crc = HEADER.unpack_from(first)
    if seq == 0 or seq % 2:
        return None
    if (magic != MAGIC or version != VERSION or not 0 < length <= CAPACITY
            or not 1 <= seq//2 <= MAX_SNAPSHOTS):
        raise ValueError('Stable heartbeat MMF header differs')
    payload = first[HEADER.size:HEADER.size+length]
    if zlib.crc32(payload) != crc:
        raise ValueError('Stable heartbeat CRC32 differs')
    value = _json_object(payload)
    if (value.get('schema') != PACKET_SCHEMA or value.get('owner_id') != owner_id
            or value.get('generation') != generation or value.get('sequence') != seq//2
            or not isinstance(value.get('snapshot_json'), str)):
        raise ValueError('Foreign or inconsistent heartbeat envelope')
    return HeartbeatSnapshot(owner_id, generation, value['sequence'], value.get('source_clock'),
                             value['snapshot_json'].encode('utf8'))


def latest_snapshot(copies, *, owner_id, generation, last_sequence=0, last_source_clock=None):
    """Read-only two-copy samples of each slot, never a camera ACK protocol."""
    if len(copies) != 2 or any(len(pair) != 2 for pair in copies):
        raise ValueError('Two original repeated copies for both heartbeat slots required')
    candidates = []
    for index, pair in enumerate(copies):
        row = decode_slot(*pair, owner_id=owner_id, generation=generation)
        if row is not None and row.sequence % 2 != index:
            raise ValueError('Heartbeat sequence appeared in the wrong owned slot')
        candidates.append(row)
    candidates = [row for row in candidates if row is not None and row.sequence > last_sequence]
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: row.sequence)
    if last_source_clock is not None and selected.source_clock <= last_source_clock:
        raise ValueError('New heartbeat sequence regressed/repeated original source clock')
    return selected


@dataclass(frozen=True)
class CachedClockLowerBound:
    """Frozen CSPClockBounds.record(), no ping or source-clock resampling here."""

    generation: str
    offset_lower_seconds: float
    offset_upper_seconds: float
    host_quantization_seconds: float
    calibrated_at_qpc_ns: int
    valid_for_ns: int
    calibration_sha256: str
    rounding_margin_ns: int = 1000

    def __post_init__(self):
        if (not isinstance(self.generation, str) or not self.generation
                or not all(_finite(x) for x in [self.offset_lower_seconds,
                           self.offset_upper_seconds, self.host_quantization_seconds])
                or not 0 <= self.offset_upper_seconds-self.offset_lower_seconds <= .1
                or not 0 <= self.host_quantization_seconds <= .001
                or type(self.calibrated_at_qpc_ns) is not int or self.calibrated_at_qpc_ns < 0
                or type(self.valid_for_ns) is not int or not 0 < self.valid_for_ns <= 60_000_000_000
                or type(self.rounding_margin_ns) is not int
                or not 1 <= self.rounding_margin_ns <= 1000000
                or not isinstance(self.calibration_sha256, str)
                or not re.fullmatch('[0-9a-f]{64}', self.calibration_sha256)):
            raise ValueError('Finite validated cached source clock bound required')

    @classmethod
    def from_record(cls, record, *, generation, maximum_uncertainty_seconds=.01,
                    valid_for_seconds=60., rounding_margin_ns=1000):
        exchanges = record.get('exchanges', [])
        quantum = record.get('host_clock_quantization_seconds', 0.)
        if (not isinstance(generation, str) or not generation
                or record.get('generation') != generation
                or not _finite(quantum) or not 0 <= quantum <= .001
                or not _finite(maximum_uncertainty_seconds)
                or not 0 < maximum_uncertainty_seconds <= .1
                or not _finite(valid_for_seconds) or not 0 < valid_for_seconds <= 60
                or type(rounding_margin_ns) is not int or not 1 <= rounding_margin_ns <= 1000000
                or not isinstance(exchanges, list) or len(exchanges) < 2):
            raise ValueError('Same-generation finite cached clock calibration required')
        lows, highs, received = [], [], []
        for row in exchanges:
            values = [row.get(key) for key in ('sent', 'received', 'producer_clock')]
            if (not all(_finite(x) for x in values) or not 0 <= values[0] <= values[1]
                    or values[2] < 0):
                raise ValueError('Original cached causal clock exchange required')
            lows.append(values[0]-values[2]-quantum)
            highs.append(values[1]-values[2]+quantum)
            received.append(values[1])
        low, high = max(lows), min(highs)
        if (low > high or high-low > maximum_uncertainty_seconds
                or any(not _finite(record.get(key)) for key in
                       ('offset_lower_seconds', 'offset_upper_seconds'))
                or abs(low-record['offset_lower_seconds']) > 1e-10
                or abs(high-record['offset_upper_seconds']) > 1e-10):
            raise ValueError('Cached clock intersection/unit-rate/uncertainty differs')
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True,
                                               allow_nan=False).encode()).hexdigest()
        return cls(generation, low, high, quantum, math.floor(max(received)*1e9),
                   math.floor(valid_for_seconds*1e9), fingerprint, rounding_margin_ns)

    def source_host_lower_ns(self, source_clock, generation, now_qpc_ns):
        if (generation != self.generation or not _finite(source_clock) or source_clock < 0
                or type(now_qpc_ns) is not int
                or not self.calibrated_at_qpc_ns <= now_qpc_ns
                <= self.calibrated_at_qpc_ns+self.valid_for_ns):
            raise ValueError('Expired/foreign/unusable cached source clock bound')
        lower = math.floor((source_clock+self.offset_lower_seconds
                            -self.host_quantization_seconds)*1e9)-self.rounding_margin_ns
        if lower > now_qpc_ns:
            raise ValueError('Original heartbeat source lies wholly in the future')
        return lower


@dataclass(frozen=True)
class WallQPCBracket:
    qpc_before_ns: int
    unix_ns: int
    qpc_after_ns: int

    def __post_init__(self):
        if (any(type(x) is not int or x <= 0 for x in
                (self.qpc_before_ns, self.unix_ns, self.qpc_after_ns))
                or not 0 <= self.qpc_after_ns-self.qpc_before_ns <= 1000000):
            raise ValueError('Fresh <=1ms QPC-before/Unix/QPC-after bracket required')

    @classmethod
    def sample(cls, *, qpc_clock=time.perf_counter_ns, unix_clock=time.time_ns):
        before = qpc_clock()
        wall = unix_clock()
        return cls(before, wall, qpc_clock())


def conservative_unix_lower_ns(source_host_lower_ns, bracket, *, rounding_margin_ns=1000,
                               wall_guard_ns=1000000):
    """Original lowerQPC + wall(mid-bracket) - QPCafter; never queue delivery time."""
    if (type(source_host_lower_ns) is not int or source_host_lower_ns < 0
            or source_host_lower_ns > bracket.qpc_after_ns
            or type(rounding_margin_ns) is not int or rounding_margin_ns < 0
            or type(wall_guard_ns) is not int or wall_guard_ns < 0):
        raise ValueError('Conservative original source bound and explicit margins required')
    lower = (source_host_lower_ns + bracket.unix_ns - bracket.qpc_after_ns
             - rounding_margin_ns - wall_guard_ns)
    if not 0 < lower <= bracket.unix_ns:
        raise ValueError('Invalid/future conservative source Unix bound')
    return lower


class OwnedWriterLease:
    """Exclusive legacy-path owner; no stale-owner deletion or implicit takeover."""

    def __init__(self, target, *, owner_id, generation):
        self.target = Path(target).resolve()
        self.owner_id, self.generation = _owner(owner_id), generation
        if not isinstance(generation, str) or not generation:
            raise ValueError('Explicit source generation required')
        self.path = self.target.with_name(self.target.name+'.flybrain-heartbeat-owner')
        self.fd = None
        self._payload = json.dumps(dict(owner_id=owner_id, generation=generation,
                                       target=str(self.target)), sort_keys=True).encode()

    def acquire(self):
        if self.fd is not None:
            raise RuntimeError('Heartbeat writer lease already acquired')
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.write(self.fd, self._payload)
            self.verify()
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def verify(self):
        if self.fd is None:
            raise RuntimeError('Explicit exclusive heartbeat writer lease required')
        held, named = os.fstat(self.fd), self.path.stat()
        if (held.st_ino != named.st_ino or held.st_dev != named.st_dev
                or self.path.read_bytes() != self._payload):
            raise RuntimeError('Heartbeat writer lease changed; no publication')

    def release(self):
        self.verify()
        os.close(self.fd)
        self.fd = None
        if self.path.read_bytes() != self._payload:
            raise RuntimeError('Foreign heartbeat lease retained; never remove it')
        self.path.unlink()


class ExclusiveHeartbeatMaps:
    """Explicit opt-in host creation only; never invoked by module load/constructors.

    Windows ERROR_ALREADY_EXISTS is rejected before mapping; host owns initial
    zero slots and retains them for Lua to open. The reader maps read-only.
    """

    def __init__(self, lease):
        self.lease, self.handles, self.views, self.api = lease, [], [], None

    def create(self):
        from ctypes import wintypes

        self.lease.verify()
        if self.handles:
            raise RuntimeError('Session heartbeat maps already created')
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.CreateFileMappingW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                           wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR]
        api.CreateFileMappingW.restype = wintypes.HANDLE
        api.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, ctypes.c_size_t]
        api.MapViewOfFile.restype = ctypes.c_void_p
        api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api = api
        try:
            for index in (0, 1):
                ctypes.set_last_error(0)
                handle = api.CreateFileMappingW(wintypes.HANDLE(-1), None, 4, 0, SLOT_SIZE,
                                               channel_name(self.lease.owner_id, index))
                error = ctypes.get_last_error()
                if not handle:
                    raise ctypes.WinError(error)
                self.handles.append(handle)
                if error == 183:
                    raise FileExistsError('Heartbeat map already exists; never reset foreign slots')
                view = api.MapViewOfFile(handle, 4, 0, 0, SLOT_SIZE)  # FILE_MAP_READ only
                if not view:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.views.append(view)
                if ctypes.string_at(view, SLOT_SIZE) != bytes(SLOT_SIZE):
                    raise ValueError('New heartbeat slot was not zero initialized')
        except BaseException:
            self.close()
            raise
        return dict(owner_id=self.lease.owner_id, generation=self.lease.generation,
                    host_created_slots=True, exclusive_creation=True,
                    slot_size=SLOT_SIZE,
                    names=[channel_name(self.lease.owner_id, i) for i in (0, 1)])

    def repeated_copies(self):
        self.lease.verify()
        if len(self.views) != 2:
            raise RuntimeError('Two explicitly created owned heartbeat maps required')
        return [(ctypes.string_at(view, SLOT_SIZE), ctypes.string_at(view, SLOT_SIZE))
                for view in self.views]

    def close(self):
        if self.api is not None:
            for view in self.views:
                self.api.UnmapViewOfFile(view)
            for handle in self.handles:
                self.api.CloseHandle(handle)
        self.views, self.handles = [], []


class OwnedLatestHeartbeatWriter:
    """Single flight plus latest queued immutable JSON, no automatic thread/process."""

    def __init__(self, lease, calibration, *, max_source_age_seconds=2.,
                 bracket_clock=None, qpc_clock=time.perf_counter_ns, maximum_attempts=256,
                 wall_guard_ns=1000000, write_private=None, stamp_private=None, replace=None):
        lease.verify()
        if (lease.generation != calibration.generation
                or not _finite(max_source_age_seconds) or not 0 < max_source_age_seconds <= 2
                or type(maximum_attempts) is not int or not 1 <= maximum_attempts <= MAX_SNAPSHOTS
                or type(wall_guard_ns) is not int or not 0 <= wall_guard_ns <= 1000000):
            raise ValueError('Owned same-generation bounded heartbeat writer required')
        self.lease, self.calibration = lease, calibration
        self.max_source_age_ns = math.floor(max_source_age_seconds*1e9)
        self.qpc_clock, self.bracket_clock = qpc_clock, bracket_clock or WallQPCBracket.sample
        self.maximum_attempts, self.wall_guard_ns = maximum_attempts, wall_guard_ns
        self.write_private = write_private or self._write_private
        self.stamp_private = stamp_private or self._stamp_private
        self.replace = replace or os.replace
        self.lock = threading.Lock()
        self.pending, self.in_flight, self.retired = None, False, False
        self.last_submitted_sequence, self.last_committed_sequence = 0, 0
        self.last_source_clock = None
        self.last_anchor = None
        self.queued_replaced = self.sequence_skips = 0
        self.records = []

    @staticmethod
    def _write_private(path, content):
        with path.open('xb') as stream:
            stream.write(content)

    @staticmethod
    def _stamp_private(path, lower_ns):
        os.utime(path, ns=(lower_ns, lower_ns))

    def submit(self, snapshot):
        with self.lock:
            if self.retired:
                raise RuntimeError('Retired heartbeat writer cannot accept work')
            if (snapshot.owner_id != self.lease.owner_id
                    or snapshot.generation != self.lease.generation
                    or snapshot.sequence <= self.last_submitted_sequence
                    or self.last_source_clock is not None
                    and snapshot.source_clock <= self.last_source_clock):
                raise ValueError('Foreign/repeated/regressing original heartbeat snapshot')
            self.sequence_skips += snapshot.sequence-self.last_submitted_sequence-1
            self.last_submitted_sequence, self.last_source_clock = (
                snapshot.sequence, snapshot.source_clock)
            if self.pending is not None:
                self.queued_replaced += 1
            self.pending = snapshot

    def _check_anchor(self, bracket):
        if self.last_anchor is not None:
            old = self.last_anchor
            if (bracket.qpc_before_ns < old.qpc_after_ns
                    or bracket.unix_ns-bracket.qpc_after_ns
                    > old.unix_ns-old.qpc_before_ns+self.wall_guard_ns
                    or bracket.unix_ns-bracket.qpc_before_ns
                    < old.unix_ns-old.qpc_after_ns-self.wall_guard_ns):
                raise ValueError('QPC/Unix bracket clock step or disagreement; no publication')
        self.last_anchor = bracket

    def pump_one(self):
        with self.lock:
            if self.in_flight:
                raise RuntimeError('Only one heartbeat file write may be in flight')
            if self.retired or self.pending is None:
                return None
            if len(self.records) >= self.maximum_attempts:
                self.retired, self.pending = True, None
                raise RuntimeError('Finite heartbeat write-attempt budget exhausted')
            work, self.pending, self.in_flight = self.pending, None, True
        part = self.lease.target.with_name(
            f'.{self.lease.target.name}.{self.lease.owner_id}.{work.sequence}.part')
        record = dict(sequence=work.sequence, source_clock=work.source_clock,
                      generation=work.generation, owner_id=work.owner_id,
                      payload_sha256=hashlib.sha256(work.snapshot_json).hexdigest(),
                      status='failed', private_file=str(part))
        try:
            self.lease.verify()
            # Blocking I/O is in the caller-owned worker, never the CSP callback.
            self.write_private(part, work.snapshot_json)
            bracket = self.bracket_clock()
            self._check_anchor(bracket)
            lower_qpc = self.calibration.source_host_lower_ns(
                work.source_clock, work.generation, bracket.qpc_after_ns)
            if bracket.qpc_after_ns-lower_qpc > self.max_source_age_ns:
                raise ValueError('Original source heartbeat too old; no publication')
            lower_unix = conservative_unix_lower_ns(
                lower_qpc, bracket, rounding_margin_ns=self.calibration.rounding_margin_ns,
                wall_guard_ns=self.wall_guard_ns)
            self.stamp_private(part, lower_unix)
            actual = part.stat().st_mtime_ns
            if actual > lower_unix:
                raise ValueError('Filesystem mtime rounded later than original lower bound')
            record.update(source_host_lower_ns=lower_qpc, source_unix_lower_ns=lower_unix,
                          actual_mtime_ns=actual, wall_qpc_bracket=vars(bracket))
            with self.lock:
                self.lease.verify()
                if self.retired or work.sequence <= self.last_committed_sequence:
                    raise RuntimeError('Retired/regressing writer cannot commit')
                before_commit = self.qpc_clock()
                if (type(before_commit) is not int or before_commit < bracket.qpc_after_ns
                        or before_commit-lower_qpc > self.max_source_age_ns):
                    raise ValueError('Original source became too old before rename')
                self.calibration.source_host_lower_ns(
                    work.source_clock, work.generation, before_commit)
                self.replace(part, self.lease.target)
                self.last_committed_sequence = work.sequence
                record.update(status='published', published_qpc_ns=self.qpc_clock(),
                              queued_replaced=self.queued_replaced,
                              sequence_skips=self.sequence_skips)
                if record['published_qpc_ns']-lower_qpc > self.max_source_age_ns:
                    record['status'] = 'late_commit_stale_mtime'  # legacy readers still reject
            return dict(record)
        except BaseException as error:
            record.update(error=type(error).__name__, reason=str(error))
            with self.lock:
                self.retired, self.pending = True, None
            raise
        finally:
            with self.lock:
                self.records.append(record)
                self.in_flight = False

    def retire(self):
        with self.lock:
            self.retired, self.pending = True, None

    def close(self):
        with self.lock:
            if self.in_flight:
                raise RuntimeError('Observe same in-flight writer to terminal before lease release')
            self.retired, self.pending = True, None
            self.lease.release()

    def run_finite(self, read_latest, *, seconds, poll_seconds=.01):
        """Explicit caller-owned loop; no thread creation or process restart here."""
        if not _finite(seconds) or not 0 < seconds <= 60 or not .001 <= poll_seconds <= .1:
            raise ValueError('Explicit <=60second finite worker budget required')
        started = self.qpc_clock()
        deadline = started+math.floor(seconds*1e9)
        try:
            while self.qpc_clock() < deadline:
                snapshot = read_latest()
                if snapshot is not None and snapshot.sequence > self.last_submitted_sequence:
                    self.submit(snapshot)
                self.pump_one()
                time.sleep(poll_seconds)
        finally:
            self.retire()
