"""Explicit finite heartbeat host context; import/construct never starts a worker.

The file lease is a persistent fail-closed fence, including after a failed join.
It is released only after the *same* writer is terminal and a fresh synchronous
Lua handoff has been checked. No vehicle API, camera acquisition or deployment.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
import threading
import time
import uuid
import zlib
from pathlib import Path

from .session_heartbeat_candidate import (
    CachedClockLowerBound,
    ExclusiveHeartbeatMaps,
    OwnedLatestHeartbeatWriter,
    latest_snapshot,
)
from .session_heartbeat_provenance import (
    QPC_DOMAIN,
    PreciseWindowsWallClock,
    ProvenanceWriterLease,
    qpc_clock_metadata,
    write_immutable_proof,
)

CONTROL_MAGIC = 0x46424843
CONTROL_CAPACITY = 2048
CONTROL_HEADER = struct.Struct('<5I')
CONTROL_SIZE = CONTROL_HEADER.size+CONTROL_CAPACITY
CONTROL_SCHEMA = 'flybrain.session_heartbeat.control.v1'
PLAN_SCHEMA = 'flybrain.session_heartbeat.arm.v1'
MAX_CONTROL_MESSAGES = 512
COMMANDS = {'run', 'retire', 'writer_fenced', 'release_confirmed'}
PHASES = {'armed', 'publishing', 'final', 'sync_return', 'failed', 'closed'}


def control_name(owner_id, direction):
    from .session_heartbeat_candidate import channel_name

    channel_name(owner_id, 0)  # same strict owner validation
    if direction not in ('host', 'lua'):
        raise ValueError('One independently owned control direction required')
    return 'Local\\FlyBrainSessionHeartbeatControl_'+owner_id+'_'+direction


def encode_control(value, sequence):
    if (type(sequence) is not int or not 1 <= sequence <= MAX_CONTROL_MESSAGES
            or value.get('schema') != CONTROL_SCHEMA):
        raise ValueError('Finite versioned control message required')
    payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if not 0 < len(payload) <= CONTROL_CAPACITY:
        raise ValueError('Control message exceeds fixed capacity')
    return (CONTROL_HEADER.pack(CONTROL_MAGIC, 1, sequence*2, len(payload), zlib.crc32(payload))
            + payload + bytes(CONTROL_CAPACITY-len(payload)))


def decode_control(first, second, *, owner_id, generation, lease_sha256):
    if len(first) != CONTROL_SIZE or len(second) != CONTROL_SIZE:
        raise ValueError('Exact control mapping length required')
    if first != second:
        return None
    magic, version, sequence, length, crc = CONTROL_HEADER.unpack_from(first)
    if sequence == 0 or sequence % 2:
        return None
    if (magic != CONTROL_MAGIC or version != 1 or not 1 <= sequence//2 <= MAX_CONTROL_MESSAGES
            or not 0 < length <= CONTROL_CAPACITY):
        raise ValueError('Stable control header differs')
    payload = first[CONTROL_HEADER.size:CONTROL_HEADER.size+length]
    if zlib.crc32(payload) != crc:
        raise ValueError('Stable control checksum differs')
    value = json.loads(payload)
    if (value.get('schema') != CONTROL_SCHEMA or value.get('owner_id') != owner_id
            or value.get('generation') != generation or value.get('lease_sha256') != lease_sha256):
        raise ValueError('Foreign control owner/generation/lease')
    return sequence//2, value


class ExclusiveControlMaps:
    """Host creates both zero maps; only host map is written by this object."""

    def __init__(self, lease):
        self.lease, self.handles, self.views, self.api = lease, [], [], None
        self.sequence = 0

    def create(self):
        from ctypes import wintypes

        self.lease.verify()
        if self.handles:
            raise RuntimeError('Control mappings already owned')
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
            for direction in ('host', 'lua'):
                ctypes.set_last_error(0)
                handle = api.CreateFileMappingW(wintypes.HANDLE(-1), None, 4, 0, CONTROL_SIZE,
                    control_name(self.lease.owner_id, direction))
                error = ctypes.get_last_error()
                if not handle:
                    raise ctypes.WinError(error)
                self.handles.append(handle)
                if error == 183:
                    raise FileExistsError('Control map exists; never reset a foreign mapping')
                access = 2 if direction == 'host' else 4
                view = api.MapViewOfFile(handle, access, 0, 0, CONTROL_SIZE)
                if not view:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.views.append(view)
                if ctypes.string_at(view, CONTROL_SIZE) != bytes(CONTROL_SIZE):
                    raise ValueError('Exclusive new control map is not zero')
        except BaseException:
            self.close()
            raise
        return self

    def send(self, value):
        self.lease.verify()
        if len(self.views) != 2:
            raise RuntimeError('Explicit control creation required')
        self.sequence += 1
        raw = encode_control(value, self.sequence)
        view = self.views[0]
        ctypes.c_uint32.from_address(view+8).value = self.sequence*2-1
        ctypes.memmove(view, raw[:8], 8)
        ctypes.memmove(view+12, raw[12:], len(raw)-12)
        ctypes.c_uint32.from_address(view+8).value = self.sequence*2

    def receive(self):
        self.lease.verify()
        view = self.views[1]
        return decode_control(ctypes.string_at(view, CONTROL_SIZE),
            ctypes.string_at(view, CONTROL_SIZE), owner_id=self.lease.owner_id,
            generation=self.lease.generation, lease_sha256=self.lease.payload_sha256)

    def close(self):
        if self.api is not None:
            for view in self.views:
                self.api.UnmapViewOfFile(view)
            for handle in self.handles:
                self.api.CloseHandle(handle)
        self.views, self.handles = [], []


def initialize_qpc_calibration(camera_root, generation, hwnd):
    """Explicit paused-service handshake; caller alone authorizes this file ping."""
    from .ac_csp_clock import CSPClockBounds, initialize_clock

    metadata = qpc_clock_metadata()
    bounds = CSPClockBounds(generation)
    initialize_clock(Path(camera_root), bounds, hwnd, host_clock=time.perf_counter)
    record = bounds.record()
    record.update(host_clock_domain=QPC_DOMAIN, host_clock_callable='time.perf_counter',
                  host_clock_metadata=metadata)
    return record


def calibration_from_qpc_record(record):
    meta = qpc_clock_metadata()
    if (record.get('host_clock_domain') != QPC_DOMAIN
            or record.get('host_clock_callable') != 'time.perf_counter'
            or record.get('host_clock_metadata') != meta):
        raise ValueError('Calibration must explicitly use this Windows QPC clock domain')
    return CachedClockLowerBound.from_record(record, generation=record['generation']), meta


def release_owned_lease_bounded(lease, *, qpc_clock=time.perf_counter_ns,
                                unlink=os.unlink, before_attempt=lambda: None):
    """Metadata-only same-name unlink retry after old writer+SYNC handoff fence.

    Keep original e5 lease implementation unchanged. A held normal reader can
    deny deletion on Windows; each retry must reprove the same file ID+bytes.
    There is no stale-owner deletion, takeover or API/activation retry.
    """
    if lease.fd is None:
        raise RuntimeError('Original held lease descriptor required before owned unlink')
    held = os.fstat(lease.fd)
    expected = lease._payload
    expected_hash = hashlib.sha256(expected).hexdigest()
    started = qpc_clock()
    deadline = started+10_000_000
    receipt = dict(owner_id=lease.owner_id, generation=lease.generation,
        expected_file_id=[held.st_dev, held.st_ino], expected_sha256=expected_hash,
        started_qpc_ns=started, deadline_qpc_ns=deadline, attempts=0,
        descriptor_closed=False, unlinked=False)
    # No worker remains. Keeping our own CRT fd open would itself deny deletion.
    os.close(lease.fd)
    lease.fd = None
    receipt['descriptor_closed'] = True
    try:
        while True:
            before_attempt()  # original source freshness/deadline/fence, never rebased
            if qpc_clock() >= deadline:
                raise TimeoutError('Original10ms owned lease unlink budget exhausted')
            try:
                with lease.path.open('rb') as stream:
                    content = stream.read(len(expected)+1)
                    named = os.fstat(stream.fileno())
                if ((named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)
                        or content != expected
                        or hashlib.sha256(content).hexdigest() != expected_hash):
                    raise RuntimeError('Foreign lease identity/content retained; never unlink it')
                before_attempt()
                if qpc_clock() >= deadline:
                    raise TimeoutError('Original10ms owned lease unlink budget exhausted')
                receipt['attempts'] += 1
                unlink(lease.path)
                receipt.update(unlinked=True, finished_qpc_ns=qpc_clock())
                return receipt
            except PermissionError as error:
                if (getattr(error, 'winerror', None) not in (5, 32, 33)
                        or qpc_clock() >= deadline):
                    raise
                time.sleep(.001)
    except BaseException as error:
        receipt.update(error=type(error).__name__, reason=str(error),
                       finished_qpc_ns=qpc_clock())
        error.heartbeat_release_receipt = receipt
        raise


class ProvenanceLatestWriter(OwnedLatestHeartbeatWriter):
    def __init__(self, lease, calibration, **kwargs):
        self.snapshots = {}
        self.rename_attempts = {}
        commit = kwargs.pop('replace', os.replace)
        commit_permitted = kwargs.pop('commit_permitted', lambda: True)
        self.legacy_wall_quantization_ns = kwargs.pop('legacy_wall_quantization_ns',
            math.ceil(time.get_clock_info('time').resolution*1e9))
        if not 1 <= self.legacy_wall_quantization_ns <= 100_000_000:
            raise ValueError('Measured bounded legacy reader wall-clock quantum required')
        # Reserve both directions of coarse wall quantization plus the precise
        # bracket's <=1ms span, existing wall guard and stamp rounding. A source
        # just under2s cannot gain ACK while its conservatively old mtime is>2s.
        self.legacy_freshness_margin_ns = (2*self.legacy_wall_quantization_ns
            +kwargs.get('wall_guard_ns', 1_000_000)+1_000_000+calibration.rounding_margin_ns)
        requested_age_ns = math.floor(kwargs.pop('max_source_age_seconds', 2.)*1e9)
        kwargs['max_source_age_seconds'] = (
            requested_age_ns-self.legacy_freshness_margin_ns)/1e9
        stamp = kwargs.pop('stamp_private', self._stamp_private)

        def conservative_legacy_stamp(part, lower):
            # Extra pastward conservatism, not an extension of either2s gate.
            # Precise UTC may lead legacy coarse GetSystemTimeAsFileTime by a tick.
            stamp(part, lower-self.legacy_wall_quantization_ns)

        def proven_replace(part, target):
            if not commit_permitted():
                raise RuntimeError('Independent context fence forbids a new commit')
            sequence = int(part.name.rsplit('.', 2)[-2])
            snapshot = self.snapshots[sequence]
            lower = calibration.source_host_lower_ns(snapshot.source_clock,
                snapshot.generation, self.qpc_clock())
            write_immutable_proof(lease, snapshot, lower)
            retry_deadline = self.qpc_clock()+10_000_000
            while True:
                if not commit_permitted():
                    raise RuntimeError('Context retired during proof write; no new rename')
                now = self.qpc_clock()
                if self.rename_attempts.get(sequence, 0) and now >= retry_deadline:
                    raise TimeoutError('Original10ms same-source rename retry budget expired')
                checked = calibration.source_host_lower_ns(
                    snapshot.source_clock, snapshot.generation, now)
                if checked != lower or now-lower > self.max_source_age_ns:
                    raise ValueError('Original source expired before same-source rename retry')
                self.rename_attempts[sequence] = self.rename_attempts.get(sequence, 0)+1
                try:
                    return commit(part, target)
                except PermissionError as error:
                    if (getattr(error, 'winerror', None) not in (5, 32, 33)
                            or self.qpc_clock() >= retry_deadline):
                        raise
                    # Metadata delivery only: same part/hash/mtime/proof/owner.
                    # A blocked syscall has no cancellation claim; deadline is
                    # checked when it returns. No API/activation is retried.
                    time.sleep(.001)

        super().__init__(lease, calibration, replace=proven_replace,
            stamp_private=conservative_legacy_stamp, **kwargs)
        self.lock = threading.RLock()

    def submit(self, snapshot):
        with self.lock:
            previous = self.snapshots.get(snapshot.sequence)
            self.snapshots[snapshot.sequence] = snapshot
            try:
                super().submit(snapshot)
            except BaseException:
                if previous is None:
                    self.snapshots.pop(snapshot.sequence, None)
                else:
                    self.snapshots[snapshot.sequence] = previous
                raise


class HeartbeatSessionContext:
    """Owned finite service session, with explicit startup and no automatic recovery.

    Failed retirement deliberately retains the ownership file and maps. A process
    exiting does not prove a blocked OS rename terminal; a new owner cannot take
    over. Root must inspect/fence the old process, then review orphan recovery.
    """

    def __init__(self, target, calibration_record, *, lifetime_seconds=60., owner_id=None,
                 qpc_clock=time.perf_counter_ns, map_factory=ExclusiveHeartbeatMaps,
                 control_factory=ExclusiveControlMaps, writer_factory=ProvenanceLatestWriter,
                 calibration=None, clock_metadata=None, bracket_clock=None):
        if not 15 <= lifetime_seconds <= 60:
            raise ValueError('Finite15..60second total lifetime required, no refresh')
        if calibration is None:
            calibration, clock_metadata = calibration_from_qpc_record(calibration_record)
        self.calibration, self.clock_metadata = calibration, clock_metadata
        self.target = Path(target).resolve()
        self.owner_id = owner_id or uuid.uuid4().hex
        self.qpc_clock, self.lifetime_seconds = qpc_clock, lifetime_seconds
        self.map_factory, self.control_factory, self.writer_factory = (
            map_factory, control_factory, writer_factory)
        self.bracket_clock = bracket_clock
        self.lease = self.maps = self.control = self.writer = None
        self.state, self.error = 'new', None
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.poll_thread = self.write_thread = None
        self.last_ack = None
        self.last_received_sequence, self.last_received_clock = 0, None
        self.command = 'run'
        self.records = []
        self._control_lock = threading.Lock()

    def prepare(self):
        if self.state != 'new':
            raise RuntimeError('Never reuse a prepared or failed heartbeat session')
        now = self.qpc_clock()
        deadline = min(now+math.floor(self.lifetime_seconds*1e9),
            self.calibration.calibrated_at_qpc_ns+self.calibration.valid_for_ns)
        if deadline-now < 45_000_000_000:
            raise ValueError('At least45seconds remain for collection30+cleanup13+handoff2')
        self.lease = ProvenanceWriterLease(self.target, owner_id=self.owner_id,
            generation=self.calibration.generation,
            calibration_sha256=self.calibration.calibration_sha256,
            registered_qpc_ns=now, deadline_qpc_ns=deadline,
            clock_metadata=self.clock_metadata).acquire()
        try:
            self.maps = self.map_factory(self.lease)
            creation = self.maps.create()
            self.control = self.control_factory(self.lease).create()
            precise = self.bracket_clock or PreciseWindowsWallClock().bracket
            self.writer = self.writer_factory(self.lease, self.calibration,
                qpc_clock=self.qpc_clock, bracket_clock=precise,
                commit_permitted=lambda: not self.stop_event.is_set())
            self.plan = dict(schema=PLAN_SCHEMA, owner_id=self.owner_id,
                generation=self.calibration.generation, lease_sha256=self.lease.payload_sha256,
                epoch=None, host_created_slots=True, max_snapshots=256,
                host_ready=False, lease_crc32=zlib.crc32(self.lease._payload),
                deadline_source_clock=deadline/1e9-self.calibration.offset_upper_seconds,
                deadline_qpc_ns=deadline, registered_qpc_ns=now,
                clock_domain=QPC_DOMAIN, slot_creation=creation,
                control_maps=[control_name(self.owner_id, x) for x in ('host', 'lua')])
            self.state = 'prepared'
            return dict(self.plan)
        except BaseException:
            if self.control is not None:
                self.control.close()
            if self.maps is not None:
                self.maps.close()
            # No arm plan has been published and no thread/rename has begun.
            self.lease.release()
            self.state = 'failed_before_arm'
            raise

    def arm_plan(self, *, epoch, minimum_remaining_seconds=45.):
        if self.state != 'prepared' or type(epoch) is not int or epoch < 0:
            raise RuntimeError('Prepared host maps and explicit observed epoch required')
        remaining = self.lease.registration['deadline_qpc_ns']-self.qpc_clock()
        if remaining < minimum_remaining_seconds*1e9:
            raise ValueError('Insufficient original lifetime; never extend the lease')
        self.plan['epoch'] = epoch
        return dict(self.plan)

    def require_collection_budget(self, *, seconds, cleanup_seconds=13., handoff_seconds=2.):
        """Call at actual activation, after camera setup; never rebase a deadline."""
        if seconds not in (3, 30) or cleanup_seconds != 13 or handoff_seconds != 2:
            raise ValueError('Only reviewed3s/30s collection and13s+2s reserves')
        remaining = (self.lease.registration['deadline_qpc_ns']-self.qpc_clock())/1e9
        if remaining < seconds+cleanup_seconds+handoff_seconds:
            raise TimeoutError('Insufficient original lifetime before activation; no extension')
        return remaining

    def start(self):
        """Explicit two owned threads; caller publishes the arm plan separately."""
        if self.state != 'prepared' or self.plan['epoch'] is None:
            raise RuntimeError('Final arm plan with epoch required before explicit start')
        self.state = 'running'
        self.poll_thread = threading.Thread(target=self._poll,
            name='heartbeat-owned-poll', daemon=False)
        self.write_thread = threading.Thread(target=self._write,
            name='heartbeat-owned-write', daemon=False)
        self.poll_thread.start()
        self.write_thread.start()
        if not self.ready_event.wait(timeout=.5) or self.error or self.stop_event.is_set():
            self._fail(TimeoutError('Initial owned host control not ready within500ms'))
            raise TimeoutError(self.error)
        self.lease.verify()
        self.plan['host_ready'] = True
        return dict(self.plan)

    def _message(self):
        committed = self.writer.last_committed_sequence
        snapshot = self.writer.snapshots.get(committed)
        message = dict(schema=CONTROL_SCHEMA, owner_id=self.owner_id,
            generation=self.calibration.generation, lease_sha256=self.lease.payload_sha256,
            command=self.command, committed_sequence=committed)
        if snapshot:
            message['committed_source_clock'] = snapshot.source_clock
        return message

    def _send(self, *, terminal=False):
        with self._control_lock:
            if self.error or self.stop_event.is_set() and not terminal:
                return False
            if terminal and (self.state not in ('awaiting_sync_return', 'closed')
                             or self.command not in ('writer_fenced', 'release_confirmed')):
                raise RuntimeError('Only terminal fenced handoff messages may follow stop')
            self.control.send(self._message())
            return True

    def _fail(self, error):
        with self._control_lock:
            self.error, self.state = str(error), 'failed_fenced'
            self.stop_event.set()
        # The lease is intentionally kept. Do not block the poller on writer's
        # commit lock or claim that a blocked Windows replace was cancelled.

    def poll_once(self):
        if self.qpc_clock() > self.lease.registration['deadline_qpc_ns']:
            raise TimeoutError('Original heartbeat lease expired; no refresh')
        ack = self.control.receive()
        if ack is not None and (self.last_ack is None or ack[0] > self.last_ack[0]):
            if ack[1].get('phase') not in PHASES:
                raise ValueError('Unknown Lua lifecycle acknowledgement')
            self.last_ack = ack
            if ack[1]['phase'] == 'failed':
                raise RuntimeError('Lua heartbeat transport failed: '+str(ack[1].get('reason')))
        snapshot = latest_snapshot(self.maps.repeated_copies(), owner_id=self.owner_id,
            generation=self.calibration.generation, last_sequence=self.last_received_sequence,
            last_source_clock=self.last_received_clock)
        if snapshot is not None:
            if json.loads(snapshot.snapshot_json).get('epoch') != self.plan['epoch']:
                raise ValueError('Session epoch changed while armed; no rebase or publication')
            self.writer.submit(snapshot)
            self.last_received_sequence = snapshot.sequence
            self.last_received_clock = snapshot.source_clock
        return snapshot

    def _poll(self):
        try:
            if not self._send():
                raise RuntimeError('Initial host control was fenced before send')
            self.ready_event.set()  # only after the actual initial control bytes were sent
            while not self.stop_event.is_set():
                self.poll_once()
                self.stop_event.wait(.005)
        except BaseException as error:
            self._fail(error)

    def _write(self):
        try:
            while not self.stop_event.is_set():
                result = self.writer.pump_one()
                if result is not None:
                    if self.stop_event.is_set() or self.error:
                        break  # old I/O outcome observed; never ACK after known failure
                    if result['status'] != 'published':
                        raise RuntimeError('Late/failed write is not a fresh commit ACK')
                    self._send()  # echo only actually committed original sequence/clock
                self.stop_event.wait(.005)
        except BaseException as error:
            self._fail(error)

    def request_retire(self):
        if self.state != 'running':
            raise RuntimeError('Only the current running context may request retirement')
        self.command, self.state = 'retire', 'retiring'
        self._send()

    def fence_writer(self, *, join_timeout=2.):
        """Call only after final Lua ACK; retain lease if the old writer is blocked."""
        if (not 0 < join_timeout <= 2 or self.last_ack is None
                or self.last_ack[1].get('phase') != 'final'
                or self.writer.last_committed_sequence != self.last_ack[1].get('final_sequence')):
            raise RuntimeError('Original final sequence must be committed before fencing')
        self.stop_event.set()
        deadline = time.perf_counter()+join_timeout
        for thread in (self.poll_thread, self.write_thread):
            if thread is not None:
                thread.join(max(0., deadline-time.perf_counter()))
        threads = (self.poll_thread, self.write_thread)
        if any(thread is not None and thread.is_alive() for thread in threads):
            self.state = 'failed_fenced'
            raise TimeoutError('Same old writer not terminal; lease retained, no sync return')
        self.writer.retire()
        if self.writer.in_flight:
            raise RuntimeError('In-flight write not terminal; never hand off')
        self.command, self.state = 'writer_fenced', 'awaiting_sync_return'
        self._send(terminal=True)

    def complete_sync_return(self):
        """No release until Lua committed fresh quiescent SYNC bytes after fence."""
        if self.state != 'awaiting_sync_return':
            raise RuntimeError('Same-writer terminal fence required')
        ack = self.control.receive()
        if ack is None or ack[1].get('phase') != 'sync_return':
            raise RuntimeError('Fresh synchronous return acknowledgement required')
        value = ack[1]
        with self.target.open('rb') as stream:
            content = stream.read()
        status = json.loads(content)
        source = status.get('script_clock')
        body = [status.get(k) for k in ('speed_kmh', 'gas', 'brake')]
        lower = self.calibration.source_host_lower_ns(source,
            self.calibration.generation, self.qpc_clock())
        if (value.get('sync_crc32') != zlib.crc32(content)
                or value.get('sync_length') != len(content)
                or value.get('sync_source_clock') != source
                or status.get('generation') != self.calibration.generation
                or status.get('schema') != 'flybrain.csp.session.status.v1'
                or status.get('epoch') != self.plan['epoch']
                or self.qpc_clock()-lower > 2_000_000_000
                or source <= self.last_ack[1].get('final_source_clock', math.inf)
                or status.get('ai_controlled') is not False or status.get('paused') is not True
                or not all(type(x) in (int, float) and math.isfinite(x) for x in body)
                or abs(body[0]) >= .1 or body[1] >= .02 or body[2] <= .98
                or any(status.get(k) for k in ('camera_active', 'expert_probe_active',
                    'brake_settle_active', 'pending_receipt_id', 'error'))):
            raise ValueError('Synchronous handoff bytes/clock/quiescence differs')
        # The old async fd still has its captured lease+immutable proof. A reader
        # starting after removal sees only these already committed SYNC bytes.
        self.command = 'release_confirmed'
        self._send(terminal=True)
        def before_release_attempt():
            now = self.qpc_clock()
            if self.error or self.state != 'awaiting_sync_return' or self.writer.in_flight:
                raise RuntimeError('Original same-writer handoff fence changed before unlink')
            if now > self.lease.registration['deadline_qpc_ns']:
                raise TimeoutError('Original lease expired before unlink; marker retained')
            lower = self.calibration.source_host_lower_ns(
                source, self.calibration.generation, now)
            if now-lower > 2_000_000_000:
                raise ValueError('Original synchronous source expired before unlink')

        try:
            self.release_receipt = release_owned_lease_bounded(self.lease,
                qpc_clock=self.qpc_clock, before_attempt=before_release_attempt)
        except BaseException as error:
            self.release_receipt = getattr(error, 'heartbeat_release_receipt', None)
            self._fail(error)  # retained marker, no new owner or silent sync takeover
            raise
        self.control.close()
        self.maps.close()
        self.state = 'closed'
        return dict(status_sha256=hashlib.sha256(content).hexdigest(),
            original_source_clock=source, same_writer_joined=True, lease_removed=True,
            owned_release=self.release_receipt)
