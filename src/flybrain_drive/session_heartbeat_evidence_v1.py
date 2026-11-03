"""Bounded RAM evidence for existing heartbeat operations; no import/start I/O.

No actor/vehicle API is used. Factory wrappers retain the exact buffers already
read/sent, without extra MMF reads, clock pings, requests or status publication.
Disk persistence requires the same owned threads to be terminal.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import threading
import time
import zlib
from pathlib import Path

from . import session_heartbeat_runtime as core
from .session_heartbeat_candidate import ExclusiveHeartbeatMaps, decode_slot

SCHEMA = 'flybrain.session_heartbeat.raw_evidence.v1'
MAX_READS_PER_CHANNEL = 16384
MAX_RAW_BYTES = 24*1024**2
MAX_STATIC_BYTES = 65536
VARIANT_LIMITS = {'data': 1024, 'lua_control': 2048, 'host_control': 512,
                  'setup': 8, 'snapshot': 256}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def require_local_c_output(path):
    path = Path(path)
    if (not path.is_absolute() or path.drive.upper() != 'C:'
            or str(path).startswith('\\\\')):
        raise ValueError('Heartbeat evidence must use an explicit local C: artifact path')
    path = path.resolve()
    if path.drive.upper() != 'C:' or str(path).startswith('\\\\'):
        raise ValueError('Heartbeat evidence must use an explicit local C: artifact path')
    return path


class EvidenceDeferred(RuntimeError):
    pass


class HeartbeatEvidence:
    def __init__(self, *, qpc_clock=time.perf_counter_ns):
        self.qpc_clock = qpc_clock
        self.lock = threading.RLock()
        self.blobs, self.blob_groups, self.group_counts, self.blob_ids = {}, {}, {}, {}
        self.reads = {key: [] for key in ('data0', 'data1', 'lua_control')}
        self.sends = []
        self.setup, self.flags = {}, {}
        self.raw_bytes = 0
        self.owner_id = self.generation = None
        self.sync_witness = self.handoff = None
        self.frozen = False
        self.deferred_registered = False

    def flag(self, name):
        with self.lock:
            self.flags[name] = self.flags.get(name, 0)+1

    def _blob(self, content, group):
        content = bytes(content)
        sha = digest(content)
        if sha in self.blobs:
            return self.blob_ids[sha]  # canonical shared reference for duplicate reads
        if (self.group_counts.get(group, 0) >= VARIANT_LIMITS[group]
                or self.raw_bytes+len(content) > MAX_RAW_BYTES):
            self.flag('raw_'+group+'_budget_exhausted')
            return None
        self.blobs[sha], self.blob_groups[sha] = content, group
        self.blob_ids[sha] = sha
        self.group_counts[group] = self.group_counts.get(group, 0)+1
        self.raw_bytes += len(content)
        return sha

    def observe_pair(self, channel, first, second):
        """Each existing read retains both copies, including odd/torn/duplicates."""
        with self.lock:
            if self.frozen:
                self.flag('read_after_freeze')
                return
            if len(self.reads[channel]) >= MAX_READS_PER_CHANNEL:
                self.flag(channel+'_read_budget_exhausted')
                return
            group = 'data' if channel.startswith('data') else 'lua_control'
            one, two = self._blob(first, group), self._blob(second, group)
            # Index is the original read ordinal. Duplicate copies are retained
            # as repeated references, not synthesized later from a newer slot.
            self.reads[channel].append((self.qpc_clock(), one, two))

    def observe_send(self, raw, sequence, *, completed, error=None):
        with self.lock:
            if self.frozen:
                self.flag('send_after_freeze')
                return
            if len(self.sends) >= 512:
                self.flag('host_control_send_budget_exhausted')
                return
            sha = self._blob(raw, 'host_control')
            if not completed:
                self.flag('incomplete_host_control_send')
            self.sends.append(dict(observed_qpc_ns=self.qpc_clock(), sha256=sha,
                sequence=sequence, completed=completed, error=error))

    def capture_static(self, name, content):
        with self.lock:
            if self.frozen or len(content) > MAX_STATIC_BYTES:
                raise ValueError('Bounded pre-freeze original metadata bytes required')
            sha = self._blob(content, 'setup')
            if name in self.setup and self.setup[name] != sha:
                raise ValueError('Original immutable evidence metadata changed: '+name)
            self.setup[name] = sha
            return sha

    def capture_calibration(self, record):
        # This is exactly CachedClockLowerBound.from_record's fingerprint input.
        return self.capture_static('calibration',
            json.dumps(record, sort_keys=True, allow_nan=False).encode())

    def capture_lease(self, lease):
        self.owner_id, self.generation = lease.owner_id, lease.generation
        return self.capture_static('lease', lease._payload)

    def capture_arm(self, path):
        # Original owned file, not a later JSON reserialization.
        with Path(path).open('rb') as stream:
            content = stream.read(MAX_STATIC_BYTES+1)
        value = json.loads(content)
        if (value.get('owner_id') != self.owner_id or value.get('generation') != self.generation
                or value.get('lease_sha256') != self.setup.get('lease')
                or value.get('host_ready') is not True):
            raise ValueError('Original owned arm bytes do not bind captured lease')
        return self.capture_static('arm', content)

    def map_factory(self, lease):
        self.capture_lease(lease)
        return EvidenceHeartbeatMaps(lease, self)

    def control_factory(self, lease):
        return EvidenceControlMaps(lease, self)

    def capture_sync(self, context):
        """Use last original ACK read; open frozen SYNC once before core handoff."""
        control = context.control
        ack, pair = control.last_decoded, control.last_pair
        if (ack is None or pair is None or pair[0] != pair[1]
                or ack[1].get('phase') != 'sync_return'):
            raise ValueError('Original stable SYNC ACK pair required before handoff')
        with context.target.open('rb') as stream:
            content = stream.read(MAX_STATIC_BYTES+1)
            modified = os.fstat(stream.fileno()).st_mtime_ns
        value = json.loads(content)
        if (len(content) > MAX_STATIC_BYTES or ack[1].get('sync_crc32') != zlib.crc32(content)
                or ack[1].get('sync_length') != len(content)
                or ack[1].get('sync_source_clock') != value.get('script_clock')
                or value.get('generation') != self.generation):
            raise ValueError('Original SYNC status bytes and original ACK differ')
        status_sha = self.capture_static('sync_status', content)
        with self.lock:
            self.sync_witness = dict(status_sha256=status_sha, original_mtime_ns=modified,
                ack_sequence=ack[0], ack_value=dict(ack[1]),
                ack_first_sha256=self._blob(pair[0], 'lua_control'),
                ack_second_sha256=self._blob(pair[1], 'lua_control'))
        return dict(self.sync_witness)

    def bind_handoff(self, handoff):
        if (self.sync_witness is None
                or handoff.get('status_sha256') != self.sync_witness['status_sha256']
                or handoff.get('same_writer_joined') is not True
                or handoff.get('lease_removed') is not True):
            self.flag('actual_handoff_witness_mismatch')
            raise ValueError('Captured original SYNC hash differs from actual handoff')
        self.handoff = dict(handoff)

    def pending_report(self, context):
        with self.lock:
            threads = [] if context is None else [t.name for t in (
                context.poll_thread, context.write_thread) if t is not None and t.is_alive()]
            lease = getattr(context, 'lease', None)
            status = 'deferred_owned_threads_live' if threads else 'partial'
            return dict(schema=SCHEMA, status=status,
                owner_id=self.owner_id, generation=self.generation,
                context_state=getattr(context, 'state', None),
                owned_lease_path=str(lease.path) if lease else None,
                original_map_creation_receipt=getattr(context, 'plan', {}).get('slot_creation'),
                owned_threads_still_live=threads, raw_blob_count=len(self.blobs),
                raw_bytes=self.raw_bytes, flags=dict(self.flags),
                original_read_counts={key: len(value) for key, value in self.reads.items()},
                setup_sha256=dict(self.setup), raw_evidence_remains_in_owned_RAM=True)

    def defer_persistence(self, path, *, context, receipt_path, register=atexit.register):
        """No new worker. Interpreter exit waits for owned non-daemon threads."""
        path, receipt_path = require_local_c_output(path), require_local_c_output(receipt_path)
        if self.deferred_registered:
            raise RuntimeError('One owned terminal evidence callback only')

        def after_owned_terminal():
            if self.pending_report(context)['owned_threads_still_live']:
                return  # retain original deferred report; never claim an archive
            try:
                report = self.persist(path, context=context)
            except BaseException as error:
                report = dict(status='terminal_persistence_failed',
                    error=type(error).__name__, reason=str(error))
            receipt_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')

        register(after_owned_terminal)
        self.deferred_registered = True
        return dict(self.pending_report(context), atexit_persistence_registered=True,
                    terminal_receipt=str(receipt_path), evidence_path=str(path))

    def persist(self, path, *, context):
        """Only after terminal joins; failure artifacts remain explicitly partial."""
        pending = self.pending_report(context)
        if pending['owned_threads_still_live']:
            raise EvidenceDeferred('Original owned threads not terminal; defer disk evidence')
        path = require_local_c_output(path)
        with self.lock:
            self.frozen = True
            snapshots, stable_data, stable_control = {}, {}, {}
            for channel in ('data0', 'data1'):
                for _, first, second in self.reads[channel]:
                    if first is None or second is None or first != second:
                        continue
                    raw = self.blobs[first]
                    try:
                        row = decode_slot(raw, raw, owner_id=self.owner_id,
                                          generation=self.generation)
                        if row is not None:
                            if row.sequence not in stable_data and len(stable_data) >= 256:
                                self.flag('stable_data_sequence_budget_exhausted')
                                continue
                            body_sha = self._blob(row.snapshot_json, 'snapshot')
                            previous = stable_data.get(row.sequence)
                            if previous is not None and previous['raw_sha256'] != first:
                                self.flag('stable_data_sequence_conflict')
                            stable_data[row.sequence] = dict(channel=channel, raw_sha256=first,
                                snapshot_json_sha256=body_sha, source_clock=row.source_clock)
                            snapshots[row.sequence] = body_sha
                    except ValueError:
                        self.flag('invalid_stable_data')
            for _, first, second in self.reads['lua_control']:
                if first is None or second is None or first != second:
                    continue
                try:
                    row = core.decode_control(self.blobs[first], self.blobs[second],
                        owner_id=self.owner_id, generation=self.generation,
                        lease_sha256=self.setup.get('lease'))
                    if row is not None:
                        if row[0] not in stable_control and len(stable_control) >= 512:
                            self.flag('stable_control_sequence_budget_exhausted')
                            continue
                        previous = stable_control.get(row[0])
                        if previous is not None and previous['raw_sha256'] != first:
                            self.flag('stable_control_sequence_conflict')
                        stable_control[row[0]] = dict(raw_sha256=first, value=row[1])
                except ValueError:
                    self.flag('invalid_stable_lua_control')
            final = 0 if context is None or context.writer is None else (
                context.writer.last_committed_sequence)
            gaps = [n for n in range(1, final+1) if n not in snapshots]
            complete = (context is not None and context.state == 'closed'
                and final > 0 and bool(stable_data)
                and self.handoff is not None and self.sync_witness is not None
                and self.setup.get('calibration') == context.calibration.calibration_sha256
                and self.setup.get('lease') == context.lease.payload_sha256
                and all(self.setup.get(k) for k in ('calibration', 'lease', 'arm'))
                and not self.flags and not gaps)
            index = dict(schema=SCHEMA, status='complete' if complete else 'partial',
                owner_id=self.owner_id, generation=self.generation,
                limits=dict(reads_per_channel=MAX_READS_PER_CHANNEL, raw_bytes=MAX_RAW_BYTES,
                    raw_variants=VARIANT_LIMITS, stable_data_sequences=256,
                    stable_control_sequences=512),
                flags=dict(self.flags), setup_sha256=dict(self.setup),
                read_columns=['observed_after_copy_qpc_ns', 'first_sha256', 'second_sha256'],
                reads=self.reads, original_host_sends=self.sends,
                stable_data=stable_data, stable_lua_control=stable_control,
                snapshot_json_sha256=snapshots, observed_data_sequence_gaps=gaps,
                sync_witness=self.sync_witness, actual_handoff=self.handoff,
                core_pristine_zero_checks='source-backed; not claimed as captured raw reads',
                blobs={sha: dict(file='blobs/'+sha+'.bin', size=len(raw),
                    group=self.blob_groups[sha]) for sha, raw in self.blobs.items()})
            blobs = dict(self.blobs)
        path.mkdir(parents=True, exist_ok=False)
        (path/'blobs').mkdir()
        for sha, raw in blobs.items():
            (path/'blobs'/(sha+'.bin')).write_bytes(raw)
        encoded = (json.dumps(index, indent=2, allow_nan=False)+'\n').encode()
        (path/'index.json').write_bytes(encoded)
        return dict(status=index['status'], index=str(path/'index.json'),
            index_sha256=digest(encoded), raw_blob_count=len(blobs),
            raw_bytes=sum(map(len, blobs.values())), flags=index['flags'])


class EvidenceHeartbeatMaps(ExclusiveHeartbeatMaps):
    def __init__(self, lease, evidence):
        super().__init__(lease)
        self.evidence = evidence

    def repeated_copies(self):
        copies = super().repeated_copies()  # exactly the old4 raw MMF reads
        for index, pair in enumerate(copies):
            self.evidence.observe_pair('data'+str(index), *pair)
        return copies


class EvidenceControlMaps(core.ExclusiveControlMaps):
    def __init__(self, lease, evidence):
        super().__init__(lease)
        self.evidence, self.last_pair, self.last_decoded = evidence, None, None

    def send(self, value):
        # Identical original core writes, retaining its exact encode buffer;
        # no extra MMF read or later reconstructed serialization.
        self.lease.verify()
        if len(self.views) != 2:
            raise RuntimeError('Explicit control creation required')
        self.sequence += 1
        raw = core.encode_control(value, self.sequence)
        view = self.views[0]
        try:
            ctypes.c_uint32.from_address(view+8).value = self.sequence*2-1
            ctypes.memmove(view, raw[:8], 8)
            ctypes.memmove(view+12, raw[12:], len(raw)-12)
            ctypes.c_uint32.from_address(view+8).value = self.sequence*2
        except BaseException as error:
            self.evidence.observe_send(raw, self.sequence, completed=False,
                                       error=type(error).__name__)
            raise
        self.evidence.observe_send(raw, self.sequence, completed=True)

    def receive(self):
        self.lease.verify()
        view = self.views[1]
        first, second = ctypes.string_at(view, core.CONTROL_SIZE), ctypes.string_at(
            view, core.CONTROL_SIZE)
        self.last_pair, self.last_decoded = (first, second), None
        self.evidence.observe_pair('lua_control', first, second)
        self.last_decoded = core.decode_control(first, second, owner_id=self.lease.owner_id,
            generation=self.lease.generation, lease_sha256=self.lease.payload_sha256)
        return self.last_decoded
