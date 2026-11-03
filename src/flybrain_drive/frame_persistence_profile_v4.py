"""Explicit host-owned payload namespace and bounded diagnostic pre-copy trace.

No filesystem/device work on import. Old V3/default writer and all old helpers
remain unchanged. The inherited persistence I/O still uses exclusive xb files.
"""

import json
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .frame_persistence_profile_v3 import (
    BufferedFrameProfile,
    BufferedImmutableFrameWriter,
    verify_buffered_persistence_archive,
)

SCHEMA = 'flybrain.frame_host_copy_profile.v4'
NAME = 'exclusive_host_payload_precopy_trace_v4'
PHASES = ('capacity_enter', 'future_done_enter', 'future_done_exit',
          'future_result_enter', 'future_result_exit', 'capacity_exit',
          'helper_enter', 'targets_reserved', 'handles_parse_enter',
          'handles_parse_exit', 'group_copy_enter', 'group_copy_exit',
          'helper_copy_returned', 'ack_finished')
TRACE_ROW = struct.Struct('<HHQQ')
EVENTS_PER_GROUP = 160
TRACE_ROWS = 1200*EVENTS_PER_GROUP


@dataclass(frozen=True)
class HostCopyProfile:
    def record(self):
        return dict(schema=SCHEMA, name=NAME, persistence=BufferedFrameProfile().record(),
            namespace='exclusive_client_request_host_payload_v4',
            collision_policy='ram_exact_sequential_targets_then_xb_no_overwrite',
            trace_record_schema='group_phase_QPCns_threadCPUns',
            trace_maximum_groups=1200, trace_events_per_group=EVENTS_PER_GROUP,
            trace_maximum_binary_bytes=TRACE_ROWS*TRACE_ROW.size,
            failed_data_retirement_minimum_remaining_seconds=8,
            original_camera_freshness_seconds=.15, original_actor_interval_seconds=.08,
            original_heartbeat_age_seconds=2.)

    @classmethod
    def from_record(cls, record):
        value = cls()
        if record != value.record():
            raise ValueError('Exact explicit hostV432/32MiB/1200 trace profile required')
        return value


def load_host_copy_profile(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise ValueError('Bounded explicit host-copy profile required')
    return HostCopyProfile.from_record(json.loads(raw))


class HostTrace:
    """Fixed preallocated RAM; timestamps are diagnostics, never sensor receipts."""
    def __init__(self, *, wall=time.perf_counter_ns, cpu=time.thread_time_ns):
        self.wall, self.cpu = wall, cpu
        self.data = bytearray(TRACE_ROWS*TRACE_ROW.size)
        self.count = 0
        self.per_group = [0]*1200
        self.producer = None

    def mark(self, group, phase):
        producer = threading.get_ident()
        if self.producer is None:
            self.producer = producer
        if (producer != self.producer or type(group) is not int or not 0 <= group < 1200
                or phase not in PHASES or self.per_group[group] >= EVENTS_PER_GROUP
                or self.count >= TRACE_ROWS):
            raise ValueError('Single-producer bounded host trace overflow/identity failure')
        wall, cpu = self.wall(), self.cpu()
        if type(wall) is not int or type(cpu) is not int or wall < 0 or cpu < 0:
            raise ValueError('Explicit nonnegative original diagnostic clock integers required')
        TRACE_ROW.pack_into(self.data, self.count*TRACE_ROW.size,
                            group, PHASES.index(phase), wall, cpu)
        self.count += 1
        self.per_group[group] += 1

    def record(self):
        return dict(schema='flybrain.host_precopy_trace.v4', phases=list(PHASES),
            clock_domains=['host_QPC_nanoseconds', 'current_thread_CPU_nanoseconds'],
            source_receipts_changed=False, serialized_after_owned_camera_terminal=True,
            maximum_groups=1200, events_per_group=EVENTS_PER_GROUP,
            binary_capacity_bytes=len(self.data), events=self.count,
            records=[list(TRACE_ROW.unpack_from(self.data, i*TRACE_ROW.size))
                     for i in range(self.count)])


class HostPayloadNamespace:
    """Atomic fresh child of an already exclusively created client request directory.

    It proves prior/internal collisions under one owned writer, not exclusion of
    malicious same-user filesystem injection. The unchanged xb boundary remains.
    """
    def __init__(self, directory, request_id, generation, maximum_groups, witness):
        self.directory, self.request_id = Path(directory), request_id
        self.generation, self.maximum_groups = generation, maximum_groups
        self.witness, self.next_group, self.producer = witness, 0, None

    @classmethod
    def create(cls, client_directory, request_id, generation, maximum_groups):
        client = Path(client_directory).resolve()  # Preactivation only.
        if (not re.fullmatch(r'mmf-stream-[0-9a-f]{32}', request_id)
                or client.name != request_id or not client.is_dir()
                or not isinstance(generation, str) or not generation
                or type(maximum_groups) is not int or not 1 <= maximum_groups <= 1200):
            raise ValueError('Exclusive client request namespace/identity required')
        directory = client/'host-payload-v4'
        directory.mkdir(exist_ok=False)  # No exists race/retry/new nonce.
        stat = directory.stat()
        witness = dict(atomic_mkdir_exclusive_succeeded=True, parent=str(client),
            directory=str(directory), creation_stat_device=stat.st_dev,
            creation_stat_inode=stat.st_ino, directory_precreated_before_camera_request=True,
            foreign_same_user_injection_excluded=False, producer_Lua_directory_unchanged=True)
        return cls(directory, request_id, generation, maximum_groups, witness)

    def reserve(self, index, views):
        producer = threading.get_ident()
        if self.producer is None:
            self.producer = producer
        if (producer != self.producer or type(index) is not int or index != self.next_group
                or not 0 <= index < self.maximum_groups or len(views) != 4
                or {v['view_id'] for v in views} != {'front','left','right','rear'}
                or any(v['file'] != f'{index:04d}-{v["view_id"]}.bmp' for v in views)):
            raise ValueError('Exact sequential once-only RAM target reservation required')
        self.next_group += 1  # Never roll back after a failed copy/ACK.

    def record(self):
        return dict(schema='flybrain.host_payload_namespace.v4', request_id=self.request_id,
            generation=self.generation, maximum_groups=self.maximum_groups,
            reserved_groups=list(range(self.next_group)),
            target_rule='{index:04d}-{front,left,right,rear}.bmp',
            metadata_rule='frame/ack/mmf-{index:04d} original bytes',
            no_per_group_filesystem_collision_query=True, actor_input=False,
            **self.witness)


class HostImmutableFrameWriter(BufferedImmutableFrameWriter):
    """Only capacity bookkeeping gains timing; inherited _persist is unchanged."""
    def __init__(self, directory, stream_path, *, namespace, trace, **kwargs):
        if Path(directory) != namespace.directory:
            raise ValueError('Writer must use the exact exclusively created host namespace')
        self.namespace, self.trace = namespace, trace
        super().__init__(directory, stream_path, profile=BufferedFrameProfile(), **kwargs)

    def check_capacity(self):
        index = self.submitted
        self.trace.mark(index, 'capacity_enter')
        producer = threading.get_ident()
        if self.producer_thread is None:
            self.producer_thread = producer
        elif producer != self.producer_thread:
            raise RuntimeError('One original camera producer owns capacity/copy/ACK/enqueue')
        if self.closed:
            raise ValueError('Frame writer is closed')
        while self.pending:
            self.trace.mark(index, 'future_done_enter')
            done = self.pending[0].done()
            self.trace.mark(index, 'future_done_exit')
            if not done:
                break
            future = self.pending.popleft()
            self.retained_payload_bytes -= self.payload_by_future.pop(future)
            self.trace.mark(index, 'future_result_enter')
            future.result()  # Only already-done futures, as in the unchanged V3 policy.
            self.trace.mark(index, 'future_result_exit')
        if self.error:
            raise RuntimeError('Immutable frame persistence failed') from self.error
        if self.submitted >= self.profile.maximum_groups:
            raise BufferError('Finite registered frame-group budget exhausted')
        if len(self.pending) >= self.maximum_pending:
            raise BufferError('Immutable frame queue full; no frame may be silently dropped')
        if self.retained_payload_bytes+self.profile.maximum_group_payload_bytes > (
                self.profile.maximum_retained_payload_bytes):
            raise BufferError('Immutable payload byte cap; no frame may be silently dropped')
        self.trace.mark(index, 'capacity_exit')

    def submit(self, group_id, files, stream_line, *, metadata_files=()):
        expected = {f'{group_id:04d}-{name}.bmp' for name in ('front','left','right','rear')}
        if (type(group_id) is not int or self.namespace.next_group != group_id+1
                or {name for name,_ in files} != expected):
            raise ValueError('Host payload submission requires its exact prior RAM reservation')
        return super().submit(group_id, files, stream_line, metadata_files=metadata_files)


@dataclass(frozen=True)
class HostFrameWriterFactory:
    profile: HostCopyProfile
    namespace: HostPayloadNamespace | None = None
    trace: HostTrace | None = None

    def __post_init__(self):
        if not isinstance(self.profile, HostCopyProfile):
            raise ValueError('Explicit hostV4 profile object required')

    def bind(self, namespace, trace):
        return HostFrameWriterFactory(self.profile, namespace, trace)

    def __call__(self, directory, stream_path, *, clock=time.perf_counter, mmf_metadata=False):
        if (not isinstance(self.namespace, HostPayloadNamespace)
                or not isinstance(self.trace, HostTrace) or mmf_metadata is not True):
            raise ValueError('Explicit bound host namespace/trace/MMF factory required')
        return HostImmutableFrameWriter(directory, stream_path, namespace=self.namespace,
            trace=self.trace, clock=clock, mmf_metadata=True)


def verify_host_persistence_archive(camera, *, expected_mmf=False):
    if expected_mmf is not True:
        raise ValueError('Explicit hostV4 MMF archive required')
    rows = verify_buffered_persistence_archive(camera, expected_mmf=True)
    namespace = camera.json('host-payload-namespace-v4.json')
    trace = camera.json('host-precopy-trace-v4.json')
    first_frame = camera.json('frame-0000.json')
    if (namespace.get('schema') != 'flybrain.host_payload_namespace.v4'
            or namespace.get('atomic_mkdir_exclusive_succeeded') is not True
            or namespace.get('directory_precreated_before_camera_request') is not True
            or namespace.get('no_per_group_filesystem_collision_query') is not True
            or namespace.get('reserved_groups') != list(range(len(rows)))
            or namespace.get('foreign_same_user_injection_excluded') is not False
            or namespace.get('request_id') != first_frame['request_id']
            or namespace.get('generation') != first_frame['generation']
            or not re.fullmatch(r'mmf-stream-[0-9a-f]{32}', namespace.get('request_id', ''))
            or type(namespace.get('maximum_groups')) is not int
            or not len(rows) <= namespace['maximum_groups'] <= 1200
            or namespace.get('actor_input') is not False
            or namespace.get('producer_Lua_directory_unchanged') is not True
            or trace.get('schema') != 'flybrain.host_precopy_trace.v4'
            or trace.get('phases') != list(PHASES) or trace.get('maximum_groups') != 1200
            or trace.get('events_per_group') != EVENTS_PER_GROUP
            or trace.get('binary_capacity_bytes') != TRACE_ROWS*TRACE_ROW.size
            or trace.get('source_receipts_changed') is not False
            or trace.get('serialized_after_owned_camera_terminal') is not True
            or trace.get('clock_domains') != ['host_QPC_nanoseconds',
                                              'current_thread_CPU_nanoseconds']
            or trace.get('events') != len(trace.get('records', []))
            or not 0 < trace.get('events', 0) <= TRACE_ROWS):
        raise ValueError('Host namespace/trace proof differs')
    counts = [0]*len(rows)
    per_group = [[] for _ in rows]
    last_group = -1
    last_wall = last_cpu = -1
    for group, phase, wall, cpu in trace['records']:
        if (type(group) is not int or not 0 <= group < len(rows)
                or type(phase) is not int or not 0 <= phase < len(PHASES)
                or type(wall) is not int or type(cpu) is not int
                or wall < last_wall or cpu < last_cpu or group < last_group):
            raise ValueError('Original bounded monotonic per-thread trace required')
        counts[group] += 1
        per_group[group].append(PHASES[phase])
        last_wall, last_cpu, last_group = wall, cpu, group
    if any(not 1 <= count <= EVENTS_PER_GROUP for count in counts):
        raise ValueError('Every original frame needs its own bounded host trace')
    mandatory = list(PHASES[6:])
    for phases in per_group:
        if (any(phases.count(p) != 1 for p in mandatory)
                or phases.count('capacity_enter') != 2 or phases.count('capacity_exit') != 2
                or [p for p in phases if p in mandatory] != mandatory
                or phases[0] != 'capacity_enter' or phases[-1] != 'capacity_exit'
                or phases.index('capacity_exit') > phases.index('helper_enter')
                or phases.index('ack_finished') > phases.index('capacity_enter', 1)
                or phases.count('future_done_enter') != phases.count('future_done_exit')
                or phases.count('future_result_enter') != phases.count('future_result_exit')):
            raise ValueError('Original capacity/helper/single-parse/copy/ACK trace order differs')
        second = phases.index('capacity_enter', 1)
        first_exit = phases.index('capacity_exit')
        if phases[first_exit+1:second] != mandatory:
            raise ValueError('Future bookkeeping may appear only inside capacity calls')
        for body in (phases[1:first_exit], phases[second+1:-1]):
            cursor = 0
            while cursor < len(body):
                if body[cursor:cursor+2] != ['future_done_enter','future_done_exit']:
                    raise ValueError('Original bounded Future.done pair required')
                cursor += 2
                if body[cursor:cursor+2] == ['future_result_enter','future_result_exit']:
                    cursor += 2
                elif cursor != len(body):
                    raise ValueError('An unfinished future ends the original capacity drain')
    for index,row in rows.items():
        if row['generation'] != namespace['generation']:
            raise ValueError('Original namespace/source generation differs')
        if not all(v['file'] == f'vision/{index:04d}-{v["view_id"]}.bmp' for v in row['views']):
            raise ValueError('Original flat host target names differ')
    return rows
