"""UNSEALED prospective actor cadence/trace; no live work on import.

QPC and current-thread CPU diagnostics never replace original sensor clocks.
Existing frame hostV4, recorder defaults and all historical sources are unchanged.
"""

import hashlib
import json
import math
import struct
import threading
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 'flybrain.actor_cadence_trace.v1'
PHASES = ('loop_enter', 'budget_started', 'child_poll', 'status', 'candidate_wait',
          'native_receive', 'candidate_select', 'decision', 'retinal_pack',
          'event_encode', 'event_write', 'event_flush', 'sleep', 'loop_exit')
ROW = struct.Struct('<IHHQQ')
POINT, START, END, ERROR = 0, 1, 2, 4
ITERATIONS, SLOTS, DECISIONS = 8192, 24, 2048


@dataclass(frozen=True)
class ActorCadenceProfile:
    nominal_seconds: float = .03

    def record(self):
        if type(self.nominal_seconds) is not float or self.nominal_seconds != .03:
            raise ValueError('Only the explicit new30ms actor profile is proposed')
        return dict(schema=SCHEMA, nominal_actor_seconds=.03,
            actor_budget_starts='before_child_poll_and_status', camera_period_seconds=.04,
            candidate_wait_seconds=.04, cached_candidate_age_seconds=.11,
            maximum_camera_age_seconds=.15, maximum_actor_interval_seconds=.08,
            maximum_heartbeat_age_seconds=2., maximum_iterations=ITERATIONS,
            records_per_iteration=SLOTS, raw_trace_capacity_bytes=ITERATIONS*SLOTS*ROW.size,
            trace_counter_bytes=ITERATIONS*2, original_budget_clock_bytes=ITERATIONS*8,
            requested_sleep_seconds_bytes=ITERATIONS*8,
            original_decision_clock_bytes=DECISIONS*8, original_clock_values_unchanged=True,
            scope='actor_timing_and_trace_only_no_policy_or_performance_claim')

    @classmethod
    def from_record(cls, value):
        profile = cls()
        if value != profile.record():
            raise ValueError('Exact new actor-cadence trace profile required')
        return profile


def load_actor_profile(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise ValueError('Bounded explicit actor profile required')
    return ActorCadenceProfile.from_record(json.loads(raw))


class _Span:
    __slots__ = ('trace', 'phase')

    def __init__(self, trace, phase):
        self.trace, self.phase = trace, phase

    def __enter__(self):
        self.trace.mark(self.phase, START)

    def __exit__(self, kind, value, traceback):
        self.trace._require_active()
        try:
            self.trace.mark(self.phase, END | (ERROR if kind else 0))
        except BaseException:
            # Only diagnostic cleanup is caught. The original source operation
            # remains primary and the iteration finisher rejects further work.
            self.trace._latch('actor diagnostic span-end mark failed')


class ActorTrace:
    def __init__(self, *, wall=time.perf_counter_ns, cpu=time.thread_time_ns):
        self.data = bytearray(ITERATIONS*SLOTS*ROW.size)
        self.per_iteration = array('H', [0])*ITERATIONS
        self.budget_clocks = array('d', [0.])*ITERATIONS
        self.sleep_requests = array('d', [-1.])*ITERATIONS
        self.decision_clocks = array('d', [0.])*DECISIONS
        self.wall, self.cpu = wall, cpu
        self.iterations = self.records = self.decisions = 0
        self.owner = threading.get_ident()
        self.active = None
        self.last = None
        self.closed = False
        self.failure = None

    def _latch(self, reason):
        self._require_owner()
        if self.failure is None:
            self.failure = reason

    def _require_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Open same-thread actor iteration required')

    def _require_active(self):
        self._require_owner()
        if self.closed or self.active is None:
            raise RuntimeError('Open same-thread actor iteration required')

    def begin_iteration(self):
        self._require_owner()
        if (self.closed or self.active is not None or self.failure
                or self.iterations >= ITERATIONS):
            raise RuntimeError('Actor trace exhausted/closed/failed before another acquisition')
        self.active = self.iterations
        self.iterations += 1
        try:
            self.mark('loop_enter')
        except BaseException:
            self._latch('actor diagnostic initial mark failed')
        if self.failure:
            self.finish_iteration()

    def mark(self, phase, flags=POINT):
        self._require_active()
        if self.failure:
            return
        if (phase not in PHASES or type(flags) is not int or not 0 <= flags <= 65535
                or self.per_iteration[self.active] >= SLOTS):
            self._latch('actor phase/flag/capacity failure')
            return
        try:
            wall, cpu = self.wall(), self.cpu()
        except BaseException:
            self._latch('actor diagnostic wall/CPU sampler failed')
            return
        if (type(wall) is not int or type(cpu) is not int
                or not 0 <= wall < (1 << 64) or not 0 <= cpu < (1 << 64)
                or self.last is not None and (wall < self.last[0] or cpu < self.last[1])):
            self._latch('actor diagnostic clock invalid/outside_uint64/decreasing')
            return
        try:
            ROW.pack_into(self.data, self.records*ROW.size,
                          self.active, PHASES.index(phase), flags, wall, cpu)
        except BaseException:
            # The unpublished slot can be partial. Previously retained records
            # are untouched and records/counters never advance on failed pack.
            self._latch('actor diagnostic row packing failed')
            return
        self.records += 1
        self.per_iteration[self.active] += 1
        self.last = wall, cpu

    def budget_started(self, original_seconds):
        self._require_active()
        if (self.active is None or type(original_seconds) not in (int, float)
                or not math.isfinite(original_seconds) or original_seconds < 0):
            raise ValueError('Original actor budget clock required')
        self.budget_clocks[self.active] = original_seconds
        self.mark('budget_started')

    def decision(self, index, original_seconds):
        self._require_active()
        if (type(index) is not int or index != self.decisions or index >= DECISIONS
                or type(original_seconds) not in (int, float)
                or not math.isfinite(original_seconds) or original_seconds < 0):
            self._latch('original decision sequence/clock capacity failure')
            return
        self.decision_clocks[index] = original_seconds
        self.decisions += 1
        self.mark('decision', index)

    def span(self, phase):
        return _Span(self, phase)

    def sleep_requested(self, seconds):
        self._require_active()
        if (self.active is None or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or not 0 <= seconds <= .03
                or self.sleep_requests[self.active] != -1.):
            self._latch('original sleep request missing/duplicate/outside explicit actor profile')
            return
        self.sleep_requests[self.active] = seconds

    def finish_iteration(self, *, original_error_active=False):
        self._require_active()  # A foreign finisher must never clear the owner.
        try:
            self.mark('loop_exit')
        except BaseException:
            self._latch('actor diagnostic final mark failed')
        finally:
            self.active = None
        if self.failure and not original_error_active:
            raise RuntimeError(self.failure+
                               '; current recorded event retained, no next acquisition')

    def seal(self):
        self._require_owner()
        if self.active is not None:
            raise RuntimeError('Cannot seal an active actor iteration')
        self.closed = True
        return self

    def index(self):
        if not self.closed:
            raise RuntimeError('Actor trace must be sealed')
        return dict(schema=SCHEMA, profile=ActorCadenceProfile().record(), phases=list(PHASES),
            trace_struct=ROW.format, record_bytes=ROW.size, records=self.records,
            iterations=self.iterations, decisions=self.decisions,
            records_per_iteration=self.per_iteration[:self.iterations].tolist(),
            original_budget_clocks=self.budget_clocks[:self.iterations].tolist(),
            requested_sleep_seconds=self.sleep_requests[:self.iterations].tolist(),
            original_decision_clocks=self.decision_clocks[:self.decisions].tolist(),
            clocks=['host_QPC_nanoseconds', 'current_actor_thread_CPU_nanoseconds'],
            failure=self.failure, complete=self.failure is None,
            sensor_receipt_or_decision_clocks_changed=False, crash_durable=False)


def cleanup_complete(value):
    camera = value.get('camera', {}) if isinstance(value, dict) else {}
    return (isinstance(value, dict)
        and value.get('schema') == 'flybrain.collection.owned_cleanup.v4'
        and value.get('complete') is True and value.get('control_process_terminal') is True
        and all(camera.get(k) is True for k in ('camera_thread_joined', 'producer_terminal_known',
            'persistence_joined', 'precompute_joined', 'shared_reader_closed')))


class ActorTraceSink:
    """Same-parent RAM handoff; uncertain cleanup keeps RAM/fence, never adds workers."""
    def __init__(self):
        self.trace, self.cleanup, self.persisted = None, None, False

    def accept(self, trace, cleanup):
        if self.trace is not None or not isinstance(trace, ActorTrace) or not trace.closed:
            raise RuntimeError('Exactly one sealed original actor trace required')
        self.trace, self.cleanup = trace, cleanup

    def persist(self, output, *, context):
        if self.persisted:
            raise RuntimeError('No repeated actor trace persistence')
        if self.trace is None:
            return dict(status='not_captured', fit_allowed=False)
        if (not cleanup_complete(self.cleanup) or context is not None
                and any(t is not None and t.is_alive() for t in (
                    context.poll_thread, context.write_thread))):
            return dict(status='deferred_in_owned_RAM_cleanup_unproven',
                trace_records=self.trace.records, crash_durable=False, fit_allowed=False)
        directory = Path(output)
        directory.mkdir(exist_ok=False)
        raw = memoryview(self.trace.data)[:self.trace.records*ROW.size].toreadonly()
        with (directory/'trace.bin').open('xb') as stream:
            stream.write(raw)
        index = self.trace.index()
        index.update(binary_file='trace.bin', binary_bytes=len(raw),
            binary_sha256=hashlib.sha256(raw).hexdigest(), owned_cleanup=self.cleanup,
            serialized_after_owned_workers_terminal=True)
        with (directory/'index.json').open('x', encoding='utf8') as stream:
            json.dump(index, stream, indent=2, allow_nan=False)
            stream.write('\n')
        self.persisted = True
        return dict(status='complete' if index['complete'] else 'failed_trace_preserved',
            index_file=str(directory/'index.json'), records=self.trace.records,
            decisions=self.trace.decisions, fit_allowed=False)


def verify_actor_trace(directory, events, *, expected_cleanup=None):
    directory = Path(directory)
    if not 0 < (directory/'index.json').stat().st_size <= 2*1024*1024:
        raise ValueError('Bounded actor trace index required')
    index = json.loads((directory/'index.json').read_bytes())
    ActorCadenceProfile.from_record(index['profile'])
    if expected_cleanup is not None and index['owned_cleanup'] != expected_cleanup:
        raise ValueError('Actor trace must bind exact same recorder cleanup/request/'
                         'control process')
    if (index['schema'] != SCHEMA or index['phases'] != list(PHASES)
            or index['complete'] is not True or index['failure'] is not None
            or index['serialized_after_owned_workers_terminal'] is not True
            or not cleanup_complete(index['owned_cleanup'])
            or index['binary_file'] != 'trace.bin'
            or not 0 < index['records'] <= ITERATIONS*SLOTS
            or not 0 < index['iterations'] <= ITERATIONS
            or type(index['records']) is not int or type(index['iterations']) is not int
            or index['binary_bytes'] != index['records']*ROW.size):
        raise ValueError('Complete exact bounded actor trace/cleanup required')
    if (directory/'trace.bin').stat().st_size != index['binary_bytes']:
        raise ValueError('Bounded original actor binary size differs')
    raw = (directory/'trace.bin').read_bytes()
    if (len(raw) != index['binary_bytes']
            or hashlib.sha256(raw).hexdigest() != index['binary_sha256']):
        raise ValueError('Original bounded actor trace bytes differ')
    rows = list(ROW.iter_unpack(raw))
    if len(rows) != index['records']:
        raise ValueError('Every actor trace record required')
    counts, decisions, previous = [0]*index['iterations'], [], None
    opened = None
    groups = [[] for _ in counts]
    last_iteration = 0
    for iteration, phase, flags, wall, cpu in rows:
        if (not 0 <= iteration < index['iterations'] or not 0 <= phase < len(PHASES)
                or iteration not in (last_iteration, last_iteration+1)
                or previous is not None and (wall < previous[0] or cpu < previous[1])):
            raise ValueError('Original actor trace clock/order differs')
        last_iteration = iteration
        previous = wall, cpu
        counts[iteration] += 1
        name = PHASES[phase]
        groups[iteration].append((name, flags))
        if name == 'decision':
            if opened is not None or flags != len(decisions):
                raise ValueError('Each original decision exactly once and ordered')
            decisions.append(flags)
        elif name in ('loop_enter', 'loop_exit', 'budget_started'):
            if flags != POINT or opened is not None:
                raise ValueError('Actor point inside open stage')
        elif flags == START:
            if opened is not None:
                raise ValueError('Unexpected nested actor stage')
            opened = iteration, phase
        elif flags in (END, END | ERROR):
            if opened != (iteration, phase):
                raise ValueError('Unmatched actor stage end')
            opened = None
        else:
            raise ValueError('Failed actor stage is not qualified')
    allowed = {
        ('child_poll',),
        ('child_poll', 'status', 'sleep'),
        ('child_poll', 'status', 'candidate_wait'),
        ('child_poll', 'status', 'candidate_wait', 'native_receive', 'status'),
        ('child_poll', 'status', 'candidate_wait', 'native_receive', 'candidate_select'),
        ('child_poll', 'status', 'candidate_wait', 'native_receive', 'candidate_select',
         'decision', 'retinal_pack', 'event_encode', 'event_write', 'event_flush', 'sleep'),
    }
    for group in groups:
        logical = tuple(name for name, flag in group
                        if (flag in (END, END | ERROR) and name != 'decision')
                        or name == 'decision')
        if (group[:2] != [('loop_enter', POINT), ('budget_started', POINT)]
                or group[-1] != ('loop_exit', POINT) or logical not in allowed
                or sum(name == 'loop_enter' for name, _ in group) != 1
                or sum(name == 'loop_exit' for name, _ in group) != 1
                or sum(name == 'budget_started' for name, _ in group) != 1):
            raise ValueError('Exact original actor stage/iteration order required')
    sleep_requests = index['requested_sleep_seconds']
    if len(sleep_requests) != len(groups):
        raise ValueError('Every original sleep request required')
    for group, request in zip(groups, sleep_requests, strict=True):
        slept = ('sleep', START) in group
        if (type(request) not in (int, float) or not math.isfinite(request)
                or slept and not 0 <= request <= .03 or not slept and request != -1.):
            raise ValueError('Original actor sleep request/branch differs')
    if (opened is not None or counts != index['records_per_iteration']
            or any(not 5 <= count <= SLOTS for count in counts)
            or decisions != list(range(len(events)))
            or index['decisions'] != len(events)
            or len(index['original_budget_clocks']) != index['iterations']
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                   for t in index['original_budget_clocks'])
            or index['original_decision_clocks'] != [
                e['decision_monotonic_seconds'] for e in events]):
        raise ValueError('Original actor event/clock/count linkage differs')
    return dict(schema=SCHEMA, iterations=len(counts), records=len(rows), decisions=len(events),
        binary_sha256=index['binary_sha256'], original_clocks_exact=True,
        CPU_less_than_wall_assumed=False, model_or_fit_qualified=False)
