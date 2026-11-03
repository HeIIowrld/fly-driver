"""One repeated native copy for body and teacher flag; authority is never actor input."""

import base64
import hashlib
import math
import time
from dataclasses import dataclass

from .ac_image_buffer import ExistingNamedMapping
from .ac_native_physics import PHYSICS_PREFIX, NativePhysicsSample, decode_physics_prefix
from .ac_teacher_status import AI_STATUS_PREFIX


@dataclass(frozen=True)
class TeacherPhysicsSample:
    physics: NativePhysicsSample
    builtin_ai_controlled: bool
    authority_prefix_bytes: bytes

    def authority_record(self):
        return dict(schema='flybrain.native.same_copy_teacher_authority.v1',
            packet_id=self.physics.packet_id,
            received_monotonic_seconds=self.physics.received_monotonic_seconds,
            builtin_ai_controlled=self.builtin_ai_controlled,
            prefix_base64=base64.b64encode(self.authority_prefix_bytes).decode(),
            prefix_sha256=hashlib.sha256(self.authority_prefix_bytes).hexdigest(),
            body_prefix_sha256=hashlib.sha256(self.physics.prefix_bytes).hexdigest(),
            same_native_copy=True, actor_input=False, source_time_seconds=None,
            atomicity_verified=False, optimal_teacher_control_verified=False)


def decode_teacher_physics(data, received_monotonic_seconds):
    if not isinstance(data, bytes) or len(data) != AI_STATUS_PREFIX.size:
        raise ValueError('Exact reviewed420-byte physics/AI prefix required')
    packet, active = AI_STATUS_PREFIX.unpack(data)
    if packet < 0 or active not in (0, 1):
        raise ValueError('Native teacher flag/packet ABI differs')
    physics = decode_physics_prefix(data[:PHYSICS_PREFIX.size], received_monotonic_seconds)
    return TeacherPhysicsSample(physics, bool(active), data)


def repeated_teacher_snapshot(mapping, *, attempts=8, clock=time.perf_counter):
    if type(attempts) is not int or not 1 <= attempts <= 64:
        raise ValueError('Bounded repeated-copy attempts required')
    for _ in range(attempts):
        first, second = mapping.read(), mapping.read()
        if first == second:
            return decode_teacher_physics(second, clock())
    raise RuntimeError('Native420-byte copy changed during every consistency attempt')


def timed_teacher_snapshot(mapping, *, attempts=8, clock=time.perf_counter):
    """Bracket both equal native copies with host times; no producer-time claim.

    Earlier mismatched copies are counted but cannot establish a coherent sample.
    This observes mapping reads, not when the engine sampled or published physics.
    """
    if type(attempts) is not int or not 1 <= attempts <= 64:
        raise ValueError('Bounded repeated-copy attempts required')
    previous_end = None
    for attempt in range(1, attempts+1):
        start = clock()
        first = mapping.read()
        middle = clock()
        second = mapping.read()
        end = clock()
        if (not all(math.isfinite(x) for x in (start, middle, end))
                or start < 0 or not start <= middle <= end
                or previous_end is not None and start < previous_end):
            raise ValueError('Monotonic finite host-copy clock required')
        if first == second:
            sample = decode_teacher_physics(second, end)
            return sample, {'first_copy_start': start, 'first_copy_end': middle,
                            'second_copy_start': middle, 'second_copy_end': end,
                            'attempts': attempt, 'engine_atomicity_verified': False,
                            'physical_sample_time_verified': False}
        previous_end = end
    raise RuntimeError('Native420-byte copy changed during every consistency attempt')


class NativeTeacherPhysicsReader:
    """Keep privileged flag outside the original actor-facing308-byte sample.

    New packet receipt is still a host bound, not a verified engine sample time.
    Duplicate IDs do not create new actor observations or refresh old receipts.
    Matching repeated copies are a consistency check, not an engine seqlock.
    """

    def __init__(self, *, timeout=.15, poll_seconds=.001, clock=time.perf_counter, mapping=None):
        if not 0 < poll_seconds <= timeout <= 60:
            raise ValueError('Positive bounded native receive deadline required')
        self.timeout, self.poll_seconds, self.clock = timeout, poll_seconds, clock
        self.mapping = mapping if mapping is not None else ExistingNamedMapping(
            'Local\\acpmf_physics', AI_STATUS_PREFIX.size)
        self.last_packet = None

    def receive(self):
        deadline = time.monotonic()+self.timeout
        while True:
            coupled = repeated_teacher_snapshot(self.mapping, clock=self.clock)
            packet = coupled.physics.packet_id
            if self.last_packet is not None and packet < self.last_packet:
                raise ValueError('Native physics sequence reset; explicit new episode required')
            if packet != self.last_packet:
                self.last_packet = packet
                return coupled
            if time.monotonic() >= deadline:
                raise TimeoutError('No advancing native teacher/body packet before deadline')
            time.sleep(self.poll_seconds)

    def close(self):
        self.mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
