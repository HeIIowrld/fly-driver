import base64
import hashlib
import struct

import pytest

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.ac_teacher_physics import (
    NativeTeacherPhysicsReader,
    decode_teacher_physics,
    repeated_teacher_snapshot,
    timed_teacher_snapshot,
)


def packet(number=7, active=1):
    raw = bytearray(420)
    struct.pack_into('<i', raw, 0, number)
    struct.pack_into('<f', raw, 4, .4)
    struct.pack_into('<f', raw, 8, .3)
    struct.pack_into('<f', raw, 28, 60.)
    struct.pack_into('<i', raw, 416, active)
    return bytes(raw)


class Mapping:
    def __init__(self, values):
        self.values = iter(values)
        self.last = None
        self.closed = False

    def read(self):
        self.last = next(self.values, self.last)
        return self.last

    def close(self):
        self.closed = True


@pytest.mark.parametrize('active', [0, 1])
def test_same_copy_preserves_exact_original_actor_prefix_receipt_and_privileged_flag(active):
    raw = packet(active=active)
    coupled = decode_teacher_physics(raw, 10.5)
    original = decode_physics_prefix(raw[:PHYSICS_PREFIX.size], 10.5)
    assert coupled.physics == original
    assert len(coupled.physics.prefix_bytes) == 308
    assert coupled.builtin_ai_controlled is bool(active)
    record = coupled.authority_record()
    assert base64.b64decode(record['prefix_base64']) == raw
    assert record['prefix_sha256'] == hashlib.sha256(raw).hexdigest()
    assert record['body_prefix_sha256'] == hashlib.sha256(original.prefix_bytes).hexdigest()
    assert record['received_monotonic_seconds'] == original.received_monotonic_seconds
    assert not record['actor_input'] and not record['atomicity_verified']
    assert not record['optimal_teacher_control_verified']
    assert coupled.physics.fields['gas'] == pytest.approx(.4)
    assert coupled.physics.fields['brake'] == pytest.approx(.3)
    assert 'builtin_ai_controlled' not in coupled.physics.fields


@pytest.mark.parametrize('raw', [packet(active=2), packet(number=-1), packet()[:-1]])
def test_invalid_flag_sequence_or_length_is_not_silently_repaired(raw):
    with pytest.raises(ValueError):
        decode_teacher_physics(raw, 10.)


def test_same_packet_with_changing_flag_does_not_pass_repeated_copy_consistency():
    with pytest.raises(RuntimeError):
        repeated_teacher_snapshot(Mapping([packet(active=0), packet(active=1)]), attempts=1)


def test_matching_full_copy_decodes_body_without_a_second_mapping_read_or_clock():
    clocks = iter([2., 3.])
    sample = repeated_teacher_snapshot(Mapping([packet(), packet()]), clock=lambda: next(clocks))
    assert sample.physics.received_monotonic_seconds == 2.
    assert next(clocks) == 3.


def test_timed_snapshot_brackets_successful_pair_after_mismatch():
    clocks = iter([1., 1.01, 1.02, 2., 2.01, 2.03])
    sample, timing = timed_teacher_snapshot(
        Mapping([packet(1), packet(2), packet(3), packet(3)]), clock=lambda: next(clocks))
    assert sample.physics.packet_id == 3
    assert sample.physics.received_monotonic_seconds == 2.03
    assert timing['first_copy_start'] == 2.
    assert timing['first_copy_end'] == timing['second_copy_start'] == 2.01
    assert timing['second_copy_end'] == 2.03 and timing['attempts'] == 2
    assert timing['physical_sample_time_verified'] is False


def test_timed_snapshot_rejects_nonmonotonic_clock():
    clocks = iter([1., .9, 1.1])
    with pytest.raises(ValueError, match='clock'):
        timed_teacher_snapshot(Mapping([packet(), packet()]), clock=lambda: next(clocks))


def test_new_packet_reader_skips_duplicate_without_creating_an_actor_observation():
    source = Mapping([packet(7), packet(7), packet(7), packet(7), packet(8), packet(8)])
    with NativeTeacherPhysicsReader(mapping=source, clock=lambda: 10.) as reader:
        assert reader.receive().physics.packet_id == 7
        assert reader.receive().physics.packet_id == 8
    assert source.closed


def test_reader_sequence_reset_requires_new_episode():
    source = Mapping([packet(8), packet(8), packet(7), packet(7)])
    with NativeTeacherPhysicsReader(mapping=source, clock=lambda: 10.) as reader:
        reader.receive()
        with pytest.raises(ValueError, match='reset'):
            reader.receive()


def test_reader_constant_packet_times_out_without_fabricating_time_or_values():
    with NativeTeacherPhysicsReader(mapping=Mapping([packet(), packet()]),
                                   timeout=.002, poll_seconds=.001, clock=lambda: 10.) as reader:
        reader.receive()
        with pytest.raises(TimeoutError):
            reader.receive()
