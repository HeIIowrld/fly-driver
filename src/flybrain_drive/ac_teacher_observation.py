"""Privileged SDK/native ->125-value SAC observations; no fly actor or actuator API."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, NativePhysicsSample
from .ac_sac_teacher import CHANNELS, PROFILES, encode_teacher_observation
from .ac_transport import TelemetryPacket


class TeacherInputUnavailable(ValueError):
    """Required SDK extension is not present; never substitute simulation truth or zero."""


@dataclass(frozen=True)
class TeacherObservation:
    state: object
    observation: object
    producer_session_id: str
    sdk_sequence: int
    native_packet_id: int
    host_receipt_skew_seconds: float
    actor_input: bool = False
    exposure_synchronization_verified: bool = False


class PrivilegedTeacherObservationBuilder:
    def __init__(self, geometry, reference_curve, *, profile, max_receipt_skew=.05,
                 minimum_interval=.03, maximum_interval=.08):
        if (profile not in PROFILES
                or not np.isfinite([max_receipt_skew, minimum_interval, maximum_interval]).all()
                or not 0 < max_receipt_skew <= .25
                or not 0 < minimum_interval < maximum_interval <= 1):
            raise ValueError('Reviewed teacher profile and finite clock bounds required')
        self.geometry, self.curve, self.profile = geometry, reference_curve, profile
        self.max_receipt_skew = max_receipt_skew
        self.minimum_interval, self.maximum_interval = minimum_interval, maximum_interval
        self.history = deque(maxlen=3)
        self.previous = None

    def build(self, packet, native):
        if not isinstance(packet, TelemetryPacket) or not isinstance(native, NativePhysicsSample):
            raise ValueError('Explicit SDK packet and native sample required')
        namespace = packet.raw.get('privileged_teacher')
        if namespace is None:
            raise TeacherInputUnavailable('SDK privileged_teacher extension not loaded')
        if (not isinstance(namespace, Mapping)
                or namespace.get('protocol') != 'flybrain.ac.privileged_teacher.v1'
                or namespace.get('actor_input') is not False):
            raise ValueError('Explicit teacher-only SDK protocol required')
        extra = namespace.get('raw', {})
        if not isinstance(extra, Mapping):
            raise ValueError('Teacher raw SDK fields must be a mapping')
        required_extra = (set(CHANNELS) - {'speed', 'gap', 'angular_velocity_y'}) | {
            'world_position_x', 'world_position_y', 'NormalizedSplinePosition', 'track_length_m'}
        required_body = {'speed', 'steerAngle', 'accStatus', 'brakeStatus', 'angular_velocity_y'}
        missing = sorted(k for k in required_extra if extra.get(k) is None)
        missing += sorted(k for k in required_body if packet.raw.get(k) is None)
        if missing:
            raise TeacherInputUnavailable('Missing teacher SDK values: ' + ', '.join(missing))
        state = {k: float(extra[k]) for k in required_extra}
        state.update({k: float(packet.raw[k]) for k in required_body})
        if not np.isfinite(list(state.values())).all():
            raise ValueError('Nonfinite teacher SDK values')
        session = packet.raw.get('producer_session_id')
        if not isinstance(session, str) or not session or len(session) > 200:
            raise ValueError('Explicit SDK producer session required')
        if not np.isfinite([packet.source_monotonic_seconds,
                            packet.received_monotonic_seconds,
                            native.received_monotonic_seconds]).all():
            raise ValueError('Finite separate source/host clocks required')
        skew = abs(packet.received_monotonic_seconds - native.received_monotonic_seconds)
        if skew > self.max_receipt_skew:
            raise ValueError('SDK/native host receipt skew exceeds declared bound')
        if self.previous is not None:
            old_session, old_sequence, old_native, old_source, old_host = self.previous
            if (session != old_session or packet.sequence <= old_sequence
                    or native.packet_id <= old_native
                    or packet.received_monotonic_seconds <= old_host):
                raise ValueError('Teacher session/sequence changed; explicit new builder required')
            elapsed = packet.source_monotonic_seconds - old_source
            if not self.minimum_interval <= elapsed <= self.maximum_interval:
                raise ValueError('Teacher source interval differs from declared decision cadence')
        values = PHYSICS_PREFIX.unpack(native.prefix_bytes)
        raw, offset = {}, 0
        for name, _, width in PHYSICS_LAYOUT:
            if width == 1:
                raw[name] = values[offset]
            offset += width
        if (raw['packetId'] != native.packet_id or not np.isfinite(raw['heading'])
                or not 0 <= raw['numberOfTyresOut'] <= 4):
            raise ValueError('Native teacher metadata ABI differs')
        if (not 0 <= state['NormalizedSplinePosition'] <= 1
                or not 100 <= state['track_length_m'] <= 100000
                or not 0 <= state['accStatus'] <= 1 or not 0 <= state['brakeStatus'] <= 1):
            raise ValueError('Teacher progress, track length or actual pedals out of range')
        state['yaw'] = -raw['heading']
        state['numberOfTyresOut'] = raw['numberOfTyresOut']
        state['out_of_track'] = int(raw['numberOfTyresOut'] > 2)
        state['LapDist'] = state['NormalizedSplinePosition'] * state['track_length_m']
        state.update(self.geometry.features(state['world_position_x'], state['world_position_y'],
                                            state['yaw']))
        curvature = self.curve.segment(state['LapDist'])
        observation = None
        if len(self.history) == 3:
            observation = encode_teacher_observation(state, list(self.history), curvature,
                                                       profile=self.profile)
            observation.setflags(write=False)
        state['sensors'].setflags(write=False)
        state = MappingProxyType(state)
        self.history.append(state)
        self.previous = (session, packet.sequence, native.packet_id,
                         packet.source_monotonic_seconds, packet.received_monotonic_seconds)
        return TeacherObservation(state, observation, session, packet.sequence,
                                  native.packet_id, skew)
