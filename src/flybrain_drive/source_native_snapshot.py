"""Decode a source-coherent teacher/body bundle and measure OS-clock age.

QPC aligns processes on this Windows machine; it does not identify a physical
simulation tick or image exposure. Actor consumers must still exclude heading.
"""
import base64
import ctypes
import math

from .ac_native_physics import decode_physics_prefix


class WindowsQpc:
    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        for name in ('QueryPerformanceCounter', 'QueryPerformanceFrequency'):
            getattr(self.api, name).argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        value = ctypes.c_longlong()
        if not self.api.QueryPerformanceFrequency(ctypes.byref(value)) or value.value <= 0:
            raise RuntimeError('QPC frequency unavailable')
        self.frequency = value.value

    def ticks(self):
        value = ctypes.c_longlong()
        if not self.api.QueryPerformanceCounter(ctypes.byref(value)):
            raise RuntimeError('QPC read failed')
        return value.value


def source_age_seconds(packet, clock):
    row = packet.raw.get('source_native_snapshot')
    if (row is None or row.get('protocol') != 'flybrain.ac.source_native.v1'
            or row.get('repeated_copy_equal') is not True
            or row.get('physical_tick_time_verified') is not False
            or row.get('camera_exposure_sync_verified') is not False):
        raise ValueError('Explicit unpromoted source-native clock contract required')
    start, end, frequency = (row.get(k) for k in (
        'qpc_started_ticks', 'qpc_finished_ticks', 'qpc_frequency_hz'))
    if (any(type(x) is not int for x in (start,end,frequency))
            or frequency != clock.frequency or not 0 < start <= end
            or not 0 <= (end-start)/frequency <= .005):
        raise ValueError('Native/QPC frequency, order or capture bracket differs')
    # Measure from the start of the bracket, including all capture and IPC work.
    age = (clock.ticks()-start)/frequency
    if not math.isfinite(age) or age < -1/frequency:
        raise ValueError('Source QPC is future or nonfinite')
    return max(0., age)


def decode_source_native(packet, clock, *, maximum_age=.05):
    if not 0 < maximum_age <= .05:
        raise ValueError('Source age bound must be within 50ms')
    if source_age_seconds(packet, clock) > maximum_age:
        raise TimeoutError('Original source QPC capture exceeds freshness budget')
    row = packet.raw['source_native_snapshot']
    data = base64.b64decode(row['prefix_base64'], validate=True)
    # This is the native byte bundle's real host receipt, not a new timestamp.
    return decode_physics_prefix(data, packet.received_monotonic_seconds)
