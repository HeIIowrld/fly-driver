"""Causal round-trip bounds for a CSP script clock; never a physical exposure clock."""

import json
import math
import time
import uuid

from .evidence import write_json


def clock_ping(root, clock, hwnd, *, host_clock=None):
    host_clock = host_clock or time.monotonic
    nonce = uuid.uuid4().hex
    sent = host_clock()
    write_json(root / 'clock-ping.json', dict(nonce=nonce))
    deadline = sent + 1.
    while host_clock() < deadline:
        path = root / 'clock-reply.json'
        if path.exists():
            try:
                reply = json.loads(path.read_text(encoding='utf8'))
            except (PermissionError, FileNotFoundError):
                # CSP atomically replaces this file. Retry this same nonce only,
                # inside its original deadline; never restart the producer.
                time.sleep(.002)
                continue
            received = host_clock()
            if reply.get('nonce') == nonce:
                if int(str(reply['window_handle']).removesuffix('ULL')) != hwnd:
                    raise ValueError('Clock reply came from another engine window')
                clock.observe(sent, received, reply['producer_clock'], reply['generation'])
                return
        time.sleep(.002)
    raise TimeoutError('No matching clock nonce within one second')


def initialize_clock(root, clock, hwnd, *, host_clock=None):
    """Intersect4..12 causal exchanges before images; never redate recorded frames."""
    for index in range(12):
        clock_ping(root, clock, hwnd, host_clock=host_clock)
        if index >= 3 and clock.high-clock.low <= .01:
            return
        # Additional exchanges change polling phase instead of widening the age budget.
        phase_delays = (.027, .031, .041, .047, .019, .007, .037, .029, .043)
        time.sleep(.04 if index < 3 else phase_delays[index-3])


class CSPClockBounds:
    def __init__(self, generation, *, maximum_width=.1, host_quantization=0.):
        if not isinstance(generation, str) or not generation:
            raise ValueError('Producer generation required')
        if not math.isfinite(maximum_width) or maximum_width <= 0:
            raise ValueError('Finite positive uncertainty budget required')
        if not math.isfinite(host_quantization) or not 0 <= host_quantization <= .1:
            raise ValueError('Bounded host clock quantization required')
        self.generation, self.maximum_width = generation, maximum_width
        self.host_quantization = host_quantization
        self.low, self.high, self.pings = -math.inf, math.inf, []

    def observe(self, sent, received, producer_clock, generation):
        if (generation != self.generation
                or not all(math.isfinite(v) for v in (sent, received, producer_clock))
                or not 0 <= sent <= received or producer_clock < 0):
            raise ValueError('Causal same-generation clock reply required')
        low = max(self.low, sent-producer_clock-self.host_quantization)
        high = min(self.high, received-producer_clock+self.host_quantization)
        if low > high:
            raise ValueError('Clock bounds disagree; unit rate/origin cannot be assumed')
        self.low, self.high = low, high
        self.pings.append(dict(sent=sent, received=received, producer_clock=producer_clock))

    def age_upper(self, decision, producer_render_started, generation):
        if (generation != self.generation or len(self.pings) < 2
                or self.high-self.low > self.maximum_width
                or not all(math.isfinite(v) for v in (decision, producer_render_started))
                or min(decision, producer_render_started) < 0):
            raise ValueError('Two consistent bounded clock exchanges required')
        age = decision-self.render_start_lower(producer_render_started, generation)
        if age < 0:
            raise ValueError('Image starts after decision even at earliest clock bound')
        return age

    def render_start_lower(self, producer_render_started, generation):
        if (generation != self.generation or not math.isfinite(producer_render_started)
                or producer_render_started < 0):
            raise ValueError('Same-generation finite nonnegative source clock required')
        if len(self.pings) < 2 or self.high-self.low > self.maximum_width:
            raise ValueError('Two consistent bounded clock exchanges required')
        # Pad once more for rounding of a later host decision timestamp.
        return producer_render_started+self.low-self.host_quantization

    def record(self):
        if not self.pings:
            raise ValueError('Clock has no observations')
        result = dict(generation=self.generation, offset_lower_seconds=self.low,
            offset_upper_seconds=self.high, uncertainty_seconds=self.high-self.low,
            maximum_width_seconds=self.maximum_width, exchanges=self.pings,
            unit_rate_assumption='tested by intersecting exchanges; limited observed duration',
            physical_exposure_sync_verified=False)
        if self.host_quantization:
            result['host_clock_quantization_seconds'] = self.host_quantization
        return result
