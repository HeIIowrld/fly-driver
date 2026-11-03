import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_csp_clock import CSPClockBounds, clock_ping


@pytest.mark.parametrize('narrow_at,expected', [(1, 4), (6, 6), (20, 12)])
def test_initial_clock_adds_bounded_real_exchanges_without_widening_bounds(
        monkeypatch, narrow_at, expected):
    import flybrain_drive.ac_csp_clock as module

    clock, calls = CSPClockBounds('one'), []

    def ping(root, bounds, hwnd, *, host_clock=None):
        calls.append(hwnd)
        sent = 100.+len(calls)
        width = .006 if len(calls) >= narrow_at else .03
        bounds.observe(sent, sent+width, sent-90+width/2, 'one')

    monkeypatch.setattr(module, 'clock_ping', ping)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    module.initialize_clock(Path('.'), clock, 17)
    assert len(calls) == expected == len(clock.pings)
    assert clock.high-clock.low == pytest.approx(.03 if narrow_at == 20 else .006)
    assert clock.maximum_width == .1


def test_clock_intersection_bounds_age_and_rejects_generation_or_clock_drift():
    c = CSPClockBounds('one')
    c.observe(100, 100.04, 10.01, 'one')  # offset in [89.99,90.03]
    with pytest.raises(ValueError, match='Two'):
        c.age_upper(100.1, 10.05, 'one')
    c.observe(100.1, 100.12, 10.105, 'one')  # intersection [89.995,90.015]
    assert c.record()['uncertainty_seconds'] == pytest.approx(.02)
    assert c.age_upper(100.2, 10.1, 'one') == pytest.approx(.105)
    before = c.record()
    with pytest.raises(ValueError, match='disagree'):
        c.observe(101, 101.01, 9, 'one')
    assert c.record() == before
    with pytest.raises(ValueError, match='generation'):
        c.observe(101, 101.01, 11, 'two')


def test_clock_wide_uncertainty_and_future_capture_are_rejected():
    c = CSPClockBounds('one')
    c.observe(100, 101, 10, 'one')
    c.observe(101, 102, 11, 'one')
    with pytest.raises(ValueError, match='bounded'):
        c.age_upper(102, 11, 'one')
    c.observe(102, 102.01, 12, 'one')
    with pytest.raises(ValueError, match='after decision'):
        c.age_upper(102, 13, 'one')


def test_direct_render_lower_requires_bounded_exchanges_and_nonnegative_source():
    c = CSPClockBounds('one')
    with pytest.raises(ValueError, match='Two consistent bounded'):
        c.render_start_lower(10, 'one')
    c.observe(100, 101, 10, 'one')
    c.observe(101, 102, 11, 'one')
    with pytest.raises(ValueError, match='Two consistent bounded'):
        c.render_start_lower(11, 'one')
    c.observe(102, 102.01, 12, 'one')
    with pytest.raises(ValueError, match='nonnegative'):
        c.render_start_lower(-1, 'one')
    assert c.render_start_lower(12, 'one') == pytest.approx(102)


def test_quantized_host_clock_expands_bounds_and_age_conservatively():
    c = CSPClockBounds('one', host_quantization=.015625)
    c.observe(100, 100.01, 10, 'one')
    c.observe(100.1, 100.11, 10.1, 'one')
    assert c.record()['uncertainty_seconds'] == pytest.approx(.04125)
    assert c.render_start_lower(10.1, 'one') == pytest.approx(100.06875)
    assert c.age_upper(100.12, 10.1, 'one') == pytest.approx(.05125)


def test_clock_ping_retries_transient_atomic_replace_with_same_nonce(tmp_path, monkeypatch):
    import flybrain_drive.ac_csp_clock as module

    monkeypatch.setattr(module.uuid, 'uuid4', lambda: SimpleNamespace(hex='fixed'))
    path = tmp_path / 'clock-reply.json'
    path.write_text(json.dumps(dict(nonce='fixed', window_handle='7ULL',
                                   producer_clock=10., generation='g')))
    original = Path.read_text
    calls = []

    def racing_read(file, *args, **kwargs):
        if file == path:
            calls.append(file)
            if len(calls) == 1:
                raise PermissionError('atomic replacement sharing race')
        return original(file, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', racing_read)
    clock = CSPClockBounds('g')
    clock_ping(tmp_path, clock, 7)
    assert len(calls) == 2 and len(clock.pings) == 1
