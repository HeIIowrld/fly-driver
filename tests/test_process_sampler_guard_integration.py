import time

import pytest
import torch

from flybrain_drive.native_process_sampler import NativeProcessSampler
from flybrain_drive.process_body_history_bridge import ProcessBodyHistoryBridge
from flybrain_drive.temporal_body_context import HostFreshnessLimits
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from test_native_process_sampler import packet
from test_unified_flight_sensorimotor import make_model


IDENTITY = dict(car_model='bmw_z4_gt3', shared_memory_version='1.7', ac_version='1.16.4')


def test_recorded_process_copy_reaches_existing_history_and_output_guard():
    sampler = NativeProcessSampler(recorded=tuple(packet(i) for i in range(1, 5)),
                                   period=.005, capacity=128, max_samples=4)
    model = make_model()
    model.freshness_limits = HostFreshnessLimits(.05, .15)
    guard = UnifiedHostGuard(model)
    bridge = ProcessBodyHistoryBridge(sampler, guard, native_identity=IDENTITY, episode_id='test')
    sampler.start()
    rows = []
    try:
        for _ in range(4):
            row = bridge.collect(timeout=3)
            assert row is not None
            rows.append(row)
        sampler.stop()
    finally:
        if sampler.process.is_alive():
            sampler.stop()
    shadow = bridge.shadow
    assert shadow.position == 4
    receipt = rows[-1]['authority']['received_monotonic_seconds']
    decision = receipt + .002
    ctx = shadow.history.build(render_start_lowers=(receipt+.0005,)*4,
                               camera_receipt_time=receipt+.001, decision_time=decision)
    assert ctx.decision_body.original_received_monotonic_seconds == receipt
    clock_values = iter((decision, decision+.001))
    guard.clock = lambda: next(clock_values)
    args = dict(decision_time=decision, racing_goals=torch.tensor([[0., 1.]]),
                scene_features=torch.zeros(1,576), scene_valid=torch.ones(1,4,dtype=torch.bool),
                body_context=ctx)
    assert guard.step(**args) is None  # Priming must not publish.
    assert not guard.failed

    # The identical source receipt becomes stale during calculation; it is
    # rejected at output even if it was fresh when the step started.
    late_model = make_model()
    late_model.freshness_limits = HostFreshnessLimits(.05, .15)
    late_clocks = iter((decision, receipt+.051))
    late_guard = UnifiedHostGuard(late_model, clock=lambda: next(late_clocks))
    with pytest.raises(ValueError, match='stale at output'):
        late_guard.step(**args)
    assert late_guard.failed
    with pytest.raises(RuntimeError, match='explicit restart'):
        late_guard.step(**args)


def test_process_failure_does_not_generate_a_history_row():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.01)
    model = make_model()
    model.freshness_limits = HostFreshnessLimits(.05, .15)
    guard = UnifiedHostGuard(model)
    bridge = ProcessBodyHistoryBridge(sampler, guard, native_identity=IDENTITY, episode_id='test')
    sampler.start()
    try:
        original_poll = sampler.poll
        sampler.poll = lambda **kwargs: original_poll(clock=lambda: time.perf_counter()+1, **kwargs)
        with pytest.raises(RuntimeError, match='stale or future'):
            bridge.collect(timeout=3)
        assert guard.failed and bridge.shadow.position == 0
        with pytest.raises(RuntimeError, match='explicit restart'):
            bridge.collect(timeout=3)
    finally:
        sampler.stop()


def test_queue_overflow_latches_existing_guard():
    sampler = NativeProcessSampler(recorded=(packet(),), period=.002, capacity=1)
    model = make_model()
    model.freshness_limits = HostFreshnessLimits(.05, .15)
    guard = UnifiedHostGuard(model)
    bridge = ProcessBodyHistoryBridge(sampler, guard, native_identity=IDENTITY, episode_id='test')
    sampler.start()
    sampler.process.join(5)
    assert not sampler.process.is_alive()
    with pytest.raises(RuntimeError, match='queue full'):
        bridge.collect(timeout=.2)
    assert guard.failed and bridge.shadow.position == 0
    with pytest.raises(RuntimeError, match='queue full'):
        sampler.stop()
