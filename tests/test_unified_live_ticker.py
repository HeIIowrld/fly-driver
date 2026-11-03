import base64
import hashlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive import unified_live_ticker as live
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from test_unified_flight_sensorimotor import context


def test_mailbox_keeps_original_scene_and_refuses_overwrite():
    mailbox = live.SceneMailbox()
    image = torch.zeros(1, 576)
    frame = context(time.perf_counter())
    published = mailbox.publish(image, frame, 1)
    image.fill_(1)
    frame.decision_body.values[0] = 9
    item, consumed = mailbox.latest()
    assert consumed == 0 and item[3] == published and item[4] == 1
    assert torch.count_nonzero(item[1]) == 0
    assert item[2].decision_body.values[0] == 0
    with pytest.raises(TimeoutError, match='not consumed'):
        mailbox.publish(image, context(time.perf_counter()), 2, timeout=.001)
    mailbox.acknowledge(item[0])
    mailbox.publish(image, context(time.perf_counter()), 2)
    with pytest.raises(ValueError, match='acknowledgment'):
        mailbox.acknowledge(item[0])
    mailbox.close()
    with pytest.raises(RuntimeError, match='closed'):
        mailbox.publish(image, context(time.perf_counter()), 3)


def test_mailbox_rejects_future_and_stale_scene():
    mailbox = live.SceneMailbox()
    with pytest.raises(ValueError, match='future or stale'):
        mailbox.publish(torch.zeros(1, 576), context(time.perf_counter() + 1), 1)
    with pytest.raises(ValueError, match='future or stale'):
        mailbox.publish(torch.zeros(1, 576), context(time.perf_counter() - 1), 1)
    assert mailbox.latest()[0] is None


def test_ticker_uses_first_native_receipt_and_one_owner(monkeypatch):
    # Fully synthetic packets and circuit: this checks scheduling, not biology or live timing.
    class Shadow(live.UnifiedMultirateShadow):
        def __init__(self):
            self.guard = SimpleNamespace(model=SimpleNamespace(previous_time=None, previous_body_receipt=None))
            self.calls = []
        def update(self, *, when, body, new_scene, new_context, published_at):
            self.calls.append((when, body, new_scene, new_context, published_at))
            previous = self.guard.model.previous_time
            self.guard.model.previous_time = when
            self.guard.model.previous_body_receipt = body.original_received_monotonic_seconds
            return None if previous is None else SimpleNamespace(intent=torch.zeros(1, 2))

    def decode(raw, receipt):
        return SimpleNamespace(packet_id=raw[0], prefix_bytes=raw)
    def nav(raw, receipt):
        return SimpleNamespace(legacy=SimpleNamespace(packet_id=raw[0], prefix_sha256=hashlib.sha256(raw).hexdigest(), received_monotonic_seconds=receipt))
    class Encoder:
        def __init__(self, *args, **kwargs): pass
        def encode(self, sample, *, episode_id, decision_monotonic_seconds):
            return NavigationBodyObservation(episode_id, sample.legacy.packet_id, sample.legacy.received_monotonic_seconds, decision_monotonic_seconds, np.zeros(21), np.ones(21, dtype=bool), 'fixture')
    monkeypatch.setattr(live, 'decode_physics_prefix', decode)
    monkeypatch.setattr(live, 'decode_navigation_body_prefix', nav)
    monkeypatch.setattr(live, 'NavigationBodyEncoder', Encoder)
    rows = []
    def add(packet):
        raw = bytes([packet]) * live.PHYSICS_PREFIX.size
        received = time.perf_counter()
        rows.append({'authority': {'prefix_base64': base64.b64encode(raw).decode(), 'prefix_sha256': hashlib.sha256(raw).hexdigest(), 'received_monotonic_seconds': received, 'packet_id': packet}, 'copy': {'second_copy_end': received}})
        return received
    shadow = Shadow()
    ticker = live.UnifiedLiveBodyTicker(shadow, native_rows=rows, native_identity={}, episode_id='synthetic')
    ticker.start()
    try:
        add(1)
        receipt2 = add(2)
        ticker.publish(torch.zeros(1, 576), context(time.perf_counter()), 7)
        deadline = time.perf_counter() + 1
        while len(shadow.calls) < 1 and time.perf_counter() < deadline:
            time.sleep(.001)
        assert shadow.calls and shadow.calls[0][1].original_received_monotonic_seconds == receipt2
        add(3)
        while len(ticker.events) < 2 and time.perf_counter() < deadline:
            time.sleep(.001)
        assert len(shadow.calls) >= 2
        assert [event['status'] for event in ticker.events[:2]] == ['priming', 'shadow_output']
        assert shadow.calls[1][2] is None and shadow.calls[1][3] is None
        assert ticker.error is None
    finally:
        ticker.stop()
    assert not ticker.thread.is_alive()


def test_mailbox_rejects_repeated_or_reversed_capture_after_consumption():
    mailbox = live.SceneMailbox()
    image = torch.zeros(1, 576)
    mailbox.publish(image, context(time.perf_counter()), 10)
    item, _ = mailbox.latest()
    mailbox.acknowledge(item[0])
    for capture in (10, 9):
        with pytest.raises(ValueError, match='strictly advance'):
            mailbox.publish(image, context(time.perf_counter()), capture)
        assert mailbox.latest()[0][4] == 10
    mailbox.publish(image, context(time.perf_counter()), 11)
    assert mailbox.latest()[0][4] == 11


def test_near_expiry_defers_without_relaxing_limit():
    bounds = (1.0, 1.01, 1.02, 1.03)
    assert not live.defer_near_expiry(1.13, bounds)
    assert live.defer_near_expiry(1.145, bounds)
    with pytest.raises(ValueError, match='stale'):
        live.defer_near_expiry(1.151, bounds)
    with pytest.raises(ValueError, match='Causal'):
        live.defer_near_expiry(.99, bounds)
