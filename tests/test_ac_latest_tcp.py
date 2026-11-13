import time
import threading
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_latest_tcp import LatestTcpTelemetryClient
from test_ac_tcp import load_app


def buffered_client(rows, *, now, previous_source):
    """Deterministic selection fixture: no worker, socket, sleep or re-dating."""
    client = object.__new__(LatestTcpTelemetryClient)
    client.clock = lambda: now
    client.timeout, client.maximum_age = .15, .05
    client.condition = threading.Condition()
    client.stopped = threading.Event()
    client.error = None
    client.packets = [SimpleNamespace(sequence=n, source_monotonic_seconds=s,
        received_monotonic_seconds=r) for n, s, r in rows]
    client.latest = client.packets[-1]
    client.last_selected, client.last_selected_source = rows[0][0]-1, previous_source
    client.cadence_constrained_selections = 0
    client.source_admission = None
    return client


def test_buffer_selects_received_fresh_sample_within_cadence_without_repair():
    client = buffered_client([(61, .0414, 10.0143), (62, .0628, 10.0292),
        (63, .0829, 10.0438)], now=10.046, previous_source=0.)
    selected = client.receive(minimum_source_interval=.02, maximum_source_interval=.08)
    assert selected is client.packets[1]
    assert selected.received_monotonic_seconds == 10.0292
    assert client.cadence_constrained_selections == 1
    # Next decision may consume the previously too-distant newest real packet.
    assert client.receive(minimum_source_interval=.02, maximum_source_interval=.08) is client.latest


def test_buffer_does_not_select_stale_intermediate_sample_to_hide_real_gap():
    client = buffered_client([(61, .0628, 9.99), (63, .1029, 10.0438)],
        now=10.046, previous_source=0.)
    with pytest.raises(ValueError, match='No fresh received TCP packet'):
        client.receive(minimum_source_interval=.02, maximum_source_interval=.08)
    assert client.last_selected_source == 0.
    observed = client.cadence_failure_observation
    assert observed['buffered_packets_in_window'] == 1
    assert observed['fresh_receipts_in_window'] == 0
    assert observed['latest_source_interval_seconds'] == pytest.approx(.1029)


def test_buffer_chooses_newest_eligible_original_packet():
    client = buffered_client([(61, .0414, 10.0143), (62, .0628, 10.0292)],
        now=10.046, previous_source=0.)
    assert client.receive(minimum_source_interval=.02, maximum_source_interval=.08) is client.latest
    assert client.cadence_constrained_selections == 0


def test_explicit_new_history_first_sample_starts_new_cadence_anchor():
    client = buffered_client([(61, .1628, 10.0292)], now=10.046,
        previous_source=0.)
    with pytest.raises(ValueError, match='No fresh received TCP packet'):
        client.receive(minimum_source_interval=.02, maximum_source_interval=.08)
    assert client.cadence_failure_observation['buffered_packets_in_window'] == 0
    # Only the caller's explicit new-builder boundary uses unconstrained cadence.
    assert client.receive(minimum_source_interval=.02, maximum_source_interval=None) is client.latest
    assert client.last_selected_source == .1628


def test_source_age_admission_waits_for_new_source_instead_of_using_old_parse():
    now=time.perf_counter()
    client=buffered_client([(1,.02,now)],now=now,previous_source=0.)
    client.clock=time.perf_counter
    client.source_admission=lambda p:p.sequence==2
    started=threading.Event()
    selected=[]
    def consume():
        started.set()
        selected.append(client.receive(minimum_source_interval=.02,maximum_source_interval=.08))
    worker=threading.Thread(target=consume)
    worker.start()
    assert started.wait(1)
    with client.condition:
        fresh=SimpleNamespace(sequence=2,source_monotonic_seconds=.04,
            received_monotonic_seconds=time.perf_counter())
        client.packets.append(fresh)
        client.latest=fresh
        client.condition.notify_all()
    worker.join(1)
    assert not worker.is_alive() and selected==[fresh]


def wait_packet(client, sequence):
    with client.condition:
        assert client.condition.wait_for(lambda: client.latest is not None and
            client.latest.sequence >= sequence, timeout=2)


def test_continuous_source_drain_preserves_original_receipts_and_raw_sequence():
    app = load_app()
    producer = app.Publisher(port=0)
    client = LatestTcpTelemetryClient(port=producer.port, timeout=1, maximum_age=.05)
    try:
        producer.publish({'speed': 2})
        wait_packet(client, 1)
        first = client.receive()
        producer.publish({'speed': 3})
        wait_packet(client, 2)
        second = client.receive()
        assert (first.sequence, second.sequence) == (1, 2)
        assert client.packets[0] is first and client.packets[1] is second
        assert first.received_monotonic_seconds <= second.received_monotonic_seconds
        assert first.raw['speed'] == 2
    finally:
        client.close()
        producer.close()
    assert not client.thread.is_alive()


def test_stale_original_receipt_and_collector_errors_are_never_repaired():
    app = load_app()
    producer = app.Publisher(port=0)
    now = [time.perf_counter()]
    client = LatestTcpTelemetryClient(port=producer.port, timeout=1,
        clock=lambda: now[0])
    try:
        producer.publish({'speed': 2})
        wait_packet(client, 1)
        original = client.latest.received_monotonic_seconds
        now[0] += .051
        with pytest.raises(TimeoutError, match='Original TCP receipt'):
            client.receive()
        assert client.latest.received_monotonic_seconds == original
        with client.condition:
            client.error = ValueError('injected sequence/session failure')
        with pytest.raises(RuntimeError, match='collector failed'):
            client.receive()
    finally:
        client.close()
        producer.close()


def test_real_capture_does_not_drop_valid_packet_during_68ms_engine_hitch():
    # Source/receipt clocks from demonstration-06; no redating or interpolation.
    now=[115821.8128723]
    client=buffered_client([(419519,8697.1007761,115821.789185)],
        now=now[0],previous_source=8697.0804813)
    client.clock=lambda:now[0]
    capture={419519:115821.7823677,419520:115821.8503285}
    client.source_admission=lambda p:0<=now[0]-capture[p.sequence]<=.04
    first=client.receive(minimum_source_interval=.02,maximum_source_interval=.08)
    assert first.sequence==419519 and now[0]-capture[first.sequence]>.03
    now[0]=115821.860
    fresh=SimpleNamespace(sequence=419520,source_monotonic_seconds=8697.1687369,
        received_monotonic_seconds=115821.8599001)
    client.packets.append(fresh);client.latest=fresh
    assert client.receive(minimum_source_interval=.02,maximum_source_interval=.08) is fresh
    assert fresh.source_monotonic_seconds-first.source_monotonic_seconds<.08


def test_socket_timeout_keeps_terminal_original_receipt():
    producer = load_app().Publisher(port=0)
    client = LatestTcpTelemetryClient(port=producer.port, timeout=.5)
    try:
        producer.publish({'speed': 1})
        wait_packet(client, 1)
        original = client.receive()
        with client.condition:
            assert client.condition.wait_for(lambda: client.error is not None, timeout=2)
        failure = client.failure_observation
        assert failure['kind'] == 'socket_timeout'
        assert failure['last_original_sequence'] == 1
        assert failure['last_original_receipt'] == original.received_monotonic_seconds
        assert failure['observed_monotonic_seconds'] > original.received_monotonic_seconds
        with pytest.raises(RuntimeError, match='collector failed'):
            client.receive()
    finally:
        client.close()
        producer.close()
