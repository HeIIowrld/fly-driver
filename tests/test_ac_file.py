"""Missing native networking must not prevent loading a read-only AC sensor publisher."""

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_file import FileTelemetryClient

APP_PATH = (Path(__file__).resolve().parents[1] / "integrations/ac/flybrain_telemetry"
            / "flybrain_telemetry.py")


def app_without_network(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("socket", "threading", "uuid"):
            raise ImportError("Simulated missing embedded networking")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("ac_file_fixture", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedded_app_loads_without_socket_and_file_packets_remain_raw(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    with pytest.raises(ImportError, match="networking"):
        app.Publisher(port=0)
    publisher = app.FilePublisher(str(tmp_path))
    client = FileTelemetryClient(tmp_path, timeout=.02)
    with pytest.raises(TimeoutError, match="No new"):
        client.receive()
    publisher.publish({"fl_wheel_load": 1024, "fl_tire_temperature_core": 75,
                       "simulation_clock_verified": False})
    first = client.receive()
    assert first.sequence == 1 and first.raw["fl_wheel_load"] == 1024
    assert not first.raw["simulation_clock_verified"]
    with pytest.raises(TimeoutError):
        client.receive()
    publisher.publish({"fl_wheel_load": None})
    assert client.receive().raw["fl_wheel_load"] is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest.json", "producer.json"]


def test_file_telemetry_uses_explicit_receipt_clock(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    client = FileTelemetryClient(tmp_path, clock=lambda: 123.456789)
    publisher.publish({'fl_wheel_load': 1000})
    assert client.receive().received_monotonic_seconds == 123.456789
    publisher.close()
    with pytest.raises(ValueError, match="closed"):
        publisher.publish({})


def test_file_transport_rejects_session_change_reset_and_oversized(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    client = FileTelemetryClient(tmp_path)
    publisher.publish({})
    client.receive()
    publisher._write("latest.json", {"steps": 0, "timestamp_ac": 1,
                                     "producer_session_id": publisher.session})
    with pytest.raises(ValueError, match="sequence reset"):
        client.receive()
    publisher._write("latest.json", {"steps": 2, "timestamp_ac": 1,
                                     "producer_session_id": "other"})
    with pytest.raises(ValueError, match="session changed"):
        client.receive()
    (tmp_path / "latest.json").write_bytes(b" " * 65536)
    with pytest.raises(ValueError, match="exceeds"):
        client.receive()
    hello = json.loads((tmp_path / "producer.json").read_text())
    hello["read_only"] = False
    (tmp_path / "producer.json").write_text(json.dumps(hello))
    with pytest.raises(ValueError, match="read-only"):
        FileTelemetryClient(tmp_path)


def test_windows_sharing_race_retries_but_persistent_denial_has_a_deadline(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    client = FileTelemetryClient(tmp_path, timeout=.02, poll_seconds=.001)
    publisher.publish({})
    real_read, attempts = client._read, []

    def transient(name):
        attempts.append(name)
        if len(attempts) == 1:
            error = PermissionError(13, 'synthetic sharing violation')
            error.winerror = 32
            raise error
        return real_read(name)

    monkeypatch.setattr(client, '_read', transient)
    assert client.receive().sequence == 1 and client.transient_read_retries == 1

    def denied(name):
        error = PermissionError(13, 'persistent sharing violation')
        error.winerror = 32
        raise error

    monkeypatch.setattr(client, '_read', denied)
    with pytest.raises(PermissionError, match='persistent'):
        client.receive()


def test_sdk_app_automatically_selects_local_files_without_native_socket(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    messages = []
    fake_ac = SimpleNamespace(newApp=lambda name: 1, setSize=lambda *args: None,
                             log=messages.append, getCarState=lambda *args: None)
    monkeypatch.setitem(sys.modules, "ac", fake_ac)
    monkeypatch.setitem(sys.modules, "acsys", SimpleNamespace(CS=SimpleNamespace()))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert app.acMain("fixture") == "FlybrainTelemetry"
    try:
        assert isinstance(app._publisher, app.FilePublisher)
        app.acUpdate(.04)
        packet = FileTelemetryClient(tmp_path / "flybrain/telemetry").receive()
        assert packet.sequence == 1 and packet.raw["fl_wheel_load"] is None
        assert packet.raw["elapsed_seconds_candidate"] == .04
        assert packet.raw["simulation_clock_verified"] is False
        assert any("local file telemetry" in message for message in messages)
    finally:
        app.acShutdown()
    assert app._publisher is None


def test_embedded_publish_sharing_error_does_not_escape_into_engine_and_next_update_recovers(
        tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    messages = []
    monkeypatch.setitem(sys.modules, 'ac', SimpleNamespace(
        newApp=lambda name: 1, setSize=lambda *args: None, log=messages.append,
        getCarState=lambda *args: None))
    monkeypatch.setitem(sys.modules, 'acsys', SimpleNamespace(CS=SimpleNamespace()))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    app.acMain('fixture')
    try:
        original = app._publisher._write
        attempts = []

        def replace_once(name, fields):
            attempts.append(name)
            if len(attempts) == 1:
                raise PermissionError(13, 'synthetic replace sharing violation')
            return original(name, fields)

        monkeypatch.setattr(app._publisher, '_write', replace_once)
        app.acUpdate(.04)
        assert app._publish_errors == 1
        app._last_emit = -1
        app.acUpdate(.04)
        packet = FileTelemetryClient(tmp_path / 'flybrain/telemetry').receive()
        assert packet.sequence == 2 and packet.raw['publisher_errors'] == 1
        assert not packet.raw['simulation_clock_verified']
    finally:
        app.acShutdown()


def test_diagnostic_snapshot_reuse_never_redates_the_original_packet(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    stamp = [10.]
    client = FileTelemetryClient(tmp_path, clock=lambda: stamp[0])
    publisher.publish({'fl_wheel_load': 1000})
    first = client.receive()
    stamp[0] += .04
    assert client.snapshot() is first
    assert client.snapshot().received_monotonic_seconds == 10.
    stamp[0] += .011
    with pytest.raises(TimeoutError, match='stale'):
        client.snapshot()
    publisher.publish({'fl_wheel_load': 1100})
    assert client.snapshot().received_monotonic_seconds == stamp[0]


def test_publisher_replace_retry_keeps_identical_serialized_measurement(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    original, attempts = app.os.replace, []

    def transient(source, destination):
        attempts.append(Path(source).read_bytes())
        if len(attempts) == 1:
            error = PermissionError(13, 'reader still holds snapshot')
            error.winerror = 5
            raise error
        return original(source, destination)

    monkeypatch.setattr(app.os, 'replace', transient)
    publisher.publish({'fl_wheel_load': 1234.})
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    row = json.loads((tmp_path/'latest.json').read_text(encoding='utf8'))
    assert row == json.loads(attempts[0]) and row['steps'] == 1
    assert row['fl_wheel_load'] == 1234. and publisher.replace_retries == 1


def test_persistent_publisher_denial_preserves_previous_file_and_is_not_suppressed(
        tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    before = (tmp_path/'latest.json').read_bytes()

    def denied(*args):
        error = PermissionError(13, 'persistent writer denial')
        error.winerror = 32
        raise error

    monkeypatch.setattr(app.os, 'replace', denied)
    with pytest.raises(PermissionError, match='persistent writer denial'):
        publisher.publish({'fl_wheel_load': 1234.})
    assert (tmp_path/'latest.json').read_bytes() == before
    assert publisher.replace_retries > 0


def test_latched_publish_timing_retains_failed_hitch_without_redating_measurement(
        tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    stamp = [100.]
    monkeypatch.setattr(app, 'time', SimpleNamespace(
        perf_counter=lambda: stamp[0], sleep=lambda seconds: None))
    original = app.os.replace

    def blocked(source, destination):
        stamp[0] += .214
        error = PermissionError(13, 'delayed replacement failed')
        error.winerror = 32
        raise error

    monkeypatch.setattr(app.os, 'replace', blocked)
    with pytest.raises(PermissionError):
        publisher.publish({'speed': 17.})
    failed = json.loads((tmp_path/'latest.json.partial').read_text())
    assert failed['timestamp_ac'] == 100. and failed['speed'] == 17.
    monkeypatch.setattr(app.os, 'replace', original)
    publisher.publish({'speed': 18.})
    publisher.publish({'speed': 19.})
    row = json.loads((tmp_path/'latest.json').read_text())
    timing = row['publisher_previous_io_timing']
    assert timing['actor_input'] is False
    replace = timing['stages']['replace_including_retry']
    assert replace['failures'] == 1
    assert replace['last']['succeeded'] is True
    assert replace['maximum']['duration_seconds'] == pytest.approx(.214)
    assert replace['maximum']['succeeded'] is False
    assert failed['timestamp_ac'] < row['timestamp_ac']


def test_file_open_delay_is_separate_from_serialization_and_replace(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    stamp = [100.]
    monkeypatch.setattr(app, 'time', SimpleNamespace(
        perf_counter=lambda: stamp[0], sleep=lambda seconds: None))

    def slow_open(*args, **kwargs):
        stamp[0] += .198
        return builtins.open(*args, **kwargs)

    monkeypatch.setattr(app, 'open', slow_open, raising=False)
    publisher.publish({'speed': 17.})
    row = json.loads((tmp_path/'latest.json').read_text())
    assert row['timestamp_ac'] == 100.
    publisher.publish({'speed': 18.})
    timing = json.loads((tmp_path/'latest.json').read_text())['publisher_previous_io_timing']
    assert timing['stages']['open_write_close']['maximum']['duration_seconds'] == pytest.approx(.198)
    assert timing['stages']['serialize']['last']['duration_seconds'] == 0.
    assert timing['stages']['replace_including_retry']['last']['duration_seconds'] == 0.


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows held-reader replacement')
def test_publisher_recovers_real_windows_read_handle_collision(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    reader = (tmp_path/'latest.json').open('rb')
    original, packets = app.os.replace, []

    def release_reader_after_first_attempt(source, destination):
        packets.append(Path(source).read_bytes())
        try:
            return original(source, destination)  # First call genuinely fails on Windows.
        finally:
            reader.close()

    monkeypatch.setattr(app.os, 'replace', release_reader_after_first_attempt)
    try:
        publisher.publish({'fl_wheel_load': 1234.})
    finally:
        reader.close()
    assert len(packets) == 2 and packets[0] == packets[1]
    assert publisher.replace_retries == 1
    assert (tmp_path/'latest.json').read_bytes() == packets[0]


def test_diagnostic_snapshot_rejects_changed_bytes_and_session(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    client = FileTelemetryClient(tmp_path, clock=lambda: 10.)
    publisher.publish({'fl_wheel_load': 1000})
    client.receive()
    value = json.loads((tmp_path/'latest.json').read_text(encoding='utf8'))
    value['fl_wheel_load'] = 999
    publisher._write('latest.json', value)
    with pytest.raises(ValueError, match='different raw bytes'):
        client.snapshot()
    value['producer_session_id'] = 'replacement'
    publisher._write('latest.json', value)
    with pytest.raises(ValueError, match='session changed'):
        client.snapshot()


def test_stale_snapshot_can_wait_for_a_genuinely_new_sequence_without_redating_old_packet(
        tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    publisher.publish({'fl_wheel_load': 1000})
    stamp = [10.]
    client = FileTelemetryClient(tmp_path, clock=lambda: stamp[0], poll_seconds=.001)
    original_packet = client.receive()
    original_read = client._read
    attempts = []

    def publishing_read(name):
        attempts.append(name)
        if len(attempts) == 2:
            publisher.publish({'fl_wheel_load': 1100})
        return original_read(name)

    monkeypatch.setattr(client, '_read', publishing_read)
    stamp[0] = 10.06
    packet = client.snapshot()
    assert packet.sequence > original_packet.sequence and client.stale_snapshot_retries == 1
    assert original_packet.received_monotonic_seconds == 10.
    assert packet.received_monotonic_seconds == 10.06


def test_snapshot_sharing_retry_preserves_original_receipt_and_still_rejects_stale(
        tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    stamp = [10.]
    client = FileTelemetryClient(tmp_path, clock=lambda: stamp[0], poll_seconds=.001)
    publisher.publish({})
    first = client.receive()
    original, attempts = client._read, []

    def transient(name):
        attempts.append(name)
        if len(attempts) == 1:
            error = PermissionError(13, 'sharing race')
            error.winerror = 32
            raise error
        return original(name)

    monkeypatch.setattr(client, '_read', transient)
    stamp[0] = 10.04
    assert client.snapshot() is first
    assert first.received_monotonic_seconds == 10. and client.transient_read_retries == 1
    stamp[0] = 10.06
    with pytest.raises(TimeoutError, match='stale'):
        client.snapshot()


def test_snapshot_persistent_or_unrelated_denial_is_not_hidden(tmp_path, monkeypatch):
    app = app_without_network(monkeypatch)
    publisher = app.FilePublisher(str(tmp_path))
    publisher.publish({})
    client = FileTelemetryClient(tmp_path, timeout=.02, poll_seconds=.001)

    def denied(name):
        error = PermissionError(13, 'persistent snapshot denial')
        error.winerror = 32
        raise error

    monkeypatch.setattr(client, '_read', denied)
    with pytest.raises(PermissionError, match='persistent snapshot'):
        client.snapshot()
