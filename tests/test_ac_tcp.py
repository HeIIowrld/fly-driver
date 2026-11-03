import ast
import hashlib
import importlib.util
import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_tcp import PROTOCOL, TcpTelemetryClient, record_packet

APP_PATH = (Path(__file__).resolve().parents[1] / "integrations" / "ac"
            / "flybrain_telemetry" / "flybrain_telemetry.py")


def load_app():
    spec = importlib.util.spec_from_file_location("flybrain_ac_fixture", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_native_tcp_publisher_reader_preserve_packets_without_control_data():
    app = load_app()
    producer = app.Publisher(port=0)
    try:
        with TcpTelemetryClient("127.0.0.1", producer.port, timeout=2) as client:
            client.subscribe()
            assert client.hello.raw["read_only"] is True
            assert client.hello.raw["simulation_clock_verified"] is False
            producer.publish({"angular_velocity_x": 3, "simulation_clock_verified": False})
            first = client.receive()
            producer.publish({"angular_velocity_x": 4})
            second = client.receive()
            assert (first.sequence, second.sequence) == (1, 2)
            assert first.raw["angular_velocity_x"] == 3
            digest = hashlib.sha256(first.wire_bytes).hexdigest()
            assert record_packet(first)["wire_sha256"] == digest
    finally:
        producer.close()
    assert not producer.thread.is_alive()


@pytest.mark.parametrize("mode", ["session", "reset", "oversized"])
def test_fragmented_tcp_and_invalid_session_or_sequence_or_length(mode):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(2)
    errors = []
    validation_done = threading.Event()

    def producer():
        try:
            with server.accept()[0] as peer:
                hello = dict(protocol=PROTOCOL, kind="hello", read_only=True,
                             producer_session_id="fixture", steps=0, timestamp_ac=1)
                wire = (json.dumps(hello) + "\n").encode()
                peer.sendall(wire[:5])
                peer.sendall(wire[5:])
                row = dict(steps=1, timestamp_ac=2, producer_session_id="fixture")
                peer.sendall((json.dumps(row) + "\n").encode())
                if mode == "oversized":
                    peer.sendall(b"x" * 65508 + b"\n")
                else:
                    row.update(producer_session_id="other" if mode == "session" else "fixture")
                    peer.sendall((json.dumps(row) + "\n").encode())
                # Keep the fixture alive until validation has consumed the bad
                # frame. Early half-close can race the Windows socket reader.
                assert validation_done.wait(5)
                peer.settimeout(2)
                try:
                    incoming = peer.recv(1)
                except ConnectionResetError:
                    # Closing with unread oversized bytes can reset TCP on Windows.
                    incoming = b""
                assert incoming == b""  # Reader sends no application/control data.
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=producer)
    worker.start()
    try:
        with TcpTelemetryClient("127.0.0.1", server.getsockname()[1], timeout=2) as client:
            client.subscribe()
            assert client.receive().sequence == 1
            with pytest.raises(ValueError, match="session|sequence|limit"):
                client.receive()
    finally:
        validation_done.set()
        worker.join(3)
        server.close()
    assert not worker.is_alive() and not errors


def test_sdk_access_is_read_only_and_missing_values_remain_missing():
    app = load_app()
    calls = []
    sources = ["LocalAngularVelocity", "WheelAngularSpeed", "SuspensionTravel",
               "CurrentTyresCoreTemp", "Load", "Steer",
               "Gas", "Brake", "SpeedMS", "LapCount"]
    cs = SimpleNamespace(**{name: name for name in sources})

    def get_state(car, source):
        calls.append((car, source))
        if source == "LocalAngularVelocity":
            return [1, 2, 3]
        raise RuntimeError("SDK field unavailable")

    row = app.read_body(SimpleNamespace(getCarState=get_state), SimpleNamespace(CS=cs))
    assert row["angular_velocity_z"] == 3 and row["steerAngle"] is None
    assert calls == [(0, source) for source in sources]
    assert not {"world_position_x", "NormalizedSplinePosition", "heading"} & row.keys()


def test_embedded_app_uses_python33_syntax_without_claiming_runtime_compatibility():
    ast.parse(APP_PATH.read_text(encoding="utf8"), feature_version=(3, 3))


def test_sdk_load_and_core_temperature_use_actual_game_feedback_fields():
    app = load_app()
    names = ("CurrentTyresCoreTemp", "Load")
    cs = SimpleNamespace(**{name: name for name in names})
    data = {"CurrentTyresCoreTemp": [30., 40., 50., 60.], "Load": [500., 600., 0., float("nan")]}
    raw = app.read_body(SimpleNamespace(getCarState=lambda car, key: data[key]),
                        SimpleNamespace(CS=cs))
    assert raw["fl_tire_temperature_core"] == 30 and raw["rr_tire_temperature_core"] == 60
    assert raw["fl_wheel_load"] == 500 and raw["rl_wheel_load"] == 0
    assert raw["rr_wheel_load"] is None
