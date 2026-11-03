"""Synthetic UDP/ABI replay and read-only existing Windows mapping probe; never control AC."""

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import threading

from flybrain_drive.ac_image_buffer import ExistingNamedMapping, decode_image_buffer
from flybrain_drive.ac_probe import inspect_installation
from flybrain_drive.ac_telemetry import ChannelCalibration, LapTimeClock
from flybrain_drive.ac_transport import TelemetryClient, decode_telemetry, encode_control_request
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.sensor_contract import SampleTime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = begin_run(args.output, {"scope": "synthetic interoperability + native read-only probe",
        "upstream_commit": "ef0fa502bdd257fc2affad9361c02b3dbdfe1632",
        "actions_sent": False, "game_connected": False})
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2)
    sent = []
    errors = []
    fixtures = [{"steps": i, "timestamp_ac": 700 + i * .01, "completedLaps": lap,
        "currentTime": current, "iLastTime": last, "angular_velocity_x": i + 1,
        "angular_velocity_y": 2, "angular_velocity_z": 3,
        "world_position_x": 100, "NormalizedSplinePosition": .7}
        for i, (lap, current, last) in enumerate([(3, 900, 0), (3, 950, 0), (4, 10, 1000)])]

    def producer():
        try:
            data, peer = server.recvfrom(1024)
            sent.append(data.decode())
            if data != b"connect":
                raise ValueError("Unexpected fixture subscription")
            server.sendto(b"identified", peer)
            for row in fixtures:
                server.sendto(json.dumps(row).encode(), peer)
        except Exception as error:
            errors.append(str(error))

    worker = threading.Thread(target=producer)
    worker.start()
    try:
        with TelemetryClient("127.0.0.1", server.getsockname()[1], timeout=2) as client:
            client.subscribe()
            packets = [client.receive() for _ in fixtures]
        worker.join(3)
        if worker.is_alive() or errors or sent != ["connect"]:
            raise RuntimeError("Synthetic UDP producer failed: " + repr(errors))
        rows = [{"kind": "synthetic_loopback_fixture", "received_monotonic_seconds":
                 packet.received_monotonic_seconds,
                 "wire_base64": base64.b64encode(packet.wire_bytes).decode(),
                 "wire_sha256": hashlib.sha256(packet.wire_bytes).hexdigest()}
                for packet in packets]
        recording = run / "synthetic-telemetry.jsonl"
        recording.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
        clock = LapTimeClock(.001)  # Explicit fixture scale, not native clock verification.
        profile = ChannelCalibration("angular_velocity", (
            "angular_velocity_x", "angular_velocity_y",
            "angular_velocity_z"), [[0, 0, 2], [1, 0, 0], [0, -1, 0]], "synthetic-axis-fixture")
        replay = []
        for row in map(json.loads, recording.read_text(encoding="utf8").splitlines()):
            wire = base64.b64decode(row["wire_base64"], validate=True)
            assert hashlib.sha256(wire).hexdigest() == row["wire_sha256"]
            packet = decode_telemetry(wire, row["received_monotonic_seconds"])
            candidate_time = clock.update(packet)
            stamp = SampleTime(candidate_time, packet.received_monotonic_seconds)
            body = profile.convert(packet, stamp)
            replay.append({"sequence": packet.sequence, "fixture_simulation_time": candidate_time,
                           "angular_velocity": body.values.tolist(), "valid": body.valid.tolist()})
        image = decode_image_buffer(struct.pack("<I", 1) + b"\xff"*6
                                     + bytes([5, 10, 20, 0, 0, 0]), 2, 1, "bgr")
        assert image.pixels[0, 1].tolist() == [0, 0, 0]
        mappings = {}
        for name in ("Local\\acpmf_physics", "Local\\acpmf_graphics", "Local\\DualBufferImage"):
            if os.name != "nt":
                mappings[name] = {"status": "not_windows"}
                continue
            try:
                with ExistingNamedMapping(name, 4) as mapping:
                    mappings[name] = {"status": "exists_unverified_source", "header_hex":
                                      mapping.read().hex()}
            except FileNotFoundError:
                mappings[name] = {"status": "absent_not_created"}
            except OSError as error:
                mappings[name] = {"status": "error", "error": type(error).__name__}
        write_json(run / "replay.json", replay)
        write_json(run / "installation.json", inspect_installation())
        summary = {"synthetic_packets_received_and_replayed": len(packets),
            "synthetic_clock_times": [r["fixture_simulation_time"] for r in replay],
            "only_fixture_subscription_sent": sent, "actuator_messages_sent": 0,
            "image_black_pixels_preserved": True, "native_mappings": mappings,
            "native_units_axes_and_steering_mapping_verified": False,
            "native_simulation_clock_verified": False, "image_exposure_metadata_available": False,
            "actor_stream_created": False, "game_runtime_verified": False,
            "dry_request_codec": json.loads(encode_control_request(.2, 0, 1, 2, steering_sign=-1))}
        write_json(run / "summary.json", summary)
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False})
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error)})
        raise
    finally:
        server.close()
        worker.join(3)


if __name__ == "__main__":
    main()
