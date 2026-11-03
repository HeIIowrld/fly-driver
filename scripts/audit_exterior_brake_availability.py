"""Audit native brake samples in retained exterior histories, not teacher authority."""

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sensors = Path(args.sensors)
    source = read(sensors / "result.json")
    if source["status"] != "completed_spatial_flight_sensor_replay":
        raise ValueError("Completed exterior sensor source required")
    run = begin_run(args.output, {
        "program_sha256": sha256(__file__),
        "sensors_result_sha256": sha256(sensors / "result.json"),
        "brake_threshold": 0.02, "teacher_authority_proven": False,
        "camera_native_physics_synchronization_proven": False, "actor_deployed": False,
    })
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    reports = []
    for entry in source["episodes"]:
        root, vision = Path(entry["recording"]), Path(entry["output"])
        events = root / "events.jsonl"
        manifest = read(vision / "manifest.json")["config"]
        bundle = read(root / "vision/bundle.json")
        archive = root / "vision" / bundle["file"]
        receipt = root / "control/api-receipt.json"
        if sha256(events) != manifest["events_sha256"] or sha256(archive) != bundle["sha256"]:
            raise ValueError("Original event and camera archive hashes required")
        telemetry = read(receipt)["telemetry"]
        if not telemetry:
            raise ValueError("Original lifecycle telemetry required")
        rows, brake = 0, []
        with zipfile.ZipFile(archive) as z, events.open(encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                native = row["native"]
                raw = base64.b64decode(native["prefix_base64"], validate=True)
                if hashlib.sha256(raw).hexdigest() != native["prefix_sha256"]:
                    raise ValueError("Original native packet hash mismatch")
                sample = decode_physics_prefix(raw, native["received_monotonic_seconds"])
                if sample.packet_id != native["packet_id"]:
                    raise ValueError("Original packet identity mismatch")
                rows += 1
                if sample.fields["brake"] <= 0.02:
                    continue
                meta = json.loads(z.read(f'frame-{row["camera"]["capture_id"]:04d}.json'))
                if meta["generation"] != row["camera"]["generation"]:
                    raise ValueError("Original camera generation mismatch")
                clock = meta["script_clock"]
                nearest = min(telemetry, key=lambda item: abs(item["script_clock"] - clock))
                brake.append({"row": row["index"], "brake": sample.fields["brake"],
                    "speed_kmh": sample.fields["speedKmh"], "camera_source_clock": clock,
                    "nearest_phase": nearest["phase"], "nearest_ai": nearest["ai_controlled"],
                    "nearest_clock_offset_seconds": clock - nearest["script_clock"]})
        if rows != entry["observations"]:
            raise ValueError("Original event count mismatch")
        reports.append({"id": entry["id"], "role": entry["role"], "rows": rows,
            "events_sha256": sha256(events), "archive_sha256": bundle["sha256"],
            "receipt_sha256": sha256(receipt), "brake_samples": brake})
    write_json(run / "result.json", {"status": "completed_brake_availability_audit",
        "episodes": reports, "nearest_telemetry_does_not_prove_authority": True,
        "physics_time_sync_verified": False, "actor_deployed": False, "goal_achieved": False})
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
