"""Replay actual coupled shadow intentions from retained pixels/native bytes and original times."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.record_ac_engine_ai_probe import verified_shadow


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--raw-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, audit = Path(args.recording), Path(args.raw_audit)
    recorded, source_config = read(root / "result.json"), read(root / "manifest.json")["config"]
    raw, raw_config = read(audit / "result.json"), read(audit / "manifest.json")["config"]
    arguments = source_config["arguments"]
    if (raw["status"] != "completed" or not raw["camera_aid_state_verified"]
            or any(raw["actor_replay_max_errors"].values())
            or raw_config["events_sha256"] != sha256(root / "events.jsonl")
            or raw_config["observation_sha256"] != sha256(root / "observations.npz")
            or raw_config["bundle_sha256"] != sha256(root / "vision/camera.zip")):
        raise ValueError("Exact aid-OFF raw replay required")
    config, proof = verified_shadow(arguments["flight_shadow_config"],
                                     arguments["flight_shadow_verification"])
    run = begin_run(args.output, dict(arguments=vars(args), seed=proof["initialization_seed"],
        runtime_configuration=config, verification=proof,
        source_manifest_sha256=sha256(root / "manifest.json"),
        events_sha256=sha256(root / "events.jsonl"),
        raw_audit_manifest_sha256=sha256(audit / "manifest.json"), actuator_messages_sent=0))
    result = dict(status="failed", goal_achieved=False, actor_qualified=False,
                  actual_actor_integration=False, student_actuator_messages_sent=0)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(proof["initialization_seed"])
        stream = FlightStream(arguments["flight_shadow_config"],
                              native_identity=read(root / "identity.json"), episode_id=root.name)
        rows = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        if len(rows) != recorded["decisions"]:
            raise ValueError("Every original observation must be retained")
        maximum, predictions, known_failure, durations, ages = 0., [], None, [], []
        with CameraArchive(root / "vision") as camera:
            for index, row in enumerate(rows):
                pixels = {}
                for view in row["camera"]["views"]:
                    raw_rgb = camera.read(Path(view["file"]).name)
                    assert hashlib.sha256(raw_rgb).hexdigest() == view["file_sha256"]
                    rgb = np.array(Image.open(io.BytesIO(raw_rgb)).convert("RGB"))
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == view["rgb_sha256"]
                    pixels[view["view_id"]] = rgb
                native = row["native"]
                sample = decode_virtual_body_prefix(base64.b64decode(native["prefix_base64"],
                    validate=True), native["received_monotonic_seconds"])
                assert sample.prefix_sha256 == native["prefix_sha256"]
                assert sample.packet_id == native["packet_id"]
                try:
                    response = stream.advance(clock=row["decision_monotonic_seconds"],
                        camera=row["camera"], pixels=pixels, native_sample=sample)
                except Exception as error:
                    observed = row.get("flight_shadow_error")
                    if (observed is None or index != len(rows) - 1
                            or observed["error"] != type(error).__name__
                            or observed["reason"] != str(error)):
                        raise
                    known_failure = dict(index=index, **observed)
                    break
                if "flight_shadow_error" in row:
                    raise ValueError("Live shadow failure did not reproduce from its input")
                actual = row["flight_shadow"]
                np.testing.assert_allclose(response["intent"], actual["intent"],
                                           atol=3e-6, rtol=2e-5)
                maximum = max(maximum, float(np.abs(np.asarray(response["intent"])
                                                    - actual["intent"]).max()))
                for key in ("decision_clock", "capture_id", "generation", "steps",
                            "visual_ticks", "flight_ticks", "recurrent_state_reset"):
                    assert response[key] == actual[key], key
                np.testing.assert_array_equal(stream.last_body.values, actual["body_values"])
                np.testing.assert_array_equal(stream.last_body.valid, actual["body_valid"])
                assert actual["actuator_messages_sent"] == 0 and row["foreground"] is False
                predictions.append(response["intent"])
                durations.append(actual["processing_seconds"])
                ages.append(actual["source_age_upper_at_output_seconds"])
        complete = recorded["status"] == "completed" and known_failure is None
        assert len(predictions) == recorded["flight_shadow"]["inferences"]
        if complete:
            assert len(predictions) == len(rows)
            assert read(root / "control/result.json")["status"] == "completed"
        np.savez_compressed(run / "replayed-intentions.npz", intent=np.array(predictions),
            original_decision_time=np.array([r["decision_monotonic_seconds"]
                                             for r in rows[:len(predictions)]]))
        intervals = np.diff([r["decision_monotonic_seconds"] for r in rows])
        result.update(status="completed_shadow_replay_audit", source_status=recorded["status"],
            complete_shadow_episode_verified=complete, original_observations=len(rows),
            matching_inferences=len(predictions), intent_max_error=maximum,
            reproduced_source_failure=known_failure, original_clocks_preserved=True,
            processing_seconds_percentiles=np.percentile(durations, [50, 95, 99, 100]).tolist()
                if durations else [],
            source_age_at_output_percentiles=np.percentile(ages, [50, 95, 99, 100]).tolist()
                if ages else [],
            original_interval_seconds_max=float(intervals.max()) if len(intervals) else None,
            intervals_above_80ms=int((intervals > .08).sum()),
            hidden_resets=0, vehicle_driver="owned finite engine-AI lease, not student",
            camera_aid_state_verified=True, online_sensory_inference_observed=bool(predictions),
            closed_loop_student_driving_verified=False, physiological_recognition_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / "result.json", result)
        raise
    write_json(run / "result.json", result)
    print(json.dumps(dict(status=result["status"], complete=complete, inferences=len(predictions),
                          error=maximum)))


if __name__ == "__main__":
    main()
