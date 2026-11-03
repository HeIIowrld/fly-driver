"""Compare one-observation coupled inference with independently retained original stages."""

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--reference-vision", required=True)
    parser.add_argument("--reference-body", required=True)
    parser.add_argument("--reference-evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, vision, body, evaluation = map(Path, [args.recording, args.reference_vision,
                                               args.reference_body, args.reference_evaluation])
    config = read(args.config)
    events_hash = sha256(root / "events.jsonl")
    visual_receipt, body_receipt, evaluated = (read(path / "result.json")
                                             for path in (vision, body, evaluation))
    for path in (vision, body):
        if read(path / "manifest.json")["config"]["events_sha256"] != events_hash:
            raise ValueError("Every reference must retain the same original episode")
    if (sha256(vision / "responses.npz") != visual_receipt["responses_sha256"]
            or sha256(body / "virtual-body.npz") != body_receipt["virtual_body_sha256"]
            or evaluated["source_record"]["source_events_sha256"] != events_hash
            or read(evaluation / "manifest.json")["config"]["checkpoint_sha256"]
            != config["checkpoint_sha256"]):
        raise ValueError("Exact retained references and fixed checkpoint required")
    run = begin_run(args.output, dict(arguments=vars(args), seed=863009,
        config_sha256=sha256(args.config), events_sha256=events_hash,
        vision_responses_sha256=visual_receipt["responses_sha256"],
        body_response_sha256=body_receipt["virtual_body_sha256"],
        evaluation_result_sha256=sha256(evaluation / "result.json"),
        scope=__doc__, optimizer_updates=0, actuator_messages_sent=0))
    result = dict(status="failed", goal_achieved=False, actor_qualified=False,
                  actual_actor_integration=False, actuator_messages_sent=0)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        stream = FlightStream(args.config, native_identity=read(root / "identity.json"),
                              episode_id=root.name)
        with load_array(vision / "responses.npz", allow_pickle=False) as archive:
            expected_retina = archive["retina_intensity"].copy()
            expected_motion = archive["motion_activity"].copy()
            clocks = archive["decision_monotonic_seconds"].copy()
        with load_array(body / "virtual-body.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["decision_monotonic_seconds"], clocks)
            expected_body, expected_valid = archive["values"].copy(), archive["valid"].copy()
        reference_features = evaluation / evaluated["source_record"]["file"]
        with load_array(reference_features, allow_pickle=False) as archive:
            expected_features = archive["scene_features"].copy()
        expected_intent = load_array(evaluation / "prediction.npy")
        errors = dict(retina=0., motion=0., body=0., scene_features=0., intent=0.)
        output, durations = [], []
        rows = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        if len(rows) != len(clocks):
            raise ValueError("Every original observation must be consumed")
        with CameraArchive(root / "vision") as camera:
            for index, row in enumerate(rows):
                clock = row["decision_monotonic_seconds"]
                if clock != clocks[index]:
                    raise ValueError("Original source clocks changed")
                pixels = {}
                for view in row["camera"]["views"]:
                    raw = camera.read(Path(view["file"]).name)
                    if hashlib.sha256(raw).hexdigest() != view["file_sha256"]:
                        raise ValueError("Original source pixel file changed")
                    rgb = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != view["rgb_sha256"]:
                        raise ValueError("Original decoded source RGB changed")
                    pixels[view["view_id"]] = rgb
                raw = base64.b64decode(row["native"]["prefix_base64"], validate=True)
                sample = decode_virtual_body_prefix(raw,
                    row["native"]["received_monotonic_seconds"])
                if (sample.prefix_sha256 != row["native"]["prefix_sha256"]
                        or sample.packet_id != row["native"]["packet_id"]):
                    raise ValueError("Original native sample changed")
                response = stream.advance(clock=clock, camera=row["camera"], pixels=pixels,
                                           native_sample=sample)
                values = dict(retina=(stream.retina, expected_retina[index]),
                    motion=(stream.last_motion.cpu().numpy(), expected_motion[index]),
                    body=(stream.last_body.values, expected_body[index]),
                    scene_features=(stream.last_features[0].cpu().numpy(),
                                    expected_features[index]),
                    intent=(np.asarray(response["intent"]), expected_intent[index]))
                for name, (actual, expected) in values.items():
                    errors[name] = max(errors[name], float(np.abs(actual - expected).max()))
                    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-5,
                                               err_msg=f"{name} at original index{index}")
                np.testing.assert_array_equal(stream.last_body.valid, expected_valid[index])
                output.append(response["intent"])
                durations.append(response["processing_seconds"])
                if index % 50 == 0:
                    write_json(run / "progress.json", dict(observations=index + 1, total=len(rows),
                        maximum_errors=errors, seconds=time.perf_counter() - started))
        # Compare parameter values, not merely the files from which they were loaded.
        public_weights = torch.load(Path(config["author_model_dir"]) / "best_chkpt",
                                    map_location="cpu", weights_only=True)["network"]
        for key, value in stream.network.state_dict().items():
            if key in public_weights:
                assert torch.equal(value.cpu(), public_weights[key].cpu()), key
        flight_receipt = read(Path(config["training"]) / "result.json")
        saved = torch.load(flight_receipt["checkpoint"], map_location="cpu", weights_only=True)
        for key, value in stream.flight.state_dict().items():
            reference = saved["model"][key]
            if value.is_sparse:
                current, original = value.coalesce().cpu(), reference.coalesce().cpu()
                assert torch.equal(current.indices(), original.indices()), key
                assert torch.equal(current.values(), original.values()), key
            else:
                assert torch.equal(value.cpu(), reference.cpu()), key
        np.savez_compressed(run / "replayed.npz", intent=np.array(output),
                            observation_time=clocks, processing_seconds=np.array(durations))
        result.update(status="completed_coupled_stream_replay", observations=len(rows),
            maximum_errors=errors, exact_body_masks=True, original_clocks_preserved=True,
            visual_ticks=stream.visual_ticks, flight_ticks=stream.flight_ticks,
            repeated_capture_observations=stream.reused_captures,
            author_and_flight_parameters_unchanged=True, hidden_resets=0,
            processing_seconds_percentiles=np.percentile(durations, [50, 95, 99, 100]).tolist(),
            host_render_load_included=False, actual_live_timing_verified=False,
            seconds=time.perf_counter() - started,
            termination_reason="complete original episode consumed one observation at a time")
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        if "output" in locals():
            result.update(observations_consumed=len(output), maximum_errors=errors)
        write_json(run / "result.json", result)
        raise
    write_json(run / "result.json", result)
    print(json.dumps(dict(status=result["status"], observations=result["observations"],
                          maximum_errors=result["maximum_errors"])))


if __name__ == "__main__":
    main()
