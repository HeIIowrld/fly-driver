"""Compare retained native tilt/rates with renderer orientation; diagnostics only."""

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def agreement(x, y, *, min_range, correlation=0.95):
    slope, offset = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    corr = float(np.corrcoef(x, y)[0, 1])
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    return dict(
        samples=len(x),
        input_range=float(np.ptp(x)),
        slope=float(slope),
        offset=float(offset),
        correlation=corr,
        uncorrected_rmse=rmse,
        passed=bool(
            np.ptp(x) >= min_range and 0.9 <= slope <= 1.1 and corr >= correlation and rmse < 0.005
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=None,
            gates=dict(
                tilt_min_range=0.005,
                tilt_correlation=0.95,
                unit_slope_range=[0.9, 1.1],
                uncorrected_rmse_max=0.005,
                angular_min_range=0.02,
                angular_correlation=0.9,
            ),
            declared_mapping=dict(
                pitch_nose_up="pitch",
                roll_left_up="-roll",
                body_angular_xyz_native_indices=[2, 0, 1],
            ),
            actuator_messages_sent=0,
            actor_pose_input=False,
        ),
    )
    summaries = []
    for recording in args.recordings:
        root = Path(recording)
        bundle = json.loads((root / "vision/bundle.json").read_text(encoding="utf8"))
        archive_path = root / "vision" / bundle["file"]
        if sha256(archive_path) != bundle["sha256"]:
            raise ValueError("Retained camera archive hash mismatch")
        records = [json.loads(line) for line in (root / "events.jsonl").open(encoding="utf8")]
        native, observations = {}, []
        for row in records:
            kind = row.get("kind")
            engine_ai_row = kind is None and all(key in row for key in (
                "native", "camera", "decision_monotonic_seconds"))
            if kind in ("raw_pair", "expert_raw_native") or engine_ai_row:
                n = row["native"]
                raw = base64.b64decode(n["prefix_base64"], validate=True)
                if hashlib.sha256(raw).hexdigest() != n["prefix_sha256"]:
                    raise ValueError("Native prefix hash mismatch")
                sample = decode_physics_prefix(raw, n["received_monotonic_seconds"])
                if sample.packet_id != n["packet_id"]:
                    raise ValueError("Native packet mismatch")
                values = PHYSICS_PREFIX.unpack(raw)
                fields, offset = {}, 0
                for key, _, count in PHYSICS_LAYOUT:
                    fields[key] = values[offset] if count == 1 else values[offset : offset + count]
                    offset += count
                native[sample.packet_id] = (fields, sample.received_monotonic_seconds)
            if kind == "student_request":
                a = row["actor_input"]
                observations.append((a["native_packet_id"], a["image"]))
            elif kind == "expert_observation" or engine_ai_row:
                packet = row["native"]["packet_id"] if engine_ai_row else row["native_packet_id"]
                observations.append((packet, row["camera"]))
        frames, attitudes, gyros, matrices, times, skipped = [], [], [], [], [], 0
        with zipfile.ZipFile(archive_path) as archive:
            for packet, image in observations:
                fields, native_time = native[packet]
                front = next(view for view in image["views"] if view["view_id"] == "front")
                if abs(native_time - front["render_started_lower"]) > 0.05:
                    skipped += 1
                    continue
                metadata = json.loads(archive.read(f"frame-{image['capture_id']:04d}.json"))
                if metadata["generation"] != image["generation"]:
                    raise ValueError("Renderer generation mismatch")
                views = {view["view_id"]: view for view in metadata["views"]}
                look = np.array(views["front"]["renderer_only_world_look"])
                left = np.array(views["left"]["renderer_only_world_look"])
                up = np.array(views["front"]["renderer_only_world_up"])
                matrix = np.column_stack([look, left, up])
                if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5):
                    raise ValueError("Renderer car basis not orthonormal")
                if not np.isclose(np.linalg.det(matrix), 1.0, atol=2e-5):
                    raise ValueError("Renderer body basis not right handed")
                attitudes.append(
                    [
                        fields["pitch"],
                        -fields["roll"],
                        np.arcsin(np.clip(look[1], -1, 1)),
                        np.arctan2(left[1], up[1]),
                    ]
                )
                frames.append(image["capture_id"])
                gyros.append(fields["localAngularVel"])
                matrices.append(matrix)
                times.append(metadata["script_clock"])
        attitudes, gyros, matrices, times = map(np.array, [attitudes, gyros, matrices, times])
        angular_expected, angular_native = [], []
        for index in range(2, len(times) - 2):
            dt = times[index + 2] - times[index - 2]
            if not 0.02 < dt < 0.2:
                continue
            world = (
                Rotation.from_matrix(matrices[index + 2] @ matrices[index - 2].T).as_rotvec() / dt
            )
            angular_expected.append(matrices[index].T @ world)
            angular_native.append(gyros[index - 2 : index + 3].mean(axis=0)[[2, 0, 1]])
        angular_expected, angular_native = map(np.array, [angular_expected, angular_native])
        summary = dict(
            recording=recording,
            events_sha256=sha256(root / "events.jsonl"),
            archive_sha256=bundle["sha256"],
            selected_samples=len(attitudes),
            skipped_for_host_interval_mismatch=skipped,
            tilt={
                name: agreement(attitudes[:, index], attitudes[:, index + 2], min_range=0.005)
                for index, name in enumerate(("pitch_nose_up", "roll_left_up"))
            },
            angular={
                name: agreement(
                    angular_native[:, index],
                    angular_expected[:, index],
                    min_range=0.02,
                    correlation=0.9,
                )
                for index, name in enumerate(("roll", "pitch", "yaw"))
            },
            source_clock_verified=False,
            scope="renderer/native consistency at observed excitation",
        )
        summaries.append(summary)
        np.savez_compressed(
            run / f"geometry-{len(summaries):02d}.npz",
            frame_ids=np.array(frames),
            native_and_renderer_tilt=attitudes,
            producer_clock=times,
            angular_native_body=angular_native,
            angular_renderer_body=angular_expected,
        )
        write_json(run / "progress.json", dict(recordings_completed=len(summaries)))
    write_json(
        run / "result.json",
        dict(
            status="completed",
            termination_reason="retained geometry comparisons completed",
            recordings=summaries,
            actuator_messages_sent=0,
            actor_pose_input=False,
            goal_achieved=False,
            physiological_registration=False,
            simulated_clock_alignment_verified=False,
        ),
    )
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
