"""Archive diagnostic four-view PNGs and project original retinal coordinates."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive, bundle_camera_files
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-host-hashes", action="store_true")
    args = parser.parse_args()
    root = Path(args.recording)
    result, rows = read(root / "result.json"), read(root / "diagnostic-frames.json")
    if (
        result["status"] != "completed"
        or result["actor_deployed"] is not False
        or sha256(root / "diagnostic-frames.json") != result["metadata_sha256"]
        or len(rows) != result["frames"]
        or not rows
    ):
        raise ValueError("Completed diagnostic with exact frame metadata required")
    source = Path(rows[0]["source_file"]).parent
    for row in rows:
        path = Path(row["source_file"])
        if path.parent != source or sha256(path) != row["sha256"]:
            raise ValueError("Original same-directory metadata hash required")
    observed = None
    if args.require_host_hashes:
        log = root / "live-frame-hashes.jsonl"
        if sha256(log) != result.get("host_observed_hashes_sha256"):
            raise ValueError("Original host-observed image hashes required")
        observed = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines()]
        if len(observed) != len(rows):
            raise ValueError("Host hash log must cover every original frame")
        for index, record in enumerate(observed):
            if (
                record["index"] != index
                or record["metadata_sha256"] != rows[index]["sha256"]
                or set(record["images"]) != set(VIEW_ORDER)
                or record["host_hash_finished"] < record["host_read_started"]
            ):
                raise ValueError("Exact causal host hash record required")
            for name in VIEW_ORDER:
                image = record["images"][name]
                if (
                    image["file"] != f"{index:04d}-{name}.png"
                    or sha256(source / image["file"]) != image["sha256"]
                ):
                    raise ValueError("Source PNG differs from host-observed digest")
    config = read(args.config)
    for key in ("camera_config", "retina_registration"):
        if sha256(config[key]) != config[key + "_sha256"]:
            raise ValueError("Original camera and retina registration required")
    camera = read(config["camera_config"])
    with load_array(config["retina_registration"], allow_pickle=False) as data:
        projector = FlyvisCameraProjector(camera, data["u"], data["v"])
        np.testing.assert_array_equal(projector.uv, data["image_uv"])
        np.testing.assert_array_equal(projector.directions_body, data["body_directions"])
    pins = {
        str(p): sha256(p)
        for p in (
            root / "result.json",
            root / "diagnostic-frames.json",
            root / "control/api-receipt.json",
            root / "control/result.json",
            Path(args.config),
        )
    }
    if observed is not None:
        pins[str(root / "live-frame-hashes.jsonl")] = sha256(root / "live-frame-hashes.jsonl")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "scope": "post-capture PNG bundle and input projection; diagnosis only",
            "hashes_are_host_observed_after_publication": observed is not None,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    bundled = bundle_camera_files([source], run / "vision")
    if bundled["files"] < len(rows) * 5:
        raise ValueError("All four PNGs and one metadata per frame required")
    retinal, timestamps, speed, pixel_hashes, distinct = [], [], [], [], []
    with CameraArchive(run / "vision") as archive:
        for row in rows:
            metadata_raw = archive.read(Path(row["source_file"]).name)
            if hashlib.sha256(metadata_raw).hexdigest() != row["sha256"]:
                raise ValueError("Bundled original metadata differs")
            frame = json.loads(metadata_raw)
            if (
                frame["index"] != row["index"]
                or frame["request_id"] != result["request_id"]
                or frame["airflow_diagnostic"] != row["diagnostic"]
                or frame.get("deferred_save_probe") is not True
                or not frame["aid_probe_only"]
            ):
                raise ValueError("Original diagnostic metadata identity required")
            mapping = {v["view_id"]: v for v in frame["views"]}
            if set(mapping) != set(VIEW_ORDER):
                raise ValueError("All four original view IDs required")
            views, hashes = [], []
            for name in VIEW_ORDER:
                raw = archive.read(mapping[name]["file"])
                if (
                    observed is not None
                    and hashlib.sha256(raw).hexdigest()
                    != observed[row["index"]]["images"][name]["sha256"]
                ):
                    raise ValueError("Bundled PNG differs from host-observed digest")
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = np.array(image.convert("RGB"))
                if rgb.shape != (180, 320, 3) or rgb.dtype != np.uint8:
                    raise ValueError("Exact 320x180 RGB source required")
                distinct.append(int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]))
                views.append(projector.sample(rgb))
                hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            retinal.append(np.stack(views))
            pixel_hashes.append(hashes)
            timestamps.append([mapping[n]["render_started_clock"] for n in VIEW_ORDER])
            speed.append(float(np.linalg.norm(row["diagnostic"]["vehicle_velocity_candidate_m_s"])))
        archive.verify_all()
    if min(distinct) <= 16 or any(sha256(p) != h for p, h in pins.items()):
        raise ValueError("Original nonconstant pixels and unchanged inputs required")
    np.savez_compressed(
        run / "retinal-inputs.npz",
        retinal=np.stack(retinal),
        render_source_clocks=np.array(timestamps),
        diagnostic_speed_m_s=np.array(speed),
        rgb_sha256=np.array(pixel_hashes),
    )
    out = {
        "status": "completed_diagnostic_owned_ai_image_archive_and_projection",
        "frames": len(rows),
        "images": 4 * len(rows),
        "moving_frames_speed_gt_1m_s": int((np.array(speed) > 1).sum()),
        "minimum_colors": min(distinct),
        "retinal_shape": list(np.stack(retinal).shape),
        "retinal_sha256": sha256(run / "retinal-inputs.npz"),
        "camera_bundle": bundled,
        "host_observed_png_hashes_verified": observed is not None,
        "pixel_hashes_created_after_publication_not_exposure_proof": True,
        "physical_exposure_or_body_sync_verified": False,
        "flyvis_network_executed": False,
        "training_admissible": False,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
