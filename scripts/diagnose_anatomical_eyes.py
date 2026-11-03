"""Check imported axes and source-FOV coverage with white input, without driving."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from flybrain_drive.compound_eye import CompoundEyeProjector, PerspectiveView
    from flybrain_drive.eye_anatomy import load_anatomical_eyes
    from flybrain_drive.graph import sha256
    from flybrain_drive.sensor_contract import SampleTime
    from flybrain_drive.vision import CameraConfig

    configuration = json.loads(Path(args.config).read_text(encoding="utf8"))
    camera = asdict(CameraConfig(**configuration["environment"]["camera"]))
    yaw, pitch = np.deg2rad(camera["yaw_degrees"]), np.deg2rad(camera["pitch_degrees"])
    origins = {side: [camera["forward"], sign * camera["baseline"] / 2, camera["height"]]
               for side, sign in (("left", 1), ("right", -1))}
    eyes = load_anatomical_eyes(args.geometry, origins, sigma_degrees=1,
                               head_to_body_rotation=np.eye(3))
    arrays, reports = {}, {}
    for eye, sign in zip(eyes, (1, -1), strict=True):
        theta = yaw * sign
        rotation = np.column_stack([
            [np.cos(pitch)*np.cos(theta), np.cos(pitch)*np.sin(theta), np.sin(pitch)],
            [-np.sin(theta), np.cos(theta), 0],
            [-np.sin(pitch)*np.cos(theta), -np.sin(pitch)*np.sin(theta), np.cos(pitch)],
        ])
        projector = CompoundEyeProjector(eye, PerspectiveView(rotation, origins[eye.side],
                                                              camera["fovy"]))
        size = camera["source_size"] or configuration["environment"].get("image_size", 32)
        response, _ = projector.project(np.full((size, size, 3), 255, np.uint8),
                                          frame_id=0, time=SampleTime(0, 0))
        np.testing.assert_allclose(response.response[response.valid], 1, atol=1e-7)
        assert not response.response[~response.valid].any()
        arrays[eye.side + "_valid"] = response.valid
        arrays[eye.side + "_responses"] = response.response
        reports[eye.side] = {"axes": len(eye.directions_body),
                             "observed": int(response.valid.sum()),
                             "observed_fraction": float(response.valid.mean()),
                             "eye_map_sha256": eye.sha256}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "white-input.npz", **arrays)
    result = {"scope": "static optical FOV and validity audit, not driving or neural registration",
              "geometry_sha256": sha256(args.geometry), "script_sha256": sha256(__file__),
              "sigma_degrees_assumption": 1, "head_to_body_rotation_assumption": np.eye(3).tolist(),
              "camera": camera, "virtual_eye_origins_m": origins,
              "single_virtual_pupil_per_eye": True, "per_facet_spatial_origins_modeled": False,
              "MaleCNS_retinal_neurons_registered": 0, "eyes": reports}
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
