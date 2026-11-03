"""Compare saved motion indices with the current FlyVis cache node type and coordinates."""

import argparse
import json
from pathlib import Path

import h5py

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "cache", "prior", "output"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    dataset, cache, prior = map(Path, (a.dataset, a.cache, a.prior))
    prior_result = read(prior / "result.json")
    if prior_result["status"] != "completed_visual_body_yaw_diagnosis":
        raise ValueError("Completed registered visual yaw evidence required")
    registrations = prior_result["checked_registration_sha256"]
    nodes = cache / "connectome/ConnectomeFromAvgFilters_0000/nodes"
    files = [nodes / (name + ".h5") for name in ("type", "u", "v")]
    hashes = {str(path): sha256(path) for path in files}
    arrays = []
    for path in files:
        with h5py.File(path, "r") as archive:
            if set(archive.keys()) != {"data"}:
                raise ValueError("Unexpected node dataset")
            arrays.append(archive["data"][:])
    types, u, v = arrays
    if len(types) != 45669:
        raise ValueError("Original FlyVis graph node count required")
    camera = read("configs/ac-csp-yaw-retina.json")
    episodes = read(dataset / "dataset.json")["episodes"]
    if len(episodes) != 8 or set(registrations) != {e["file"] for e in episodes}:
        raise ValueError("Exact eight audited histories required")
    report = []
    for entry in episodes:
        visual = Path(entry["original"]["scene_replay"])
        manifest, result = read(visual / "manifest.json"), read(visual / "result.json")
        args = manifest["config"]["arguments"]
        if (
            Path(args["cache_root"]).resolve() != cache.resolve()
            or result["status"] != "completed"
            or sha256(visual / "responses.npz") != result["responses_sha256"]
            or result["responses_sha256"] != entry["source_visual_sha256"]
            or sha256(visual / "retina-registration.npz") != registrations[entry["file"]]
            or sha256("configs/ac-csp-yaw-retina.json")
            != manifest["config"]["camera_config_sha256"]
        ):
            raise ValueError("Original replay provenance or camera changed")
        with load_array(visual / "retina-registration.npz", allow_pickle=False) as registration:
            VisualSelfMotionPool.from_registration(camera, registration)
            VisualSelfMotionPool.verify_motion_node_identities(registration, types, u, v)
        report.append(
            {
                "file": entry["file"],
                "role": entry["role"],
                "registration_sha256": registrations[entry["file"]],
                "all_5768_type_and_position_pairs_match": True,
            }
        )
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "node_array_hashes_current": hashes,
            "dataset_sha256": sha256(dataset / "dataset.json"),
            "prior_result_sha256": sha256(prior / "result.json"),
            "source_scope": ("current cache referenced by historical replay; "
                "historical node hashes absent"),
            "flyvis_network_executed": False,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        run / "result.json",
        {
            "status": "completed_cached_flyvis_node_identity_check",
            "reports": report,
            "node_array_hashes_current": hashes,
            "historical_cache_node_hashes_verified": False,
            "actual_fly_physiology_verified": False,
            "flyvis_network_executed": False,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "episodes": len(report),
                "node_count": len(types),
                "matches_per_episode": 5768,
                "node_hashes": hashes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
