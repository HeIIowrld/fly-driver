"""Extract original and reflected causal FlyVis features from named camera archives."""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from extract_next_corner_flyvis import extract_pair, setup
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.flyvis_camera import VIEW_ORDER
from infer_frozen_next_corner import FEATURE_DIM
from train_vision_teacher_line import digest, jsonl, read, write


def capture_index(directory, *, expected_track=None):
    if expected_track is not None and re.fullmatch(r"[a-z0-9_-]+", expected_track) is None:
        raise ValueError("An exact AC track ID is required")
    with CameraArchive(directory) as archive:
        stream = [json.loads(line) for line in archive.read("stream.jsonl").splitlines()]
        captures = {row["capture_id"]: row for row in stream}
        if len(captures) != len(stream):
            raise ValueError("Repeated capture ID")
        if expected_track is not None:
            generations = {row.get("generation") for row in stream}
            if not archive.bundle_sha256 or len(generations) != 1 or not next(iter(generations)):
                raise ValueError("Expected a verified camera bundle from one session generation")
            for frame, row in captures.items():
                meta = archive.json(Path(row["file"]).name)
                if (meta.get("track_id") != expected_track or meta.get("index") != frame
                        or meta.get("generation") != row["generation"]):
                    raise ValueError(f"Camera track, generation or frame differs at capture {frame}")
        accepted = {
            frame: row["generation"] for frame, row in captures.items()
            if not row["ideal_line_shown"] and not row["aid_probe_only"]
            and len(row["views"]) == len(VIEW_ORDER)
            and {view["view_id"] for view in row["views"]} == set(VIEW_ORDER)
        }
        available = {
            frame for frame, generation in accepted.items()
            if frame-2 in accepted and frame-1 in accepted
            and accepted[frame-2] == generation == accepted[frame-1]
        }
        return list(captures), available, archive.bundle_sha256


def run(archives, index, cache, output, batch_pair=False, expected_track=None):
    folders = {name: Path(path) for name, path in read(archives).items()}
    inventory = {name: capture_index(path, expected_track=expected_track) for name, path in folders.items()}
    if index:
        records = [json.loads(line) for line in index.read_text(encoding="utf8").splitlines()]
        keys = [(row["run"], row["frame"]) for row in records]
    else:
        keys = [(name, frame) for name, (frames, _, _) in inventory.items() for frame in frames]
    if not keys or len(set(keys)) != len(keys) or set(folders) != {name for name, _ in keys}:
        raise ValueError("Expected unique frames and matching named archives")
    for name, (frames, _, _) in inventory.items():
        if not {frame for key, frame in keys if key == name}.issubset(frames):
            raise ValueError(f"Requested frame absent in archive: {name}")

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    network, projector, indices, pool, step, provenance = setup(cache, batch_size=8 if batch_pair else 4)
    output.mkdir(parents=True, exist_ok=False)
    index_path = output/"index.jsonl"
    jsonl(index_path, (dict(run=name, frame=frame) for name, frame in keys))
    features = np.zeros((2, len(keys), FEATURE_DIM), np.float32)
    valid = np.zeros((2, len(keys)), bool)
    reports = {}, {}
    for name, directory in folders.items():
        _, eligible, expected_bundle = inventory[name]
        subset = [(j, frame) for j, (key, frame) in enumerate(keys) if key == name]
        required = {i for _, frame in subset if frame in eligible for i in range(frame-2, frame+1)}
        pair, bundle = ({}, {}), expected_bundle
        if required:
            pair, bundle = extract_pair(directory,required,network,projector,indices,pool,step,batched=batch_pair)
            if bundle != expected_bundle:
                raise ValueError("Camera bundle changed during extraction")
        for branch, found in enumerate(pair):
            for j, frame in subset:
                history = range(frame-2, frame+1)
                if frame in eligible and all(i in found for i in history):
                    features[branch,j] = np.concatenate([found[i] for i in history])
                    valid[branch,j] = True
            reports[branch][name] = dict(bundle_sha256=bundle, requested=len(subset),
                                         available=sum(bool(valid[branch,j]) for j, _ in subset))
    for branch, variant in enumerate(("original","reflected")):
        dest = output/variant
        dest.mkdir()
        np.savez_compressed(dest/"features.npz", features=features[branch], available=valid[branch])
        proof = dict(**provenance, seed=None, inventory_sha256=digest(archives), rows_sha256=digest(index_path),
                     features_sha256=digest(dest/"features.npz"), reports=reports[branch], read_at="last-render",
                     program_sha256=digest(Path(__file__)), batch_size=8 if batch_pair else 4, termination="completed")
        if expected_track is not None:
            proof["verified_track_id"] = expected_track
        if branch == 1:
            proof["reflection"] = "input receptor permutation and swapped side eyes; unchanged visual network"
        write(dest/"flyvis-result.json", proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archives", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--batch-pair", action="store_true")
    parser.add_argument("--expected-track", help="Verify original frame track and session before extraction")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
