"""Read admitted front video and join the offline surface labels by capture identity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
ROOT = RUNS.parent
LABELS = RUNS / "vision-corner-physics-surface-20261112-01"
SILVER = "ks_silverstone1967"
TRAIN = ("magione", "spa")
CROP = (72, 152)
SIZE = (160, 40)
sys.path.insert(0, str(RUNS / "vision-corner-image-row-trace-20261112-01"))
import replay as old
from replay_learned import baseline_data, frame_key


def preprocess(image):
    if image is None or image.shape != (180, 320, 3):
        raise ValueError("Original 320x180 front image required")
    return cv2.resize(image[CROP[0]:CROP[1]], SIZE, interpolation=cv2.INTER_AREA)


def labels(track):
    path = LABELS / f"{track}-train-labels.npz"
    receipt = old.read(path.with_suffix(".json"))
    verify = old.read(LABELS / "training-labels-verification.json")
    assert old.digest(path) == receipt["labels_sha256"] == verify["tracks"][track]["labels_sha256"]
    with np.load(path, allow_pickle=False) as data:
        assert data["views"].tolist() == ["front", "left", "right", "rear"]
        return {key: data[key] for key in ("run", "frame", "stations", "lateral_m", "adjacent_invalid", "image_px")}, path


def extract(track):
    truth, label_path = labels(track)
    hashes = {Path(__file__), label_path, label_path.with_suffix(".json"),
              LABELS / "training-labels-verification.json"}
    group, _, mapping, bundles = baseline_data(track, hashes)
    keys = [frame_key(row) for row in group["rows"]]
    assert keys == list(zip(truth["run"], truth["frame"], strict=True))
    expected = {key: i for i, key in enumerate(keys)}
    seen = set()
    images = np.zeros((len(keys), SIZE[1], SIZE[0], 3), dtype=np.uint8)
    available = np.zeros(len(keys), dtype=bool)
    for run, folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[run]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = run, item["capture_id"]
                if key not in expected:
                    continue
                assert key not in seen
                seen.add(key)
                index = expected[key]
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, key[1], item["generation"])
                by_view = {v["view_id"]: v for v in item["views"]}
                if (set(by_view) != {"front", "left", "right", "rear"} or len(item["views"]) != 4
                    or any(r.get(flag) for r in (meta,item) for flag in ("ideal_line_shown", "aid_probe_only"))):
                    continue
                blob = archive.read(Path(by_view["front"]["file"]).name)
                image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                images[index] = preprocess(image)
                available[index] = True
    assert seen == set(expected)
    output = OUT / f"{track}-front.npz"
    np.savez_compressed(output, images=images, available=available, run=truth["run"], frame=truth["frame"])
    old.write(output.with_suffix(".json"), dict(
        track=track, frames=len(keys), frames_with_video=int(available.sum()),
        images_sha256=old.digest(output), crop_native_y=CROP, size_wh=SIZE, seed=None,
        source_sha256={str(p): old.digest(p) for p in sorted(hashes)},
        camera_bundle_sha256_by_run=bundles, source_revision=old.revision(), termination="archive_end"))
    print(track, len(keys), int(available.sum()), flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    tracks = tuple(sys.argv[1:]) or TRAIN
    if not set(tracks) <= {SILVER, *TRAIN} or len(set(tracks)) != len(tracks):
        raise ValueError("Unknown or duplicate track")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(extract, tracks))
