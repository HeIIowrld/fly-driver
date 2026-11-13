"""Run the frozen corner observer directly from named four-camera archives."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from replay_corner_boundary_states import ROOT, old
from collect import SHAPE, admitted, combined, image, inference, metadata, preprocess
from evaluate_vallelunga import features as oriented_motion_features
from extract_frozen_corner_archive import capture_index

EXTRACTOR = ROOT/"scripts/extract_frozen_corner_archive.py"
MOTION_SETTINGS = ROOT/"configs/flight-stream-batched-v1-20261029.json"


def stream(folder, bundle, expected_track):
    with old.CameraArchive(folder) as archive:
        if archive.bundle_sha256 != bundle:
            raise ValueError("Camera archive changed during replay")
        track = expected_track
        for line in archive.read("stream.jsonl").splitlines():
            item = json.loads(line)
            if track is None:
                track = archive.json(Path(item["file"]).name)["track_id"]
            meta = metadata(archive, item, track)
            yield archive, item, meta, admitted(item, meta)


def extract_boundaries(folders, bundles, output, expected_track):
    keys, rows, images, available, motion = [], [], [], [], []
    absent_crop = np.zeros((40, 160, 3), np.uint8)
    for name, folder in folders.items():
        for archive, item, meta, views in stream(folder, bundles[name], expected_track):
            frame = item["capture_id"]
            keys.append((name, frame))
            rows.append(dict(run=name, frame=frame, generation=item["generation"], received_s=item["received"]))
            images.append(preprocess(image(archive, views["front"])) if views else absent_crop)
            available.append(bool(views))
            body = old.read_motion(meta) if views else None
            motion.append(body if body is not None else (np.nan, np.nan, np.nan))
    if not keys:
        raise ValueError("No captures in the archives")
    run, frame = (np.asarray(col) for col in zip(*keys, strict=True))
    crops = output/"front-crops.npz"
    np.savez_compressed(crops, run=run, frame=frame, images=images, available=available)
    weights, training = SHAPE/"shape.pt", SHAPE/"training.json"
    if old.digest(weights) != old.read(training)["checkpoint_sha256"]:
        raise ValueError("Frozen road boundary checkpoint differs")
    mount = old.read(old.MOUNT)
    inferred, mask, estimates, heads = inference(crops, weights, mount["views"]["front"])
    if inferred != keys or not np.array_equal(mask, available):
        raise ValueError("Boundary model changed the capture order or admission mask")
    near, far = (old.SurfaceContrast(mount, near=value) for value in (True, False))
    far.stations = old.STATIONS[3:]
    far.lookup = far._sampling(mount, far.stations)
    sections = np.full((len(keys), len(old.STATIONS), 2), np.nan)
    cursor = 0
    for name, folder in folders.items():
        for archive, item, _, views in stream(folder, bundles[name], expected_track):
            i = cursor
            cursor += 1
            if keys[i] != (name, item["capture_id"]):
                raise ValueError("Capture order changed during boundary extraction")
            body = motion[i]
            if views is None or not np.isfinite(body).all() or body[1] < old.MIN_SPEED_MPS:
                continue
            pixels = {view: image(archive, entry) for view, entry in views.items()}
            lab = {view: cv2.cvtColor(frame_image, cv2.COLOR_BGR2LAB) for view, frame_image in pixels.items()}
            base = old.selections(near, far, pixels, lab, near.detect(pixels), old.edge_scores(near, lab))
            sections[i] = combined(base, estimates[i])["shape_geometry"]
    if cursor != len(keys):
        raise ValueError("Capture count changed during boundary extraction")
    sensor = dict(run=run, frame=frame, generation=np.asarray([r["generation"] for r in rows]),
                  received_s=np.asarray([r["received_s"] for r in rows]), stations=old.STATIONS,
                  motion=np.asarray(motion), sections=sections, heads=heads)
    sources = {crops, weights, training, old.MOUNT, Path(inference.__code__.co_filename),
               Path(preprocess.__code__.co_filename), Path(metadata.__code__.co_filename),
               Path(old.selections.__code__.co_filename), Path(old.read_motion.__code__.co_filename)}
    return rows, sensor, sources


def pack_motion(folder, archives, rows, bundles, hashes):
    for name in ("original", "reflected"):
        proof = old.read(folder/name/"flyvis-result.json")
        if (proof["inventory_sha256"] != old.digest(archives) or proof["program_sha256"] != old.digest(EXTRACTOR)
                or proof["termination"] != "completed" or proof["read_at"] != "last-render"):
            raise ValueError("Motion features are not a matching completed frozen archive extraction")
    group = oriented_motion_features(folder, rows, bundles, hashes)
    index = {(r["run"], r["frame"]): f for r, f in zip(group["selected"], group["x"], strict=True)}
    missing = np.full((2, 768), np.nan)
    return np.stack([index.get((r["run"], r["frame"]), missing) for r in rows])


def run(archives, cache, output, expected_track=None, motion_extraction=None):
    if not archives.is_file():
        raise FileNotFoundError(archives)
    folders = {name: Path(folder) for name, folder in old.read(archives).items()}
    if not folders:
        raise ValueError("Expected at least one named archive")
    inventory = {name: capture_index(folder, expected_track=expected_track) for name, folder in folders.items()}
    bundles = {name: value[2] for name, value in inventory.items()}
    if not all(bundles.values()):
        raise ValueError("Camera bundle lacks an integrity hash")
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if motion_extraction is None:
        motion_extraction = output/"flyvis"
        command = [sys.executable, str(EXTRACTOR), "--archives", str(archives), "--cache", str(cache),
                   "--output", str(motion_extraction), "--batch-pair"]
        if expected_track is not None:
            command.extend(("--expected-track", expected_track))
        subprocess.run(command, cwd=ROOT, check=True)
    rows, sensor, sources = extract_boundaries(folders, bundles, output, expected_track)
    if [(r["run"], r["frame"]) for r in rows] != [(name, f) for name, v in inventory.items() for f in v[0]]:
        raise ValueError("Boundary and motion extraction inventories differ")
    sensor["t4t5"] = pack_motion(motion_extraction, archives, rows, bundles, sources)
    bundle = output/"sensors.npz"
    np.savez_compressed(bundle, **sensor)
    replay = ROOT/"scripts/replay_corner_boundary_states.py"
    dest = output/"predictions"
    subprocess.run([sys.executable, str(replay), "--input", str(bundle), "--output", str(dest)],
                   cwd=ROOT, check=True)
    result = dest/"result.json"
    sources.update((Path(__file__), archives, EXTRACTOR, MOTION_SETTINGS, result))
    old.write(output/"result.json", dict(status="archive_replayed", source_revision=old.revision(),
              frames=len(rows), archive_bundles=bundles, expected_track=expected_track,
              capture_source="raw_four_camera_archive", inputs_sha256=old.digest(bundle),
              prediction_receipt_sha256=old.digest(result),
              source_sha256={str(p): old.digest(p) for p in sorted(sources)}, termination="archive_end"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT/"runs/vision-next-corner-flyvis-cache-20261110")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-track")
    parser.add_argument("--motion-extraction", type=Path, help="A matching completed original/reflected extraction")
    args = parser.parse_args()
    args.archives, args.cache, args.output = (p.resolve() for p in (args.archives, args.cache, args.output))
    if args.motion_extraction is not None:
        args.motion_extraction = args.motion_extraction.resolve()
    run(**vars(args))


if __name__ == "__main__":
    main()
