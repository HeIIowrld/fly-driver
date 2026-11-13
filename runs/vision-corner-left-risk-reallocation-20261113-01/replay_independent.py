"""Replay the fixed left-turn policy from a saved four-camera capture, then score it."""
import argparse
from pathlib import Path
import sys

import joblib

from left_policy import OUT, apply_policy, dependencies, describe
from fixed_left import CLI, EVAL, RISK, PER_RECORDING, digests, keys, load_sensor, old, score
from evaluate_live_corner_geometry import common_rows, load_track_truth
from replay_corner_camera_archive import EXTRACTOR, ROOT, SHAPE, run as replay_archive

INVENTORY = OUT.parent / "vision-corner-fixed-left-20261113-01/independent-inventory.json"
CACHE = OUT.parent / "vision-next-corner-flyvis-cache-20261110"
AC_TRACKS = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")


def load_fixed_policy():
    receipt, checkpoint = OUT / "frozen.json", OUT / "policy.joblib"
    policy = old.read(receipt)
    if old.digest(checkpoint) != policy["model_sha256"]:
        raise ValueError("Fixed checkpoint changed")
    for path, digest in policy["source_sha256"].items():
        if old.digest(Path(path)) != digest:
            raise ValueError(f"Fixed input changed: {path}")
    observer, settings, structured, _ = dependencies()
    if settings != policy["settings"]:
        raise ValueError("Fixed observer settings changed")
    return receipt, policy, joblib.load(checkpoint), observer, settings, structured


def validate_transfer(track, layout):
    inventory = old.read(INVENTORY)
    training = {t for component in inventory["recorded_training"].values() for t in component["circuits"]}
    comparisons = {track for track, _ in EVAL}
    if track in training | comparisons:
        raise ValueError(f"Track {track} has been used in training, calibration or the comparison")
    if Path(track).name != track or (layout and Path(layout).name != layout):
        raise ValueError("Expected exact track and optional layout IDs")
    spline = AC_TRACKS / track / layout / "ai/fast_lane.ai"
    if not spline.is_file():
        raise FileNotFoundError(spline)
    if (track, layout) == (inventory["reserved_circuit"], inventory["reserved_layout"]):
        recorded = inventory["installed_files"][str(spline)]
        if old.digest(spline) != recorded:
            raise ValueError("Reserved layout geometry has changed")
    return spline


def check_extraction(policy, archive_receipt=None):
    inputs = None if archive_receipt is None else {
        Path(p).resolve(): h for p, h in archive_receipt["source_sha256"].items()
    }
    prior_path = CLI / "silverstone/result.json"
    prior = {Path(p).resolve(): h for p, h in old.read(prior_path)["source_sha256"].items()}
    pinned = (EXTRACTOR, SHAPE / "shape.pt", SHAPE / "training.json", ROOT / "scripts/corner_archive_recovery.py")
    current = {Path(p).resolve(): h for p, h in policy["source_sha256"].items()}
    for p in pinned:
        expected = current.get(p.resolve(), prior[p.resolve()])
        if old.digest(p) != expected:
            raise ValueError(f"Extraction dependency changed: {p}")
        if inputs is not None and inputs.get(p.resolve()) != expected:
            raise ValueError(f"Different dependency used for archive: {p}")
    return prior_path


def validate_layout_receipt(archives, capture_receipt, track, layout):
    if not layout:
        return None
    if capture_receipt is None:
        raise ValueError("A multi-layout circuit requires its original capture receipt")
    proof = old.read(capture_receipt)
    if (proof.get("track") != track or proof.get("layout") != layout
            or proof.get("status") != "captured"
            or proof.get("archives_sha256") != old.digest(archives)
            or proof.get("layout_source") != "unchanged_active_race_ini"):
        raise ValueError("Capture receipt does not verify this archive and layout")
    return capture_receipt


def run(args):
    receipt, policy, model, observer, settings, structured = load_fixed_policy()
    prior_extraction = check_extraction(policy)
    spline = validate_transfer(args.track, args.layout)
    capture_receipt = validate_layout_receipt(args.archives, args.capture_receipt, args.track, args.layout)
    args.output.mkdir(parents=True, exist_ok=False)
    replay = args.output / "archive-replay"
    replay_archive(args.archives, args.cache, replay, args.track, args.motion_extraction, "local-image")
    archive_receipt = replay / "result.json"
    proof = old.read(archive_receipt)
    check_extraction(policy, proof)
    raw = load_sensor(replay / "sensors.npz", proof["inputs_sha256"])
    restored = load_sensor(replay / "recovered-sensors.npz", proof["recovered_input_sha256"])
    baseline, offers = describe(raw, restored, observer, settings, structured)
    revised, changes = apply_policy(baseline, offers, policy, model)
    witness = args.output / "forecasts.jsonl"
    old.jsonl(witness, revised)

    labeler, _, _ = load_track_truth(spline)
    labels = []
    for run, path in old.read(args.archives).items():
        with old.CameraArchive(path) as camera:
            if camera.bundle_sha256 != proof["archive_bundles"][run]:
                raise ValueError("Archive changed before scoring")
            labels.extend(common_rows(run, camera, {}, args.track, labeler))
    if keys(labels) != keys(revised):
        raise ValueError("Forecast and ground truth timelines differ")
    labels_path = args.output / "labels.jsonl"
    old.jsonl(labels_path, labels)
    before, after = score(baseline, labels), score(revised, labels)
    delta = {key: after["counts"].get(key, 0) - before["counts"].get(key, 0) for key in RISK}
    held = [after["approaches"].get(side, {}).get("frames", 0) for side in ("1", "2")]
    risks_pass = all(delta[key] <= cap for key, cap in PER_RECORDING.items())
    sources = [Path(__file__), receipt, INVENTORY, args.archives, archive_receipt, prior_extraction, spline]
    if capture_receipt is not None:
        sources.append(capture_receipt)
    old.write(args.output / "result.json", dict(
        track=args.track, layout=args.layout, before=before, after=after, risk_delta=delta,
        per_recording_risk_pass=risks_pass, both_directions_evaluated=all(held),
        candidate_changes=len(changes), forecasts_sha256=old.digest(witness), labels_sha256=old.digest(labels_path),
        source_revision=old.revision(), source_sha256=digests(sources), seed=None, termination="archive_end"))
    print(args.track, after["approaches"], delta, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--layout", default="")
    parser.add_argument("--capture-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--motion-extraction", type=Path)
    args = parser.parse_args()
    for key in ("archives", "capture_receipt", "output", "cache", "motion_extraction"):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.resolve())
    run(args)


if __name__ == "__main__":
    main()
