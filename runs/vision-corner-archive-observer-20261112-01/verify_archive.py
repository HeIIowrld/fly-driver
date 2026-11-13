"""Compare raw-camera replay to the earlier sensor bundle and score after inference."""
from collections import Counter
from pathlib import Path
import argparse
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from replay_corner_boundary_states import ROUTING, old
from verify_directional import frames

REFERENCE = OUT.parent/"vision-corner-sensor-only-20261112-01"
METHODS = ("temporal", "t4t5_confirm")


def compare(a, b):
    if a.dtype.kind not in "f":
        return dict(equal=bool(np.array_equal(a, b)))
    shared = np.isfinite(a) & np.isfinite(b)
    delta = abs(a[shared]-b[shared])
    return dict(equal=bool(np.array_equal(a, b, equal_nan=True)),
                finite_mask_equal=bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                max_abs_delta=float(delta.max()) if len(delta) else None)


def score(predicted, reference):
    metrics = {name: Counter() for name in METHODS}
    mismatches = []
    for new, expected in zip(predicted, reference, strict=True):
        assert (new["run"], new["frame"]) == (expected["run"], expected["frame"])
        if new["forecasts"] != expected["forecasts"]:
            mismatches.append(dict(frame=new["frame"], actual=new["forecasts"], prior=expected["forecasts"]))
        for name, forecast in new["forecasts"].items():
            state, distance = forecast["state"], forecast["distance_m"]
            offer = ({"left": 1, "right": 2}[state], distance) if state in ("left", "right") else None
            if offer is None:
                assert distance is None
            result = old.category(expected, offer)
            metrics[name][result] += 1
            if result == "joint_hit" and state == "right":
                metrics[name]["right_joint_hit"] += 1
            metrics[name][expected["truth"] + "/" + state] += 1
    return {name: dict(counter) for name, counter in metrics.items()}, mismatches


def verify(track, directory):
    receipt = directory/"result.json"
    captured = old.read(receipt)
    for name, digest in captured["source_sha256"].items():
        assert old.digest(Path(name)) == digest
    sensor_path = directory/"sensors.npz"
    assert old.digest(sensor_path) == captured["inputs_sha256"]
    reference = REFERENCE/f"{track}-sensors.npz"
    fields = {}
    with np.load(sensor_path, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            assert a[name].shape == b[name].shape
            fields[name] = compare(a[name], b[name])
        assert all(fields[n]["equal"] for n in ("run", "frame", "generation", "received_s", "stations"))
    actual = directory/"predictions/result.json"
    assert old.digest(actual) == captured["prediction_receipt_sha256"]
    replay = old.read(actual)
    predicted = frames(actual.with_name("forecasts.jsonl"), replay["forecasts_sha256"])
    earlier = old.read(ROUTING/"straight-verification.json")
    prior = ROUTING/f"{track}-state-distance.jsonl"
    truth = frames(prior, earlier["tracks"][track]["rows_sha256"])
    metrics, mismatches = score(predicted, truth)
    boundaries = actual.with_name("boundaries.npz")
    assert old.digest(boundaries) == replay["boundaries_sha256"]
    original = REFERENCE/track/"boundaries.npz"
    with np.load(boundaries, allow_pickle=False) as a, np.load(original, allow_pickle=False) as b:
        edges = {name: compare(a[name], b[name]) for name in ("observed", "estimated", "restored")}
    report = dict(status="archive_and_label_separation_verified", track=track, frames=len(truth),
                  sensor_fields=fields, edge_outputs=edges, parity_frames=len(truth)-len(mismatches),
                  metrics=metrics, mismatches=mismatches,
                  source_sha256={str(p): old.digest(p) for p in (Path(__file__), receipt, reference, original, prior)})
    old.write(directory/"verification.json", report)
    print(track, len(truth), len(mismatches), fields, edges, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.track, args.directory.resolve())
