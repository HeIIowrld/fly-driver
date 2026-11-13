"""Verify exported geometry labels and their source receipts."""
from pathlib import Path
import json
import numpy as np

from kn5_surfaces import OUT, digest

ROOT = OUT.parents[1]
TRACKS = ("ks_silverstone1967", "spa", "magione")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    result = {}
    checked = {}
    for track in TRACKS:
        path = OUT / f"{track}-train-labels.npz"
        receipt = read(path.with_suffix(".json"))
        assert receipt["track"] == track and digest(path) == receipt["labels_sha256"]
        assert receipt["raw_assets_copied"] == 0 and not receipt["training_applied"]
        for name, sha in receipt["source_sha256"].items():
            if name not in checked:
                checked[name] = digest(ROOT / name)
            assert checked[name] == sha, name
        with np.load(path, allow_pickle=False) as data:
            n = receipt["frames"]
            assert len(set(zip(data["run"], data["frame"], strict=True))) == n
            assert data["lateral_m"].shape == (n, 17, 2)
            assert data["image_px"].shape == (n, 4, 17, 2, 2)
            assert data["views"].tolist() == ["front", "left", "right", "rear"]
            assert data["adjacent_invalid"].dtype == bool
            assert not (data["adjacent_invalid"] & ~np.isfinite(data["lateral_m"])).any()
            support = np.isfinite(data["image_px"]).all(axis=-1)
            assert not (support & ~np.isfinite(data["lateral_m"])[:, None]).any()
            counts = (support & data["adjacent_invalid"][:, None]).sum(axis=(0, 2)).tolist()
            assert dict(zip(data["views"], counts, strict=True)) == receipt["adjacent_invalid_counts_by_view_and_side"]
            if track == TRACKS[0]:
                raw = read(OUT / "surface-labels.json")
                indices = {int(f): i for i, f in enumerate(data["frame"])}
                assert len(indices) == n
                for frame, row in raw["labels"].items():
                    index = indices[int(frame)]
                    for side in row["labels"]:
                        station = int(side["station_m"] / 2.) - 1
                        j = ("right", "left").index(side["side"])
                        np.testing.assert_allclose(data["lateral_m"][index, station, j], side["lateral_m"], atol=1e-6)
                        assert data["adjacent_invalid"][index, station, j] == side["witnessed"]
                        if side["pixel"] is not None:
                            np.testing.assert_allclose(data["image_px"][index, 0, station, j], side["pixel"],
                                                       atol=3e-5)
            result[track] = dict(frames=n, labels_sha256=receipt["labels_sha256"],
                                 adjacent_invalid_pairs=receipt["adjacent_invalid_pairs"],
                                 projected_labels=dict(zip(data["views"], counts, strict=True)))
    report = dict(status="verified_offline_labels", tracks=result, source_files=len(checked),
                  verifier_sha256=digest(Path(__file__)), actor_input=False)
    (OUT / "training-labels-verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
