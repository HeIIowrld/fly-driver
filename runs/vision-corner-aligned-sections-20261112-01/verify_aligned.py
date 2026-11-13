"""Verify causal transport and score recovered edges against held-out mesh labels."""
from pathlib import Path

import numpy as np

from aligned_sections import OUT, LISTWISE, old, fill, project_sections, sequence, valid_pairs
from listwise import SILVER, VAL, DEVELOP, METHODS
from verify_directional import frames, counts


def check_transport():
    x = np.arange(2., 18., 2.)
    raw = np.tile([-3., 3.], (len(x), 1))
    sections = np.repeat(raw[None], 3, axis=0)
    sections[2, 2, 1] = np.nan
    motion = np.array([[1., 10., 0.], [1.1, 10., 0.], [1.2, 10., 0.]])
    filled = sequence(x, [False, True, True], sections, motion)
    np.testing.assert_array_equal(filled[2], raw)
    broken = sequence(x, [False, True, False], sections, motion)
    np.testing.assert_array_equal(broken, sections)
    bad_clock = motion.copy()
    bad_clock[2, 0] = 1.4
    np.testing.assert_array_equal(sequence(x, [False, True, True], sections, bad_clock), sections)
    changed_future = sections.copy()
    changed_future[2] += 9.
    np.testing.assert_array_equal(sequence(x, [False, True, True], changed_future, motion)[:2], filled[:2])
    points = np.stack((np.repeat(x[2:6, None], 2, axis=1), raw[2:6]), axis=-1)
    projection = project_sections(x, points)
    assert np.isnan(projection[:2]).all() and np.isnan(projection[6:]).all()
    wrong = raw.copy()
    wrong[2] += 4.
    bad = fill(x, sections[2], [wrong])
    assert np.isnan(bad[2, 1])
    # Conflicting histories with no agreement do not silently become an average.
    hole = sections[2].copy()
    hole[2] = np.nan
    disagree = raw.copy()
    disagree[2] += 2.
    assert np.isnan(fill(x, hole, [raw, disagree])[2]).all()


def mesh_score(data, rows):
    path = OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{SILVER}-train-labels.npz"
    receipt = path.with_suffix(".json")
    assert old.digest(path) == old.read(receipt)["labels_sha256"]
    with np.load(path, allow_pickle=False) as labels:
        assert list(zip(labels["run"], labels["frame"], strict=True)) == list(zip(data["run"], data["frame"], strict=True))
        np.testing.assert_array_equal(labels["stations"], data["stations"])
        witness = labels["adjacent_invalid"]
        valid = witness & np.isfinite(labels["image_px"]).all(axis=-1).any(axis=1)
        truth = labels["lateral_m"]
    right = np.array([r["truth"] == "approach" and r["label"] == "2" for r in rows])
    existing = np.isfinite(data["current"])
    new = np.isfinite(data["aligned"]) & ~existing
    errors = abs(data["aligned"] - truth)
    reports = {}
    for name, include in (("all", np.ones(len(rows), bool)), ("right_approach", right)):
        mask = valid & include[:, None, None]
        fresh = mask & new
        reports[name] = dict(
            labelled=mask.sum(axis=(0, 1)).tolist(),
            observed=(mask & existing).sum(axis=(0, 1)).tolist(),
            recovered=fresh.sum(axis=(0, 1)).tolist(),
            added_mae_m=[float(errors[..., i][fresh[..., i]].mean()) if fresh[..., i].any() else None for i in range(2)])
    return dict(source_sha256={str(p): old.digest(p) for p in (path, receipt)}, counts=reports)


def main():
    check_transport()
    result_path, training_path = OUT/"result.json", OUT/"training.json"
    training, result = old.read(training_path), old.read(result_path)
    assert result["training_sha256"] == old.digest(training_path)
    assert training["checkpoint_sha256"] == old.digest(OUT/"rankers.joblib")
    assert training["development"] == list(DEVELOP) and training["excluded"] == [SILVER, VAL]
    for name, digest in training["source_sha256"].items():
        assert SILVER not in name and VAL not in name and old.digest(Path(name)) == digest
    reports = {}
    for track, report in result["tracks"].items():
        for name, digest in report["source_sha256"].items():
            assert old.digest(Path(name)) == digest
        variants = {name: frames(OUT/f"{track}-{name}.jsonl", r["rows_sha256"])
                    for name, r in report["variants"].items()}
        assert all(len(rows) == 1200 for rows in variants.values())
        for direct, fixed, refit in zip(*variants.values(), strict=True):
            for name in ("run", "frame", "truth", "label", "truth_endpoints_m", "common_edge"):
                assert direct[name] == fixed[name] == refit[name]
            assert fixed["pool"] == refit["pool"]
            assert all(o in fixed["pool"] for o in direct["pool"])
            for row in (direct, fixed, refit):
                assert row["scores"]["temporal"]["candidate"] == row["scores"]["t4t5_confirm"]["candidate"]
                for score in row["scores"].values():
                    offer = score["candidate"]
                    assert offer is None or (offer in row["pool"] and type(offer[0]) is int)
            for method in METHODS:
                a, b = direct["scores"][method], fixed["scores"][method]
                if a["candidate"] == b["candidate"] and a["score"] is not None:
                    np.testing.assert_allclose(a["score"], b["score"], rtol=1.e-11, atol=1.e-12)
        path = OUT/f"{track}-sections.npz"
        assert old.digest(path) == report["sections_sha256"]
        track_report = {name: {m: counts(rows, m) for m in METHODS} for name, rows in variants.items()}
        with np.load(path, allow_pickle=False) as data:
            assert list(zip(data["run"], data["frame"], strict=True)) == [(r["run"], r["frame"]) for r in variants["direct"]]
            current, aligned = data["current"], data["aligned"]
            np.testing.assert_array_equal(current[np.isfinite(current)], aligned[np.isfinite(current)])
            np.testing.assert_array_equal(data["added"], valid_pairs(aligned) & ~valid_pairs(current))
            if track == SILVER:
                track_report["mesh"] = mesh_score(data, variants["direct"])
        reports[track] = track_report
    old.write(OUT/"verification.json", dict(status="verified", paired_frames=2400, causal_transport=True,
              result_sha256=old.digest(result_path), tracks=reports,
              termination="causal_transport_frozen_replay_and_mesh_scored"))


if __name__ == "__main__":
    main()
