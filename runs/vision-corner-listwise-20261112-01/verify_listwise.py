"""Verify the rank loss and compare frozen predictions on identical recorded frames."""
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

from listwise import OUT, PREVIOUS, SILVER, VAL, DEVELOP, METHODS, objective, old
from verify_directional import frames, counts


def main():
    features = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    sample = [(features, np.array([True, True, False]), 1.)]
    assert check_grad(lambda w: objective(w, sample)[0], lambda w: objective(w, sample)[1],
                      np.array([.3, -.1])) < 1.e-6
    assert objective(np.array([1., .5]), sample)[0] < objective(np.zeros(2), sample)[0]
    train_path, result_path = OUT/"training.json", OUT/"result.json"
    training, result = old.read(train_path), old.read(result_path)
    previous = old.read(PREVIOUS/"result.json")
    assert training["development"] == list(DEVELOP) and training["excluded"] == [SILVER, VAL]
    assert result["training_sha256"] == old.digest(train_path)
    assert training["checkpoint_sha256"] == old.digest(OUT/"rankers.joblib")
    for path, digest in training["source_sha256"].items():
        assert SILVER not in path and VAL not in path and old.digest(Path(path)) == digest
    reports = {}
    for track, report in result["tracks"].items():
        prior = previous["tracks"][track]
        before = frames(PREVIOUS/f"{track}-veto-frames.jsonl", prior["veto_rows_sha256"])
        after = frames(OUT/f"{track}-veto-frames.jsonl", report["veto_rows_sha256"])
        raw = frames(OUT/f"{track}-frames.jsonl", report["rows_sha256"])
        assert len(before) == len(after) == len(raw) == 1200
        for path, digest in report["source_sha256"].items():
            assert old.digest(Path(path)) == digest
        for base, new, unguarded in zip(before, after, raw, strict=True):
            for key in ("run", "frame", "truth", "label", "truth_endpoints_m", "pool", "primary", "common_edge"):
                assert base[key] == new[key] == unguarded[key]
            scores = new["scores"]
            assert scores["temporal"]["candidate"] == scores["t4t5_confirm"]["candidate"]
            for method in METHODS:
                assert scores[method]["candidate"] == unguarded["scores"][method]["candidate"]
        changes = []
        for row in after:
            scores = row["scores"]
            edge = old.category(row, scores["temporal"]["accepted"])
            motion = old.category(row, scores["t4t5_confirm"]["accepted"])
            if edge != motion:
                changes.append(dict(run=row["run"], frame=row["frame"], label=row["label"],
                                    truth=row["truth"], edge=edge, t4t5=motion))
        reports[track] = dict(prior={m: counts(before, m) for m in METHODS},
                              listwise={m: counts(after, m) for m in METHODS},
                              paired_outcome_changes=changes)
    old.write(OUT/"verification.json", dict(status="verified", gradient=True, paired_frames=2400,
              result_sha256=old.digest(result_path), tracks=reports,
              termination="loss_gradient_and_frozen_paired_replay_verified"))


if __name__ == "__main__":
    main()
