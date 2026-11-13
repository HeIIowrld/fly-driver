"""Verify directional geometry and the frozen paired replay."""
from collections import Counter
from pathlib import Path
import json

import numpy as np

from directional_features import OUT, PARENT, PHASE, DEVELOP, SILVER, VAL, hinge, relative_features, old


def frames(path, digest):
    assert old.digest(path) == digest, path
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def check_geometry():
    x = np.arange(2., 36., 2.)
    d = 14.
    local = x - d
    offsets = np.array([-4., 4.])
    y = offsets + .002 * np.maximum(local[:, None], 0.)**2
    motion = np.array([[.9, 20., .08], [1., 20., .1]])
    left = relative_features(x, y, motion[-1], motion, (1, d))
    mirrored = motion * [1., 1., -1.]
    right = relative_features(x, -y[:, ::-1], mirrored[-1], mirrored, (2, d))
    np.testing.assert_allclose(left, right)
    use = abs(local) <= 12.
    step = hinge(local[use], y[use])
    uniform = hinge(local[use], (offsets + .002 * local[:, None]**2)[use])
    assert step[0] == uniform[0] == 1. and step[1] > .0039 and step[-1] > .01
    assert abs(uniform[1]) < 1.e-12 and abs(uniform[-1]) < 1.e-12
    assert left[0] > 0. and left[1] > 0.


def counts(rows, method):
    total, right = Counter(), []
    for row in rows:
        score = row["scores"][method]
        outcome = old.category(row, score["accepted"])
        total[outcome] += 1
        if row["truth"] != "approach" or row["label"] != "2":
            continue
        if not any(old.category(row, candidate) == "joint_hit" for candidate in row["pool"]):
            stage = "absent"
        elif old.category(row, score["candidate"]) != "joint_hit":
            stage = "misranked"
        elif outcome != "joint_hit":
            stage = "rejected"
        else:
            stage = "joint_hit"
        total["right_" + stage] += 1
        right.append(dict(frame=row["frame"], stage=stage, **score))
    return dict(counts=dict(total), right_frames=right)


def main():
    check_geometry()
    training_path, result_path = OUT/"training.json", OUT/"result.json"
    training, result = old.read(training_path), old.read(result_path)
    assert result["training_sha256"] == old.digest(training_path)
    assert training["checkpoint_sha256"] == old.digest(OUT/"rankers.joblib")
    assert training["development"] == list(DEVELOP) and training["excluded"] == [SILVER, VAL]
    for path, digest in training["source_sha256"].items():
        assert SILVER not in path and VAL not in path and old.digest(Path(path)) == digest
    previous = old.read(PARENT/"result.json")
    phase_report = old.read(PHASE/"result.json")
    summary = {}
    for track, report in result["tracks"].items():
        base = frames(PARENT/f"{track}-frames.jsonl", previous["tracks"][track]["rows_sha256"])
        actual = frames(OUT/f"{track}-frames.jsonl", report["rows_sha256"])
        veto = frames(OUT/f"{track}-veto-frames.jsonl", report["veto_rows_sha256"])
        phase = frames(PHASE/f"{track}-frames.jsonl", phase_report["tracks"][track]["rows_sha256"])
        assert len(actual) == len(veto) == len(base) == len(phase) == 1200
        for path, digest in report["source_sha256"].items():
            assert old.digest(Path(path)) == digest
        for row, masked, prior, q in zip(actual, veto, base, phase, strict=True):
            for key in ("run", "frame", "truth", "label", "truth_endpoints_m", "pool", "primary", "common_edge"):
                assert row[key] == masked[key] == prior[key]
            assert (row["run"], row["frame"]) == (q["run"], q["frame"])
            scores = row["scores"]
            assert scores["temporal"]["candidate"] == scores["t4t5_confirm"]["candidate"]
            for name, score in scores.items():
                guarded = masked["scores"][name]
                assert guarded["candidate"] == score["candidate"]
                if score["candidate"] is not None:
                    suppress = q["phase"][2] > training["shared_phase_veto"]
                    assert guarded["score"] == (-1. if suppress else score["score"])
        variants = (("temporal_baseline", base), ("signed", actual), ("signed_veto", veto))
        summary[track] = {method: {name: counts(rows, method) for name, rows in variants}
                          for method in ("temporal", "t4t5_confirm", "t4t5_rerank")}
    old.write(OUT/"verification.json", dict(status="verified", reflection=True,
              frozen_result_sha256=old.digest(result_path),
              termination="geometry_and_paired_candidate_replay_verified", tracks=summary))


if __name__ == "__main__":
    main()
