"""Verify routing invariants and summarize the four states and distances."""
from collections import Counter
from pathlib import Path

import numpy as np

from routing import OUT, ALIGNED, POLICIES, DEVELOP, SILVER, VAL, METHODS, old
from verify_directional import frames, counts

EDGE = "temporal"
MOTION = "t4t5_confirm"
STATES = ("straight", "left", "right", "abstain")


def diagnose(rows, method):
    result = counts(rows, method)
    confusion, distances = {}, []
    for row in rows:
        key = row["truth"] + "/" + str(row["label"])
        states = confusion.setdefault(key, dict.fromkeys(STATES, 0))
        observed = row["states"][method+"_candidate_only"]
        states[observed] += 1
        prediction = row["scores"][method]["accepted"]
        if old.category(row, prediction) == "joint_hit":
            error = max(abs(prediction[1]-x) for x in row["truth_endpoints_m"])
            distances.append(dict(frame=row["frame"], side=row["label"], distance=prediction[1],
                                  truth_m=row["truth_endpoints_m"], max_error_m=error))
    result.update(confusion=confusion, hits=distances,
                  joint_only_mean_max_error_m=float(np.mean([d["max_error_m"] for d in distances])) if distances else None)
    return result


def main():
    training_path, result_path = OUT/"training.json", OUT/"result.json"
    training, result = old.read(training_path), old.read(result_path)
    assert training["development"] == list(DEVELOP) and training["excluded"] == [SILVER, VAL]
    assert result["training_sha256"] == old.digest(training_path)
    for name, digest in training["source_sha256"].items():
        assert SILVER not in name and VAL not in name and old.digest(Path(name)) == digest
    aligned = old.read(ALIGNED/"result.json")
    reports = {}
    for track, record in result["tracks"].items():
        for name, digest in record["source_sha256"].items():
            assert old.digest(Path(name)) == digest
        variants = {name: frames(OUT/f"{track}-{name}.jsonl", info["rows_sha256"])
                    for name, info in record["variants"].items()}
        for name, before in (("direct", "direct"), ("union", "carry_frozen")):
            assert record["variants"][name]["rows_sha256"] == aligned["tracks"][track]["variants"][before]["rows_sha256"]
        assert all(len(rows) == 1200 for rows in variants.values())
        changes = {p: Counter() for p in POLICIES}
        for i, current in enumerate(variants["direct"]):
            full = variants["union"][i]
            for name, stream in variants.items():
                row = stream[i]
                for key in ("run", "frame", "truth", "label", "truth_endpoints_m", "common_edge"):
                    assert row[key] == current[key]
                assert row["scores"][EDGE]["candidate"] == row["scores"][MOTION]["candidate"]
                assert row["pool"] == (current if name == "direct" else full)["pool"]
                if name in ("direct", "union"):
                    continue
                policy, label = name.rsplit("_", 1)
                assert policy in POLICIES and label in ("fixed", "calibrated")
                for method in METHODS:
                    a, b, chosen = (v["scores"][method] for v in (current, full, row))
                    c, proposed = a["candidate"], b["candidate"]
                    match = (c is not None and proposed is not None and c[0] == proposed[0] and
                             abs(c[1]-proposed[1]) <= training["max_agreement_distance_m"])
                    use_recovery = not current["pool"] or (policy == "empty_or_consistent" and match)
                    expected = b if use_recovery else a
                    assert chosen["candidate"] == expected["candidate"]
                    if chosen["score"] is None:
                        assert expected["score"] is None
                    else:
                        np.testing.assert_allclose(chosen["score"], expected["score"], rtol=1.e-11, atol=1.e-12)
                    cuts = training["baseline_cuts"] if label == "fixed" else training["cuts"][policy]
                    accepted = chosen["candidate"] if chosen["score"] is not None and chosen["score"] >= cuts[method] else None
                    assert chosen["accepted"] == accepted
                    if label == "fixed" and method == EDGE and chosen["candidate"] != c:
                        changes[policy]["changed_choice"] += 1
                        changes[policy]["from_empty"] += int(c is None)
        reports[track] = dict(routes={p: dict(c) for p, c in changes.items()},
                              variants={n: {m: diagnose(rows, m) for m in (EDGE, MOTION)}
                                        for n, rows in variants.items()})
    old.write(OUT/"verification.json", dict(status="verified", paired_frames=2400,
              result_sha256=old.digest(result_path), tracks=reports,
              termination="frozen_routes_four_state_and_distance_verified"))


if __name__ == "__main__":
    main()
