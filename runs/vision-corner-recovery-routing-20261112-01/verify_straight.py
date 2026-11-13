"""Verify the final state and distance stream against the frozen turn replay."""
from pathlib import Path

from straight import OUT, ROUTE, METHODS, DEVELOP, SILVER, VAL, old, frames


def main():
    settings = OUT/"straight-training.json"
    trained = old.read(settings)
    report_path = OUT/"straight-result.json"
    report = old.read(report_path)
    route_path = OUT/"result.json"
    route = old.read(route_path)
    assert report["training_sha256"] == old.digest(settings)
    assert report["route_sha256"] == old.digest(route_path)
    assert trained["checkpoint_sha256"] == old.digest(OUT/"straight.joblib")
    assert trained["development"] == list(DEVELOP) and trained["excluded"] == [SILVER, VAL]
    for name, digest in trained["source_sha256"].items():
        assert SILVER not in name and VAL not in name and old.digest(Path(name)) == digest
    outputs = {}
    for track, evidence in report["tracks"].items():
        for name, digest in evidence["source_sha256"].items():
            assert old.digest(Path(name)) == digest
        turns = route["tracks"][track]["variants"][ROUTE]
        reference = frames(OUT/f"{track}-{ROUTE}.jsonl", turns["rows_sha256"])
        actual = frames(OUT/f"{track}-straight-frames.jsonl", evidence["rows_sha256"])
        assert len(reference) == len(actual) == 1200
        rows = []
        for base, updated in zip(reference, actual, strict=True):
            for key in ("run", "frame", "truth", "label"):
                assert base[key] == updated[key]
            forecasts = {}
            for method in METHODS:
                prior = base["states"][method+"_candidate_only"]
                target = updated["states"][method]
                assert prior == target["old"]
                accepted = base["scores"][method]["accepted"]
                if accepted is not None:
                    assert prior == target["new"] == {1: "left", 2: "right"}[accepted[0]]
                else:
                    q = updated["straight_score"]
                    assert target["new"] == ("straight" if q is not None and q >= trained["straight_cut"] else "abstain")
                forecasts[method] = dict(state=target["new"],
                                         distance_m=accepted[1] if accepted is not None else None)
            rows.append(dict(run=base["run"], frame=base["frame"], truth=base["truth"], label=base["label"],
                             truth_endpoints_m=base["truth_endpoints_m"], forecasts=forecasts))
        output = OUT/f"{track}-state-distance.jsonl"
        old.jsonl(output, rows)
        outputs[track] = dict(frames=len(rows), rows_sha256=old.digest(output))
    old.write(OUT/"straight-verification.json", dict(status="verified", paired_frames=2400,
              result_sha256=old.digest(report_path), tracks=outputs,
              termination="complete_four_state_and_distance_stream_verified"))


if __name__ == "__main__":
    main()
