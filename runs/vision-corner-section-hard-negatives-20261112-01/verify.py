"""Verify experiment receipts and summarize paired changes in corner warnings."""
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONTROL = ROOT / "runs/vision-corner-local-curvature-20261112-01"


def sha(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as source:
        return list(map(json.loads, source))


def brief(result):
    right = result["by_direction"]["right"]
    left = result["by_direction"]["left"]
    return dict(frames=result["frames"], left=left["direction_hits"], right=right["direction_hits"],
                joint=left["joint_hits"] + right["joint_hits"], right_joint=right["joint_hits"],
                false_alerts=result["false_alerts"]["outside_band"]["frames"],
                stable=sum(item["first_stable_frame"] is not None for item in result["events"]))


def direction(proposal):
    return "none" if proposal is None else {1: "left", 2: "right"}[proposal[0]]


def main():
    base = read(CONTROL / "result.json")
    results = [OUT / "result.json", OUT / "local/result.json"]
    hinge, local = map(read, results)
    assert sha(OUT / "patch-results.json") == hinge["patch_results_sha256"]
    assert sha(OUT / "hard-negatives.joblib") == hinge["checkpoint_sha256"]
    sources = {}
    for receipt, folder in zip((hinge, local), (OUT, OUT / "local"), strict=True):
        sources.update(receipt["source_sha256"])
        for track, report in receipt["tracks"].items():
            sources.update(report["source_sha256"])
            assert sha(folder / f"{track}-frames.jsonl") == report["rows_sha256"]
            if "sections_sha256" in report:
                assert sha(folder / f"{track}-sections.npz") == report["sections_sha256"]
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, name
    analysis = {}
    modes = ("ranked_local", "ranked_local_boundary_hold", "ranked_local_t4t5_hold")
    for track, result in local["tracks"].items():
        reference = base["tracks"][track]
        path = CONTROL / f"{track}-frames.jsonl"
        assert sha(path) == reference["rows_sha256"]
        old, new = rows(path), rows(OUT / "local" / path.name)
        assert len(old) == len(new) == 1200
        right = Counter()
        changed = []
        for a, b in zip(old, new, strict=True):
            for key in ("run", "frame", "truth", "label", "physical_corner", "truth_endpoints_m"):
                assert a[key] == b[key], key
            for mode in ("baseline", "normal_local"):
                assert a["proposals"][mode] == b["proposals"][mode], (track, a["frame"], mode)
            if a["truth"] != "approach" or a["label"] != "2":
                continue
            before, after = a["proposals"]["ranked_local"], b["proposals"]["ranked_local"]
            right[f"{direction(before)} -> {direction(after)}"] += 1
            if before != after:
                changed.append(dict(frame=a["frame"], truth_endpoints_m=a["truth_endpoints_m"],
                                    before=before, after=after))
        report = {}
        for mode in modes:
            original = [brief(run) for run in reference["methods"][mode].values()]
            modified = [brief(run) for run in result["methods"][mode].values()]
            report[mode] = dict(random=original, hard=modified)
        analysis[track] = dict(modes=report, right_candidate_transitions=dict(right),
                               changed_right_approaches=changed)
    evidence = dict(status="verified", seed=hinge["seed"], source_revision=hinge["source_revision"],
                    result_sha256={str(p.relative_to(OUT)): sha(p) for p in results},
                    tracks=analysis, termination="sources_and_frame_pairs_verified")
    (OUT / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
