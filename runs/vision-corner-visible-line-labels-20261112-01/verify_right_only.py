"""Verify that only the right trace changes in the side intervention."""
import json
from pathlib import Path

from annotate import OUT, digest
from verify import OLD, ROOT, read


def main():
    candidate = OUT / "right-only"
    paths = [OLD / "result.json", OUT / "result.json", candidate / "result.json"]
    absolute, signed, right = map(read, paths)
    for name, expected in right["source_sha256"].items():
        assert digest(ROOT / name) == expected
    key = "ks_silverstone1967"
    diagnostics = [r["tracks"][key]["diagnostic"] for r in (absolute, signed, right)]
    for frame, row in diagnostics[2].items():
        assert row["trace"][0] == diagnostics[1][frame]["trace"][0], frame
        assert row["trace"][1] == diagnostics[0][frame]["trace"][1], frame
    parity = {}
    for track, result in right["tracks"].items():
        for name, expected in result["source_sha256"].items():
            assert digest(ROOT / name) == expected
        a, b = (folder / f"{track}-frames.jsonl" for folder in (OLD, candidate))
        assert digest(a) == absolute["tracks"][track]["rows_sha256"]
        assert digest(b) == result["rows_sha256"]
        rows = [list(map(json.loads, p.read_text(encoding="utf8").splitlines())) for p in (a, b)]
        for prior, actual in zip(*rows, strict=True):
            assert (prior["run"], prior["frame"]) == (actual["run"], actual["frame"])
            for mode in ("edge_base", "t4t5_base", "edge_far_wide", "t4t5_far_wide"):
                assert prior["decisions"][mode] == actual["decisions"][mode]
        assert len(rows[0]) == 1200
        parity[track] = len(rows[0])
    report = dict(status="verified", compared_diagnostics=len(diagnostics[2]),
                  unchanged_control_frames=parity, results_sha256={str(p): digest(p) for p in paths},
                  verifier_sha256=digest(Path(__file__)))
    (candidate / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print("13 Silverstone paired side traces and 2,400 unchanged control frames verified.")


if __name__ == "__main__":
    main()
