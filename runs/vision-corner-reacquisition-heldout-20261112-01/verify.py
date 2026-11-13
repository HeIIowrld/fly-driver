"""Verify every replay row and compare held-history and current confirmation swaps."""

import ast
from collections import Counter
from pathlib import Path
import subprocess

from ablate import count
from evaluate_fused_corner import read_rows
from replay import OUT, key
from train_vision_teacher_line import digest, read, write


def main():
    replay = OUT / "result.json"
    ablation = OUT / "silverstone-623-ablation.json"
    known = read(replay)
    for item in (known,read(ablation)):
        for name, sha in item["source_sha256"].items():
            assert digest(Path(name)) == sha, name
    totals, comparisons = {}, {}
    for track, summary in known["tracks"].items():
        frames = OUT / f"{track}-frames.jsonl"
        rows = read_rows(frames)
        assert digest(frames) == summary["rows_sha256"]
        assert len({key(r) for r in rows}) == len(rows)
        assert track not in summary["training"]
        comparisons[track] = count(rows)
        for mode, metrics in summary["metrics"].items():
            assert metrics["frames"] == len(rows)
            assert sum(metrics["outputs"].values()) == len(rows)
            totals.setdefault(mode, Counter()).update({k:v for k,v in metrics.items() if k!="outputs"})
        assert sum(r["old"]!=r["new"] for r in rows) == len(summary["changes"])
    safe = Path.cwd().resolve().as_posix()
    revision = subprocess.check_output(["git","-c",f"safe.directory={safe}","rev-parse","HEAD"],text=True).strip()
    for p in OUT.glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8-sig"))
    write(OUT/"verification.json",dict(status="verified",source_revision=revision,seed=None,
          totals=totals,component_swaps_on_quiet_frames=comparisons,
          hashes={str(p):digest(p) for p in (replay,ablation,Path(__file__))},
          termination="full_frame_scoring_and_component_swaps_verified"))
    print(dict(totals=totals,component_swaps=comparisons),flush=True)


if __name__ == "__main__":
    main()
