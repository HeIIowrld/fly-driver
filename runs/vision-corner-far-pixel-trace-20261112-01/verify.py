"""Verify plotted pixels, intervention receipts and full-window comparisons."""

import ast
from collections import Counter
from pathlib import Path
import subprocess

from evaluate_fused_corner import read_rows
from trace import OUT
from train_vision_teacher_line import digest, read, write


def main():
    names=("pixel-trace.json","visible-transitions.json","projection-counterfactual.json",
           "perturbation-result.json","transfer-result.json")
    receipts={name:read(OUT/name) for name in names}
    for result in receipts.values():
        for name,expected in result["source_sha256"].items():
            assert digest(Path(name))==expected,name
    for frame,row in receipts["pixel-trace.json"]["frames"].items():
        for kind,sha in row["images"].items():
            assert digest(OUT/f"{frame}-{kind}.png")==sha
    totals={}
    for source in names[-2:]:
        for track,item in receipts[source]["tracks"].items():
            path=OUT/f"{track}-perturbations.jsonl"
            rows=read_rows(path)
            assert digest(path)==item["rows_sha256"]
            assert len({(r["run"],r["frame"]) for r in rows})==len(rows)
            for mode,values in item["metrics"].items():
                assert values["frames"]==len(rows) and values["no_output"]==0
                if source==names[-1]:
                    totals.setdefault(mode,Counter()).update(values)
            for row in rows:
                for model in ("edge","t4t5"):
                    d=row["decisions"]
                    assert d[model+"_robust"] in (d[model+"_base"],d[model+"_fixed"])
                    if d[model+"_base"] is not None:
                        assert d[model+"_robust"]==d[model+"_base"]
    assert totals["t4t5_base"]["frames"]==6087
    for p in OUT.glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8-sig"))
    safe=Path.cwd().resolve().as_posix()
    revision=subprocess.check_output(["git","-c",f"safe.directory={safe}","rev-parse","HEAD"],text=True).strip()
    write(OUT/"verification.json",dict(status="verified",seed=None,source_revision=revision,
        hashes={name:digest(OUT/name) for name in names},development=totals,
        termination="pixel_overlays_and_full_timelines_verified"))
    print(totals,flush=True)


if __name__=="__main__":
    main()
