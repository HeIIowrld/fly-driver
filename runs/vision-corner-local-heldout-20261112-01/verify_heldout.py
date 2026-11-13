"""Verify held-out fold evidence and summarize stable physical corners by side."""

from collections import Counter
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).parent
BASE="baseline"
GEOMETRY="ranked_local_boundary_hold"
T4T5="ranked_local_t4t5_hold"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(mapping):
    for p,h in mapping.items():
        assert sha(p)==h,p


def unique_corners(summaries):
    return {(event["label"],event["physical_corner"]) for r in summaries.values() for event in r["events"]
            if event["first_stable_frame"] is not None}


def main():
    source=OUT/"result.json"
    d=read(source)
    assert d["status"]=="evaluated" and d["termination"]=="all_five_circuits_scored"
    verified(d["source_sha256"])
    tracks=set(d["track_receipts_sha256"])
    assert len(tracks)==5 and set(d["model_training_receipts"])==tracks
    for source_hashes in d["model_training_receipts"].values():
        verified(source_hashes)
    result={}
    for track, h in d["track_receipts_sha256"].items():
        path=OUT/f"{track}-result.json"
        assert sha(path)==h
        fold=read(path)
        assert fold["held_out"]==track and set(fold["training"])==tracks-{track}
        assert sha(OUT/f"{track}-fold-models.joblib")==fold["fold_models_sha256"]
        assert sha(OUT/f"{track}-frames.jsonl")==fold["replay"]["rows_sha256"]
        assert sha(OUT/f"{track}-sections.npz")==fold["replay"]["sections_sha256"]
        verified(fold["replay"]["source_sha256"])
        changes=Counter({(r["before"],r["after"]):r["frames"] for r in fold["paired"]["transitions"]})
        false=lambda s:s in ("straight_alert","out_of_range_alert")
        delta=sum(n*(int(false(b))-int(false(a))) for (a,b),n in changes.items())
        assert delta==fold["counts"][T4T5]["outside"]-fold["counts"][GEOMETRY]["outside"]
        corners={name:sorted(unique_corners(fold["replay"]["methods"][name])) for name in (BASE,GEOMETRY,T4T5)}
        result[track]=dict(counts={m:fold["counts"][m] for m in (BASE,GEOMETRY,T4T5)},
                           stable_physical_corners=corners,paired_changes=fold["paired"]["transitions"])
    assert sum(r["counts"][BASE]["frames"] for r in result.values())==6087
    (OUT/"verified-summary.json").write_text(json.dumps(dict(status="verified",tracks=result,
        totals={k:d["totals"][k] for k in (BASE,GEOMETRY,T4T5)},result_sha256=sha(source),
        program_sha256=sha(__file__)),indent=2)+"\n",encoding="utf-8")
    print("Verified five splits, 6087 frames, model and source hashes, paired alert counts.")


if __name__=="__main__":
    main()
