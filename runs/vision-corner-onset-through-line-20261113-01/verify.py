"""Check saved input/decision identity and quantify coverage and errors without excluding misses."""
from collections import Counter
from pathlib import Path

import numpy as np

from dense_onset import CLI,EVAL,METHODS,OUT,TURNS,VAL,frames,old,reference,score_archive


def index(rows):
    return {(r["run"],r["frame"]):r for r in rows}


def run():
    onset=old.read(OUT/"onset-result.json")
    line=old.read(OUT/"line-result.json")
    scored=old.read(OUT/"monza-through-corner.json")
    training=OUT/"onset-training.json"
    assert old.digest(training)==onset["training_sha256"]
    assert old.digest(OUT/"dense-onset.joblib")==old.read(training)["checkpoint_sha256"]
    for receipt in (old.read(training),line,scored):
        for name,digest in receipt["source_sha256"].items():
            assert old.digest(Path(name))==digest, name
    results={}
    for track,folder in EVAL:
        forecast=frames(OUT/f"{track}-onset.jsonl",onset["tracks"][track]["forecasts_sha256"])
        planned=frames(OUT/f"{track}-lines.jsonl",line["tracks"][track]["lines_sha256"])
        root=CLI/("vallelunga-retry" if track==VAL else folder)
        prior=root/"predictions/forecasts.jsonl"
        control=frames(prior,old.read(prior.with_name("result.json"))["forecasts_sha256"])
        truth,_=reference(track)
        totals=Counter(r["label"] for r in truth if r["truth"]=="approach")
        out={}
        forecasts_index=index(forecast)
        for method in METHODS:
            assert score_archive(truth,forecasts_index,method)==onset["tracks"][track]["revised"][method]
            errors=[]
            for row in truth:
                if row["truth"]!="approach":
                    continue
                decision=indexed_forecast(forecasts_index,row,method)
                if TURNS.get(decision["state"])==int(row["label"]):
                    errors.append(max(abs(decision["distance_m"]-v) for v in row["truth_endpoints_m"]))
            out[method]=dict(distance_mae=float(np.mean(errors)) if errors else None,direction_correct=len(errors))
        variants=Counter()
        for forecast_row, old_row, path in zip(forecast,control,planned,strict=True):
            assert (forecast_row["run"],forecast_row["frame"])==(path["run"],path["frame"])
            assert path["forecast"]==forecast_row["forecasts"]["t4t5_confirm"]
            if any(f["state"] in TURNS for f in old_row["forecasts"].values()):
                assert forecast_row==old_row
            for method,f in forecast_row["forecasts"].items():
                side=TURNS.get(f["state"])
                if side:
                    assert forecast_row["candidate"]==[side,f["distance_m"]] and 4<=f["distance_m"]<=20
                else:
                    assert f["state"] in ("straight","abstain") and f["distance_m"] is None
            for name, curve in path["lines"].items():
                if curve is None:
                    continue
                x,y,k=(np.asarray(curve[name]) for name in ("x_m","y_m","curvature_inv_m"))
                assert np.isfinite([x,y,k]).all() and np.all(np.diff(x)>0)
                assert np.max(abs(k))<=line["limits"]["max_curvature"]+1.e-9
                variants[name]+=1
            variants["preview_changed"]+=path["lines"]["geometry"]!=path["lines"]["preview"]
        assert all(variants[m]==n for m,n in line["tracks"][track]["coverage"].items())
        results[track]=dict(approaches=totals,correct_direction_distance=out,line_counts=variants)
    old.write(OUT/"verification.json",dict(status="verified",tracks=results,
        source_sha256={str(OUT/f):old.digest(OUT/f) for f in ("onset-result.json","line-result.json","monza-through-corner.json")}))
    print(results,flush=True)


def indexed_forecast(rows,truth,method):
    return rows[truth["run"],truth["frame"]]["forecasts"][method]


if __name__=="__main__":
    run()
