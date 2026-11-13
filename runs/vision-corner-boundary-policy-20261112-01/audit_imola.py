"""Locate lost true alerts under the more permissive boundary threshold."""
from collections import Counter
from calibrate_policy import (OUT, DEVELOP, EXTRA, PREV, POLICIES, SensorArrays, examples, features,
    load_labels, load_observer, np, observe, old, patch, read_development, selected, train)


def turn(row, method):
    prediction = row["forecasts"][method]
    side = {"left":1, "right":2}.get(prediction["state"])
    return None if side is None else (side, prediction["distance_m"])


def run():
    calibration=OUT/"training.json"
    assert old.read(calibration)["chosen"]=="raw"
    observer,settings,sources=load_observer()
    sources.update((calibration,Path(__file__)))
    dev={}
    for track in DEVELOP:
        data,truth,receipt=read_development(track,sources,settings)
        edge,visible=load_labels(EXTRA/f"{track}-train-labels.npz",data,sources,receipt["camera_bundle_sha256_by_run"])
        ex=examples(features(SensorArrays(data),settings)[0])
        dev[track]=((ex,edge,visible),data,truth)
    test="imola"
    (ex,edge,visible),data,truth=dev[test]
    choice=selected(ex,train([v[0] for t,v in dev.items() if t!=test]))
    patched=patch(data["sections"],ex,choice,POLICIES["score_65"])
    before=observe(SensorArrays(data),observer,settings)[0]
    after=observe(SensorArrays(dict(data,sections=patched)),observer,settings)[0]
    outputs={}
    for method in ("temporal","t4t5_confirm"):
        hits=Counter()
        dropped=[]
        for i,(ref,a,b) in enumerate(zip(truth,before,after,strict=True)):
            assert (ref["run"],ref["frame"])==(a["run"],a["frame"])==(b["run"],b["frame"])
            a_hit=old.category(ref,turn(a,method))=="joint_hit"
            b_hit=old.category(ref,turn(b,method))=="joint_hit"
            hits.update(before=int(a_hit),after=int(b_hit))
            if a_hit and not b_hit:
                candidate=old.category(ref,b["candidate"])
                stage="confirmation" if candidate=="joint_hit" else "candidate"
                accepted=~np.isfinite(data["sections"][i]) & np.isfinite(patched[i])
                labelled=accepted & visible[i]
                errors=abs(patched[i][labelled]-edge[i][labelled])
                dropped.append(dict(frame=ref["frame"],before=a["candidate"],after=b["candidate"],
                     cause=stage,after_candidate_category=candidate,added_points=int(accepted.sum()),
                     labelled_errors_m=errors.tolist()))
        outputs[method]=dict(hits=hits,dropped=dropped)
        frozen=old.read(calibration)["development_results"][test]
        assert hits["before"]==frozen["raw"]["forecasts"][method]["joint_hit"]
        assert hits["after"]==frozen["score_65"]["forecasts"][method]["joint_hit"]
    old.write(OUT/"imola-dropped.json",dict(track=test,policy="score_65",outputs=outputs,
            source_sha256={str(p):old.digest(p) for p in sorted(sources)}))
    print(outputs,flush=True)


if __name__=="__main__":
    from pathlib import Path
    run()
