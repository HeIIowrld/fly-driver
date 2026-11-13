"""Verify the frozen policy and the missing boundary and alert changes."""
from collections import Counter
from calibrate_policy import (ARCHIVE, EVALUATION, METHODS, OUT, POLICIES, PREV, Path, SILVER, SensorArrays,
    features, examples, frames, joblib, load_labels, load_observer, np, old, patch, reference, row_map, select_policy, selected)


def run():
    report = old.read(OUT/"result.json")
    file = OUT/"training.json"
    receipt = old.read(file)
    assert old.digest(file)==report["training_sha256"]
    policy, eligible, _ = select_policy(receipt["development_results"])
    assert policy==report["policy"]==receipt["chosen"]
    assert eligible==receipt["eligible"]
    assert {k:tuple(v) for k,v in receipt["policies"].items()}==POLICIES
    for path,digest in receipt["source_sha256"].items():
        assert old.digest(Path(path))==digest, path
    observer, settings, _ = load_observer()
    model = joblib.load(PREV/"selector.joblib")["model"]
    verification = {}
    for track,folder in EVALUATION:
        item = report["tracks"][track]
        for path,digest in item["source_sha256"].items():
            assert old.digest(Path(path))==digest, path
        rows=frames(OUT/f"{track}-frames.jsonl",item["trace_sha256"])
        root=ARCHIVE/folder
        archive=root/"predictions/forecasts.jsonl"
        original=frames(archive,old.read(archive.with_name("result.json"))["forecasts_sha256"])
        by_key=row_map(original)
        states=Counter()
        for row in rows:
            prior=by_key[row["run"],row["frame"]]
            assert row["before"]["candidate"]==prior["candidate"]
            for method in METHODS:
                for key,value in prior["forecasts"][method].items():
                    assert row["before"]["forecasts"][method][key]==value
                decision=row["after"]["forecasts"][method]
                direction={"left":1,"right":2}.get(decision["state"])
                distance=decision["distance_m"]
                if direction is None:
                    assert decision["state"] in ("straight","abstain") and distance is None
                else:
                    assert row["after"]["candidate"]==[direction,distance] and 4<=distance<=20
                states[f"{method}/{decision['state']}"]+=1
        with np.load(root/"sensors.npz",allow_pickle=False) as data:
            sensor=SensorArrays({k:data[k] for k in data.files})
        output=OUT/f"{track}-boundaries.npz"
        assert old.digest(output)==item["boundaries_sha256"]
        with np.load(output,allow_pickle=False) as data:
            for key in ("stations","run","frame"):
                np.testing.assert_array_equal(data[key],sensor[key])
            stored=data["sections"]
        raw=sensor["sections"]
        ex=examples(features(sensor,settings)[0])
        choice=selected(ex,model)
        actual=patch(raw,ex,choice,POLICIES[policy])
        np.testing.assert_equal(stored,actual)
        verified=dict(states=states,new=int((~np.isfinite(raw)&np.isfinite(actual)).sum()))
        if track==SILVER:
            test=SensorArrays({k:v if k=="stations" else v[:810] for k,v in sensor.items()})
            short=examples(features(test,settings)[0])
            result=patch(test["sections"],short,selected(short,model),POLICIES[policy])
            np.testing.assert_equal(actual[:810],result)
            mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge, visible=load_labels(mesh,sensor,set())
            labels=row_map(reference(track)[0])
            additions=[]
            mask=np.zeros(raw.shape,bool)
            for i,row in enumerate(original):
                ref=labels[row["run"],row["frame"]]
                if ref["truth"]!="approach" or ref["label"]!="2":
                    continue
                mask[i]=visible[i] & ~np.isfinite(raw[i])
                for j,k in np.argwhere(visible[i] & ~np.isfinite(raw[i]) & np.isfinite(actual[i])):
                    additions.append(dict(frame=row["frame"],station_m=float(sensor["stations"][j]),
                                          side=("right","left")[k], error_m=float(abs(actual[i,j,k]-edge[i,j,k]))))
            verified["right_approach_additions"]=additions
            verified["diagnostic_not_used_for_policy_choice"]={}
            for name,candidate in POLICIES.items():
                boundary=patch(raw,ex,choice,candidate)
                use=mask & np.isfinite(boundary)
                errors=abs(boundary-edge)
                verified["diagnostic_not_used_for_policy_choice"][name]={
                    side:dict(accepted=int(use[:,:,k].sum()),
                              within_125cm=int((use[:,:,k] & (errors[:,:,k]<=1.25)).sum()))
                    for k,side in enumerate(("right","left"))
                }
        verification[track]=verified
    old.write(OUT/"verification.json",dict(status="verified",chosen=policy,tracks=verification,
                                          result_sha256=old.digest(OUT/"result.json")))
    print("Verified policy selection, source files, replay contracts, and boundary coordinates.",flush=True)


if __name__=="__main__":
    run()
