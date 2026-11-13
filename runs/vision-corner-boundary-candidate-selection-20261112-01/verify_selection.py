"""Verify unchanged observations, causal selection, and the paired corner replay."""
from select_boundary import (ARCHIVE, EVALUATION, OUT, Path, SensorArrays, examples, features,
    frames, joblib, load_observer, np, old, selected, reference, SILVER, observe)
from replay_corner_boundary_states import proposals, sensor_rows


def run():
    training = old.read(OUT/"training.json")
    result = old.read(OUT/"result.json")
    checkpoint = OUT/"selector.joblib"
    assert old.digest(checkpoint)==training["checkpoint_sha256"]
    assert old.digest(OUT/"training.json")==result["training_sha256"]
    for path, digest in training["source_sha256"].items():
        assert old.digest(Path(path))==digest
    head = joblib.load(checkpoint)
    assert head["cut"]==training["threshold"]
    observer, settings, _ = load_observer()
    silver_pool = {}
    verified = {}
    for track,folder in EVALUATION:
        report = result["tracks"][track]
        for path, digest in report["source_sha256"].items():
            assert old.digest(Path(path))==digest
        path = ARCHIVE/folder/"sensors.npz"
        with np.load(path,allow_pickle=False) as arrays:
            data=SensorArrays({k:arrays[k] for k in arrays.files})
        trace = frames(OUT/f"{track}-frames.jsonl",report["trace_sha256"])
        keys = list(zip(data["run"],data["frame"],strict=True))
        assert {(r["run"],r["frame"]) for r in trace}.issubset(keys)
        file = OUT/f"{track}-boundaries.npz"
        assert old.digest(file)==report["boundaries_sha256"]
        with np.load(file,allow_pickle=False) as stored:
            for key in ("run","frame","stations"):
                np.testing.assert_array_equal(stored[key],data[key])
            before,after = data["sections"],stored["sections"]
            known = np.isfinite(before)
            np.testing.assert_array_equal(before[known],after[known])
            assert np.count_nonzero(np.isfinite(after)&~known)==report["new_points"]
        states = {}
        for mode in ("before","after"):
            for method in ("temporal","t4t5_confirm"):
                label = f"{mode}/{method}"
                states[label] = dict.fromkeys(("straight","left","right","abstain"),0)
                for row in trace:
                    decision = row[mode]["forecasts"][method]
                    state = decision["state"]
                    distance = decision["distance_m"]
                    states[label][state] += 1
                    if state in ("left","right"):
                        assert 4<=distance<=20
                        assert row[mode]["candidate"]==[{"left":1,"right":2}[state],distance]
                    else:
                        assert distance is None
        if folder=="silverstone":
            subset = SensorArrays({k:v if k=="stations" else v[:812] for k,v in data.items()})
            ex = examples(features(subset,settings)[0])
            f = features(data,settings)[0][:812]
            ex_full = examples(f)
            a,b = selected(ex,head["model"]),selected(ex_full,head["model"])
            for x,y in zip(a,b,strict=True):
                np.testing.assert_array_equal(x,y)
            assert len(result["tracks"][track]["right_missing_points"])==10
            truth = reference(SILVER)[0]
            replay = { (r["run"],r["frame"]):r for r in trace }
            metadata = sensor_rows(data,settings)
            for mode, raw in (("before",before),("after",after)):
                sensors = SensorArrays(dict(data,sections=raw))
                output, aligned, _ = observe(sensors,observer,settings)
                direct, past = [proposals(metadata,bounds,data["motion"],data["heads"])[0] for bounds in (raw,aligned)]
                matched = []
                for i, (ref, prediction) in enumerate(zip(truth,output,strict=True)):
                    key = ref["run"],ref["frame"]
                    assert key == (prediction["run"],prediction["frame"])
                    row = replay[key][mode]
                    assert tuple(row["candidate"] or ()) == tuple(prediction["candidate"] or ())
                    for method,forecast in row["forecasts"].items():
                        assert all(forecast[k] == v for k,v in prediction["forecasts"][method].items())
                    if ref["truth"] == "approach" and ref["label"] == "2":
                        options = {c["offer"] for c in direct[i]+past[i]}
                        correct = sorted(o for o in options if old.category(ref,o) == "joint_hit")
                        if correct:
                            matched.append(dict(frame=ref["frame"],offers=correct))
                silver_pool[mode] = matched
        verified[track] = states
    old.write(OUT/"verification.json",dict(status="verified",result_sha256=old.digest(OUT/"result.json"),
                                          states=verified,causality="recording_prefix",silverstone_right_candidates=silver_pool))
    print("Verified sources, state and distance contracts, paired offers, unchanged observations, and causality.",flush=True)


if __name__=="__main__":
    run()
