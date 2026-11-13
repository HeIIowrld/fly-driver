"""Verify causal transport, archived identity, and paired forecast changes."""
from collections import Counter

from fit_fusion import *
from replay_corner_boundary_states import proposals


def audit():
    training = old.read(OUT/"training.json")
    report = old.read(OUT/"result.json")
    assert old.digest(OUT/"boundary.joblib") == training["checkpoint_sha256"]
    assert old.digest(OUT/"training.json") == report["training_sha256"]
    for source, sha in training["source_sha256"].items():
        assert old.digest(Path(source)) == sha, source
    model, settings, _ = load_observer()
    paired = {}
    missing_right = []
    for track, name in ((SILVER, "silverstone"), (VAL, "vallelunga"), (MONZA, "monza")):
        root = ARCHIVE/name
        result = report["tracks"][track]
        for source, sha in result["source_sha256"].items():
            assert old.digest(Path(source)) == sha, source
        with np.load(root/"sensors.npz", allow_pickle=False) as source:
            sensors = SensorArrays({k:source[k] for k in source.files})
        archived = root/"predictions/forecasts.jsonl"
        baseline = frames(archived, old.read(archived.with_name("result.json"))["forecasts_sha256"])
        control, original_aligned, _ = observe(sensors, model, settings)
        for a, b in zip(control, baseline, strict=True):
            assert (a["run"], a["frame"], a["forecasts"]) == (b["run"], b["frame"], b["forecasts"])
            assert tuple(a["candidate"] or ()) == tuple(b["candidate"] or ())
        if track == SILVER:
            stop = 812
            prefix = SensorArrays({k:v if k == "stations" else v[:stop] for k,v in sensors.items()})
            np.testing.assert_equal(features(prefix, settings)[0], features(sensors, settings)[0][:stop])
            reset = SensorArrays(prefix)
            reset["generation"] = reset["generation"].astype("U128")
            reset["generation"][-1] = "new_test_generation"
            assert np.isnan(history(reset, settings)[-1]).all()
        trace_path = OUT/f"{track}-frames.jsonl"
        trace = frames(trace_path, result["trace_sha256"])
        boundary_path = OUT/f"{track}-boundaries.npz"
        assert old.digest(boundary_path) == result["boundaries_sha256"]
        with np.load(boundary_path, allow_pickle=False) as d:
            np.testing.assert_array_equal(d["frame"], sensors["frame"])
            assert np.all(np.isfinite(d["sections"])[np.isfinite(sensors["sections"])])
            updated, updated_aligned = d["sections"], d["aligned"]
        reference_rows, _ = reference(track)
        truth_by_key = {(r["run"],r["frame"]):r for r in reference_rows}
        metadata = sensor_rows(sensors, settings)
        ids = {(r["run"],r["frame"]):i for i,r in enumerate(metadata)}
        if track == SILVER:
            mesh = OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge, visible = load_labels(mesh, sensors, set())
            f, support = features(sensors, settings)
            checkpoint = joblib.load(OUT/"boundary.joblib")
            prediction = estimate(checkpoint["coordinate"], f, support)
            probability = confidence(checkpoint["confidence"], f, support, prediction)
            for key, ref in truth_by_key.items():
                if ref["truth"] != "approach" or ref["label"] != "2":
                    continue
                i = ids[key]
                for station in np.flatnonzero(visible[i,:,0] & ~np.isfinite(sensors["sections"][i,:,0])):
                    options = f[i,station,0,[1,11,12,13,14]]
                    errors = abs(options + edge[i,station,0])
                    missing_right.append(dict(frame=key[1], station_m=float(sensors["stations"][station]),
                        sources=int(np.isfinite(options).sum()), available_within_125cm=bool(np.any(errors<=1.25)),
                        estimate_error_m=float(abs(prediction[i,station,0]+edge[i,station,0])) if support[i,station,0] else None,
                        confidence=float(probability[i,station,0]), accepted=bool(np.isfinite(updated[i,station,0]))))
            assert len(missing_right) == 10
        sections = {"before": (sensors["sections"], original_aligned), "after": (updated, updated_aligned)}
        transition = {}
        for mode in ("before","after"):
            changes = Counter()
            right_frames = []
            direct, aligned = [proposals(metadata, bounds, sensors["motion"], sensors["heads"])[0] for bounds in sections[mode]]
            for r in trace:
                a = r["forecasts"][f"{mode}_temporal"]
                b = r["forecasts"][f"{mode}_t4t5_confirm"]
                if a["category"] != b["category"]:
                    changes[f"{a['category']} -> {b['category']}"] += 1
                if r["truth"] == "approach" and r["label"] == "2":
                    key = r["run"],r["frame"]
                    i = ids[key]
                    ref = truth_by_key[key]
                    offers = sorted({candidate["offer"] for candidate in direct[i]+aligned[i]})
                    hits = [offer for offer in offers if old.category(ref, offer) == "joint_hit"]
                    selection = r[f"{mode}_candidate"]
                    right_frames.append(dict(frame=r["frame"], candidate=selection, matching_candidates=hits,
                                             selection_category=old.category(ref, tuple(selection) if selection else None),
                                             edge=a, combined=b))
                for item in (a,b):
                    if item["state"] in ("left", "right"):
                        assert r[f"{mode}_candidate"] == [{"left":1,"right":2}[item["state"]], item["distance_m"]]
            transition[mode] = dict(category_changes=changes, right_frames=right_frames)
        paired[track] = transition
    old.write(OUT/"verification.json", dict(status="verified", causality="prefix_and_session_reset",
              result_sha256=old.digest(OUT/"result.json"), paired=paired, missing_silverstone_right=missing_right))
    print("Verified control replay, source hashes, prefix causality, and identical paired candidates.", flush=True)


if __name__ == "__main__":
    audit()
