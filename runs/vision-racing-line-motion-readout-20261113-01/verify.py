"""Verify line symmetry, causal prefix invariance, missing data, and exported frame coverage."""
import json
from pathlib import Path

import numpy as np

from evaluate_readout import CLI, EVAL, OUT, digests, load_fixed, load_sensor, metadata, old
from line_readout import HORIZONS, MODES, lateral_prediction, sensor_features, bounded_path
from teacher_labels import future_targets


def reflect(data):
    result = dict(data)
    result["sections"] = -data["sections"][:,:,::-1]
    result["motion"] = data["motion"].copy()
    result["motion"][:,2] *= -1
    result["t4t5"] = data["t4t5"][:,::-1,:]
    return result


def verify_labels():
    rows = [dict(run="synthetic", frame=i, received_s=i*.1, generation="initial") for i in range(7)]
    poses = [(.1*i, 30., np.array([3.*i,0.]), np.array([1.,0.])) for i in range(7)]
    labels, _ = future_targets(rows,poses,np.array([5.,10.]))
    np.testing.assert_allclose(labels[0],[[5.,0.],[10.,0.]])
    rows[3]["generation"]="restart"
    split, _ = future_targets(rows,poses,np.array([5.,10.]))
    assert np.isnan(split[2]).all() and np.isfinite(split[0,0]).all() and np.isnan(split[0,1]).all()
    for points in ([0.,0.,0.], [5.,-5.,5.], [-5.,5.,-5.]):
        path = bounded_path(points)
        assert np.all(np.diff(path[:,0])>0)
        np.testing.assert_allclose(np.linalg.norm(np.diff(path,axis=0),axis=1),.25,atol=1.e-13)
    assert bounded_path([0.,np.nan,0.]) is None


def verify_tracks(policy,models):
    result = {}
    for track,name in EVAL:
        d=CLI/("vallelunga-retry" if name=="vallelunga" else name)
        bundle = load_sensor(d/"sensors.npz",old.read(d/"result.json")["inputs_sha256"])
        f = sensor_features(bundle)
        reflected = sensor_features(reflect(bundle))
        prefix_length = len(bundle["frame"])//2
        prefix = {key: value if key=="stations" else value[:prefix_length] for key,value in bundle.items()}
        missing = dict(bundle,t4t5=np.full_like(bundle["t4t5"],np.nan))
        missing = sensor_features(missing)
        errors = {}
        for mode in MODES:
            full = lateral_prediction(f,mode,models[mode])
            mirror = lateral_prediction(reflected,mode,models[mode])
            partial = lateral_prediction(sensor_features(prefix),mode,models[mode])
            np.testing.assert_allclose(full,-mirror,equal_nan=True,rtol=0.,atol=1.e-9)
            np.testing.assert_allclose(full[:prefix_length],partial,equal_nan=True,rtol=0.,atol=1.e-9)
            loss = lateral_prediction(missing,mode,models[mode])
            if mode in ("motion","boundary_motion"):
                assert np.isnan(loss).all()
            else:
                np.testing.assert_array_equal(full,loss)
            errors[mode] = float(np.nanmax(abs(full+mirror)))
        file = OUT/f"{track}-line.jsonl"
        exported = [json.loads(row) for row in file.read_text(encoding="utf-8").splitlines()]
        assert len(exported)==len(bundle["frame"])
        emitted = 0
        expected = lateral_prediction(f,policy["selected_vision_readout"],models[policy["selected_vision_readout"]])
        for i,(ref,row) in enumerate(zip(metadata(bundle),exported,strict=True)):
            assert (ref["run"],ref["frame"],ref["generation"],ref["received_s"]) == (
                row["episode_id"],row["frame"],row["generation"],row["observation_time_s"])
            assert row["schema"] == "flybrain.racing_line.v1"
            assert [p["s_m"] for p in row["points"]]==HORIZONS.tolist()
            available=np.isfinite(expected[i]).all()
            if available:
                assert row["status"]=="available" and all(p["valid"] for p in row["points"])
                xy = np.array([(p["x_m"],p["y_m"]) for p in row["points"]])
                np.testing.assert_allclose(xy,bounded_path(expected[i])[(HORIZONS/.25).astype(int)])
            else:
                assert row["status"]=="abstain" and all(not p["valid"] and p["x_m"] is None and p["y_m"] is None for p in row["points"])
            emitted+=int(available)
        result[track]=dict(frames=len(exported),available=emitted,abstain=len(exported)-emitted,
                           reflection_max_lateral_difference_m=errors,output_sha256=old.digest(file))
        print(track,result[track],flush=True)
    return result


def main():
    verify_labels()
    policy, models = load_fixed()
    result = verify_tracks(policy, models)
    frozen = OUT/"frozen.json"
    comparison = OUT.parent/"vision-corner-left-risk-reallocation-20261113-01/comparison.json"
    evaluation = OUT/"evaluation.json"
    receipt = old.read(evaluation)
    assert all(t["output_sha256"]==result[name]["output_sha256"] for name,t in receipt["tracks"].items())
    corner_frozen = comparison.with_name("frozen.json")
    assert old.read(comparison)["freeze_sha256"] == old.digest(corner_frozen)
    old.write(OUT/"verification.json",dict(tracks=result, teacher_gap_reset_verified=True,
              causal_prefix_matches=True,t4t5_missing_abstains=True,
              source_sha256=digests((Path(__file__),frozen,evaluation,comparison,corner_frozen)),
              termination="verified",seed=None))


if __name__=="__main__":
    main()
