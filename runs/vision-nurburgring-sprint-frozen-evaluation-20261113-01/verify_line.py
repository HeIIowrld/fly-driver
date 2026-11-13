"""Verify that the optimized path readout is causal and traceable to its saved input."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
TEACHER = RUNS / "vision-optimized-line-teacher-20261113-01"
BASE = RUNS / "vision-racing-line-motion-readout-20261113-01"
sys.path[:0] = [str(TEACHER),str(BASE)]
from evaluate_optimized import old, digests, load_sensor, metadata, sensor_features, lateral_prediction
from train_optimized import FROZEN, HORIZONS
from verify import reflect


def check(replay, output, frozen, models):
    receipt = replay/"result.json"
    inputs = old.read(receipt)
    source = replay/"sensors.npz"
    data = load_sensor(source,inputs["inputs_sha256"])
    frames = metadata(data)
    selected = frozen["selected_vision"]
    model = models[selected]
    predictions = lateral_prediction(sensor_features(data),selected,model)
    swapped = lateral_prediction(sensor_features(reflect(data)),selected,model)
    np.testing.assert_allclose(predictions,-swapped,equal_nan=True,atol=1e-10,rtol=0.)
    cut = len(frames)//2
    prefix = {key:value if key=="stations" else value[:cut] for key,value in data.items()}
    partial = lateral_prediction(sensor_features(prefix),selected,model)
    np.testing.assert_allclose(predictions[:cut],partial,equal_nan=True,atol=1e-10,rtol=0.)
    absent = dict(data,sections=np.full_like(data["sections"],np.nan))
    assert np.isnan(lateral_prediction(sensor_features(absent),selected,model)).all()
    items = [json.loads(line) for line in output.read_bytes().splitlines()]
    count = 0
    for row,label,prediction in zip(frames,items,predictions,strict=True):
        assert (row["run"],row["frame"],row["generation"],row["received_s"]) == (
            label["run"],label["frame"],label["generation"],label["observation_time_s"])
        available = np.isfinite(prediction).all()
        assert label["status"] == ("available" if available else "abstain")
        assert [p["forward_m"] for p in label["points"]] == HORIZONS.tolist()
        if available:
            np.testing.assert_allclose([p["left_m"] for p in label["points"]],prediction,atol=1e-10,rtol=0.)
            count += 1
        else:
            assert all(p["left_m"] is None for p in label["points"])
    return dict(frames=len(frames),available=count,abstain=len(frames)-count,
                causal_prefix=True,reflection=True,missing_boundary_abstains=True,
                sources=digests((receipt,source,output)))


def main():
    freeze = old.read(FROZEN)
    models = TEACHER/"optimized-readouts.joblib"
    assert old.digest(models)==freeze["checkpoint_sha256"]
    models = joblib.load(models)
    tracks = check(OUT/"archive-replay",OUT/"ks_nurburgring-optimized-readout.jsonl",freeze,models)
    result = OUT/"optimized-line-result.json"
    assert old.read(result)["evaluation"]["output_sha256"] == old.digest(OUT/"ks_nurburgring-optimized-readout.jsonl")
    old.write(OUT/"line-verification.json",dict(
        track="ks_nurburgring",checks=tracks,
        sources=digests((FROZEN,result,Path(__file__))),seed=None,termination="verified"))
    print(tracks,flush=True)


if __name__ == "__main__":
    main()
