"""Select and freeze a racing-line readout using leave-one-circuit-out development."""
from pathlib import Path
import sys

import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "vision-corner-fixed-left-20261113-01"))
from fixed_left import DEVELOP, EVAL, digests, old
from line_readout import ALPHAS, MODES, WEIGHTS, HORIZONS, availability, lateral_prediction, sensor_features, train

SEED = None
DEVELOPMENT = OUT / "development-labels.json"


def validation_target(prediction, target, use):
    lateral = target[use,:,1]
    delta = abs(prediction[use]-lateral)
    if not len(delta):
        raise ValueError("No eligible validation frames")
    curved = abs(lateral[:,-1]) >= 1.
    straight = ~curved
    reports = {}
    for name, selection in (("all",np.ones(len(lateral),dtype=bool)),("curved",curved),("straight",straight)):
        if not selection.any():
            raise ValueError("Validation split lacks straight or curved trajectory targets")
        reports[name] = dict(n=int(selection.sum()), lateral_mae_m=delta[selection].mean(axis=0).tolist())
    return reports


def selection_cost(folds):
    # Give both turn and straight frames weight without making straight samples dominate.
    return float(np.mean([.7*f["curved"]["lateral_mae_m"][-1] + .3*f["straight"]["lateral_mae_m"][-1]
                          for f in folds.values()]))


def main():
    freeze_path = OUT / "frozen.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    labels = old.read(DEVELOPMENT)
    inputs = {DEVELOPMENT, Path(__file__), Path(train.__code__.co_filename)}
    data = {}
    for track in DEVELOP:
        record = labels["artifacts"][track]
        path = Path(record["file"])
        if old.digest(path) != record["sha256"]:
            raise ValueError("Development artifact hash changed")
        inputs.add(path)
        with np.load(path, allow_pickle=False) as bundle:
            data[track] = sensor_features(bundle), bundle["teacher_xy_m"]
    trials = {}
    for mode in MODES:
        candidates = []
        for alpha in ALPHAS:
            for balance in WEIGHTS:
                fold = {}
                for name, (features, target) in data.items():
                    fit = train([v for t,v in data.items() if t != name], mode, alpha, balance)
                    pred = lateral_prediction(features, mode, fit)
                    # Identical validation frames across every readout.
                    valid = availability(features,"boundary_motion") & np.isfinite(target).all(axis=(1,2))
                    fold[name] = validation_target(pred,target,valid)
                candidates.append(dict(alpha=alpha, curvature_balance=balance, folds=fold,
                                       cost=selection_cost(fold)))
        trials[mode] = sorted(candidates,key=lambda c:c["cost"])
        print(mode, {k:trials[mode][0][k] for k in ("alpha","curvature_balance","cost")}, flush=True)
    mode = min(("boundary","boundary_motion","motion"), key=lambda m:trials[m][0]["cost"])
    models = {}
    for method, options in trials.items():
        selected = options[0]
        models[method] = train(list(data.values()),method,selected["alpha"],selected["curvature_balance"])
    file = OUT / "readouts.joblib"
    joblib.dump(models,file)
    old.write(freeze_path,dict(development=list(DEVELOP), comparisons=[t for t,_ in EVAL],
              held_for_independent_replay="ks_highlands", selected_vision_readout=mode,
              development_trials=trials, horizons_m=HORIZONS.tolist(), seed=SEED,
              model_sha256=old.digest(file), sources=digests(inputs), source_revision=old.revision(),
              termination="frozen_before_evaluation"))
    print("Selected",mode,flush=True)


if __name__ == "__main__":
    main()
