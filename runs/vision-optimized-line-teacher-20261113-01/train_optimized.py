"""Train fixed local path estimates on four low-curvature teacher loops."""
from pathlib import Path
import sys

import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / "vision-racing-line-motion-readout-20261113-01"
sys.path.insert(0, str(BASE))
from evaluate_readout import load_fixed, metadata, old, digests, reference_poses
from fixed_left import DEVELOP, EVAL, archive_mapping
from line_readout import HORIZONS, availability, lateral_prediction, sensor_features, train
from score_frozen import teacher_points
from train_readout import validation_target, selection_cost

MODES = ("body", "boundary", "boundary_motion")
ALPHAS = (10., 100., 1000., 10000.)
TEACHER = OUT / "teacher-evaluation.json"
DEVELOPMENT = BASE / "development-labels.json"
FROZEN = OUT / "readout-frozen.json"


def examples():
    teacher = old.read(TEACHER)["tracks"]
    labels = old.read(DEVELOPMENT)["artifacts"]
    mapping, mapping_source = archive_mapping()
    sources = {TEACHER, DEVELOPMENT, mapping_source, Path(__file__),
               Path(teacher_points.__code__.co_filename), Path(train.__code__.co_filename),
               BASE / "frozen.json"}
    result = {}
    for name in DEVELOP:
        entry = labels[name]
        path = Path(entry["file"])
        reference_file = OUT / f"{name}-teacher.npz"
        if old.digest(path) != entry["sha256"] or old.digest(reference_file) != teacher[name]["teacher_sha256"]:
            raise ValueError("Development data or teacher path changed")
        with np.load(path, allow_pickle=False) as sample, np.load(reference_file, allow_pickle=False) as reference:
            feature = sensor_features(sample)
            poses = reference_poses(mapping[name], name, entry["bundle_sha256"], metadata(sample))
            line = reference["optimized_xy_m"]
            labels_xy = np.full((len(poses),len(HORIZONS),2),np.nan)
            for i, pose in enumerate(poses):
                if pose is None:
                    continue
                _,_,origin,forward = pose
                target = teacher_points(line,origin,forward,HORIZONS)
                if target is not None:
                    labels_xy[i] = np.column_stack((HORIZONS,target))
        result[name] = feature, labels_xy
        sources.update((path,reference_file,mapping[name]))
        print(name, "optimized line labels", np.isfinite(labels_xy).all(axis=2).sum(axis=0).tolist(), flush=True)
    return result, sources


def main():
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    load_fixed()
    data, sources = examples()
    selected = {}
    trials = {}
    for mode in MODES:
        scores = []
        for alpha in ALPHAS:
            folds = {}
            for name, (features, target) in data.items():
                readout = train([value for track,value in data.items() if track != name],mode,alpha,0.)
                pred = lateral_prediction(features,mode,readout)
                use = availability(features,"boundary_motion") & np.isfinite(target).all(axis=(1,2))
                folds[name] = validation_target(pred,target,use)
            scores.append(dict(alpha=alpha,cost=selection_cost(folds),folds=folds))
        trials[mode] = sorted(scores,key=lambda trial:trial["cost"])
        selected[mode] = train(list(data.values()),mode,trials[mode][0]["alpha"],0.)
        print(mode,"development cost",trials[mode][0]["cost"],flush=True)
    mode = min((name for name in MODES if name != "body"),key=lambda name:trials[name][0]["cost"])
    checkpoint = OUT / "optimized-readouts.joblib"
    joblib.dump(selected,checkpoint)
    old.write(FROZEN,dict(
        development=list(DEVELOP), comparisons=[t for t,_ in EVAL] + ["ks_highlands"],
        selected_vision=mode, models=list(MODES), trials=trials,
        horizon_contract="fixed forward meters in ego frame", horizons_m=HORIZONS.tolist(),
        checkpoint_sha256=old.digest(checkpoint), sources=digests(sources),
        seed=None, source_revision=old.revision(), termination="frozen_before_archive_score"))


if __name__ == "__main__":
    main()
