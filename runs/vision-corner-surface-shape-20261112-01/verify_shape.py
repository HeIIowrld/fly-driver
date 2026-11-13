"""Verify frozen data separation and compare three-station shape errors."""
from hashlib import file_digest
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = OUT.parent/"vision-corner-surface-supervised-20261112-01"
SILVER = "ks_silverstone1967"


def digest(path):
    with path.open("rb") as source:
        return file_digest(source,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def verify_run(folder, reference):
    train_path, result_path = folder/"training.json", folder/"result.json"
    training, result = read(train_path), read(result_path)
    assert training["excluded_circuits"] == [SILVER]
    assert SILVER not in training["training_circuits"]
    assert training["training_circuits"] == result["training_circuits"]
    assert digest(folder/"shape.pt") == training["checkpoint_sha256"]
    assert training["best_epoch"] == min(training["history"],key=lambda r:r["score"])["epoch"]
    for name,sha in training["source_sha256"].items():
        assert SILVER not in name and digest(ROOT/name) == sha
    for name,sha in result["source_sha256"].items():
        assert digest(ROOT/name) == sha
    frames, path = folder/"silverstone-frames.jsonl", folder/"cross-sections.npz"
    assert digest(frames) == result["rows_sha256"] and digest(path) == result["sections_sha256"]
    records = rows(frames)
    keys = [(r["run"],r["frame"]) for r in records]
    assert len(keys)==len(set(keys))==1200
    for row,base in zip(records,reference,strict=True):
        for name in ("run","frame","truth","label","truth_endpoints_m"):
            assert row[name] == base[name]
        for m in ("edge","t4t5"):
            assert row["decisions"][f"{m}_far_wide"] == base["decisions"][f"{m}_far_wide"]
    with np.load(path,allow_pickle=False) as data:
        assert keys == list(zip(data["run"],data["frame"],strict=True))
        boundaries = data["shape_geometry"]
    return training, result, records, boundaries


def three_valid(mask):
    return mask[:,:-2] & mask[:,1:-1] & mask[:,2:]


def curve_metrics(reference, witness, baseline, shape, frames, station):
    subset = three_valid(witness & np.isfinite(reference) & np.isfinite(baseline) & np.isfinite(shape))
    subset &= frames[:,None,None] & (station[1:-1]>=16.)[None,:,None]
    expected = np.diff(reference,n=2,axis=1)/4.
    output = {}
    for side,name in enumerate(("right","left")):
        take = subset[...,side]
        errors = [abs(np.diff(v,n=2,axis=1)/4.-expected)[...,side][take] for v in (baseline,shape)]
        output[name] = dict(points=int(take.sum()), point_only_mean=float(errors[0].mean()) if take.any() else None,
                            learned_shape_mean=float(errors[1].mean()) if take.any() else None)
    return output


def main():
    old_result = read(PREVIOUS/"result.json")
    path, old_frames = PREVIOUS/"observed-cross-sections.npz", PREVIOUS/"silverstone-frames.jsonl"
    assert digest(path) == old_result["sections_sha256"] and digest(old_frames) == old_result["rows_sha256"]
    old_rows = rows(old_frames)
    with np.load(path,allow_pickle=False) as data:
        baseline = data["direct_replace"]
        stations = data["stations"]
        keys = list(zip(data["run"],data["frame"],strict=True))
    labels = OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{SILVER}-train-labels.npz"
    assert digest(labels) == read(labels.with_suffix(".json"))["labels_sha256"]
    with np.load(labels,allow_pickle=False) as data:
        assert keys == list(zip(data["run"],data["frame"],strict=True))
        np.testing.assert_array_equal(data["stations"],stations)
        expected, witness = data["lateral_m"], data["adjacent_invalid"]
    reports = {}
    for name,folder in (("shape",OUT),("entry_supervised",OUT/"entry-supervised")):
        training, result, recorded, shape = verify_run(folder,old_rows)
        subset = np.array([r["truth"]=="approach" and r["label"]=="2" for r in recorded])
        assert subset.sum()==13
        reports[name] = dict(
            checkpoint_epoch=training["best_epoch"], validation=training["history"][training["best_epoch"]-1]["validation"],
            curvature_difference_mae_per_m=curve_metrics(expected,witness,baseline,shape,subset,stations),
            right_approaches=[r for r,include in zip(recorded,subset,strict=True) if include],
            training_sha256=digest(folder/"training.json"), result_sha256=digest(folder/"result.json"))
    data = dict(status="verified",seed=training["seed"],source_revision=result["source_revision"],runs=reports,
                labels_sha256=digest(labels),point_baseline_sha256=digest(path), verifier_sha256=digest(Path(__file__)),
                termination="both_frozen_checkpoints_and_1200_controls_per_replay_verified")
    (OUT/"verification.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v["curvature_difference_mae_per_m"] for k,v in reports.items()},indent=2))


if __name__ == "__main__":
    main()
