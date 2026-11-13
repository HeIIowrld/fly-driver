"""Audit spatial accuracy and paired decisions after side-camera additions."""
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_side import OUT, RAW_ARCHIVE, ROUTING, MESH, TEST, VAL, METHODS, VARIANTS, old
from aligned_sections import sequence, valid_pairs
from replay_corner_boundary_states import sensor_rows, load_observer
from verify_directional import frames
from alternatives import adjacent


def label_score(previous, current, truth, mask):
    keep = mask & np.isfinite(truth)
    before, after = np.isfinite(previous)&keep, np.isfinite(current)&keep
    both = before & after
    new, lost = after & ~before, before & ~after
    err_old, err_new = abs(previous-truth), abs(current-truth)
    def mae(error, mask):
        return float(error[mask].mean()) if mask.any() else None
    return dict(labelled=int(keep.sum()), before=int(before.sum()), after=int(after.sum()),
                added=int(new.sum()), lost=int(lost.sum()), added_mae_m=mae(err_new,new),
                added_within_1_25_m=int((new&(err_new<=1.25)).sum()),
                common_before_mae_m=mae(err_old,both), common_after_mae_m=mae(err_new,both))


def classify(row, ref, method):
    p = row["forecasts"][method]
    direction = {"left": 1, "right": 2}.get(p["state"])
    return old.category(ref,(direction,p["distance_m"]) if direction is not None else None)


def main():
    track_paths = {TEST:"silverstone",VAL:"vallelunga"}
    report_path = OUT/"side-result.json"
    result = old.read(report_path)
    model, settings, _ = load_observer()
    route = old.read(ROUTING/"straight-verification.json")
    assert old.digest(MESH)==old.read(MESH.with_suffix(".json"))["labels_sha256"]
    output = {}
    for track, report in result["tracks"].items():
        baseline = RAW_ARCHIVE/track_paths[track]/"predictions/boundaries.npz"
        prior = ROUTING/f"{track}-state-distance.jsonl"
        reference = frames(prior,route["tracks"][track]["rows_sha256"])
        for source, sha in report["source_sha256"].items():
            assert old.digest(Path(source))==sha
        with np.load(baseline,allow_pickle=False) as arrays:
            observed = arrays["observed"]
            old_edges = arrays["estimated"]
        collected = {}
        for variant in VARIANTS:
            proof = report["methods"][variant]
            path = OUT/f"{track}-{variant}-sensors.npz"
            assert old.digest(path)==proof["sensor_sha256"]
            with np.load(path,allow_pickle=False) as sensor:
                rows = sensor_rows(sensor,settings)
                assert len(rows)==len(reference)==1200
                assert [(r["run"],r["frame"]) for r in rows]==[(r["run"],r["frame"]) for r in reference]
                raw = sensor["sections"]
                np.testing.assert_array_equal(raw[np.isfinite(observed)],observed[np.isfinite(observed)])
                continuity = [False,*(adjacent(a,b) for a,b in zip(rows,rows[1:]))]
                aligned = sequence(sensor["stations"],continuity,raw,sensor["motion"])
            replay = frames(OUT/f"{track}-{variant}-frames.jsonl",proof["frames_sha256"])
            paired = Counter()
            for item, ref in zip(replay,reference,strict=True):
                assert (item["run"],item["frame"])==(ref["run"],ref["frame"])
                edge, motion = (classify(item,ref,m) for m in METHODS)
                if edge!=motion:
                    paired[(ref["truth"],ref["label"],edge,motion)] += 1
            entry = dict(paired=[dict(truth=k[0], label=k[1], edge=k[2], motion=k[3], frames=v)
                                 for k,v in sorted(paired.items())])
            if track==TEST:
                with np.load(MESH,allow_pickle=False) as mesh:
                    assert [(r["run"],r["frame"]) for r in rows]==list(zip(mesh["run"],mesh["frame"],strict=True))
                    physical = mesh["lateral_m"]
                    mask = mesh["adjacent_invalid"] & np.isfinite(mesh["image_px"]).all(-1).any(axis=1)
                approaches = np.array([r["truth"]=="approach" and r["label"]=="2" for r in reference])
                entry["mesh"] = {name:label_score(old_edges[:,:,side],aligned[:,:,side],physical[:,:,side],
                                                 mask[:,:,side]&frame_mask[:,None])
                                 for name,side,frame_mask in (("all_right",0,np.ones(1200,bool)),
                                     ("all_left",1,np.ones(1200,bool)),("right_approach_right",0,approaches))}
                assert entry["mesh"]["right_approach_right"]["added"]==proof["mesh"]["new"]
            collected[variant]=entry
        output[track]=collected
    old.write(OUT/"side-verification.json",dict(status="verified", result_sha256=old.digest(report_path),
              source_sha256={str(p):old.digest(p) for p in (Path(__file__),MESH)},tracks=output,
              termination="camera_pair_and_physical_boundary_score_verified"))


if __name__=="__main__":
    main()
