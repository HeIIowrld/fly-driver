"""Apply the two frozen confirmation models to the Vallelunga Extended recording."""
from pathlib import Path
import json

import cv2
import numpy as np

from evaluate_gate import OUT, MODES, evaluate, comparable_budget, joblib, old
from collect import SHAPE, admitted, crop_track, image, inference, metadata, combined
from features import assemble
from fit_gate import scores
from infer_frozen_next_corner import load_features
from pair_audit import motion_features

TRACK = "ks_vallelunga"
CAPTURE = OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"
PREVIOUS = OUT.parent/"vision-vallelunga-local-corner-evaluation-20261112-05"
PROTOCOL = OUT.parent/"vision-vallelunga-local-corner-protocol-20261112-01"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def features(directory, truth, bundles, hashes):
    index = directory/"index.jsonl"
    keys = [(r["run"],r["frame"]) for r in rows(index)]
    assert keys==[(r["run"],r["frame"]) for r in truth]
    batches = []
    extractor = Path("configs/flight-stream-batched-v1-20261029.json")
    hashes.update((index,extractor,Path(load_features.__code__.co_filename),
                   Path(motion_features.__code__.co_filename)))
    for branch in ("original","reflected"):
        source = directory/branch/"features.npz"
        x,available,proof = load_features(source,old.digest(index),len(keys))
        assert {run:r["bundle_sha256"] for run,r in proof["reports"].items()}==bundles
        assert proof["author_checkpoint_sha256"]==old.read(extractor)["author_weights_sha256"]
        assert proof["camera_config_sha256"]==old.digest(old.MOUNT)
        assert proof["batch_size"]==8
        hashes.update((source,source.with_name("flyvis-result.json")))
        batches.append((x,available))
    (a,av),(b,bv) = batches
    mask = av & bv
    a,b = motion_features(a[mask]),motion_features(b[mask])
    shared,odd = .5*(a+b),.5*(a-b)
    motion = np.stack((np.c_[shared,odd],np.c_[shared,-odd]),axis=1)
    return dict(rows=truth,selected=[r for r,v in zip(truth,mask,strict=True) if v],x=motion)


def collect_offers(keys, available, estimates, mapping, bundles):
    entry = old.context()
    mount = old.read(old.MOUNT)
    near,far = (old.SurfaceContrast(mount,near=value) for value in (True,False))
    far.stations = old.STATIONS[3:]
    far.lookup = far._sampling(mount,far.stations)
    expected = {key:i for i,key in enumerate(keys)}
    seen = set()
    boundaries = np.full((len(keys),len(old.STATIONS),2),np.nan)
    offers = [None]*len(keys)
    for tag,folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key = tag,item["capture_id"]
                if key not in expected:
                    continue
                assert key not in seen
                seen.add(key)
                i = expected[key]
                meta = metadata(archive,item,TRACK)
                views = admitted(item,meta)
                assert bool(views)==available[i]
                motion = old.read_motion(meta) if views else None
                if motion is None or motion[1]<old.MIN_SPEED_MPS:
                    continue
                images = {name:image(archive,view) for name,view in views.items()}
                lab = {name:cv2.cvtColor(data,cv2.COLOR_BGR2LAB) for name,data in images.items()}
                base = old.selections(near,far,images,lab,near.detect(images),old.edge_scores(near,lab))
                boundaries[i] = combined(base,estimates[i])["shape_geometry"]
                offers[i] = entry.local(old.STATIONS,boundaries[i],10.)
    assert seen==set(keys)
    return offers,boundaries


def main():
    cv2.setNumThreads(1)
    freeze, confirmation = OUT/"training.json", OUT/"confirmation.joblib"
    trained = old.read(freeze)
    assert old.digest(confirmation)==trained["checkpoint_sha256"]
    assert TRACK not in trained["development"]
    shape_freeze, shape = SHAPE/"training.json",SHAPE/"shape.pt"
    shape_training = old.read(shape_freeze)
    assert TRACK not in shape_training["training_circuits"] and old.digest(shape)==shape_training["checkpoint_sha256"]
    admission, truth_path = PREVIOUS/"admission/result.json", PREVIOUS/"admission/rows.jsonl"
    mapping, capture = CAPTURE/"archive-map.json",CAPTURE/"result.json"
    older, earlier_rows, lock = PREVIOUS/"result.json",PREVIOUS/f"{TRACK}-frames.jsonl",PROTOCOL/"lock.json"
    proof, previous, captured = (old.read(p) for p in (admission,older,capture))
    assert proof["ready_for_scoring"] and previous["layout"]=="extended_circuit"
    assert old.digest(admission)==previous["admission_sha256"] and old.digest(lock)==previous["lock_sha256"]
    assert old.digest(truth_path)==proof["rows_sha256"] and old.digest(mapping)==proof["archives_sha256"]
    assert old.digest(earlier_rows)==previous["replay"]["rows_sha256"]
    truth, prior = rows(truth_path),rows(earlier_rows)
    assert [(r["run"],r["frame"]) for r in truth]==[(r["run"],r["frame"]) for r in prior]
    bundles = {TRACK:captured["archive"]["bundle_sha256"]}
    hashes = {Path(__file__),freeze,confirmation,shape_freeze,shape,admission,truth_path,mapping,capture,
              older,earlier_rows,lock,old.MOUNT,Path(inference.__code__.co_filename),
              Path(crop_track.__code__.co_filename),Path(assemble.__code__.co_filename),
              Path(evaluate.__code__.co_filename),Path(old.context().__file__)}
    cache = crop_track(TRACK,dict(rows=truth),mapping,bundles)
    hashes.add(cache)
    mount = old.read(old.MOUNT)
    keys, available, estimates, _ = inference(cache,shape,mount["views"]["front"])
    offers, sections = collect_offers(keys,available,estimates,mapping,bundles)
    base = [r["proposals"]["ranked_local_boundary_hold"] for r in prior]
    raw = features(PREVIOUS/"features",truth,bundles,hashes)
    g = assemble(raw,keys,offers,sections,base)
    models = joblib.load(confirmation)
    prediction = {m:scores(g,models[m]) for m in MODES}
    metrics,detail,records,paired = evaluate(g,prediction,trained["threshold_by_model"],sections)
    assert len(records)==1200
    path = OUT/"vallelunga-frames.jsonl"
    old.jsonl(path,records)
    cap = min(metrics[f"{m}_candidate_only"]["outside"] for m in MODES)
    report = dict(status="second_excluded_circuit_replay",track=TRACK,layout=previous["layout"],
                  seed=trained["seed"],source_revision=old.revision(),frames=len(records),
                  thresholds=trained["threshold_by_model"],paired_available=len(g["index"]),
                  raw_candidates=sum(o is not None for o in offers),metrics=metrics,detailed=detail,paired=paired,
                  equal_outside_diagnostic=dict(outside_cap=cap,result=comparable_budget(g,prediction,cap)),
                  rows_sha256=old.digest(path),source_sha256={str(p):old.digest(p) for p in sorted(hashes)},
                  termination="frozen_models_replayed_on_vallelunga")
    old.write(OUT/"vallelunga-result.json",report)
    print(metrics,report["equal_outside_diagnostic"],flush=True)


if __name__ == "__main__":
    main()
