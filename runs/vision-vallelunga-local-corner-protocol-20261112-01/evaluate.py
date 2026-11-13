"""Freeze and verify the Vallelunga Extended independent paired replay."""

import argparse
import configparser
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import joblib
import numpy as np

from audit_corner_capture_admission import review
from evaluate_fused_corner import read_rows
from evaluate_independent_corner import compare
from infer_frozen_next_corner import load_features
from paired_motion import inputs
from pair_audit import motion_features
import profile_replay as profile
import replay_local as replay
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
TRACK, LAYOUT = "ks_vallelunga", "extended_circuit"
INSTALL = profile.TRACK_ROOT/TRACK
PAST = Path("runs/vision-silverstone1967-independent-protocol-20261111-01")
SILVER = replay.SILVER
MODES = ("boundaries", "boundaries_temporal_t4t5")
PRIMARY = "ranked_local_t4t5_hold"
CONTROL = "ranked_local_boundary_hold"
GEOMETRY = INSTALL/LAYOUT/"ai/fast_lane.ai"
MOUNT = Path("configs/ac-csp-exterior-front-v3-retina.json")


def hashed(paths):
    return {str(p): digest(p) for p in sorted(paths)}


def verify(items):
    for path, expected in items.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def freeze():
    lock = OUT/"lock.json"
    if lock.exists():
        raise FileExistsError(lock)
    excluded = ("monza", "magione", "imola", "ks_laguna_seca", "ks_zandvoort",
                "mugello", "spa", SILVER)
    assert TRACK not in excluded
    ini = INSTALL/f"models_{LAYOUT}.ini"
    definition = configparser.ConfigParser(interpolation=None)
    definition.read(ini, encoding="utf-8-sig")
    meshes = {INSTALL/s["FILE"] for s in definition.values() if "FILE" in s}
    assert meshes and all(p.is_file() for p in meshes)
    files = meshes | {ini, GEOMETRY, INSTALL/"ui"/LAYOUT/"ui_track.json"}
    prior = read(PAST/"base.json")
    base = OUT/"base.json"
    write(base, dict(proposed_track=TRACK, layout=LAYOUT, excluded_development_tracks=excluded,
                     data_admission=prior["data_admission"], comparison=prior["comparison"],
                     hashes=hashed(files)))
    candidate = OUT/"candidate.json"
    write(candidate, dict(base_protocol=str(base), camera_config=str(MOUNT), development_rehearsal=False,
                          hashes=hashed({base, MOUNT})))
    route = OUT/"route.json"
    write(route, dict(fusion_protocol=str(candidate), hashes=hashed({candidate})))
    capture = OUT/"capture.json"
    old_capture = read(PAST/"capture.json")
    write(capture, dict(warning_end_protocol=str(route), capture=old_capture["capture"],
                       hashes=hashed({route, *(Path(p) for p in old_capture["hashes"] if not p.endswith("route.json"))})))
    admission = OUT/"admission-review.json"
    checker = Path("scripts/check_corner_stability_coverage.py")
    write(admission, dict(minimum_consecutive_corners_per_side=2, checker_sha256=digest(checker),
          immutable_candidate_sha256=digest(candidate), capture_protocol_sha256=digest(capture)))
    frozen = replay.PREVIOUS/"frozen.json"
    patch_model = replay.PREVIOUS/"patch-model.joblib"
    old = read(frozen)
    verify(old["source_sha256"])
    assert digest(patch_model) == old["checkpoint_sha256"]
    training = tuple(old["training"])
    assert set(training).issubset(excluded)
    hashes = {Path(__file__), base, candidate, route, capture, admission, checker, MOUNT, frozen, patch_model,
              Path(replay.__file__), Path(profile.__file__), Path(inputs.__code__.co_filename),
              Path(motion_features.__code__.co_filename), Path(load_features.__code__.co_filename),
              Path(replay.ranked_patches.__code__.co_filename), Path(replay.signals.__code__.co_filename),
              Path(replay.curvature.__code__.co_filename), Path(replay.summarize.__code__.co_filename),
              Path(replay.SurfaceContrast.detect.__code__.co_filename), Path(replay.edge_scores.__code__.co_filename),
              Path(replay.fit_boundaries.__code__.co_filename), Path(review.__code__.co_filename),
              Path(compare.__code__.co_filename), Path("scripts/evaluate_live_corner_geometry.py"),
              Path("scripts/extract_frozen_corner_archive.py"), Path("scripts/extract_next_corner_flyvis.py"),
              Path("configs/flight-stream-batched-v1-20261029.json"), OUT/"run-evaluation.ps1", OUT/"capture_layout.py"}
    data = [inputs(t, hashes) for t in training]
    models = {mode: profile.fit([g[mode] for g in data], 1) for mode in MODES}
    model_path = OUT/"confirmation-models.joblib"
    joblib.dump(models, model_path)
    hashes.add(model_path)
    write(lock, dict(status="frozen_before_independent_pixels", source_revision=prior["revision"],
          track=TRACK, layout=LAYOUT, training=training, seed=old["seed"], control=CONTROL, candidate=PRIMARY,
          visual_threshold=old["patch_threshold"], curvature_cut=replay.CUT, confirmation_cut=replay.CONFIRM_THRESHOLD,
          one_frame_hold=True, hashes=hashed(hashes), termination="prepared_for_independent_capture"))


def boundaries_for_replay(track, archive_map, rows):
    near, far = replay.SurfaceContrast(read(MOUNT), near=True), replay.SurfaceContrast(read(MOUNT))
    keys, sections = [], []
    for run, folder in read(archive_map).items():
        with replay.CameraArchive(folder) as archive:
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                path = Path(item["file"]).name
                assert archive.by_name[path]["sha256"] == item["file_sha256"]
                meta = archive.json(path)
                frame = item["capture_id"]
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, frame, item["generation"])
                views = {v["view_id"]: v for v in item["views"]}
                usable = set(views) == {"front","rear","left","right"} and not any(
                    x.get(flag) for x in (item, meta) for flag in ("ideal_line_shown","aid_probe_only"))
                values = np.full((len(replay.stations), 2), np.nan)
                if usable:
                    images = {side: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                                 cv2.IMREAD_COLOR) for side, v in views.items()}
                    assert all(image is not None for image in images.values())
                    raw = near.detect(images) + far.detect({"front":images["front"]})[1:]
                    values = np.array([[s[f] if s["valid"] else np.nan for f in replay.FIELDS] for s in raw])
                keys.append((run, frame))
                sections.append(values)
    assert keys == [replay.frame_key(r) for r in rows]
    return np.stack(sections)


def predict(rows, features, boundaries, bundles):
    index = features/"index.jsonl"
    keys = [replay.frame_key(r) for r in read_rows(index)]
    assert keys == [replay.frame_key(r) for r in rows]
    found = [load_features(features/branch/"features.npz", digest(index), len(keys))
             for branch in ("original","reflected")]
    for _, _, proof in found:
        assert {run:r["bundle_sha256"] for run,r in proof["reports"].items()} == bundles
        extractor = read(Path("configs/flight-stream-batched-v1-20261029.json"))
        assert proof["author_checkpoint_sha256"] == extractor["author_weights_sha256"]
        assert proof["camera_config_sha256"] == digest(MOUNT)
        assert proof["batch_size"] == 8
    (a,av,_), (b,bv,_) = found
    valid = av & bv
    if not valid.any():
        return ({}, {}), 0
    a,b = motion_features(a[valid]), motion_features(b[valid])
    shared,odd = .5*(a+b),.5*(a-b)
    motion = np.stack((np.c_[shared,odd],np.c_[shared,-odd]),axis=1)
    values = boundaries[valid]
    edges = np.stack([np.concatenate((np.nan_to_num(v,nan=0.).reshape(len(v),-1),
                                    np.isfinite(v).reshape(len(v),-1)),axis=1)
                      for v in (values,-values[:,:,::-1])],axis=1)
    selected = [key for key, ok in zip(keys,valid,strict=True) if ok]
    models = joblib.load(OUT/"confirmation-models.joblib")
    results = tuple(dict(zip(selected,profile.predict(dict(x=x), models[m]),strict=True))
                    for m,x in zip(MODES,(edges,np.concatenate((edges,motion),axis=2)),strict=True))
    return results, len(selected)


def evaluate(track, archive_map, rows, features, output, cached=None):
    with_bundle = {}
    for run, path in read(archive_map).items():
        with replay.CameraArchive(path) as archive:
            with_bundle[run] = archive.bundle_sha256
    boundaries = boundaries_for_replay(track, archive_map, rows) if cached is None else cached
    scores, feature_count = predict(rows,features,boundaries,with_bundle)
    keys = [replay.frame_key(r) for r in rows]
    by_key = dict(zip(keys,boundaries,strict=True))

    def replay_input(requested, hashes):
        assert requested == track
        hashes.add(archive_map)
        return dict(rows=rows),by_key,archive_map,with_bundle

    mount = read(MOUNT)
    with patch.object(replay,"baseline_data",replay_input), patch.object(replay,"OUT",output):
        result = replay.replay(track,mount,joblib.load(replay.PREVIOUS/"patch-model.joblib"),scores)
    with np.load(output/f"{track}-sections.npz",allow_pickle=False) as cache:
        assert np.array_equal(cache["sections"][:,0],boundaries,equal_nan=True)
    saved = read_rows(output/f"{track}-frames.jsonl")
    timelines = {}
    for name in ("baseline",CONTROL,PRIMARY):
        by_run = {}
        for row, result_row in zip(rows,saved,strict=True):
            key = replay.frame_key(row)
            assert key == replay.frame_key(result_row)
            proposal = result_row["proposals"][name]
            by_run.setdefault(key[0],[]).append(replay.publish(row,proposal,
                result_row["baseline_observable"] or proposal is not None))
        timelines[name] = by_run
    rules = read(OUT/"base.json")["comparison"]
    return dict(feature_frames=feature_count, replay=result,
                versus_boundary=compare(dict(parent=timelines[CONTROL],candidate=timelines[PRIMARY]),rules),
                versus_baseline=compare(dict(parent=timelines["baseline"],candidate=timelines[PRIMARY]),rules))


def verify_layout(race_ini):
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(race_ini,encoding="utf-8-sig")
    if (settings.get("RACE","TRACK",fallback=None),settings.get("RACE","CONFIG_TRACK",fallback=None)) != (TRACK,LAYOUT):
        raise ValueError("Race configuration must name Vallelunga Extended")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","verify","rehearse","evaluate"))
    for name in ("capture-dir","output","cache","race-ini"):
        parser.add_argument("--"+name,type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.mode == "freeze":
        freeze()
        return
    lock = read(OUT/"lock.json")
    verify(lock["hashes"])
    verify(read(OUT/"base.json")["hashes"])
    if args.mode == "verify":
        print("Vallelunga Extended protocol, models and acquisition hashes verified")
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output
    if args.mode == "rehearse":
        previous = Path("runs/vision-silverstone1967-independent-evaluation-20261112-01")
        features = previous/"visual/base/features"
        rows = read_rows(previous/"admission/rows.jsonl")
        mapping = Path("runs/vision-silverstone1967-independent-capture-20261112-05/archive-map.json")
        known = read(replay.OUT/"result.json")["tracks"][SILVER]
        cache = replay.OUT/f"{SILVER}-sections.npz"
        assert digest(cache) == known["sections_sha256"]
        with np.load(cache,allow_pickle=False) as data:
            assert list(zip(data["run"],data["frame"],strict=True)) == [replay.frame_key(r) for r in rows]
            cached = data["sections"][:,0]
        output.mkdir(parents=True,exist_ok=False)
        result = evaluate(SILVER,mapping,rows,features,output,cached)
        assert result["replay"]["methods"] == known["methods"], "Frozen replay parity differs"
        write(output/"result.json",dict(status="rehearsal_verified",track=SILVER,independent=False,**result))
        print("All nine Silverstone timelines reproduced")
        return
    if any(p is None for p in (args.capture_dir,args.cache,args.race_ini)):
        parser.error("Independent evaluation requires --capture-dir, --cache and the captured --race-ini")
    verify_layout(args.race_ini)
    capture = args.capture_dir/"result.json"
    receipt = read(capture)
    if (receipt.get("race_ini_sha256") != digest(args.race_ini)
            or receipt.get("capture_wrapper_sha256") != digest(OUT/"capture_layout.py")):
        raise ValueError("Capture must include the pre/post verified Vallelunga layout receipt")
    output.mkdir(parents=True,exist_ok=False)
    admitted = review(args.capture_dir,OUT/"capture.json",OUT/"admission-review.json",output/"admission")
    if not admitted["ready_for_scoring"]:
        write(output/"result.json",dict(status="not_admitted",track=TRACK,admission=admitted))
        return
    from extract_frozen_corner_archive import run as extract
    mapping = args.capture_dir/"archive-map.json"
    index = output/"admission/rows.jsonl"
    features = output/"features"
    extract(mapping,index,args.cache,features,batch_pair=True,expected_track=TRACK)
    result = evaluate(TRACK,mapping,read_rows(index),features,output)
    write(output/"result.json",dict(status="evaluated",track=TRACK,layout=LAYOUT,independent=True,
          admission_sha256=digest(output/"admission/result.json"), lock_sha256=digest(OUT/"lock.json"),**result))


if __name__ == "__main__":
    main()
