"""Identify whether the retained patches or the fixed onset rule rejects a warning."""

import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from corner_fusion_boundaries import curvature, decide, stations
from flybrain_drive.ac_camera_archive import CameraArchive
from hinge import fit
from replay_learned import FIELDS, OUT, PROTOCOL, SILVER, PatchGrid, SurfaceContrast, choose, edge_scores
from replay_learned import baseline_data, digest, learned, load_frozen, probabilities, ranked_patches, read, write

TARGETS = {SILVER: {620, 621, 622, 623}, "spa": {325}}


def run():
    cv2.setNumThreads(1)
    proof = OUT/"frozen.json"
    checkpoint = OUT/"patch-model.joblib"
    assert digest(checkpoint) == read(proof)["checkpoint_sha256"]
    hashes = {Path(__file__), proof, checkpoint, Path(fit.__code__.co_filename),
              Path(decide.__code__.co_filename), Path(ranked_patches.__code__.co_filename)}
    model = joblib.load(checkpoint)
    config = Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    mount = read(config)
    hashes.add(config)
    near, far, detector = SurfaceContrast(mount, near=True), SurfaceContrast(mount), PatchGrid(mount)
    result = {}
    for track, frames in TARGETS.items():
        group, _, mapping, bundles = baseline_data(track, hashes)
        for run, path in read(mapping).items():
            truth = {r["frame"]: r for r in group["rows"] if r["run"] == run and r["frame"] in frames}
            with CameraArchive(path) as archive:
                assert archive.bundle_sha256 == bundles[run]
                for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                    frame = item["capture_id"]
                    if frame not in truth:
                        continue
                    name = Path(item["file"]).name
                    assert archive.by_name[name]["sha256"] == item["file_sha256"]
                    meta = archive.json(name)
                    assert (meta["track_id"], meta["index"], meta["generation"]) == (track, frame, item["generation"])
                    views = {v["view_id"]: v for v in item["views"]}
                    assert set(views) == {"front", "rear", "left", "right"} and not any(
                        r.get(k) for r in (item, meta) for k in ("ideal_line_shown", "aid_probe_only"))
                    images = {k: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                              cv2.IMREAD_COLOR) for k, v in views.items()}
                    raw = near.detect(images) + far.detect({"front": images["front"]})[1:]
                    lab = {k: cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for k, im in images.items()}
                    contrast = np.r_[edge_scores(near, lab), edge_scores(far, lab)[1:]]
                    p = probabilities(model, detector, lab)
                    variants = dict(normal=choose(raw, contrast, near.lateral, "normal_fill"),
                                    learned=learned.sections(p), ranked=ranked_patches(raw, contrast, near.lateral, p))
                    record = {}
                    for mode, edge in variants.items():
                        observed, proposal = fit(stations, edge)
                        profile = curvature(edge)
                        record[mode] = dict(observed=observed, unthresholded_hinge=proposal,
                                            local_curvature=decide(profile, .002))
                    result[f"{run}/{frame}"] = dict(truth=truth[frame]["truth_endpoints_m"], variants=record)
    assert len(result) == sum(map(len, TARGETS.values()))
    write(OUT/"fit-diagnostic.json", dict(status="evaluated", rows=result,
          source_sha256={str(p): digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    run()
