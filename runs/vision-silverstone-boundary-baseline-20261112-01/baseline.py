"""Compare archived road boundaries and supervised FlyVis curvature on Silverstone."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from audit_road_edge_projection import section_points
from corner_fusion_boundaries import curvature, decide, stations
from evaluate_fused_corner import frame_key, load_frozen, read_rows
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from heads import SILVER, SILVER_RESULT, PROTOCOL, load_silver, review
from hinge import fit as hinge_fit
import profile_replay as profile
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import score

OUT = Path(__file__).parent
CAPTURE = Path("runs/vision-silverstone1967-independent-capture-20261112-05")
ARCHIVES = CAPTURE/"archive-map.json"
HINGE_CUT = .04


def publish(row, proposal, observable, unavailable="boundary_unobserved"):
    if observable and proposal and proposal[0]:
        return dict(score(row,proposal), abstain_reason=None)
    return dict(row, state="straight" if observable else "abstain",
                predicted_direction=0 if observable else None, predicted_distance_m=None,
                abstain_reason=None if observable else unavailable, direction_hit=False, joint_hit=False,
                signed_distance_error_m=None, absolute_distance_error_m=None, presence_score=None)


def fit_boundaries(edges, cut):
    fit = decide(curvature(edges),cut)
    observable, proposed = hinge_fit(stations,edges)
    hinge = proposed[1:] if proposed and proposed[0] >= HINGE_CUT else None
    return {"curvature":(fit is not None, fit), "hinge":(observable,hinge)}


def read_sections(near, far_sections, images):
    items = near.detect(images) + far_sections
    assert [item["forward_x_m"] for item in items] == stations.tolist()
    return np.array([[item["right_y_m"],item["left_y_m"]] if item["valid"] else [np.nan,np.nan]
                     for item in items])


def reference(edges, body):
    if not body or not body.get("physics_available"):
        return np.full((len(stations),2),np.nan)
    xyz = section_points(edges,body,stations)[:,::-1]
    origin = np.asarray(body["renderer_only_position"])
    look = np.asarray(body["renderer_only_look"])[[2,0]]
    norm = np.linalg.norm(look)
    if not np.isfinite(norm) or norm < .1:
        return np.full((len(stations),2),np.nan)
    normal = np.array([-look[1],look[0]]) / norm
    return (xyz-origin)[:,:,[2,0]] @ normal


def summarize_approaches(rows):
    return {side:dict(frames=len(selected), states=dict(Counter(r["state"] for r in selected)),
                      direction=sum(r["direction_hit"] for r in selected),
                      joint=sum(r["joint_hit"] for r in selected))
            for side in ("1","2")
            for selected in ([r for r in rows if r["truth"]=="approach" and r["label"]==side],)}


def inspect_video(group, hashes):
    candidate_path = PROTOCOL.with_name("candidate.json")
    candidate = load_frozen(candidate_path)
    mount = Path(candidate["camera_config"])
    geometry = profile.TRACK_ROOT/SILVER/"ai/fast_lane.ai"
    assert read(PROTOCOL)["hashes"][str(geometry.resolve())] == digest(geometry)
    pins_path = Path(candidate["boundary_fusion_pins"])
    cut = load_frozen(pins_path)["curvature_threshold_per_m"]
    proof = read(SILVER_RESULT/"visual/result.json")
    assert digest(ARCHIVES)==read(SILVER_RESULT/"visual/base/result.json")["archives_sha256"]
    near, far = SurfaceContrast(read(mount),near=True),SurfaceContrast(read(mount))
    front_near = SurfaceContrast(read(mount),near=True)
    front_near.lookup = {"front":front_near.lookup["front"]}
    edges,_ = load_edges(geometry)
    make_target = profile.track_reference(geometry)
    rows_by_mode = {f"{mode}_{rule}":[] for mode in ("front","multiview") for rule in ("curvature","hinge")}
    profiles = {}
    full_sections = []
    diagnostic = []
    with CameraArchive(read(ARCHIVES)[SILVER]) as archive:
        assert archive.bundle_sha256==proof["camera_bundle_sha256_by_run"][SILVER]
        timeline = map(json.loads,archive.read("stream.jsonl").splitlines())
        for row, capture in zip(group["rows"],timeline,strict=True):
            key = (SILVER,capture["capture_id"])
            assert frame_key(row)==key
            name = Path(capture["file"]).name
            assert archive.by_name[name]["sha256"]==capture["file_sha256"]
            meta = archive.json(name)
            assert meta["track_id"]==SILVER and meta["generation"]==capture["generation"]
            views = {v["view_id"]:v for v in meta["views"]}
            images = {}
            for side,view in views.items():
                raw = archive.read(Path(view["file"]).name)
                images[side] = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if set(images)!={"front","left","right","rear"} or any(v is None for v in images.values()):
                raise ValueError("Admitted camera group cannot be decoded")
            view = max(views.values(),key=lambda v:v["render_call_finished_clock"])
            body = view.get("source_body_after_render")
            profiles[key] = make_target(body)
            detail = {"frame":key[1],"label":row["label"],"variants":{}}
            ref = reference(edges,body) if row["truth"]=="approach" else None
            front = {"front":images["front"]}
            far_sections = far.detect(front)[1:]
            for mode, detector, inputs in (("front",front_near,front),("multiview",near,images)):
                detected = read_sections(detector,far_sections,inputs)
                if mode == "multiview":
                    full_sections.append(detected)
                decisions = fit_boundaries(detected,cut)
                for rule,(available,proposal) in decisions.items():
                    rows_by_mode[f"{mode}_{rule}"].append(publish(row,proposal,available))
                if ref is not None:
                    masked = np.where(np.isfinite(detected),ref,np.nan)
                    shared = np.isfinite(detected)&np.isfinite(ref)
                    detail["variants"][mode] = dict(sections=int(np.isfinite(detected).all(axis=1).sum()),
                        observations=decisions, map_same_sections=fit_boundaries(masked,cut),
                        map_full=fit_boundaries(ref,cut),
                        edge_error_m=np.abs(detected-ref)[shared].tolist())
            if ref is not None:
                diagnostic.append(detail)
    hashes.update((candidate_path,mount,geometry,pins_path,ARCHIVES,CAPTURE/"result.json",
                   Path("scripts/relative_corner_boundaries.py"),Path("scripts/corner_fusion_boundaries.py"),
                   Path("scripts/audit_road_edge_projection.py"),Path(hinge_fit.__code__.co_filename)))
    group["labels"] = np.array([profiles[frame_key(row)] if profiles[frame_key(row)] is not None
                               else np.full((2,len(profile.OFFSETS)),np.nan) for row in group["selected"]])
    np.savez_compressed(OUT/"multiview-sections.npz", values=np.stack(full_sections), stations=stations,
                        frame=np.array([r["frame"] for r in group["rows"]]))
    rows_file = OUT/"geometry-rows.jsonl"
    jsonl(rows_file,[dict(method=name,**row) for name,rows in rows_by_mode.items() for row in rows])
    jsonl(OUT/"approach-reference.jsonl",diagnostic)
    return {mode:dict(score=review(group,rows),approach=summarize_approaches(rows))
            for mode,rows in rows_by_mode.items()}


def train_profiles(group, hashes):
    development = {}
    for track in profile.CALIBRATION_TRACKS:
        g = profile.load(track,hashes)
        g["labels"] = profile.labels(track,g,hashes)
        development[track] = g
        print("Loaded geometry supervision",track,flush=True)
    names = tuple(development)
    results = {}
    for kind,name in enumerate(("curvature","span")):
        model = profile.fit(list(development.values()),kind)
        prediction = profile.predict(group,model)
        nested = {t:profile.predict(g,profile.fit([development[k] for k in names if k!=t],kind))
                  for t,g in development.items()}
        cutoff,shape = profile.select(nested,development)
        diagnostic_cut = .10 if kind==0 else .50
        def standalone(chosen, approach_shape):
            estimates = iter(prediction)
            rows = []
            for row,valid in zip(group["rows"],group["valid"],strict=True):
                proposal = profile.onset(next(estimates),chosen,approach_shape) if valid else None
                rows.append(publish(row,proposal,valid,"flyvis_feature_unavailable"))
            return dict(score=review(group,rows),approach=summarize_approaches(rows))
        results[name] = dict(training=names, nested_cutoff=cutoff, nested_shape=shape,
                diagnostic_cutoff=diagnostic_cut, profile=profile.diagnostics(group,prediction,kind),
                diagnostic=standalone(diagnostic_cut,True),
                selected_append=profile.score_profiles(group,prediction,cutoff,shape))
    return results


def main():
    cv2.setNumThreads(1)
    hashes = {Path(__file__),Path(profile.__file__),Path(load_silver.__code__.co_filename),
              Path("scripts/score_corner_captures.py")}
    group = load_silver(hashes)
    geometry = inspect_video(group,hashes)
    write(OUT/"geometry-result.json",dict(status="evaluated",track=SILVER,seed=None,
          frames=len(group["rows"]),rows_sha256=digest(OUT/"geometry-rows.jsonl"),
          reference_sha256=digest(OUT/"approach-reference.jsonl"),
          sections_sha256=digest(OUT/"multiview-sections.npz"),
          methods=geometry,source_sha256={str(p):digest(p) for p in sorted(hashes)}))
    print("Standalone pixel boundary scores saved",flush=True)
    trained = train_profiles(group,hashes)
    write(OUT/"profile-result.json",dict(status="evaluated",track=SILVER,seed=None,
          methods=trained,source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
