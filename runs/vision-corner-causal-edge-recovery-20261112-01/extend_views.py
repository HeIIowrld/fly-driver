"""Use the outward side camera when the front view loses one far road edge."""
import json
from pathlib import Path

import cv2
import numpy as np

from baseline import PROTOCOL, SILVER, fit_boundaries, publish, summarize_approaches
from corner_fusion_boundaries import stations
from evaluate_fused_corner import frame_key,load_frozen
from flybrain_drive.ac_camera_archive import CameraArchive
from heads import review
from recover import baseline_data
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
SIDES=("right","left")
FIELDS=("right_y_m","left_y_m")
MODES=("current","side_fill","side_fill_supported")


def complete(raw, extras, mode):
    values=[]
    for i,(station, row) in enumerate(zip(stations,raw,strict=True)):
        pair=[row[key] for key in FIELDS]
        if mode!="current" and station>=10.:
            for side in range(2):
                if pair[side] is not None:
                    continue
                alternative=extras[side][i-4][FIELDS[side]]
                if alternative is None:
                    continue
                if mode=="side_fill_supported":
                    neighbors=[raw[j][FIELDS[side]] for j in (i-1,i+1)
                               if 0<=j<len(raw) and raw[j][FIELDS[side]] is not None]
                    if not any(abs(alternative-neighbor)<=1.25 for neighbor in neighbors):
                        continue
                pair[side]=alternative
        if None not in pair and 3.<pair[1]-pair[0]<20.:
            values.append(pair)
        else:
            values.append([np.nan,np.nan])
    return np.asarray(values)


def run_track(track, hashes, mount):
    group, original, mapping, bundles = baseline_data(track,hashes)
    near=SurfaceContrast(mount,near=True)
    far=SurfaceContrast(mount)
    side_detectors=[]
    for side in SIDES:
        eye=SurfaceContrast(mount)
        eye.lookup={"front":eye.lookup[side]}
        side_detectors.append(eye)
    by_key={frame_key(row):row for row in group["rows"]}
    scores={mode:{} for mode in MODES}
    details=[]
    for run,folder in read(mapping).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[run]
            for capture in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key=(run,capture["capture_id"])
                if key not in by_key:
                    continue
                row=by_key[key]
                path=Path(capture["file"]).name
                assert archive.by_name[path]["sha256"]==capture["file_sha256"]
                meta=archive.json(path)
                assert meta["track_id"]==track and meta["index"]==key[1] and meta["generation"]==capture["generation"]
                views={v["view_id"]:v for v in capture["views"]}
                if set(views)=={"front","left","right","rear"} and not any(
                        source.get(k) for source in (capture,meta) for k in ("ideal_line_shown","aid_probe_only")):
                    images={side:cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name),np.uint8),
                                             cv2.IMREAD_COLOR) for side,view in views.items()}
                    assert all(image is not None for image in images.values())
                    raw=near.detect(images)+far.detect({"front":images["front"]})[1:]
                    extras=[eye.detect({"front":images[side]})[1:] for side,eye in zip(SIDES,side_detectors,strict=True)]
                    edges={mode:complete(raw,extras,mode) for mode in MODES}
                else:
                    edges={mode:np.full((len(stations),2),np.nan) for mode in MODES}
                if track==SILVER:
                    assert np.array_equal(original[key],edges["current"],equal_nan=True)
                base=np.isfinite(edges["current"]).all(axis=1)
                detail=dict(run=run,frame=key[1],truth=row["truth"],label=row["label"],
                            physical_corner=row["physical_corner"],methods={})
                for mode,array in edges.items():
                    current=np.isfinite(array).all(axis=1)
                    assert np.array_equal(array[base],edges["current"][base])
                    available,proposal=fit_boundaries(array,.002)["hinge"]
                    scored=publish(row,proposal,available)
                    scores[mode][key]=scored
                    detail["methods"][mode]=dict(added=stations[current&~base].tolist(),
                        observable=available,state=scored["state"],distance_m=scored["predicted_distance_m"],
                        direction_hit=scored["direction_hit"],joint_hit=scored["joint_hit"])
                if row["truth"]=="approach" or any(v["added"] for v in detail["methods"].values()):
                    details.append(detail)
    results={}
    for mode,index in scores.items():
        assert set(index)==set(by_key)
        rows=[index[frame_key(r)] for r in group["rows"]]
        results[mode]=dict(score=review(group,rows),approach=summarize_approaches(rows))
    path=OUT/f"{track}-side-fill.jsonl"
    jsonl(path,details)
    return dict(methods=results,rows_sha256=digest(path))


def main():
    cv2.setNumThreads(1)
    candidate=load_frozen(PROTOCOL.with_name("candidate.json"))
    mount=Path(candidate["camera_config"])
    hashes={Path(__file__),mount,Path(fit_boundaries.__code__.co_filename),
            Path(SurfaceContrast.detect.__code__.co_filename)}
    outputs={}
    for track in (SILVER,"spa"):
        outputs[track]=run_track(track,hashes,read(mount))
        print(track,{k:r["score"]["metrics"]["candidate"] for k,r in outputs[track]["methods"].items()},flush=True)
    write(OUT/"side-fill-result.json",dict(status="evaluated",seed=None,tracks=outputs,
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
