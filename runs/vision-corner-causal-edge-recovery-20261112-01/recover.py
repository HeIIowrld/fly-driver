"""Recover missing road sections from earlier, motion-aligned observations."""
import json
from collections import deque
from pathlib import Path

import numpy as np

from baseline import ARCHIVES, CAPTURE, OUT as SILVER_BASE, SILVER, fit_boundaries, publish, reference, summarize_approaches
from corner_fusion_boundaries import advance, read_motion, section_values, stations, transform
from evaluate_fused_corner import frame_key, load_score
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from heads import SILVER_RESULT, load_silver, review
from profile_replay import TRACKS, TRACK_ROOT, archive_path, sources
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
CACHE = Path("runs/vision-corner-boundary-hinge-20261111-01")
MODES = ("current", "carry_confirmed", "carry_supported")
HISTORY_S = .45
EDGE_AGREEMENT_M = 1.0
ANCHOR_SPACING_M = 4.


def as_points(sections):
    valid = np.isfinite(sections).all(axis=1)
    x = np.repeat(stations[valid,None],2,axis=1)
    return np.stack((x,sections[valid]),axis=-1)


def recover(current, history, mode):
    result = current.copy()
    known = np.isfinite(current).all(axis=1)
    if not history or mode=="current":
        return result
    for i in np.flatnonzero(~known):
        observations = [past[i] for past in history if np.isfinite(past[i]).all()]
        if not observations:
            continue
        close = np.flatnonzero(known & (np.abs(stations-stations[i])<=ANCHOR_SPACING_M))
        agreeing = [v for v in observations if any(np.max(np.abs(v-other))<=EDGE_AGREEMENT_M
                   for other in observations if other is not v)]
        if agreeing:
            chosen = np.median(agreeing,axis=0)
        elif mode=="carry_supported" and len(close):
            chosen=observations[0]
        else:
            continue
        if len(close):
            nearest=close[np.argmin(np.abs(stations[close]-stations[i]))]
            delta=stations[i]-stations[nearest]
            if np.max(np.abs(chosen-current[nearest])) > .5+.35*abs(delta):
                continue
        elif mode=="carry_supported" and not agreeing:
            continue
        if 3. < chosen[1]-chosen[0] < 20.:
            result[i]=chosen
    return result


def baseline_data(track, hashes):
    if track==SILVER:
        group=load_silver(hashes)
        data=SILVER_BASE/"multiview-sections.npz"
        receipt=SILVER_BASE/"geometry-result.json"
        bundles={track:read(CAPTURE/"result.json")["archive"]["bundle_sha256"]}
        mapping=ARCHIVES
    else:
        final,base,protocol=sources(track)
        result=final/"result.json"
        base_result=base/"result.json"
        full,raw=read(result),read(base_result)
        assert full["base_receipt_sha256"]==digest(base_result)
        mapping=archive_path(track)
        assert raw["archives_sha256"]==digest(mapping)
        comparison=Path(read(protocol)["base_protocol"])
        group=dict(rows=load_score(final/"score",full["score_receipt_sha256"]),
                   original=load_score(base/"parent/score",raw["score_receipt_sha256"]["parent"]),
                   rules=read(comparison)["comparison"])
        data=CACHE/f"{track}-sections.npz"
        receipt=data.with_suffix(".json")
        bundles=full["camera_bundle_sha256_by_run"]
        hashes.update((result,base_result,comparison,protocol))
    assert digest(data)==read(receipt)["sections_sha256"]
    with np.load(data,allow_pickle=False) as f:
        keys=[(track,int(frame)) for frame in f["frame"]] if track==SILVER else list(zip(f["run"],f["frame"],strict=True))
        assert keys==[frame_key(r) for r in group["rows"]]
        assert np.array_equal(f["stations"],stations)
        edges=f["values"] if track==SILVER else f["values"][:,0]
    hashes.update((data,receipt,mapping))
    return group,dict(zip(keys,edges,strict=True)),mapping,bundles


def run_track(track, hashes):
    group,sections,mapping,bundles=baseline_data(track,hashes)
    row_by_key={frame_key(r):r for r in group["rows"]}
    curve_rows={m:{} for m in MODES}
    details=[]
    ref_edges=None
    if track==SILVER:
        geometry=TRACK_ROOT/track/"ai/fast_lane.ai"
        ref_edges,_=load_edges(geometry)
        hashes.add(geometry)
    for run,folder in read(mapping).items():
        history=deque()
        previous=None
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[run]
            for capture in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key=(run,capture["capture_id"])
                if key not in row_by_key:
                    previous=None
                    history.clear()
                    continue
                row=row_by_key[key]
                path=Path(capture["file"]).name
                assert archive.by_name[path]["sha256"]==capture["file_sha256"]
                meta=archive.json(path)
                assert meta["track_id"]==track and meta["index"]==key[1]
                motion=read_motion(meta)
                generation=capture["generation"]
                contiguous=(previous is not None and motion is not None
                            and key[1]==previous[0]+1 and generation==previous[1])
                step=transform(motion,previous[2]) if contiguous else None
                if step is None:
                    history.clear()
                else:
                    rotation,move=step
                    history=deque((stamp,advance(points,rotation,move)) for stamp,points in history
                                  if motion[0]-stamp <= HISTORY_S)
                current=sections[key]
                past=[section_values(points) for _,points in reversed(history)]
                ref=None
                if ref_edges is not None and row["truth"]=="approach":
                    latest=max(meta["views"],key=lambda v:v["render_call_finished_clock"])
                    ref=reference(ref_edges,latest.get("source_body_after_render"))
                frame=dict(run=run,frame=key[1],track=track,truth=row["truth"],label=row["label"],
                           physical_corner=row["physical_corner"],truth_distance_m=row["truth_distance_m"],methods={})
                current_valid=np.isfinite(current).all(axis=1)
                for mode in MODES:
                    restored=recover(current,past,mode)
                    valid=np.isfinite(restored).all(axis=1)
                    assert np.array_equal(restored[current_valid],current[current_valid])
                    available,proposal=fit_boundaries(restored,.002)["hinge"]
                    curve_rows[mode][key]=publish(row,proposal,available)
                    new=valid & ~current_valid
                    data=dict(paired_sections_m=stations[valid].tolist(), restored_m=stations[new].tolist(),
                              observable=available,proposal=proposal)
                    if ref is not None:
                        shared=new&np.isfinite(ref).all(axis=1)
                        data["added_edge_abs_error_m"]=np.abs(restored-ref)[shared].ravel().tolist()
                    frame["methods"][mode]=data
                if row["truth"]=="approach" or any(m["restored_m"] for m in frame["methods"].values()):
                    details.append(frame)
                if motion is not None:
                    points=as_points(current)
                    if len(points):
                        history.append((motion[0],points))
                    previous=(key[1],generation,motion)
                else:
                    previous=None
    outputs={}
    for mode,index in curve_rows.items():
        assert set(index)==set(row_by_key)
        rows=[index[frame_key(row)] for row in group["rows"]]
        outputs[mode]=dict(score=review(group,rows),approaches=summarize_approaches(rows))
    return outputs,details


def checks():
    current=np.full((len(stations),2),np.nan)
    previous=current.copy()
    current[3]=[-2.,3.]
    previous[4]=[-2.2,2.8]
    assert np.isnan(recover(current,[previous],"carry_confirmed")[4]).all()
    assert np.array_equal(recover(current,[previous],"carry_supported")[4],previous[4])
    assert np.array_equal(recover(current,[previous,previous.copy()],"carry_confirmed")[4],previous[4])
    assert np.array_equal(recover(current,[],"carry_supported"),current,equal_nan=True)


def main():
    checks()
    hashes={Path(__file__),Path(fit_boundaries.__code__.co_filename),Path(transform.__code__.co_filename),
            Path("scripts/score_corner_captures.py")}
    results={}
    for track in (*TRACKS,SILVER):
        scores,details=run_track(track,hashes)
        path=OUT/f"{track}-sections.jsonl"
        jsonl(path,details)
        results[track]=dict(methods=scores,rows_sha256=digest(path))
        print(track,{k:v["score"]["metrics"]["candidate"] for k,v in scores.items()},flush=True)
    write(OUT/"result.json",dict(status="evaluated",seed=None,history_s=HISTORY_S,tracks=results,
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
