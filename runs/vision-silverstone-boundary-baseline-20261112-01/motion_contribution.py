"""Measure the contribution of T4/T5 time features to the same supervised profile."""
import json
from pathlib import Path

import numpy as np

from baseline import ARCHIVES, OUT, PROTOCOL, SILVER, load_silver, publish, summarize_approaches
from evaluate_fused_corner import frame_key
from flybrain_drive.ac_camera_archive import CameraArchive
from heads import review
import profile_replay as profile
from train_vision_teacher_line import digest, read, write

BOUNDARY = Path("runs/vision-corner-boundary-hinge-20261111-01")
MODES = ("boundaries", "t4t5", "boundaries_latest_t4t5", "boundaries_temporal_t4t5")
CUTOFF = .50


def silver_labels(group, hashes):
    geometry = profile.TRACK_ROOT/SILVER/"ai/fast_lane.ai"
    assert digest(geometry)==read(PROTOCOL)["hashes"][str(geometry.resolve())]
    label = profile.track_reference(geometry)
    indexed = {}
    with CameraArchive(read(ARCHIVES)[SILVER]) as archive:
        receipt = read(ARCHIVES.with_name("result.json"))
        assert archive.bundle_sha256==receipt["archive"]["bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            path=Path(item["file"]).name
            assert archive.by_name[path]["sha256"]==item["file_sha256"]
            meta=archive.json(path)
            assert meta["track_id"]==SILVER and meta["index"]==item["capture_id"]
            view=max(meta["views"],key=lambda v:v["render_call_finished_clock"])
            result=label(view.get("source_body_after_render"))
            if result is not None:
                indexed[(SILVER,item["capture_id"])]=result
    hashes.update((ARCHIVES,ARCHIVES.with_name("result.json"),geometry))
    missing=np.full((2,len(profile.OFFSETS)),np.nan)
    return np.array([indexed.get(frame_key(r),missing) for r in group["selected"]])


def boundary_features(track, group, hashes):
    is_silver=track==SILVER
    path=OUT/"multiview-sections.npz" if is_silver else BOUNDARY/f"{track}-sections.npz"
    receipt=OUT/"geometry-result.json" if is_silver else path.with_suffix(".json")
    assert digest(path)==read(receipt)["sections_sha256"]
    hashes.update((path,receipt))
    with np.load(path,allow_pickle=False) as data:
        keys=[(track,int(f)) for f in data["frame"]] if is_silver else list(zip(data["run"],data["frame"],strict=True))
        assert keys==[frame_key(r) for r in group["rows"]]
        assert np.array_equal(data["stations"],np.arange(2.,25.,2.))
        boundaries=data["values"] if is_silver else data["values"][:,0]
    values=boundaries[group["valid"]]
    branches=(values,-values[:,:,::-1])
    return np.stack([np.concatenate((np.nan_to_num(v,nan=0.).reshape(len(v),-1),
                                     np.isfinite(v).reshape(len(v),-1)),axis=1) for v in branches],axis=1)


def run():
    hashes={Path(__file__),Path(profile.__file__),Path(publish.__code__.co_filename),
            Path("scripts/score_corner_captures.py")}
    groups={}
    for track in (*profile.TRACKS,SILVER):
        g=load_silver(hashes) if track==SILVER else profile.load(track,hashes)
        g["labels"]=silver_labels(g,hashes) if track==SILVER else profile.labels(track,g,hashes)
        edge=boundary_features(track,g,hashes)
        motion=g["x"]
        half=motion.shape[-1]//2
        step=half//3
        assert step*3==half
        latest=np.r_[0:step,half:half+step]
        groups[track]=(g,dict(zip(MODES,(edge,motion,np.concatenate((edge,motion[:,:,latest]),axis=2),
                                     np.concatenate((edge,motion),axis=2)),strict=True)))
        print("Aligned boundary and motion",track,flush=True)
    results={}
    for track,(g,features) in groups.items():
        training=tuple(name for name in profile.CALIBRATION_TRACKS if name!=track)
        result={}
        for mode,x in features.items():
            source=[dict(groups[t][0],x=groups[t][1][mode]) for t in training]
            group=dict(g,x=x)
            predicted=profile.predict(group,profile.fit(source,1))
            values=iter(predicted)
            rows=[]
            for row,valid in zip(g["rows"],g["valid"],strict=True):
                proposal=profile.onset(next(values),CUTOFF,True) if valid else None
                rows.append(publish(row,proposal,valid,"features_unavailable"))
            result[mode]=dict(score=review(group,rows),approach=summarize_approaches(rows),
                              profile=profile.diagnostics(group,predicted,1))
        results[track]=dict(training=training,methods=result)
        print(track,{mode:data["score"]["metrics"]["candidate"] for mode,data in result.items()},flush=True)
    write(OUT/"motion-contribution.json",dict(status="evaluated",diagnostic_cutoff=CUTOFF,seed=None,
          tracks=results,source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    run()
