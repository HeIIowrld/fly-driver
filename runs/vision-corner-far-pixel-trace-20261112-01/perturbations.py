"""Require a far candidate to survive a two-pixel projection perturbation."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from corner_fusion_boundaries import read_motion
from flybrain_drive.ac_camera_archive import CameraArchive
from image_normal import edge_scores
from projection_counterfactual import selections, lookup_with_offset
from relative_corner_boundaries import SurfaceContrast
from replay_learned import publish, signals
from score_corner_captures import MIN_SPEED_MPS
from summarize_paired import category
from trace import MOUNT, OLD, OUT, STATIONS, context
from train_vision_teacher_line import digest, jsonl, read, write


def agrees(candidate, alternatives):
    return candidate is not None and all(p and p[0]==candidate[0] and abs(p[1]-candidate[1])<=3.
                                         for p in alternatives)


def run(track, entry, replay, hashes):
    rows,old,mapping,bundles,profiles,training=replay.setup(track,hashes)
    prior=OLD/f"{track}-frames.jsonl"
    receipt=OLD/"result.json"
    assert digest(prior)==read(receipt)["tracks"][track]["rows_sha256"]
    hashes.update((prior,receipt))
    observed={replay.key(r):r for r in map(json.loads,prior.open())}
    expected={replay.key(r):r for r in rows}
    old={replay.key(r):r for r in old}
    mount=read(MOUNT)
    near,far=(SurfaceContrast(mount,near=k) for k in (True,False))
    far.stations=STATIONS[3:]
    far.lookup=far._sampling(mount,far.stations)
    original=far.lookup["front"]
    modes=("base","fixed","robust")
    timelines={f"{m}_{mode}":defaultdict(list) for m in replay.MODELS for mode in modes}
    details,removed=[],Counter()
    for tag,folder in read(mapping).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key=tag,item["capture_id"]
                if key not in expected:
                    continue
                t,ref=expected[key],old[key]
                base={m:ref["proposals"][spec[1]] for m,spec in replay.MODELS.items()}
                fixed=observed[key]["offers"].get("far_wide")
                stable=False
                variants=[]
                if fixed and not all(base.values()):
                    filename=Path(item["file"]).name
                    assert archive.by_name[filename]["sha256"]==item["file_sha256"]
                    meta=archive.json(filename)
                    assert (meta["track_id"],meta["index"],meta["generation"])==(track,key[1],item["generation"])
                    motion=read_motion(meta)
                    assert motion is not None and motion[1]>=MIN_SPEED_MPS
                    images={v["view_id"]:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),
                                                     cv2.IMREAD_COLOR) for v in item["views"]}
                    assert set(images)=={"front","left","right","rear"} and not any(
                        rec.get(flag) for rec in (item,meta) for flag in ("ideal_line_shown","aid_probe_only"))
                    labs={n:cv2.cvtColor(im,cv2.COLOR_BGR2LAB) for n,im in images.items()}
                    raw,scores=near.detect(images),edge_scores(near,labs)
                    for offset in (-2.,2.):
                        lookup_with_offset(far,original,0.,offset/(images["front"].shape[0]-1))
                        edges=selections(near,far,images,labs,raw,scores)
                        variants.append(entry.local(STATIONS,edges,10.))
                    stable=agrees(fixed,variants)
                decisions={}
                for m in replay.MODELS:
                    values=profiles[m].get(key)
                    score=signals(values,*fixed)["local"] if fixed and base[m] is None else None
                    accepted=fixed if score is not None and score>=.25 else None
                    outcomes=(base[m],base[m] or accepted,base[m] or (accepted if stable else None))
                    for mode,offer in zip(modes,outcomes,strict=True):
                        label=f"{m}_{mode}"
                        decisions[label]=offer
                        timelines[label][tag].append(publish(t,offer,ref["baseline_observable"] or offer is not None))
                    if accepted and not stable:
                        removed[m,category(t,accepted)]+=1
                    assert decisions[f"{m}_fixed"]==observed[key]["final"][f"{m}_far_wide"]
                details.append(dict(run=tag,frame=key[1],truth=t["truth"],label=t["label"],fixed=fixed,
                                    perturbed=variants,stable=stable,decisions=decisions))
    assert [replay.key(r) for r in details]==list(expected)
    path=OUT/f"{track}-perturbations.jsonl"
    jsonl(path,details)
    return dict(training=training,rows_sha256=digest(path),
                metrics={k:replay.totals(v) for k,v in timelines.items()},
                removed=[dict(model=k[0],category=k[1],frames=v) for k,v in sorted(removed.items())])


def main():
    cv2.setNumThreads(1)
    entry=context()
    replay=entry.import_replay()
    hashes={Path(__file__),MOUNT,Path(selections.__code__.co_filename),
            Path(entry.__file__),Path(replay.__file__)}
    results={}
    for track in replay.CHECK:
        results[track]=run(track,entry,replay,hashes)
        print(track,results[track],flush=True)
    write(OUT/"perturbation-result.json",dict(status="evaluated",seed=None,posthoc=True,tracks=results,
         shifts_px=[-2.,2.],distance_tolerance_m=3.,
         source_sha256={str(p):digest(p) for p in sorted(hashes)},termination="two_full_timelines_scored"))


if __name__=="__main__":
    main()
