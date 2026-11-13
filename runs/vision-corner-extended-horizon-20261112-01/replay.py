"""Replay longer image-supported road horizons without changing the warning models."""

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from corner_fusion_boundaries import decide, read_motion, stations
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from relative_corner_boundaries import SurfaceContrast
from replay_learned import publish, signals
from score_corner_captures import MIN_SPEED_MPS
from summarize_paired import category
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
PAST = OUT.parent/"vision-corner-complementary-history-20261112-01"
MOUNT = Path("configs/ac-csp-exterior-front-v3-retina.json")
LOCK = OUT.parent/"vision-silverstone1967-independent-protocol-20261111-01/base.json"
LONG = np.arange(2.,35.,2.)
MODES = ("old","near_local","near_wide","far_wide","near_hinge","far_hinge")


def import_replay():
    spec = importlib.util.spec_from_file_location("complementary_replay",PAST/"replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    return replay


def local(x, edges, radius):
    valid = np.isfinite(edges).all(axis=1)
    width = edges[:,1]-edges[:,0]
    valid &= (width>3.) & (width<20.)
    samples=[]
    for center in np.arange(4.,22.):
        offset=x-center
        use=valid & (abs(offset)<=radius)
        z=offset[use]
        if len(z)<4 or z[0]>=0 or z[-1]<=0 or np.max(np.diff(z))>4.:
            continue
        design=np.column_stack((np.ones(len(z)),z,z*z))
        co=np.linalg.lstsq(design,edges[use],rcond=None)[0]
        k=2.*co[2]/(1.+co[1]**2)**1.5
        samples.append((float(center),float(np.copysign(min(abs(k)),k[0])) if k[0]*k[1]>0 else 0.))
    offer=decide(samples,.002)
    return offer if offer and offer[0] and 4.<=offer[1]<=20. else None


def shared_hinge(x, edges):
    """Select the onset of two supported edges with a common signed bend."""
    valid=np.isfinite(edges).all(axis=1)
    width=edges[:,1]-edges[:,0]
    valid &= (width>3.) & (width<20.)
    x,y=x[valid],edges[valid]
    if len(x)<6 or np.max(np.diff(x))>4.:
        return None
    base=np.column_stack((np.ones(len(x)),x))
    residual=np.eye(len(x))-base@np.linalg.pinv(base)
    y=residual@y

    def fit(z):
        z=residual@z
        norm=z@z
        if norm<=1e-9:
            return float(np.sqrt(np.mean(y**2))),0.
        c=(z@y).mean()/norm
        return float(np.sqrt(np.mean((y-z[:,None]*c)**2))),float(c)

    null=fit(x*x)[0]
    best=None
    for onset in np.arange(4.,21.):
        before=x<=onset
        if before.sum()<2 or (~before).sum()<3:
            continue
        if onset-x[before][-1]>2. or x[~before][0]-onset>2.:
            continue
        error,bend=fit(np.maximum(x-onset,0.)**2)
        if abs(bend)<.001 or error>.75 or error>.8*null:
            continue
        rank=error,1 if bend>0 else 2,float(onset)
        if best is None or rank<best:
            best=rank
    return best[1:] if best else None


def proposals(edges):
    short=edges[:len(stations)]
    return dict(near_local=local(stations,short,6.),near_wide=local(stations,short,10.),
                far_wide=local(LONG,edges,10.),near_hinge=shared_hinge(stations,short),
                far_hinge=shared_hinge(LONG,edges))


def map_reference(hashes):
    from audit_road_edge_projection import section_points
    pins=read(LOCK)["hashes"]
    source=next(Path(p) for p in pins if Path(p).name=="fast_lane.ai")
    assert digest(source)==pins[str(source.resolve())]
    hashes.update((source,LOCK,Path(section_points.__code__.co_filename)))
    edges,_=load_edges(source)

    def positions(meta):
        view=max(meta["views"],key=lambda v:v["render_call_finished_clock"])
        body=view["source_body_after_render"]
        xyz=section_points(edges,body,LONG)[:,::-1]
        p=np.asarray(body["renderer_only_position"])
        look=np.asarray(body["renderer_only_look"])[[2,0]]
        normal=np.array([-look[1],look[0]])/np.linalg.norm(look)
        return (xyz-p)[:,:,[2,0]]@normal
    return positions


def run(track, replay, hashes):
    rows,old,mapping,bundles,profiles,training=replay.setup(track,hashes)
    expected={replay.key(r):r for r in rows}
    old={replay.key(r):r for r in old}
    parent=PAST/f"{track}-frames.jsonl"
    assert digest(parent)==read(PAST/"result.json")["tracks"][track]["rows_sha256"]
    hashes.add(parent)
    old_current={replay.key(r):r["proposals"].get("current") for r in map(json.loads,parent.open())}
    mount=read(MOUNT)
    near,far=(SurfaceContrast(mount,near=k) for k in (True,False))
    far.stations=LONG[3:]
    far.lookup=far._sampling(mount,far.stations)
    ref=map_reference(hashes) if track=="ks_silverstone1967" else None
    timelines={f"{m}_{g}":defaultdict(list) for m in replay.MODELS for g in MODES}
    records,matched,oracle=[],Counter(),{}
    for tag,folder in read(mapping).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[tag]
            for capture in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key=tag,capture["capture_id"]
                if key not in expected:
                    continue
                truth,reference=expected[key],old[key]
                filename=Path(capture["file"]).name
                assert archive.by_name[filename]["sha256"]==capture["file_sha256"]
                meta=archive.json(filename)
                assert (meta["track_id"],meta["index"],meta["generation"])==(track,key[1],capture["generation"])
                views={v["view_id"]:v for v in capture["views"]}
                usable=set(views)=={"front","left","right","rear"} and not any(
                    x.get(f) for x in (capture,meta) for f in ("ideal_line_shown","aid_probe_only"))
                motion=read_motion(meta) if usable else None
                moving=motion is not None and motion[1]>=MIN_SPEED_MPS
                offers={}
                support=0
                if moving:
                    edges=replay.read_edges(archive,views,near,far)
                    offers=proposals(edges)
                    support=int(np.isfinite(edges[len(stations):]).all(axis=1).sum())
                prior=offers.get("near_local")
                assert (list(prior) if prior else None)==old_current[key],(key,prior,old_current[key])
                if ref and truth["truth"]=="approach" and truth["label"]=="2":
                    ideal=ref(meta)
                    oracle[str(key[1])]=dict(reference=proposals(ideal),observed=offers,
                        truth_m=truth["truth_endpoints_m"],far_pairs=support)
                base={m:reference["proposals"][spec[1]] for m,spec in replay.MODELS.items()}
                decisions={}
                for model in replay.MODELS:
                    for mode in MODES:
                        offer=offers.get(mode)
                        score=signals(profiles[model].get(key),*offer)["local"] if offer else None
                        final=base[model] or (offer if score is not None and score>=.25 else None)
                        name=f"{model}_{mode}"
                        timelines[name][tag].append(publish(truth,final,reference["baseline_observable"] or final is not None))
                        decisions[name]=final
                if not any(base.values()):
                    for mode in MODES[1:]:
                        edge,t4t5=(decisions[f"{m}_{mode}"] for m in replay.MODELS)
                        if edge!=t4t5:
                            hit=edge or t4t5
                            matched[mode,"t4t5" if t4t5 else "edge",category(truth,hit)]+=1
                records.append(dict(run=tag,frame=key[1],truth=truth["truth"],label=truth["label"],
                     truth_endpoints_m=truth["truth_endpoints_m"],far_pairs=support,offers=offers,final=decisions))
    assert [replay.key(r) for r in records]==list(expected)
    path=OUT/f"{track}-frames.jsonl"
    jsonl(path,records)
    return dict(metrics={name:replay.totals(r) for name,r in timelines.items()},training=training,
         rows_sha256=digest(path),oracle=oracle,
         paired=[dict(mode=k[0],confirmed_by=k[1],category=k[2],frames=v) for k,v in sorted(matched.items())])


def main():
    cv2.setNumThreads(1)
    replay=import_replay()
    hashes={Path(__file__),MOUNT,PAST/"replay.py",PAST/"result.json",
            Path(SurfaceContrast.detect.__code__.co_filename),Path(signals.__code__.co_filename)}
    results={}
    for track in (*replay.CHECK,*replay.profile.CALIBRATION_TRACKS):
        results[track]=run(track,replay,hashes)
        print(track,{k:v for k,v in results[track]["metrics"].items() if k.startswith("t4t5")},flush=True)
        write(OUT/"progress.json",dict(complete=list(results)))
    write(OUT/"result.json",dict(status="evaluated",seed=None,posthoc=True,tracks=results,
        source_sha256={str(p):digest(p) for p in sorted(hashes)},termination="all_timelines_scored"))


if __name__=="__main__":
    main()
