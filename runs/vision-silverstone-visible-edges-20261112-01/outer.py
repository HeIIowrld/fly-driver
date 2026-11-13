"""Test a geometrically supported onset when only the outer road edge is observable."""
import json
from pathlib import Path

import cv2
import numpy as np

from baseline import PROTOCOL,SILVER,fit_boundaries,publish,summarize_approaches
from corner_fusion_boundaries import read_motion,stations,transform
from evaluate_fused_corner import frame_key,load_frozen
from flybrain_drive.ac_camera_archive import CameraArchive
from heads import review
from hinge import squared_error
from recover import baseline_data
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
MODES=("paired","outer_edge","outer_edge_temporal")
FIELDS=("right_y_m","left_y_m")
GAIN=.12


def fit_single(y):
    known=np.isfinite(y)
    x,y=stations[known],y[known]
    if len(x)<7 or x[0]>8. or np.max(np.diff(x))>4.:
        return None
    null=min(squared_error(x,y)[0],squared_error(x,y,x*x)[0])
    best=None
    for onset in range(4,19):
        before=x<=onset
        if before.sum()<3 or (~before).sum()<3:
            continue
        if onset-x[before][-1]>2. or x[~before][0]-onset>2.:
            continue
        error,coeff=squared_error(x,y,np.maximum(0.,x-onset)**2)
        k=2.*coeff[2]/(1.+coeff[1]**2)**1.5
        gain=(np.sqrt(null)-np.sqrt(error))/(.2+np.sqrt(error))
        if abs(k)<.003 or gain<GAIN or np.sqrt(error)>.50:
            continue
        candidate=float(gain),1 if k>0 else 2,float(onset)
        if best is None or candidate[0]>best[0]:
            best=candidate
    return best


def proposals(raw):
    y=np.array([[np.nan if r[f] is None else r[f] for f in FIELDS] for r in raw])
    pairs=np.array([r["valid"] for r in raw],dtype=bool)
    observed,base=fit_boundaries(np.where(pairs[:,None],y,np.nan),.002)["hinge"]
    outer=[]
    for side in (0,1):
        fit=fit_single(y[:,side])
        if fit is not None and fit[1]==side+1:
            outer.append(fit)
    # When two independent outer-edge hypotheses disagree, leave the paired decision in place.
    fallback=outer[0][1:] if len(outer)==1 and not observed else None
    return observed,base,fallback


def replay(track,mount,hashes):
    group,_,mapping,bundles=baseline_data(track,hashes)
    near,far=SurfaceContrast(mount,near=True),SurfaceContrast(mount)
    source={frame_key(r):r for r in group["rows"]}
    all_outputs={name:{} for name in MODES}
    details=[]
    for run,path in read(mapping).items():
        previous=None
        with CameraArchive(path) as archive:
            assert archive.bundle_sha256==bundles[run]
            for capture in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key=run,capture["capture_id"]
                if key not in source:
                    previous=None
                    continue
                row=source[key]
                name=Path(capture["file"]).name
                assert archive.by_name[name]["sha256"]==capture["file_sha256"]
                meta=archive.json(name)
                assert meta["track_id"]==track and meta["index"]==key[1] and meta["generation"]==capture["generation"]
                views={v["view_id"]:v for v in capture["views"]}
                usable=set(views)=={"front","rear","left","right"} and not any(
                   obj.get(flag) for obj in (capture,meta) for flag in ("ideal_line_shown","aid_probe_only"))
                if usable:
                    images={side:cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name),np.uint8),
                                              cv2.IMREAD_COLOR) for side,view in views.items()}
                    assert all(im is not None for im in images.values())
                    raw=near.detect(images)+far.detect({"front":images["front"]})[1:]
                    observed,base,outer=proposals(raw)
                else:
                    observed,base,outer=False,None,None
                motion=read_motion(meta) if usable else None
                temporal=None
                if (outer and motion and previous and previous[3] and previous[2]
                        and key[1]==previous[0]+1 and capture["generation"]==previous[1]):
                    step=transform(motion,previous[2])
                    prior=previous[3]
                    if (step is not None and prior[0]==outer[0] and
                        abs(prior[1]-float(step[1][0])-outer[1])<=4.):
                        temporal=outer
                previous=(key[1],capture["generation"],motion,outer)
                variants=(base,base or outer,base or temporal)
                for mode,forecast in zip(MODES,variants,strict=True):
                    all_outputs[mode][key]=publish(row,forecast,observed or forecast is not None)
                if row["truth"]=="approach" or outer:
                    details.append(dict(run=run,frame=key[1],truth=row["truth"],label=row["label"],
                        physical_corner=row["physical_corner"],base=base,outer=outer,temporal=temporal,
                        states={mode:all_outputs[mode][key]["state"] for mode in MODES}))
    result={}
    for mode,lookup in all_outputs.items():
        assert set(lookup)==set(source)
        rows=[lookup[frame_key(r)] for r in group["rows"]]
        result[mode]=dict(score=review(group,rows),approach=summarize_approaches(rows))
    path=OUT/f"{track}-outer.jsonl"
    jsonl(path,details)
    return dict(methods=result,rows_sha256=digest(path))


def main():
    cv2.setNumThreads(1)
    mount=Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    hashes={Path(__file__),mount,Path(SurfaceContrast.detect.__code__.co_filename),
            Path(fit_boundaries.__code__.co_filename),Path(squared_error.__code__.co_filename),
            Path(baseline_data.__code__.co_filename),Path(transform.__code__.co_filename)}
    results={}
    for track in (SILVER,"spa"):
        results[track]=replay(track,read(mount),hashes)
        print(track,{k:v["score"]["metrics"]["candidate"] for k,v in results[track]["methods"].items()},flush=True)
    write(OUT/"outer-result.json",dict(status="evaluated",seed=None,tracks=results,
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
