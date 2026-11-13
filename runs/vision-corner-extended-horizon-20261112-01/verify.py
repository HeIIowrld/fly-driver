
"""Verify full archived replays and quantify the longer horizon's pixel spacing."""

from collections import Counter
from pathlib import Path
import subprocess

import numpy as np

from replay import LONG, MOUNT, OUT, proposals
from relative_corner_boundaries import SurfaceContrast
from train_vision_teacher_line import digest, read, write
from evaluate_fused_corner import read_rows


def main():
    receipt=OUT/"result.json"
    result=read(receipt)
    for name,sha in result["source_sha256"].items():
        assert digest(Path(name))==sha,name
    dev,coverage={},{}
    for track,proof in result["tracks"].items():
        path=OUT/f"{track}-frames.jsonl"
        rows=read_rows(path)
        assert digest(path)==proof["rows_sha256"]
        assert len({(r["run"],r["frame"]) for r in rows})==len(rows)
        assert all(r["frames"]==len(rows) and r["no_output"]==0 for r in proof["metrics"].values())
        rights=[r for r in rows if r["truth"]=="approach" and r["label"]=="2"]
        coverage[track]=dict(right_frames=len(rights),with_far_pairs=sum(r["far_pairs"]>0 for r in rights),
            far_pairs=sum(r["far_pairs"] for r in rights))
        if track in ("ks_silverstone1967","spa"):
            continue
        assert track not in proof["training"]
        for mode,values in proof["metrics"].items():
            dev.setdefault(mode,Counter()).update(values)
    assert dev["t4t5_near_local"]["frames"]==6087
    previous=read(OUT.parent/"vision-corner-complementary-history-20261112-01/verification.json")["development"]
    for mode in ("edge","t4t5"):
        assert dev[mode+"_near_local"]==previous[mode+"_current"]
    mount=read(MOUNT)
    camera=SurfaceContrast(mount)
    u,v,ok=camera._sampling(mount,LONG)["front"]
    column=int(np.argmin(abs(camera.lateral-4.)))
    projection=[dict(station_m=float(x),px=[float(u[i,column]*319),float(v[i,column]*179)],visible=bool(ok[i,column]))
                for i,x in enumerate(LONG) if x>=24.]
    y=-.01*np.maximum(LONG-19.,0.)**2
    synthetic=proposals(np.c_[y-4.,y+4.])
    assert synthetic["far_hinge"]==(2,19.)
    safe=Path.cwd().resolve().as_posix()
    revision=subprocess.check_output(["git","-c",f"safe.directory={safe}","rev-parse","HEAD"],text=True).strip()
    write(OUT/"verification.json",dict(status="verified",seed=None,source_revision=revision,
       result_sha256=digest(receipt),development=dev,far_observation_coverage=coverage,
       projection_flat_road_lateral_4m=projection,synthetic_onset_19m=synthetic,
       termination="full_timelines_and_near_control_verified"))
    print(dict(development=dev,coverage=coverage,synthetic=synthetic),flush=True)


if __name__=="__main__":
    main()
