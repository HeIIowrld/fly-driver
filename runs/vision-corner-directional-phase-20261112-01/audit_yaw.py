"""Check the local yaw sign against frame-to-frame orientation on development recordings."""
from pathlib import Path
import json
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PHASE = OUT.parent/"vision-corner-phase-confirmation-20261112-01"
sys.path.insert(0,str(PHASE))
from phase_features import DEVELOP, old
from collect import metadata, admitted
from replay_learned import baseline_data


def signed_step(a,b):
    return float(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b))


def run(track,sources):
    _,_,mapping,bundles = baseline_data(track,sources)
    records = []
    for tag,folder in old.read(mapping).items():
        prev = None
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                meta = metadata(archive,item,track)
                views = admitted(item,meta)
                if not views:
                    prev = None
                    continue
                witness = max(meta["views"],key=lambda v:v["render_call_finished_clock"])
                body = witness.get("source_body_after_render")
                motion = old.read_motion(meta)
                if motion is None or motion[1]<old.MIN_SPEED_MPS or not body:
                    prev = None
                    continue
                look = np.asarray(body.get("renderer_only_look"),dtype=float)
                if look.shape!=(3,) or not np.isfinite(look).all():
                    prev = None
                    continue
                x = look[[2,0]]
                key = (item["capture_id"],item["generation"])
                if prev is not None and key==(prev[0][0]+1,prev[0][1]):
                    time = motion[0]-prev[1][0]
                    if .045<time<=.18:
                        measured = signed_step(prev[2],x)/time
                        signal = .5*(motion[2]+prev[1][2])
                        if min(abs(measured),abs(signal))>=.05:
                            records.append((signal,measured))
                prev = key,motion,x
    a,b = np.asarray(records).T
    return dict(frames=len(a),positive_left_agreement=float(np.mean(a*b>0)),
                pearson=float(np.corrcoef(a,b)[0,1]))


def main():
    sources = {Path(__file__),Path(metadata.__code__.co_filename),Path(old.read_motion.__code__.co_filename)}
    tracks = {t:run(t,sources) for t in DEVELOP}
    assert all(v["frames"]>=20 and v["positive_left_agreement"]>=.85 for v in tracks.values()),tracks
    old.write(OUT/"yaw-sign.json",dict(status="development_sign_verified",seed=None,source_revision=old.revision(),
                 convention="positive local yaw is vehicle-left; recorded world look is used only for this audit",
                 tracks=tracks,source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                 termination="development_frame_to_frame_sign_verified"))
    print(tracks,flush=True)


if __name__=="__main__":
    main()
