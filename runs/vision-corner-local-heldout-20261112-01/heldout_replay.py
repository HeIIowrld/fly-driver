"""Cross-fit the complete local-curvature warning pipeline by circuit."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import cv2
import joblib

from paired_motion import inputs
import profile_replay as profile
from replay_learned import learned, training_set
import replay_local as local
from summarize_paired import category, count

OUT = Path(__file__).parent
TRACKS = tuple(profile.CALIBRATION_TRACKS)
MODES = ("boundaries", "boundaries_temporal_t4t5")
CONTROL = "ranked_local_boundary_hold"
CANDIDATE = "ranked_local_t4t5_hold"


def train_group(track):
    hashes=set()
    data=inputs(track,hashes)
    examples=training_set(track,hashes)
    print("loaded",track,flush=True)
    return examples,data,{str(p):local.digest(p) for p in sorted(hashes)}


def compare_rows(path, baseline, candidate):
    transition=Counter()
    important=[]
    for row in map(json.loads,path.open()):
        a,b=(row["proposals"][mode] for mode in (baseline,candidate))
        if a==b:
            continue
        x,y=category(row,a),category(row,b)
        transition[x,y]+=1
        if row["truth"]=="approach":
            important.append(dict(run=row["run"],frame=row["frame"],before=x,after=y,
                                  base_proposal=a,new_proposal=b,truth_m=row["truth_endpoints_m"]))
    return dict(transitions=[dict(before=a,after=b,frames=n) for (a,b),n in sorted(transition.items())],
                approach_changes=important)


def run_fold(track, data, mount):
    peers=tuple(t for t in TRACKS if t!=track)
    assert len(peers)==4 and track not in peers
    patch=learned.train([data[t][0] for t in peers])
    models={mode:profile.fit([data[t][1][mode] for t in peers],1) for mode in MODES}
    checkpoint=OUT/f"{track}-fold-models.joblib"
    joblib.dump(dict(patch=patch,**models),checkpoint)
    profiles=[]
    for mode in MODES:
        g=data[track][1][mode]
        profiles.append(dict(zip((local.frame_key(r) for r in g["selected"]),
                                 profile.predict(g,models[mode]),strict=True)))
    report=local.replay(track,mount,patch,profiles)
    counts={name:count(runs) for name,runs in report["methods"].items()}
    assert all(x["frames"]==len(data[track][1]["boundaries"]["rows"]) for x in counts.values())
    path=OUT/f"{track}-frames.jsonl"
    assert local.digest(path)==report["rows_sha256"]
    transitions=compare_rows(path,CONTROL,CANDIDATE)
    summary=dict(training=peers,held_out=track,fold_models_sha256=local.digest(checkpoint),
                 counts=counts,replay=report,paired=transitions)
    local.write(OUT/f"{track}-result.json",summary)
    print("scored",track,{m:counts[m] for m in ("baseline",CONTROL,CANDIDATE)},flush=True)
    return summary


def main():
    cv2.setNumThreads(1)
    local.checks()
    assert set(TRACKS).isdisjoint(local.TEST_TRACKS)
    local.OUT=OUT
    mount_path=Path(local.load_frozen(local.PROTOCOL.with_name("candidate.json"))["camera_config"])
    mount=local.read(mount_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        data=dict(zip(TRACKS,pool.map(train_group,TRACKS),strict=True))
        futures={pool.submit(run_fold,t,data,mount):t for t in TRACKS}
        reports={}
        for future in as_completed(futures):
            track=futures[future]
            reports[track]=future.result()
    artifacts={t:local.digest(OUT/f"{t}-result.json") for t in TRACKS}
    source={Path(__file__),mount_path,Path(local.__file__),Path(training_set.__code__.co_filename),
            Path(learned.__file__),Path(inputs.__code__.co_filename),Path(profile.__file__),
            Path(category.__code__.co_filename)}
    totals={mode:{metric:sum(r["counts"][mode][metric] for r in reports.values())
                  for metric in reports[TRACKS[0]]["counts"][mode]}
            for mode in reports[TRACKS[0]]["counts"]}
    local.write(OUT/"result.json",dict(status="evaluated",seed=learned.SEED,
         source_revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",posthoc=True,independent_capture=False,
         termination="all_five_circuits_scored",confirmation_cut=local.CONFIRM_THRESHOLD,
         curvature_cut=local.CUT,patch_cut=learned.PATCH_THRESHOLD,
         model_training_receipts={t:data[t][2] for t in TRACKS},track_receipts_sha256=artifacts,
         source_sha256={str(p):local.digest(p) for p in sorted(source)},totals=totals))


if __name__=="__main__":
    main()
