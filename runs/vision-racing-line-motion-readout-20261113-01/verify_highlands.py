"""Verify the sealed Highlands results, right-output invariance, and all replay denominators."""
from collections import Counter
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
RUNS=OUT.parent
sys.path.insert(0,str(RUNS/"vision-corner-left-risk-reallocation-20261113-01"))
from replay_independent import load_fixed_policy, validate_transfer
from fixed_left import METHOD, digests, keys, old, score
from evaluate_readout import load_fixed

CAPTURE=RUNS/"vision-highlands-short-fixed-corner-capture-20261113-06"
CORNER=RUNS/"vision-highlands-short-fixed-corner-evaluation-20261113-01"
LINE=RUNS/"vision-highlands-short-fixed-line-evaluation-20261113-01"
TRACK="ks_highlands"
LAYOUT="layout_short"


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def main():
    frozen, policy, *_=load_fixed_policy()
    line_policy,_=load_fixed()
    spline=validate_transfer(TRACK,LAYOUT)
    capture=old.read(CAPTURE/"result.json")
    admission=old.read(CAPTURE/"admission-summary.json")
    corner=old.read(CORNER/"result.json")
    line=old.read(LINE/"evaluation.json")
    line_export=old.read(LINE/"result.json")
    archive=old.read(CORNER/"archive-replay/result.json")
    assert capture["archive"]["bundle_sha256"]==admission["bundle_sha256"]==archive["archive_bundles"][TRACK]
    assert archive["archive_bundles"]==line_export["camera_bundle_sha256_by_run"]
    assert TRACK not in set(line_policy["development"]+policy["development"]+policy["exclusions"])
    assert corner["source_sha256"][str(frozen)]==old.digest(frozen)
    assert line_export["policy_sha256"]==old.digest(OUT/"frozen.json")
    assert old.digest(spline)==admission["spline_sha256"]
    assert capture["layout"]==corner["layout"]==line["layout"]==LAYOUT
    assert capture["track"]==corner["track"]==line["track"]==TRACK
    for sources in (corner["source_sha256"],line["sources"]):
        for path, checksum in sources.items():
            assert old.digest(Path(path))==checksum,path
    forecasts=CORNER/"forecasts.jsonl"
    baseline=CORNER/"archive-replay/predictions/forecasts.jsonl"
    truth=CORNER/"labels.jsonl"
    path_file=LINE/"racing-line.jsonl"
    assert old.digest(forecasts)==corner["forecasts_sha256"]
    assert old.digest(truth)==corner["labels_sha256"]
    assert old.digest(path_file)==line["line_sha256"]==line_export["racing_line_sha256"]
    base,revised,labels,paths=map(rows,(baseline,forecasts,truth,path_file))
    assert len(base)==len(revised)==len(labels)==len(paths)==1200==capture["frames"]==line["evaluated_frames"]
    assert keys(base)==keys(revised)==keys(labels)==[(p["episode_id"],p["frame"]) for p in paths]
    assert all(r["camera_complete"] for r in labels)
    assert Counter(row["truth"] for row in labels)==admission["truth"]
    assert Counter(row["label"] for row in labels if row["truth"]=="approach")==admission["approach_by_direction"]
    assert all(a==b for a,b in zip(base,revised,strict=True) if a["forecasts"][METHOD]["state"]=="right")
    before,after=score(base,labels),score(revised,labels)
    assert (before,after)==(corner["before"],corner["after"])
    for stage in (before,after):
        assert sum(stage["counts"].values())==sum(stage["states"].values())==1200
        assert all(s["joint_hit"]+s["miss"]==s["frames"] for s in stage["approaches"].values())
    risk={k:after["counts"].get(k,0)-before["counts"].get(k,0) for k in policy["per_track_caps"]}
    assert risk==corner["risk_delta"]
    assert corner["per_recording_risk_pass"]==all(v<=policy["per_track_caps"][k] for k,v in risk.items())
    assert Counter(p["status"] for p in paths)=={"available":line["line_outputs"],"abstain":line["line_abstentions"]}
    for p in paths:
        assert [x["s_m"] for x in p["points"]]==[5.,10.,20.]
        assert all(x["valid"]==(p["status"]=="available") for x in p["points"])
    old.write(LINE/"verification.json",dict(track=TRACK,layout=LAYOUT,frames=1200,approaches=admission["approach_by_direction"],
        before=before,after=after,risks=risk,risk_pass=corner["per_recording_risk_pass"],
        line_results=line["results"],clearance=line["clearance"],
        source_sha256=digests((Path(__file__),frozen,OUT/"frozen.json",CAPTURE/"result.json",CORNER/"result.json",LINE/"evaluation.json")),
        seed=None,termination="verified_saved_archive"))
    print("Highlands capture, fixed policy, every label and forecast, and line replay verified",flush=True)


if __name__=="__main__":
    main()
