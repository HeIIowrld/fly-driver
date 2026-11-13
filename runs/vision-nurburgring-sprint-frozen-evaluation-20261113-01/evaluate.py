"""Replay the frozen corner setting on the new Sprint archive, then score it."""
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
CAPTURE = RUNS / "vision-nurburgring-sprint-capture-20261113-01"
FROZEN = RUNS / "vision-corner-phase-veto-frontier-20261113-01/frozen.json"
BASE = RUNS / "vision-corner-left-risk-reallocation-20261113-01"
FILTER = RUNS / "vision-corner-added-left-phase-veto-20261113-01"
sys.path[:0] = [str(BASE),str(FILTER)]
from replay_independent import CACHE, check_extraction, validate_layout_receipt, validate_transfer, replay_archive
from veto_added_left import (
    METHOD, RISK, apply_policy, available, counts, dependencies, describe,
    digests, keys, load_sensor, old, score,
)
from evaluate_live_corner_geometry import common_rows, load_track_truth

TRACK = "ks_nurburgring"
LAYOUT = "layout_sprint_a"


def main():
    policy = old.read(FROZEN)
    for source, checksum in policy["sources"].items():
        if old.digest(Path(source)) != checksum:
            raise ValueError("Frozen policy dependency changed")
    if TRACK in policy["selection_recordings"]:
        raise ValueError("Sprint is already used in selection")
    base_file = BASE / "frozen.json"
    base_policy = old.read(base_file)
    if old.digest(base_file) != policy["base_policy_sha256"]:
        raise ValueError("Base policy checksum differs")
    for source, checksum in base_policy["source_sha256"].items():
        if old.digest(Path(source)) != checksum:
            raise ValueError("Original base dependency changed")
    observer,settings,structured,_ = dependencies()
    if base_policy["settings"] != settings:
        raise ValueError("Observer settings differ")
    spline = validate_transfer(TRACK,LAYOUT)
    archives = CAPTURE/"archive-map.json"
    capture = CAPTURE/"result.json"
    validate_layout_receipt(archives,capture,TRACK,LAYOUT)
    capture_proof = old.read(capture)
    if capture_proof["inputs"].get(str(FROZEN)) != old.digest(FROZEN):
        raise ValueError("Capture did not record this frozen policy")
    check_extraction(base_policy)
    replay = OUT/"archive-replay"
    receipt = replay/"result.json"
    if not receipt.exists():
        replay_archive(archives,CACHE,replay,TRACK,None,"local-image")
    proof = old.read(receipt)
    check_extraction(base_policy,proof)
    raw,restored = (replay/name for name in ("sensors.npz","recovered-sensors.npz"))
    original,offers = describe(load_sensor(raw,proof["inputs_sha256"]),
                               load_sensor(restored,proof["recovered_input_sha256"]),
                               observer,settings,structured)
    setting = policy["chosen"]
    filtered = available(original,offers,setting["predicate"],setting["threshold"])
    output,_ = apply_policy(original,filtered,base_policy,None)
    if any(a!=b for a,b in zip(original,output,strict=True)
           if a["forecasts"][METHOD]["state"]=="right"):
        raise ValueError("An original right forecast changed")
    predictions = OUT/"forecasts.jsonl"
    old.jsonl(predictions,output)
    labeler,_,_ = load_track_truth(spline)
    labels = []
    for tag,directory in old.read(archives).items():
        with old.CameraArchive(directory) as archive:
            if archive.bundle_sha256 != proof["archive_bundles"][tag]:
                raise ValueError("Original archive changed")
            labels.extend(common_rows(tag,archive,{},TRACK,labeler))
    if keys(labels) != keys(output) or len(output) != capture_proof["frames"]:
        raise ValueError("Forecast and label coverage differ from capture")
    label_path = OUT/"labels.jsonl"
    old.jsonl(label_path,labels)
    before,after = (score(rows,labels) for rows in (original,output))
    delta = counts(before["counts"],after["counts"],(*RISK,"hit_1","hit_2"))
    pass_caps = all(delta[k] <= cap for k,cap in policy["deployment_risk_caps"].items())
    coverage = all(after["approaches"].get(side,{}).get("frames",0) for side in ("1","2"))
    old.write(OUT/"result.json",dict(track=TRACK,layout=LAYOUT,before=before,after=after,delta=delta,
        both_directions_evaluated=coverage,per_recording_risk_pass=pass_caps,
        sources=digests((FROZEN,base_file,Path(__file__),receipt,capture,archives,spline)),
        forecasts_sha256=old.digest(predictions),labels_sha256=old.digest(label_path),
        source_revision=old.revision(),seed=None,termination="archive_end"))
    print(TRACK,after["approaches"],delta,pass_caps,flush=True)


if __name__ == "__main__":
    main()
