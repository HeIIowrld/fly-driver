"""Evaluate a frozen racing-line readout on the same capture as the corner replay."""
import argparse
from pathlib import Path

import numpy as np

from audit_approaches import TRACK_DIR, borders_from_spline, evaluate_clearance
from evaluate_readout import (
    HORIZONS, STRAIGHT, digests, error_metrics, load_fixed, load_sensor, metadata, old,
    predict_all, reference_poses,
)
from line_readout import sensor_features
from replay_readout import run as replay_line
from teacher_labels import future_targets


def run(archives, track, layout, replay, output):
    receipt = replay / "result.json"
    proof = old.read(receipt)
    if proof["expected_track"] != track:
        raise ValueError("Track differs from the camera replay identity")
    if not track or Path(track).name != track or (layout and Path(layout).name != layout):
        raise ValueError("Expected exact track and optional layout IDs")
    spline = TRACK_DIR / track / layout / "ai/fast_lane.ai"
    if not spline.is_file():
        raise FileNotFoundError(spline)

    replay_line(replay,output)
    export_receipt = output/"result.json"
    export = old.read(export_receipt)
    policy, models = load_fixed()
    source = replay / "sensors.npz"
    sensor = load_sensor(source,proof["inputs_sha256"])
    rows = metadata(sensor)
    poses = reference_poses(archives,track,proof["archive_bundles"],rows)
    targets,_ = future_targets(rows,poses)
    predictions = predict_all(sensor,models)
    predictions["straight"] = np.broadcast_to(STRAIGHT,(len(rows),len(HORIZONS),2))
    turn = targets[:,-1,1]
    groups = dict(all=np.ones(len(rows),dtype=bool), left=turn>=1., right=turn<=-1., straight=abs(turn)<1.)
    mode = policy["selected_vision_readout"]
    report = error_metrics(targets,predictions,groups)
    clearance = evaluate_clearance(sensor_features(sensor),models[mode],mode,poses,borders_from_spline(spline))
    program = Path(__file__)
    source_paths = (program,archives,receipt,source,spline,export_receipt,
                    Path(error_metrics.__code__.co_filename),Path(evaluate_clearance.__code__.co_filename),
                    Path(future_targets.__code__.co_filename))
    old.write(output/"evaluation.json",dict(
        track=track, layout=layout, evaluated_frames=len(rows), horizons_m=HORIZONS.tolist(),
        selected=mode, line_outputs=export["available"], line_abstentions=export["abstain"],
        line_sha256=export["racing_line_sha256"], results=report, clearance=clearance,
        label_kind="recorded future vehicle travel", sources=digests(source_paths),
        source_revision=old.revision(),seed=None,termination="archive_end"))
    print(track,report["all"]["20"]["methods"][mode],clearance["curve"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives",type=Path,required=True)
    parser.add_argument("--track",required=True)
    parser.add_argument("--layout",default="")
    parser.add_argument("--input",type=Path,required=True,help="Completed corner archive-replay directory")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.archives.resolve(),args.track,args.layout,args.input.resolve(),args.output.resolve())


if __name__=="__main__":
    main()
