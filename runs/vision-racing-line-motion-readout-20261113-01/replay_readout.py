"""Apply the frozen line readout to a verified four-camera replay sensor bundle."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_readout import export_rows, load_fixed, load_sensor, old
from line_readout import lateral_prediction, metadata, sensor_features


def run(input_dir, output):
    policy, models = load_fixed()
    mode = policy["selected_vision_readout"]
    receipt = input_dir / "result.json"
    proof = old.read(receipt)
    if proof["status"] != "archive_replayed" or proof["capture_source"] != "raw_four_camera_archive":
        raise ValueError("The input must be a completed four-camera archive replay")
    source = input_dir / "sensors.npz"
    sensor = load_sensor(source,proof["inputs_sha256"])
    rows = metadata(sensor)
    if len(rows) != proof["frames"]:
        raise ValueError("Sensor timeline length differs from the archive replay")
    lateral = lateral_prediction(sensor_features(sensor),mode,models[mode])
    output.mkdir(parents=True,exist_ok=False)
    records = output/"racing-line.jsonl"
    export_rows(records,rows,sensor,lateral)
    available = int(np.isfinite(lateral).all(axis=1).sum())
    old.write(output/"result.json",dict(
        status="line_replayed",mode=mode,frames=len(rows),available=available,abstain=len(rows)-available,
        input_sha256=proof["inputs_sha256"],input_receipt_sha256=old.digest(receipt),
        camera_bundle_sha256_by_run=proof["archive_bundles"],
        policy_sha256=old.digest(Path(__file__).with_name("frozen.json")),
        code_sha256=old.digest(Path(__file__)),racing_line_sha256=old.digest(records),
        source_revision=old.revision(),seed=None,termination="archive_end"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True,help="Completed replay_corner_camera_archive output directory")
    parser.add_argument("--output",type=Path,required=True,help="New output directory")
    args=parser.parse_args()
    run(args.input.resolve(),args.output.resolve())


if __name__=="__main__":
    main()
