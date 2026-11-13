"""Verify the terminal Sprint capture and its offline corner-label denominators."""
from collections import Counter
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
CAPTURE = RUNS / "vision-nurburgring-sprint-capture-20261113-01"
sys.path.insert(0,str(RUNS/"vision-corner-left-risk-reallocation-20261113-01"))
from replay_independent import validate_layout_receipt,validate_transfer,old
from evaluate_live_corner_geometry import common_rows,load_track_truth
from flybrain_drive.ac_metadata_mmf import verify_mmf_archive_row

TRACK = "ks_nurburgring"
LAYOUT = "layout_sprint_a"


def main():
    receipt = CAPTURE/"result.json"
    mapping = CAPTURE/"archive-map.json"
    validate_layout_receipt(mapping,receipt,TRACK,LAYOUT)
    capture = old.read(receipt)
    labeler,_,_ = load_track_truth(validate_transfer(TRACK,LAYOUT))
    labels = []
    for name,folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            if archive.bundle_sha256 != capture["archive"]["bundle_sha256"]:
                raise ValueError("Capture checksum differs")
            raw = archive.read("stream.jsonl").splitlines()
            journal = archive.json("frame-persistence-journal.json")
            terminal = archive.json("result.json")
            consumer = archive.json("consumer-result.json")
            required = capture["frames"]
            if (len(raw) != required or len(journal["records"]) != required
                    or terminal["frames"] != required or terminal["status"] != "completed"
                    or consumer["frames"] != required or consumer["error"] is not None
                    or journal["failed"] != 0 or journal["completed"] != required or not journal["closed"]):
                raise ValueError("Camera archive is incomplete")
            for index,(line,proof) in enumerate(zip(raw,journal["records"],strict=True)):
                row = json.loads(line)
                if (row["capture_id"] != index or proof["group_id"] != index
                        or proof["status"] != "persisted" or row["ideal_line_shown"] or row["aid_probe_only"]):
                    raise ValueError("Persisted frame identity or aids differ")
                verify_mmf_archive_row(archive,row)
            labels.extend(common_rows(name,archive,{},TRACK,labeler))
    count = Counter(row["truth"] for row in labels)
    directions = Counter(row["label"] for row in labels if row["truth"]=="approach")
    approach = {side:len({r["physical_corner"] for r in labels if r["truth"]=="approach" and r["label"]==side})
                for side in directions}
    if not all(r["camera_complete"] for r in labels):
        raise ValueError("Camera views are incomplete")
    output = dict(track=TRACK,layout=LAYOUT,frames=len(labels),truth=dict(count),
                  approaches_by_direction=dict(directions),corners_by_direction=approach,
                  capture_sha256=old.digest(receipt),mapping_sha256=old.digest(mapping),
                  seed=None,termination="verified")
    old.write(OUT/"capture-verification.json",output)
    print(output,flush=True)


if __name__ == "__main__":
    main()
