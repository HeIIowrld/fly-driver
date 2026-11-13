"""Screen the newly inventoried Imola bundles for moving 4–20 m approaches."""
from collections import Counter
from pathlib import Path

from evaluate_live_corner_geometry import common_rows, load_track_truth
from flybrain_drive.ac_camera_archive import CameraArchive
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
inventory = out.parent / "vision-cross-track-source-inventory-20261111-01/result.json"
track = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks/imola/ai/fast_lane.ai")
truth, _, _ = load_track_truth(track)
sources = [r for r in read(inventory)["records"] if r.get("track_ids") == ["imola"]]
rows, reports = [], {}
for source in sources:
    directory = Path(source["directory"])
    tag = directory.parent.name
    try:
        with CameraArchive(directory) as archive:
            if archive.bundle_sha256 != source["bundle_sha256"]:
                raise ValueError("Camera archive differs from inventoried hash")
            captures = list(common_rows(tag, archive, {}, "imola", truth))
        rows.extend(captures)
        reports[tag] = dict(frames=len(captures), truth=dict(Counter(r["truth"] for r in captures)),
                            bundle_sha256=source["bundle_sha256"])
    except (OSError, ValueError, KeyError) as error:
        reports[tag] = dict(error=str(error))
jsonl(out / "rows.jsonl", rows)
write(out / "result.json", dict(frames=len(rows), truth=dict(Counter(r["truth"] for r in rows)),
     reports=reports, hashes={str(p): digest(p) for p in (inventory, track, Path(__file__),
         Path("scripts/evaluate_live_corner_geometry.py"), Path("scripts/score_corner_captures.py"))},
     rows_sha256=digest(out / "rows.jsonl"), seed=None, revision=read(inventory)["revision"],
     termination="archive_end"))
print(len(rows), Counter(r["truth"] for r in rows), {k:v for k,v in reports.items() if "error" in v})
