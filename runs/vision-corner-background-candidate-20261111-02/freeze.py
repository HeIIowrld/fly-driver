"""Freeze a development-selected background offset and exercise the inference API."""
from pathlib import Path

from infer_frozen_next_corner import infer
from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
base = out.parent / "vision-frozen-next-corner-20261110-01"
pins = read(base / "pins.json")
pins["background_score_offset"] = -.02
pins["development_selection"] = dict(
    source="runs/vision-corner-background-margin-20261111-01/result.json",
    sha256=digest(out.parent / "vision-corner-background-margin-20261111-01/result.json"),
    rule="Smallest tested offset magnitude improving Magione left recall without added development false alarms",
    independent_evaluation_used=False)
write(out / "pins.json", pins)
archive = base / "archive-parity"
infer(out / "pins.json", archive / "index.jsonl", archive / "original/features.npz",
      archive / "reflected/features.npz", out / "ordinal")
infer(out / "pins.json", archive / "index.jsonl", archive / "original/features.npz",
      archive / "reflected/features.npz", out / "joint-distance",
      joint_distance_pins=out.parent / "vision-magione-joint-distance-fusion-20261111-01/joint-pins.json")
write(out / "result.json", dict(
      pins_sha256=digest(out / "pins.json"),
      program_sha256=digest(Path("scripts/infer_frozen_next_corner.py")),
      selection_program_sha256=digest(Path(__file__)),
      ordinal_result_sha256=digest(out / "ordinal/result.json"),
      joint_result_sha256=digest(out / "joint-distance/result.json"),
      revision=pins["revision"], seed=None, status="frozen_development_candidate",
      termination="candidate inference archived"))
