"""Verify common scoring inputs, regression parity and focused tests."""
from collections import Counter
from pathlib import Path
import subprocess
import sys

from train_vision_teacher_line import digest, read, write

out = Path(__file__).parent
root = out.parent
files = []
totals = {}
for name in ("magione-20261111-03", "magione-20261111-04", "imola-20261111-03"):
    base = root / ("vision-corner-common-" + name)
    proof = read(base / "result.json")
    assert digest(base / "rows.jsonl") == proof["rows_sha256"]
    for path, expected in proof["hashes"].items():
        assert digest(Path(path)) == expected, path
    counts = Counter()
    by_side = Counter()
    for report in proof["report"].values():
        assert sum(report["outputs"].values()) == report["frames"]
        assert sum(sum(v.values()) for v in report["truth_by_output"].values()) == report["frames"]
        counts.update(report["outputs"])
        for side, values in report["by_direction"].items():
            by_side.update({side+"_"+key: values[key] for key in ("frames","direction_hits","joint_hits")})
    totals[name] = dict(frames=proof["capture_frames"], outputs=dict(counts), by_side=dict(by_side))
    files.extend((base / "rows.jsonl", base / "result.json"))

legacy = root / "vision-corner-common-legacy-parity-20261111-02"
previous = root / "vision-imola-portable-corner-evaluation-20261110-01"
assert (legacy / "rows.jsonl").read_bytes() == (previous / "rows.jsonl").read_bytes()
assert read(legacy / "result.json")["report"] == read(previous / "result.json")["report"]
stages = root / "vision-magione-left-readout-stages-20261111-02"
assert (stages / "rows.jsonl").read_bytes() == (root / "vision-magione-left-readout-stages-20261111-01/rows.jsonl").read_bytes()

tests = subprocess.run([sys.executable, "-m", "pytest", "tests/test_corner_capture_scoring.py", "-q"],
                       text=True, capture_output=True)
(out / "pytest.log").write_text(tests.stdout + tests.stderr, encoding="utf8")
assert tests.returncode == 0, tests.stdout + tests.stderr
pins_path = root / "vision-frozen-next-corner-20261110-01/pins.json"
pins = read(pins_path)
for model in pins["models"].values():
    assert digest(Path(model["path"])) == model["sha256"]
files.extend([pins_path, legacy / "result.json", stages / "result.json", stages / "rows.jsonl",
              root / "vision-imola-full-capture-admission-20261111-01/result.json",
              root / "vision-cross-track-source-inventory-20261111-01/result.json",
              Path("tests/test_corner_capture_scoring.py"), Path(__file__), out / "pytest.log"])
write(out / "result.json", dict(status="passed", totals=totals, tests_returncode=tests.returncode,
      legacy_rows_identical=True, legacy_reports_identical=True, left_stage_rows_identical=True,
      hashes={str(path): digest(path) for path in files}, revision=pins["revision"], seed=None,
      termination="all checks passed"))
write(out / "protocol.json", dict(base_pins=dict(path=str(pins_path), sha256=digest(pins_path)),
      ordinal_forecasts=dict(path=str(root/"vision-corner-common-magione-20261111-03/result.json"),
                              sha256=digest(root/"vision-corner-common-magione-20261111-03/result.json")),
      joint_distance_forecasts=dict(path=str(root/"vision-corner-common-magione-20261111-04/result.json"),
                                    sha256=digest(root/"vision-corner-common-magione-20261111-04/result.json")),
      geometry_parameters=read(root/"vision-corner-common-magione-20261111-04/result.json")["parameters"],
      scope="saved forecast coverage audit; forecast configuration and teacher-label variants remain explicit",
      independent_positive_track_status="unavailable in screened repository sources",
      hashes={str(p): digest(p) for p in (Path("scripts/evaluate_live_corner_geometry.py"),
                                        Path("scripts/score_corner_captures.py"))}))
print(totals)
print(tests.stdout)
