"""Pin Silverstone 1967 acquisition, the original gates and an exploratory distance arm."""
from pathlib import Path

from evaluate_fused_corner import load_frozen
from evaluate_live_corner_geometry import load_track_truth
from profile_replay import TRACK_ROOT
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
SPA = Path("runs/vision-spa-independent-protocol-20261111-01")
TRACK = TRACK_ROOT / "ks_silverstone1967"
BUFFERED = Path("runs/vision-imola-approach-capture-protocol-20261111-01/capture_buffered.py")
ASSOCIATION = Path("runs/vision-corner-onset-association-20261111-01/result.json")
DISTANCE = Path("runs/vision-corner-motion-anchored-distance-20261111-01/anchor_distance.py")
SOURCES = (Path(__file__), OUT/"evaluate_silverstone.py", BUFFERED,
           Path("scripts/audit_corner_capture_admission.py"), Path("scripts/check_corner_stability_coverage.py"),
           Path("scripts/evaluate_fused_corner_warning_end.py"), DISTANCE, ASSOCIATION)


def replace_hash(config, old, new):
    config["hashes"].pop(str(old))
    config["hashes"][str(new)] = digest(new)


def run():
    if (OUT/"lock.json").exists():
        raise ValueError("Silverstone protocol already exists")
    track = TRACK / "ai/fast_lane.ai"
    ini = TRACK / "models.ini"
    model_files = [(TRACK / line.partition("=")[2].strip()).resolve() for line in ini.read_text().splitlines()
                   if line.strip().upper().startswith("FILE=")]
    if not model_files or not all(p.is_file() for p in model_files):
        raise ValueError("Silverstone 1967 model files are incomplete")
    _, _, spans = load_track_truth(track)
    sides = {name:sum(s[2] == side for s in spans) for side,name in ((1,"left"),(-1,"right"))}
    if min(sides.values()) < 2:
        raise ValueError("Silverstone geometry lacks both physical corner directions")
    base = load_frozen(SPA/"base.json")
    geometry = next(Path(p) for p in base["hashes"] if Path(p).name == "fast_lane.ai")
    replace_hash(base, geometry, track.resolve())
    base["hashes"][str(ini.resolve())] = digest(ini)
    base["excluded_development_tracks"].append("spa")
    base.update(proposed_track=TRACK.name, track_status="installed, independent acquisition pending",
                supersedes_protocol_sha256=digest(SPA/"base.json"),
                termination="independent_silverstone_geometry_and_gates_pinned")
    base_path = OUT/"base.json"
    write(base_path, base)

    candidate = load_frozen(SPA/"candidate.json")
    replace_hash(candidate, Path(candidate["base_protocol"]), base_path)
    candidate["base_protocol"] = str(base_path)
    candidate_path = OUT/"candidate.json"
    write(candidate_path, candidate)
    route = OUT/"route.json"
    write(route, dict(fusion_protocol=str(candidate_path), hashes={str(candidate_path):digest(candidate_path)}))

    acquisition = load_frozen(SPA/"capture.json")
    replace_hash(acquisition, Path(acquisition["warning_end_protocol"]), route)
    acquisition["warning_end_protocol"] = str(route)
    acquisition["capture"]["period_s"] = .09
    capture = OUT/"capture.json"
    write(capture, acquisition)
    review = read(SPA/"admission-review.json")
    review.update(immutable_candidate_sha256=digest(candidate_path), capture_protocol_sha256=digest(capture),
                  evaluation_program_sha256=digest(OUT/"evaluate_silverstone.py"),
                  source="Independent Silverstone 1967; original Spa gates carried forward; frozen before pixels",
                  exploratory="visual candidate with bounded 2m correction after relative travel")
    review_path = OUT/"admission-review.json"
    write(review_path, review)
    files = (*SOURCES, base_path, candidate_path, route, capture, review_path)
    write(OUT/"lock.json", dict(track=TRACK.name, geometry_counts=sides,
          expected_model_files=[p.name for p in model_files], exploratory_distance_filter=["bounded",2.,8.],
          source_revision=base["revision"], seed=None, independent_pixels_read=False,
          hashes={str(p):digest(p) for p in files},
          termination="silverstone_acquisition_evaluation_and_thresholds_pinned"))
    print(TRACK.name, sides, "requested camera cadence", acquisition["capture"]["period_s"])


if __name__ == "__main__":
    run()
