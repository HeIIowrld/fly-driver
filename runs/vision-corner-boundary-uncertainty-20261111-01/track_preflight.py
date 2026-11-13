"""Read-only course file and geometry preflight for future independent capture."""
from collections import Counter
import configparser
from pathlib import Path

from evaluate_live_corner_geometry import load_track_truth
from train_vision_teacher_line import digest, write

out = Path(__file__).parent
root = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")
targets = {
    "ks_zandvoort": ("",),
    "spa": ("",),
    "ks_silverstone": ("gp",),
    "ks_vallelunga": ("extended_circuit",),
    "ks_brands_hatch": ("",),
    "ks_barcelona": ("",),
}
reports = {}
for track, layouts in targets.items():
    for layout in layouts:
        folder = root/track
        name = f"{track}/{layout}" if layout else track
        suffix = f"_{layout}" if layout else ""
        model_ini = folder/f"models{suffix}.ini"
        spline = folder/layout/"ai"/"fast_lane.ai"
        ui = folder/"ui"/layout/"ui_track.json"
        paths = (model_ini, spline, ui)
        absent = [str(p.relative_to(folder)) for p in paths if not p.is_file()]
        if absent:
            reports[name] = dict(files_passed=False, absent=absent)
            continue
        models = configparser.ConfigParser(interpolation=None)
        assert models.read(model_ini, encoding="utf-8-sig")
        members = [folder/models[section]["file"] for section in models if "file" in models[section]]
        missing = [p.name for p in members if not p.is_file()]
        _, points, spans = load_track_truth(spline)
        side_counts = dict(Counter("left" if sign > 0 else "right" for _, _, sign in spans))
        reports[name] = dict(files_passed=bool(members) and not missing, missing_models=missing,
                             spline_points=points, physical_spans_by_side=side_counts,
                             minimum_two_spans_per_side=all(side_counts.get(s, 0) >= 2
                                                            for s in ("left", "right")),
                             files={str(p):digest(p) for p in paths},
                             model_member_count=len(members), menu_and_engine_verified=False)
write(out/"track-preflight.json", dict(tracks=reports, seed=None,
     source_sha256=digest(Path(__file__)), data_use="evaluation geometry preflight only",
     termination="read_only", frozen_candidate_selected=False))
for name, info in reports.items():
    print(name, info["files_passed"], info.get("physical_spans_by_side"), info.get("absent"))
