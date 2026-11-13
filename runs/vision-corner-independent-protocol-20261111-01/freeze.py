"""Record the frozen comparison before independent capture pixels are viewed."""
from collections import Counter
from pathlib import Path

from evaluate_live_corner_geometry import load_track_truth
from flybrain_drive.win_process_receipt import engine_inventory
from train_vision_teacher_line import digest, read, write

out=Path(__file__).parent
root=out.parent
track=Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks/mugello/ai/fast_lane.ai")
_,points,spans=load_track_truth(track)
sides=Counter("left" if sign>0 else "right" for _,_,sign in spans)
base=root/"vision-frozen-next-corner-20261110-01/pins.json"
candidate=root/"vision-corner-blended-candidate-20261111-02/pins.json"
distance=root/"vision-magione-joint-distance-fusion-20261111-01/joint-pins.json"
protocol=root/"vision-corner-common-verification-20261111-01/protocol.json"
inputs=[base,candidate,distance,protocol,track,Path(__file__),
        Path("scripts/infer_frozen_next_corner.py"),Path("scripts/extract_frozen_corner_archive.py"),
        Path("scripts/extract_next_corner_flyvis.py"),Path("scripts/evaluate_live_corner_geometry.py"),
        Path("scripts/score_corner_captures.py"),Path("configs/ac-csp-exterior-front-v3-retina.json")]
write(out/"protocol.json",dict(
    revision=read(candidate)["revision"],seed=None,stage="frozen_candidate_for_independent_capture",
    excluded_development_tracks=["monza","magione","imola"],
    proposed_track="mugello",track_status="installed geometry inspected; no admitted independent camera capture",
    geometry_preflight=dict(installed_spline_points=points,physical_spans_by_side=sides,actor_input=False),
    data_admission=dict(minimum_independent_tracks=1,minimum_distinct_physical_corners_per_side=2,
        four_view_aid_off=True,moving_approach_band_m=[4,20],
        requires_straight_and_out_of_range=True,requires_source_clocks_and_generation=True,
        include_all_captures=True),
    comparison=dict(parent_pins=str(base),candidate_pins=str(candidate),joint_distance_pins=str(distance),
        same_capture_stream=True,same_feature_history=True,score_parameters=read(protocol)["geometry_parameters"],
        minimum_left_direction_hit_increase=1,maximum_right_direction_hit_loss=0,
        maximum_joint_hit_loss=0,maximum_outside_alert_increase=0,
        maximum_new_missing_decisions=0,
        minimum_stable_warning_per_physical_corner=1,
        distance_change_mae_must_improve_on_shared_direction_hit_pairs=True,
        receding_fraction_must_not_increase_on_shared_closing_pairs=True,
        measure_first_stable_lead_per_approach=True,
        use_common_endpoint_tolerance=True),
    capture_execution=dict(current_engine_inventory=engine_inventory(),
        existing_verified_cold_launch_track_ids=["monza","magione","imola"],
        independent_capture_started=False,desktop_input_sent=False),
    independent_pixels_read_for_selection=False,
    hashes={str(p):digest(p) for p in inputs},termination="independent protocol and installed-geometry preflight recorded"))
print(sides)
