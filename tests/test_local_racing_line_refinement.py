from flybrain_drive.local_racing_line import plan_local_line


LIMITS = dict(clearance_m=1.1, max_curvature=.15, lateral_accel_mps2=4.,
              braking_mps2=4., speed_ceiling_mps=16.)


def test_refinement_recovers_curve_without_waiving_entry_speed_bound():
    sections = [dict(forward_x_m=x, right_y_m=r, left_y_m=l, valid=True)
                for x, r, l in [(4., -1.8218, 4.506), (8., .1136, 5.0733),
                                (14., 1.7367, 6.3412), (20., 3.6863, 8.0397)]]
    assert plan_local_line(sections, 10., **LIMITS) == (None, "no_feasible_curve")
    line, status = plan_local_line(sections, 10., refine_on_failure=True, **LIMITS)
    assert status == "above_entry_speed_bound"
    assert line["candidate_grid_points_per_axis"] == 17
    assert 0. < line["speed_bound_mps"][0] < 10.


def test_refinement_preserves_coarse_result_on_supported_corridor():
    sections = [dict(forward_x_m=x, right_y_m=-4., left_y_m=4., valid=True)
                for x in (4., 8., 14., 20.)]
    expected = plan_local_line(sections, 10., **LIMITS)
    assert expected[0] is not None
    assert plan_local_line(sections, 10., refine_on_failure=True, **LIMITS) == expected
