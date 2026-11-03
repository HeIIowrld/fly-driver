"""Early termination must not turn unequal driving horizons into a false improvement."""

import json
import sys

import numpy as np

from scripts.compare_multiview_evaluations import main


def test_comparison_uses_shared_prefix_and_exposes_short_failed_run(tmp_path, monkeypatch):
    config = {"control_dt_seconds": .04, "fixed_speed_mps": 1.5,
              "brain_ticks_per_decision": 4, "period_decisions": [1, 2],
              "view_maps_sha256": "a" * 64,
              "evaluation": {"steps": 4, "cases": [{"name": "a"}],
                             "conditions": ["normal", "vision_missing"]}}
    for name, conditions in (("old", [[1., 1.], [3.] * 4]),
                             ("new", [[.5, .5, 100., 100.], [2.] * 4])):
        root = tmp_path / name
        root.mkdir()
        rows = []
        for i, (condition, errors) in enumerate(zip(("normal", "vision_missing"),
                                                   conditions, strict=True)):
            n = len(errors)
            rows.append({"condition": condition, "case": "a", "steps": n,
                         "reason": "track_exit" if n < 4 else "time_limit",
                         "progress_delta_m": float(n)})
            values = np.column_stack([np.arange(1, n+1) * .04, np.zeros(n),
                                      np.arange(n), errors])
            np.savez(root / f"trace-{i:02d}.npz", trace=values,
                     columns=np.array(["simulation_time", "steering_proxy",
                                       "diagnostic_progress", "diagnostic_lateral"]))
        (root / "evaluation.json").write_text(json.dumps({"cases": rows}))
        (root / "manifest.json").write_text(json.dumps({"config": {"configuration": config}}))
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["compare", "--previous", str(tmp_path / "old"),
                                     "--current", str(tmp_path / "new"), "--output", str(output)])
    main()
    result = json.loads((output / "comparison.json").read_text())
    assert result["cases"][0]["matched_decisions"] == 2
    assert not result["cases"][0]["previous_normal"]["full_budget_without_exit"]
    assert result["cases"][0]["current_normal"]["full_budget_without_exit"]
    assert result["pooled_common_horizon_lateral_rms_m"]["current_normal"] == .5
    assert result["current_visual_error_lower_than_missing"]
