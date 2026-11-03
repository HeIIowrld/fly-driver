"""New evaluation geometry must be applied without contaminating subsequent cases."""

import pytest

from scripts.evaluate_multiview_steering import evaluation_environment


def test_new_curve_is_applied_only_to_its_declared_case():
    environment = {"mode": "V", "road": {"length": 100, "turn_period": 70},
                   "vehicle": {"max_speed": 4}}
    selected = evaluation_environment(environment, {"road": {"turn_period": 50}}, 2000)
    assert selected["road"] == {"length": 100, "turn_period": 50}
    assert selected["max_steps"] == 2000
    selected["vehicle"]["max_speed"] = 10
    assert environment["vehicle"]["max_speed"] == 4
    assert evaluation_environment(environment, {}, 600)["road"]["turn_period"] == 70
    assert "max_steps" not in environment


def test_invalid_scenario_budget_or_geometry_is_rejected():
    for steps, case in ((0, {}), (True, {}), (600, {"road": []})):
        with pytest.raises(ValueError, match="scenario budget"):
            evaluation_environment({}, case, steps)
