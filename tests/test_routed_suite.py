import copy
import json

import pytest

from scripts.evaluate_routed_controller import read_suite


def test_evaluation_override_is_separate_from_frozen_checkpoint_configuration(tmp_path):
    config = {"evaluation": {"steps": 8000, "cases": [{"name": "old", "seed": 1}]}}
    original = copy.deepcopy(config)
    replacement = {"role": "independent_cases", "steps": 3,
                   "cases": [{"name": "new", "seed": 720101, "start_x": 5}]}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(replacement))
    assert read_suite(config, path) == replacement
    assert config == original
    assert read_suite(config) == original["evaluation"]


@pytest.mark.parametrize("suite", [
    {"steps": 0, "cases": [{"name": "a", "seed": 1}]},
    {"steps": True, "cases": [{"name": "a", "seed": 1}]},
    {"steps": 3, "cases": [{"name": "a", "seed": True}]},
    {"steps": 3, "cases": [{"name": "a", "seed": 1}, {"name": "a", "seed": 2}]},
    {"steps": 3, "cases": [{"name": "a", "seed": 1, "oracle_heading": 2}]},
])
def test_ambiguous_or_unlisted_evaluation_cases_are_rejected(tmp_path, suite):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        read_suite({}, path)
