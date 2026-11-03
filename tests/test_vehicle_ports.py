import copy

import numpy as np
import pytest
import torch

from flybrain_drive.modality_ports import ModalityPortAdapter
from flybrain_drive.vehicle_ports import extend_organ_ports


def setup():
    graph = {"ids": np.array(list(map(str, range(1, 9)))), "types": np.array(["s"] * 8),
             "roles": np.array(["input"] * 7 + ["output"]),
             "pre": np.arange(7), "post": np.full(7, 7), "weights": np.ones(7)}
    old = {"visual_ids": ["1"], "organ_size": 2,
           "organ_groups": {"old": {"body_ids": ["2"], "feature_indices": [0, 1]}}}
    old_names = ("position.x", "rotation.z.+")
    names = ("position.x",) + tuple(f"tyre_core_temperature.{i}.level_0" for i in range(4))
    names += tuple(f"wheel_normal_load.{i}.level_0" for i in range(4)) + ("rotation.z.+",)
    annotations = [
        {"bodyId": 3, "rootSide": "L", "class": "thermosensory", "type": "TRN_VP2"},
        {"bodyId": 4, "rootSide": "R", "class": "thermosensory", "type": "TRN_VP2"},
        *[{"bodyId": i, "rootSide": "L", "subclass": "campaniform sensilla", "entryNerve": n}
          for i, n in ((5, "ProLN"), (6, "MesoLN"), (7, "MetaLN"))],
    ]
    candidates = {"thermal_receptor_candidates": {"body_ids": ["3", "4"]},
                  "leg_load_receptor_candidates": {"body_ids": ["5", "6", "7"]}}
    return graph, annotations, old, old_names, names, candidates


def test_name_mapping_preserves_old_gyro_and_routes_only_same_side_wheel_features():
    args = setup()
    groups, old_map = extend_organ_ports(*args)
    assert old_map == [0, 9] and groups["old"]["feature_indices"] == [0, 9]
    assert groups["load_ProLN_L"]["feature_indices"] == [5]
    assert groups["load_MesoLN_L"]["feature_indices"] == [5, 7]
    assert groups["load_MetaLN_L"]["feature_indices"] == [7]
    adapter = ModalityPortAdapter(args[0], ["1"], groups, 10, 1)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.fill_(1)
    values, valid = torch.zeros(1, 10), torch.zeros(1, 10)
    values[:, 2], valid[:, 2] = .2, 1  # FR temperature only.
    drive = adapter(torch.zeros(1, 1), torch.zeros(1, 1), values, valid)
    assert drive[0, 3] > 0  # Right thermal port 4.
    assert not drive[:, [0, 1, 2, 4, 5, 6, 7]].any()


def test_unknown_side_or_incomplete_input_assignment_cannot_be_silently_repaired():
    args = list(setup())
    bad = copy.deepcopy(args)
    bad[1][0]["rootSide"] = "unknown"
    with pytest.raises(ValueError, match="left/right"):
        extend_organ_ports(*bad)
    bad = copy.deepcopy(args)
    bad[5]["leg_load_receptor_candidates"]["body_ids"].pop()
    with pytest.raises(ValueError, match="all disjoint"):
        extend_organ_ports(*bad)
