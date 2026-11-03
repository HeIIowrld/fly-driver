import copy

import numpy as np
import pytest
import torch

from flybrain_drive.local_retinal_circuit import LocalRetinalController
from flybrain_drive.organ_feedback import calibrate_local_feedback
from scripts.evaluate_local_retinal_controller import apply_body_condition


def controller():
    graph = {"ids": np.array(["v1", "v2", "mid", "dn1", "dn2", "pos", "rot"]),
             "types": np.array(["MeTu", "MeTu", "EPG", "DN", "DN", "organ", "organ"]),
             "roles": np.array(["input", "input", "internal", "output", "output", "input",
                                 "input"]),
             "pre": np.array([0, 1, 2, 2, 5, 6]), "post": np.array([2, 2, 3, 4, 3, 4]),
             "weights": np.array([2., 1., 4., 3., 1., 1.], dtype=np.float32)}
    directions = np.array([[[1., 0, 0], [0, 1, 0]], [[1., 0, 0], [0, -1, 0]]])
    model = LocalRetinalController(graph, ["v1", "v2"], {
        "chordotonal_test": {"body_ids": ["pos"], "feature_indices": [0]},
        "haltere_test": {"body_ids": ["rot"], "feature_indices": list(range(1, 7))}},
        directions, np.ones((2, 2)), 7, output_signs=[-1, 1], neighbors=1,
        output_activation="linear")
    model.readout_gain.fill_(100)
    return model


def response(model, organs, valid):
    drive = model.adapter(torch.zeros(1, 12), torch.zeros(1, 1), organs, valid)
    state = model.initial_state()
    for _ in range(200):
        steering, state = model.advance(drive, state)
    return float(steering.detach()[0])


def test_feedback_has_negative_sign_and_uses_separate_measured_input_paths():
    model = controller()
    original = copy.deepcopy(model.state_dict())
    calibrate_local_feedback(model, [1, 0, 0, 0, 0, 0, 0], 5)
    torch.testing.assert_close(model.adapter.visual_adapter.weight,
                               original["adapter.visual_adapter.weight"] * 1.2, atol=0, rtol=0)
    torch.testing.assert_close(model.circuit.matrix.to_dense(),
                               original["circuit.matrix"].to_dense(), atol=0, rtol=0)
    for channel, value, expected in [(0, 1., -.2), (5, .2, -.1), (6, .2, .1)]:
        organs = torch.zeros(1, 7)
        organs[0, channel] = value
        assert response(model, organs, torch.ones_like(organs)) == pytest.approx(expected, abs=.001)
        assert response(model, organs, torch.zeros_like(organs)) == 0
    with pytest.raises(ValueError, match="zero initial"):
        calibrate_local_feedback(model, [1, 0, 0, 0, 0, 0, 0], 5)


def test_feedback_rejects_decoder_that_consumes_rotation_without_mutation():
    model = controller()
    old = model.adapter.visual_adapter.weight.detach().clone()
    with pytest.raises(ValueError, match="[Pp]osition-only"):
        calibrate_local_feedback(model, [1, 0, 0, 0, 0, 0, 1], 5)
    torch.testing.assert_close(old, model.adapter.visual_adapter.weight, atol=0, rtol=0)
    assert all(not p.any() for layer in model.adapter.organ_adapters for p in layer.parameters())


@pytest.mark.parametrize("condition", ["body_missing", "position_missing", "rotation_missing"])
def test_body_ablation_keeps_other_modalities_available(condition):
    packed = {k: torch.ones(1, 7) for k in ("organs", "organ_valid", "auxiliary", "auxiliary_valid",
                                          "applied_control", "applied_valid")}
    packed["ray_values"] = torch.ones(1, 2, 2, 3)
    apply_body_condition(packed, condition, 1)
    assert packed["ray_values"].all()
    if condition == "body_missing":
        assert not packed["organs"].any() and not packed["auxiliary_valid"].any()
    elif condition == "position_missing":
        assert not packed["organs"][:, :1].any() and packed["organs"][:, 1:].all()
    else:
        assert packed["organs"][:, :1].all() and not packed["organs"][:, 1:].any()
