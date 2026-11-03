import numpy as np
import pytest
import torch

from flybrain_drive.modality_ports import ModalityPortAdapter
from flybrain_drive.policy import SparseCircuit


def graph():
    return {"ids": np.array(["visual", "rotation", "position", "relay", "output"]),
            "types": np.array(["v", "r", "p", "n", "o"]),
            "roles": np.array(["input", "input", "input", "internal", "output"]),
            "pre": np.array([0, 1, 2, 3]), "post": np.array([4, 3, 3, 4]),
            "weights": np.ones(4, np.float32)}


def adapter(**options):
    result = ModalityPortAdapter(graph(), ["visual"], {
        "rotation": {"body_ids": ["rotation"], "feature_indices": [0]},
        "position": {"body_ids": ["position"], "feature_indices": [1]},
    }, organ_size=2, visual_size=1, **options)
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.fill_(1)
    return result


def test_modality_isolation_missing_masks_and_no_internal_or_output_injection():
    model = adapter()
    visual, organs = torch.tensor([[.2]]), torch.tensor([[.3, .4]])
    drive = model(visual, torch.ones(1, 1), organs, torch.ones(1, 2))
    np.testing.assert_allclose(drive.detach(), [[.2, 1.3, 1.4, 0, 0]], rtol=1e-6)
    moved = model(visual + 1, torch.ones(1, 1), organs, torch.ones(1, 2)) - drive
    assert torch.count_nonzero(moved[:, 1:]) == 0
    moved = model(visual, torch.ones(1, 1), organs + torch.tensor([[1., 0.]]),
                  torch.ones(1, 2)) - drive
    assert moved[0, 1] == 1 and torch.count_nonzero(moved[:, [0, 2, 3, 4]]) == 0
    assert not model(visual, torch.zeros(1, 1), organs, torch.zeros(1, 2)).any()
    with pytest.raises(ValueError, match="contract"):
        model(visual, torch.full((1, 1), .5), organs, torch.ones(1, 2))


def test_circuit_uses_measured_paths_persistent_state_and_blocks_output_bypass():
    circuit = SparseCircuit(graph(), .01)
    state = torch.zeros(1, 5)
    drive = torch.tensor([[0., 1., 0., 0., 0.]])
    first, state = circuit.step_with_drive(drive, state)
    assert not first.any()
    second, state = circuit.step_with_drive(drive, state)
    assert not second.any() and state[0, 3] > 0
    third, state = circuit.step_with_drive(drive, state)
    assert third[0, 0] > 0
    third.sum().backward()
    assert circuit.gain.grad is not None and circuit.gain.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="only declared input"):
        circuit.step_with_drive(torch.ones(1, 5), state)


def test_gate_only_validity_cannot_drive_organs_and_still_blocks_invalid_values():
    model = adapter(organ_validity_mode='gate_only')
    zero = torch.zeros(1, 1)
    valid = torch.ones(1, 2)
    assert not model(zero, zero, torch.zeros(1, 2), valid).any()
    drive = model(zero, zero, torch.tensor([[.3, .4]]), valid)
    torch.testing.assert_close(drive, torch.tensor([[0., .3, .4, 0., 0.]]))
    masked = model(zero, zero, torch.tensor([[100., .4]]), torch.tensor([[0., 1.]]))
    torch.testing.assert_close(masked, torch.tensor([[0., 0., .4, 0., 0.]]))
    drive.sum().backward()
    for layer in model.organ_adapters:
        assert not layer.weight.grad[:, layer.in_features // 2:].any()
