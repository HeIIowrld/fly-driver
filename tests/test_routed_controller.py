import numpy as np
import torch

from flybrain_drive.body_organs import PositionOrganSpec
from flybrain_drive.control import VehicleConfig
from flybrain_drive.routed_controller import RoutedController, steering_population_decoder


def inputs():
    graph = {"ids": np.array(["vL", "vR", "r", "p", "relay", "outL", "outR"]),
             "types": np.array(["v", "v", "r", "p", "n", "o", "o"]),
             "roles": np.array(["input"] * 4 + ["internal", "output", "output"]),
             "pre": np.array([0, 1, 2, 3, 4]), "post": np.array([5, 6, 4, 4, 5]),
             "weights": np.ones(5, np.float32)}
    ports = {"visual_CX_cut_boundary_ids": ["vL", "vR"],
             "organ_input_groups": {"haltere_L": ["r"], "chordotonal_ProLN_L": ["p"]}}
    return graph, ports, [PositionOrganSpec("steering_angle", 2, -.5, .5, .15, 1)]


def test_commands_traverse_outputs_and_cut_connections_block_visual_readthrough():
    torch.set_num_threads(1)
    graph, ports, specs = inputs()
    model = RoutedController(graph, ports, 18, 24, .01, [-1, 1])
    report = model.calibrate(specs, VehicleConfig(), steps=100, position_gain=0, yaw_gain=0)
    assert report["visual_steady_rmse"] < .01
    state = model.initial_state()
    with torch.no_grad():
        for _ in range(150):
            command, state, _ = model.advance(torch.tensor([[.6]]), torch.ones(1, 1),
                                               torch.zeros(1, 24), torch.zeros(1, 24), state)
    assert abs(float(command[0]) - .6) < .01
    model.circuit.structural_weight.zero_()
    state = model.initial_state()
    for _ in range(10):
        command, state, _ = model.advance(torch.ones(1, 1), torch.ones(1, 1),
                                           torch.ones(1, 24), torch.ones(1, 24), state)
        assert not command.any()


def test_position_code_calibration_and_model_state_restore(tmp_path):
    graph, ports, specs = inputs()
    weight, error = steering_population_decoder(specs, VehicleConfig(), 24)
    assert error < .003 and np.isfinite(weight).all()
    model = RoutedController(graph, ports, 18, 24, .01, [-1, 1])
    model.calibrate(specs, VehicleConfig(), steps=100)
    assert sum(float(p.abs().sum()) for p in model.adapter.organ_adapters.parameters()) > 0
    path = tmp_path / "controller.pt"
    torch.save(model.state_dict(), path)
    restored = RoutedController(graph, ports, 18, 24, .01, [-1, 1])
    restored.load_state_dict(torch.load(path, weights_only=True))
    args = (torch.tensor([[.3]]), torch.ones(1, 1), torch.ones(1, 24), torch.ones(1, 24))
    a, b = model.initial_state(), restored.initial_state()
    for _ in range(5):
        out_a, a, _ = model.advance(*args, a)
        out_b, b, _ = restored.advance(*args, b)
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
