"""Engineering-calibrated steering through a measured structural rate circuit.

This is a cut-boundary component model. It does not supply upstream visual/CX
biology, transmitter signs, measured time constants or physiological organ tuning.
"""

import numpy as np
import torch
from torch import nn

from .modality_ports import ModalityPortAdapter
from .policy import SparseCircuit


class RoutedController(nn.Module):
    def __init__(self, graph, ports, position_size, organ_size, dt, output_signs):
        super().__init__()
        if organ_size - position_size != 6:
            raise ValueError("Six signed rotation features must follow position/movement features")
        groups = {}
        for name, bodies in ports["organ_input_groups"].items():
            if name.startswith("haltere_"):
                indices = list(range(position_size, organ_size))
            elif name.startswith("chordotonal_"):
                indices = list(range(position_size))
            else:
                raise ValueError("Unknown organ-group mapping hypothesis")
            groups[name] = {"body_ids": bodies, "feature_indices": indices}
        self.adapter = ModalityPortAdapter(graph, ports["visual_CX_cut_boundary_ids"], groups,
                                            organ_size, 1)
        self.circuit = SparseCircuit(graph, dt)
        if len(output_signs) != len(self.circuit.output_ports):
            raise ValueError("Explicit differential output signs are required")
        self.register_buffer("output_signs", torch.tensor(output_signs, dtype=torch.float32))
        self.register_buffer("readout_gain", torch.tensor(1.))
        self.register_buffer("position_decoder", torch.zeros(organ_size))
        self.position_size, self.organ_size = position_size, organ_size
        self.requires_grad_(False)
        with torch.no_grad():
            for parameter in self.adapter.parameters():
                parameter.zero_()

    def initial_state(self, batch=1):
        return self.readout_gain.new_zeros((batch, self.circuit.size))

    def advance(self, visual, visual_valid, organs, organ_valid, state):
        drive = self.adapter(visual, visual_valid, organs, organ_valid)
        outputs, state = self.circuit.step_with_drive(drive, state)
        # Every command depends on output-neuron state only. No visual residual/read-through.
        steering = torch.clamp((outputs @ self.output_signs) * self.readout_gain, -1, 1)
        return steering, state, outputs

    def _steady_response(self, drive, steps):
        state = self.initial_state()
        with torch.no_grad():
            for _ in range(steps):
                output, state = self.circuit.step_with_drive(drive, state)
        return float((output @ self.output_signs)[0])

    def _input_pattern(self, allowed, steps, amplitude):
        with torch.enable_grad():
            drive = self.initial_state().requires_grad_()
            state = self.initial_state()
            for _ in range(steps):
                output, state = self.circuit.step_with_drive(drive, state)
            gradient, = torch.autograd.grad((output @ self.output_signs).sum(), drive)
        pattern = torch.zeros_like(gradient)
        selected = gradient[:, allowed]
        maximum = selected.abs().max()
        if maximum < 1e-10:
            raise ValueError("Declared input group cannot affect the differential output")
        pattern[:, allowed] = selected / maximum * amplitude
        return pattern.detach()

    def calibrate(self, position_specs, vehicle, *, position_gain=.2, yaw_gain=.1,
                  rotation_scale=5., steps=200, amplitude=.03):
        """Calibrate engineering transduction/readout with unit stimuli, not route truth.

        Position feedback boosts visual feedforward by 1+k and subtracts k times
        measured steering. Rotation feedback supplies a declared yaw damping term.
        Cell preferences come from structural sensitivity, not measured physiology.
        """
        if (min(position_gain, yaw_gain) < 0 or rotation_scale <= 0 or steps < 3
                or not np.isfinite([position_gain, yaw_gain, rotation_scale, amplitude]).all()
                or amplitude <= 0):
            raise ValueError("Invalid engineering calibration settings")
        visual = self._input_pattern(self.adapter.visual_ports, steps, amplitude)
        levels = np.linspace(-1, 1, 9)
        responses = np.array([self._steady_response(visual * float(level), steps)
                              for level in levels])
        if responses @ responses < 1e-14:
            raise ValueError("Visual calibration has no differential output response")
        gain = float(responses @ levels / (responses @ responses))
        with torch.no_grad():
            self.readout_gain.fill_(gain)
            self.adapter.visual_adapter.weight[:, 0] = visual[0, self.adapter.visual_ports] * (
                1 + position_gain)
        body_reports = {}
        decoder, rmse = steering_population_decoder(position_specs, vehicle, self.organ_size)
        with torch.no_grad():
            self.position_decoder.copy_(torch.tensor(decoder, dtype=torch.float32))
        for prefix, multiplier in (("haltere_", yaw_gain), ("chordotonal_", position_gain)):
            members = [i for i, name in enumerate(self.adapter.group_names)
                       if name.startswith(prefix)]
            if not members:
                raise ValueError("Both declared organ path classes are required")
            allowed = torch.cat([getattr(self.adapter, f"group_ports_{i}") for i in members])
            pattern = self._input_pattern(allowed, steps, amplitude)
            equivalent_gain = gain * self._steady_response(pattern, steps)
            if abs(equivalent_gain) < 1e-8:
                raise ValueError("Body path has no calibrated output response")
            feature = np.zeros(self.organ_size)
            if prefix == "haltere_":
                feature[-2:] = [rotation_scale, -rotation_scale]
            else:
                feature[:] = decoder
            for i in members:
                indices = getattr(self.adapter, f"group_features_{i}")
                ports = getattr(self.adapter, f"group_ports_{i}")
                projection = torch.tensor(feature, dtype=torch.float32)[indices]
                with torch.no_grad():
                    self.adapter.organ_adapters[i].weight.zero_()
                    self.adapter.organ_adapters[i].weight[:, :len(indices)] = (
                        -multiplier * pattern[0, ports, None] / equivalent_gain
                        * projection[None, :])
            body_reports[prefix] = {"input_neurons": len(allowed),
                                    "unit_equivalent_gain": equivalent_gain,
                                    "feedback_gain": multiplier}
        return {"calibration_steps": steps, "amplitude": amplitude,
                "visual_steady_rmse": float(np.sqrt(np.mean((responses * gain - levels)**2))),
                "readout_gain": gain, "position_decoder_rmse": rmse,
                "body_paths": body_reports, "physiological_preferences_verified": False,
                "biophysics": "normalized counts, engineered signed-rate dynamics and gains"}


def steering_population_decoder(specs, vehicle, organ_size):
    """Fit the declared Gaussian position code to Ackermann steering, without route labels."""
    offset = 0
    for spec in specs:
        if spec.sensor == "steering_angle":
            break
        offset += spec.channels * (spec.populations + 2)
    else:
        raise ValueError("Steering position populations are required")
    if spec.channels != 2:
        raise ValueError("Two roadwheel position populations are required")
    curvature = np.linspace(-vehicle.max_curvature, vehicle.max_curvature, 401)
    angles = np.arctan(vehicle.wheelbase * curvature[:, None] / np.column_stack([
        1 - vehicle.track * curvature / 2, 1 + vehicle.track * curvature / 2]))
    angles = np.clip(angles, -vehicle.max_steer, vehicle.max_steer)
    centers = np.linspace(spec.minimum, spec.maximum, spec.populations)
    features = np.zeros((len(curvature), organ_size))
    for channel in range(2):
        start = offset + channel * (spec.populations + 2)
        features[:, start:start + spec.populations] = np.exp(-.5 * (
            (angles[:, channel, None] - centers) / spec.tuning_width)**2)
    tangent = np.tan(angles)
    target = (tangent[:, 0] / (vehicle.wheelbase + vehicle.track / 2 * tangent[:, 0])
              + tangent[:, 1] / (vehicle.wheelbase - vehicle.track / 2 * tangent[:, 1])) / (
                  2 * vehicle.max_curvature)
    weight = np.linalg.solve(features.T @ features + 1e-5 * np.eye(organ_size), features.T @ target)
    return weight, float(np.sqrt(np.mean((features @ weight - target)**2)))
