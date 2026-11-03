"""Engineering organ feedback through explicit sensory ports, without route labels."""

import numpy as np
import torch


def calibrate_local_feedback(model, position_decoder, rotation_scale, *, position_gain=.2,
                             yaw_gain=.1, steps=200, amplitude=.03):
    """Assign sensory preferences by structural sensitivity, not measured physiology.

    The visual spatial pattern is retained with gain 1+k. Measured steering and yaw
    produce negative feedback through their own input-port groups. No direct motor
    residual or internal/output-neuron injection is added.
    """
    n = model.adapter.organ_size
    decoder = np.asarray(position_decoder, dtype=float)
    if (n < 7 or decoder.shape != (n,) or not np.isfinite(decoder).all()
            or np.any(decoder[-6:]) or model.output_activation != "linear"
            or not np.isfinite([rotation_scale, position_gain, yaw_gain, amplitude]).all()
            or min(rotation_scale, amplitude) <= 0 or min(position_gain, yaw_gain) < 0
            or type(steps) is not int or steps < 3):
        raise ValueError("Position-only decoder and finite linear feedback settings required")
    if any(torch.any(p != 0) for layer in model.adapter.organ_adapters for p in layer.parameters()):
        raise ValueError("Feedback calibration requires zero initial organ adapters")
    classes = {}
    for i, name in enumerate(model.adapter.group_names):
        prefix = next((p for p in ("chordotonal_", "haltere_") if name.startswith(p)), None)
        if prefix is None:
            raise ValueError("Only declared chordotonal and haltere port hypotheses are supported")
        indices = getattr(model.adapter, f"group_features_{i}")
        if (prefix == "chordotonal_" and torch.any(indices >= n - 6)
                or prefix == "haltere_" and torch.any(indices < n - 6)):
            raise ValueError("Position and rotation features must remain in separate port groups")
        classes.setdefault(prefix, []).append(i)
    if set(classes) != {"chordotonal_", "haltere_"}:
        raise ValueError("Both position and rotation input groups required")

    def response(drive):
        state = model.initial_state()
        with torch.no_grad():
            for _ in range(steps):
                outputs, state = model.circuit(drive, state)
            return float((outputs @ model.output_signs)[0] * model.readout_gain)

    planned, reports = {}, {}
    with torch.enable_grad():
        drive = model.initial_state().requires_grad_()
        state = model.initial_state()
        for _ in range(steps):
            outputs, state = model.circuit(drive, state)
        sensitivity, = torch.autograd.grad((outputs @ model.output_signs).sum(), drive)
    for prefix, multiplier in (("chordotonal_", position_gain), ("haltere_", yaw_gain)):
        members = classes[prefix]
        ports = torch.cat([getattr(model.adapter, f"group_ports_{i}") for i in members])
        maximum = sensitivity[:, ports].abs().max()
        if maximum < 1e-14:
            raise ValueError("Organ ports have no usable differential-output sensitivity")
        pattern = torch.zeros_like(sensitivity)
        pattern[:, ports] = sensitivity[:, ports] / maximum * amplitude
        equivalent = response(pattern)
        if not np.isfinite(equivalent) or abs(equivalent) < 1e-8:
            raise ValueError("Organ unit stimulus has no usable output response")
        feature = decoder.copy() if prefix == "chordotonal_" else np.zeros(n)
        if prefix == "haltere_":
            feature[-2:] = [rotation_scale, -rotation_scale]
        for i in members:
            indices = getattr(model.adapter, f"group_features_{i}")
            local_ports = getattr(model.adapter, f"group_ports_{i}")
            weight = torch.zeros_like(model.adapter.organ_adapters[i].weight)
            projection = torch.tensor(feature, dtype=weight.dtype, device=weight.device)[indices]
            weight[:, :len(indices)] = (-multiplier * pattern[0, local_ports, None]
                                        / equivalent * projection[None])
            planned[i] = weight
        reports[prefix] = {"input_ports": len(ports), "unit_output_response": equivalent,
                           "feedback_gain": multiplier}
    with torch.no_grad():
        for i, weight in planned.items():
            model.adapter.organ_adapters[i].weight.copy_(weight)
        model.adapter.visual_adapter.weight.mul_(1 + position_gain)
    return {"kind": "engineering_structural_sensitivity_not_physiological_tuning",
            "steps": steps, "amplitude": amplitude, "position_gain": position_gain,
            "yaw_gain": yaw_gain, "rotation_scale": rotation_scale,
            "visual_pattern_scale": 1 + position_gain, "organ_groups": reports,
            "route_labels_used": False, "direct_motor_residual": False}
