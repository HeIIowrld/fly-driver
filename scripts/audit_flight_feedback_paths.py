"""Attribute virtual yaw feedback and inject named descending neurons in the fixed rate graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.flight_learning import GOAL_CONTRACT, speed_goals
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def fixed_inputs(arrays, count):
    inputs = {key: torch.tensor(np.repeat(arrays[key][:1], count, axis=0), device="cuda")
              for key in ("visual", "visual_valid", "body", "body_valid", "scene_features")}
    inputs["body"][:, :5] = 0
    inputs["goals"] = torch.tensor(speed_goals(15, count), device="cuda")
    return inputs


def encode(model, inputs):
    return model._drive_tensor(inputs["visual"], inputs["visual_valid"], inputs["body"],
                               inputs["body_valid"], inputs["goals"], inputs["scene_features"])


def settle(model, drive, steps=200):
    state = model.initial_state(len(drive))
    for _ in range(steps):
        state = model._advance_elapsed_tensor(drive, state, .01)
    return state


def attribution(model, arrays, annotations):
    inputs = fixed_inputs(arrays, 1)
    drive = encode(model, inputs).detach().requires_grad_(True)
    state = settle(model, drive)
    turn = model._read_state_tensor(state)[0][0, 0]
    influence = torch.autograd.grad(turn, drive)[0][0, model.body_ports]
    # Only yaw has been perturbed: tilt/load/temperature context is held fixed.
    rate = drive.detach()[0, model.body_ports]
    afferent_yaw = rate * (1 - rate) * model.body.weight[:, 2]
    contribution = (influence * afferent_yaw).detach().cpu().numpy()
    source = {}
    for bid, value, gradient, encoding in zip(model.spec["body_input_ids"], contribution,
                                            influence.detach().cpu().numpy(),
                                            afferent_yaw.detach().cpu().numpy(), strict=True):
        row = annotations[bid]
        source[bid] = dict(type=row["type"], rootSide=row["rootSide"],
                           output_sensitivity_per_port=float(gradient),
                           port_sensitivity_per_yaw=float(encoding),
                           chain_contribution=float(value))
    groups = {}
    for row in source.values():
        key = str(row["type"]) + "/rootSide=" + str(row["rootSide"])
        group = groups.setdefault(key, dict(count=0, sum=0., absolute_sum=0.))
        group["count"] += 1
        group["sum"] += row["chain_contribution"]
        group["absolute_sum"] += abs(row["chain_contribution"])
    with torch.no_grad():
        perturbed = fixed_inputs(arrays, 2)
        perturbed["body"][:, 4] = torch.tensor([-.001, .001], device="cuda")
        output = model._read_state_tensor(settle(model, encode(model, perturbed)))[0][:, 0]
        finite_difference = float((output[1] - output[0]) / .002)
    chain = float(contribution.sum())
    if not np.isclose(chain, finite_difference, rtol=.015, atol=3e-4):
        raise ValueError("Yaw chain attribution differs from central finite difference")
    return dict(steady_yaw_derivative=chain, finite_difference=finite_difference,
                groups=groups, per_afferent=source)


@torch.no_grad()
def descending_injections(model, arrays, annotations, graph_ids):
    cases = [("baseline", [])]
    for name in ("DNa04", "DNa05"):
        for bid in graph_ids:
            if annotations[bid]["type"] == name:
                cases.append((name + "/" + bid, [bid]))
    for side in ("L", "R"):
        cases.append(("DNg02/somaSide=" + side,
                      [bid for bid in model.spec["goal_input_ids"]
                       if annotations[bid]["somaSide"] == side]))
    if len(cases) != 7 or any(not ids for _, ids in cases[1:]):
        raise ValueError("Two pairs of steering candidates and both power groups required")
    base = encode(model, fixed_inputs(arrays, len(cases)))
    pulse = base.clone()
    for index, (_, ids) in enumerate(cases):
        for bid in ids:
            pulse[index, graph_ids.index(bid)] += .1
    state, held = settle(model, base), base
    states, intents = [], []
    for index in range(150):
        state = model._advance_elapsed_tensor(held, state, .01)
        states.append(state[:, model.circuit.output_ports].cpu().numpy())
        intents.append(model._read_state_tensor(state)[0].cpu().numpy())
        held = pulse if 50 <= index < 100 else base
    activity, intent = np.stack(states), np.stack(intents)
    delta = activity - activity[:, :1]
    output_ids = [graph_ids[int(index)] for index in model.circuit.output_ports.cpu()]
    reports = []
    for index, (name, ids) in enumerate(cases):
        muscles = {}
        for muscle in ("b1", "b2", "b3", "i1", "i2", "iii3"):
            muscles[muscle] = {}
            for side in ("L", "R"):
                selected = [i for i, bid in enumerate(output_ids)
                            if annotations[bid]["type"] == muscle + " MN"
                            and annotations[bid]["somaSide"] == side]
                if len(selected) != 1:
                    raise ValueError("Exact original bilateral motor IDs required")
                muscles[muscle][side] = float(delta[80:100, index, selected[0]].mean())
        # A diagnostic hypothesis only, not a new actor or a verified force/torque decoder.
        contrast = .25 * (muscles["b1"]["R"] - muscles["b1"]["L"]
                          + muscles["b2"]["R"] - muscles["b2"]["L"]
                          + muscles["b3"]["L"] - muscles["b3"]["R"]
                          + muscles["i1"]["L"] - muscles["i1"]["R"])
        reports.append(dict(case=name, injected_ids=ids, injected_current_per_node=.1,
                            somaSide=[annotations[bid]["somaSide"] for bid in ids],
                            neck_projection_side_verified=False, motor_deltas=muscles,
                            four_muscle_turn_contrast_hypothesis=contrast,
                            current_readout_delta=(intent[80:100, index]
                                                   - intent[80:100, 0]).mean(0).tolist()))
    return reports, activity, intent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    result, configuration = read(root / "result.json"), read(root / "manifest.json")["config"]
    checkpoint, spec = Path(result["checkpoint"]), configuration["plan"]["spec"]
    if (sha256(checkpoint) != result["checkpoint_sha256"]
            or sha256(spec) != configuration["spec_sha256"]):
        raise ValueError("Original checkpoint/spec changed")
    data_root = Path(configuration["arguments"]["data"])
    if sha256(data_root / "dataset.json") != configuration["dataset_manifest_sha256"]:
        raise ValueError("Original data manifest changed")
    entry = read(data_root / "dataset.json")["episodes"][0]
    if sha256(data_root / entry["file"]) != entry["sha256"]:
        raise ValueError("Original fixed-scene source changed")
    with load_array(data_root / entry["file"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    run = begin_run(args.output, dict(arguments=vars(args), seed=configuration["plan"]["seed"],
        checkpoint_sha256=sha256(checkpoint), scope=__doc__,
        derivative_policy="fixed scene, body zero rotation, same model inputs, central epsilon.001",
        neural_injection="named existing DNa04/05 or DNg02 nodes; diagnostic extra current only",
        pulse_indices=[50, 100], measurement_indices=[80, 100], model_neutral_steps=200,
        dt=.01, structural_graph_modified=False, optimizer_updates=0, actuator_messages_sent=0,
        side_policy="original somaSide only; no inference that DN soma side equals neck side"))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(configuration["plan"]["seed"])
    model = FlightCircuit(spec, glutamate_sign=configuration["plan"]["glutamate_sign"]).to("cuda")
    initial_body = {key: value.detach().clone() for key, value in model.body.state_dict().items()}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["goal_contract"] != GOAL_CONTRACT:
        raise ValueError("Explicit v4 checkpoint required")
    matrix = model.circuit.matrix.coalesce().detach().clone()
    model.load_state_dict(payload["model"], strict=True)
    if (not torch.equal(matrix.indices(), model.circuit.matrix.coalesce().indices())
            or not torch.equal(matrix.values(), model.circuit.matrix.coalesce().values())):
        raise ValueError("Checkpoint changed measured graph")
    model.eval()
    annotations = {str(row["bodyId"]): row for row in read(model.spec["annotations_file"])}
    with load_array(model.spec["graph_file"], allow_pickle=False) as graph:
        graph_ids = graph["ids"].tolist()
    learned = attribution(model, arrays, annotations)
    trained_body = {key: value.detach().clone() for key, value in model.body.state_dict().items()}
    model.body.load_state_dict(initial_body)
    reverted = attribution(model, arrays, annotations)
    model.body.load_state_dict(trained_body)
    injections, activity, intent = descending_injections(model, arrays, annotations, graph_ids)
    write_json(run / "yaw-attribution.json", dict(trained=learned, initial_body_only=reverted))
    write_json(run / "descending-injections.json", injections)
    np.savez_compressed(run / "descending-responses.npz", activity=activity, intent=intent)
    write_json(run / "result.json", dict(status="completed", termination_reason=__doc__,
        trained_yaw_derivative=learned["steady_yaw_derivative"],
        initial_body_only_yaw_derivative=reverted["steady_yaw_derivative"],
        attribution_sha256=sha256(run / "yaw-attribution.json"),
        injections_sha256=sha256(run / "descending-injections.json"),
        original_graph_unchanged=True, physiological_claim=False, actor_qualified=False,
        optimizer_updates=0, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(derivative=learned["steady_yaw_derivative"],
                         reverted=reverted["steady_yaw_derivative"])))


if __name__ == "__main__":
    main()
