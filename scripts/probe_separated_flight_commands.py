"""Verify v5 command routing, explicit v4 conversion and optimizer-safe body feedback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT, SeparatedFlightCircuit
from flybrain_drive.flight_learning import bounded_intent_loss
from flybrain_drive.graph import sha256
from scripts.audit_flight_feedback_paths import encode, fixed_inputs, read, settle
from scripts.audit_flight_yaw_boundary import pulse
from scripts.train_flight_initial import drives_for, replay, substeps


@torch.no_grad()
def command_probe(model, arrays):
    goals = torch.tensor([[0, .5], [.5, .5], [-.5, .5], [0, 1.], [.5, 1.]], device="cuda")
    inputs = fixed_inputs(arrays, len(goals))
    inputs["goals"] = goals
    drive = encode(model, inputs)
    turn_leak = float((drive[1, model.goal_ports] - drive[0, model.goal_ports]).abs().max())
    speed_leak = float((drive[4, model.steering_ports]
                        - drive[1, model.steering_ports]).abs().max())
    if turn_leak or speed_leak:
        raise ValueError("Goal coordinates leaked into the wrong direct boundary")
    current = drive[:, model.steering_ports].cpu().numpy()
    outputs = model._read_state_tensor(settle(model, drive))[0].cpu().numpy()
    flags = [bool(outputs[1, 0] > outputs[0, 0] + 1e-5),
             bool(outputs[2, 0] < outputs[0, 0] - 1e-5),
             bool(outputs[3, 1] > outputs[0, 1] + 1e-5)]
    return dict(goals=goals.cpu().tolist(), direct_turn_to_power_leak=turn_leak,
                direct_speed_to_steering_leak=speed_leak, steering_currents=current.tolist(),
                settled_intents=outputs.tolist(), left_right_speed_response_passed=flags,
                scope="direct-boundary separation; recurrent motor cross-coupling is retained")


@torch.no_grad()
def feedback_probe(model, episodes):
    yaw = np.array([0., -.22, -.07, .07, .22], np.float32)
    reports = []
    for number, index in [(1, 100), (2, 300), (3, 200), (4, 400)]:
        entry, _, _, arrays = episodes[number]
        response = pulse(model, arrays, index, entry["desired_speed_kmh"], yaw)
        means = (response[:, :, 0] - response[:, :1, 0])[65:90].mean(0)
        reports.append(dict(file=entry["file"], index=index, mean_delta=means.tolist(),
                            signs_passed=(-np.sign(yaw[1:]) * means[1:] > 1e-5).tolist()))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/flight-circuit-separated-v5-20261029.json")
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, root = Path(args.source), Path(args.data)
    receipt, data_receipt = read(source / "result.json"), read(root / "result.json")
    checkpoint = Path(receipt["checkpoint"])
    if (receipt["status"] != "completed_model_calibration"
            or sha256(checkpoint) != receipt["checkpoint_sha256"]
            or sha256(root / "dataset.json") != data_receipt["dataset_manifest_sha256"]):
        raise ValueError("Unchanged calibrated model and complete dataset required")
    run = begin_run(args.output, dict(arguments=vars(args), seed=863130,
        source_checkpoint_sha256=sha256(checkpoint), spec_sha256=sha256(args.spec),
        dataset_manifest_sha256=sha256(root / "dataset.json"),
        goal_transition="same numerical turn/speed fields, explicit v4-to-v5 boundary conversion",
        optimizer_probe=dict(episode=2, first_observation=0, chunks=5, chunk_observations=64,
                             learning_rate=.0001, weight_decay=.01, target_scale=[.1, .1]),
        scope=__doc__, actuator_messages_sent=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(863130)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    v4_spec = read(source / "manifest.json")["config"]["arguments"]["training"]
    v4_spec = read(Path(v4_spec) / "manifest.json")["config"]["plan"]["spec"]
    old = FlightCircuit(v4_spec, glutamate_sign=payload["glutamate_sign"]).to("cuda")
    old.load_state_dict(payload["model"], strict=True)
    try:
        FlightCircuit(args.spec, glutamate_sign=payload["glutamate_sign"])
    except ValueError as error:
        if "SeparatedFlightCircuit" not in str(error):
            raise
    else:
        raise ValueError("Legacy loader silently accepted new command routing")
    model = SeparatedFlightCircuit(args.spec, glutamate_sign=payload["glutamate_sign"])
    model.import_calibrated_v4(payload)
    model = model.to("cuda")
    episodes, equivalence = [], []
    for entry in read(root / "dataset.json")["episodes"]:
        if sha256(root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original episode changed")
        with load_array(root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        episode = (entry, data, schedule, arrays)
        episodes.append(episode)
        previous = replay(old, data, schedule)[0]
        converted = replay(model, data, schedule)[0]
        error = float((previous - converted).abs().max())
        torch.testing.assert_close(previous, converted, atol=3e-6, rtol=2e-5)
        equivalence.append(dict(file=entry["file"], observations=len(schedule), max_error=error))
        write_json(run / "progress.json", dict(stage="whole-history conversion",
                   completed=len(episodes), total=5))
    before_commands = command_probe(model, episodes[0][3])
    before_feedback = feedback_probe(model, episodes)
    frozen = {key: value.detach().clone() for key, value in model.body.state_dict().items()}
    readout = {key: value.detach().clone() for key, value in model.readout.state_dict().items()}
    original_matrix = model.circuit.matrix.coalesce().detach().clone()
    trainables = {name: value.detach().clone() for name, value in model.named_parameters()
                  if value.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.0001, weight_decay=.01)
    entry, data, schedule, arrays = episodes[2]
    if entry["role"] != "train":
        raise ValueError("Only explicit training episode may supply optimizer probe targets")
    state, held, losses = model.initial_state(1), None, []
    for start in range(0, 320, 64):
        stop = start + 64
        optimizer.zero_grad(set_to_none=True)
        drive = drives_for(model, data, start, stop)
        outputs, selected = [], []
        for local, index in enumerate(range(start, stop)):
            if index:
                for dt in schedule[index]:
                    state = model._advance_elapsed_tensor(held, state, dt)
            if arrays["target_valid"][index].any():
                outputs.append(model._read_state_preactivation_tensor(state)[0][0])
                selected.append(index)
            held = drive[local:local + 1]
        loss = bounded_intent_loss(torch.stack(outputs), data["target"][selected],
                                   data["target_valid"][selected],
                                   torch.tensor([.1, .1], device="cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.assert_body_boundary_protected()
        state, held = state.detach(), held.detach()
        losses.append(float(loss.detach()))
        write_json(run / "progress.json", dict(stage="five optimizer probes", chunk_stop=stop))
    body_error = max(float((value - frozen[key]).abs().max())
                     for key, value in model.body.state_dict().items())
    readout_unchanged = all(torch.equal(value, readout[key])
                            for key, value in model.readout.state_dict().items())
    readout_error = max(float((value - readout[key]).abs().max())
                        for key, value in model.readout.state_dict().items()
                        if value.dtype != torch.bool)
    trainable_change = max(float((value.detach() - trainables[key]).abs().max())
                           for key, value in model.named_parameters() if value.requires_grad)
    matrix = model.circuit.matrix.coalesce()
    if (body_error or readout_error or not readout_unchanged or trainable_change == 0
            or not torch.equal(matrix.indices(), original_matrix.indices())
            or not torch.equal(matrix.values(), original_matrix.values())):
        raise ValueError("Optimizer protection or fixed-edge contract failed")
    after_commands = command_probe(model, episodes[0][3])
    after_feedback = feedback_probe(model, episodes)
    output_checkpoint = run / "checkpoint.pt"
    torch.save(dict(schema="flybrain.flight.separated_commands.v5", model=model.state_dict(),
        spec_sha256=sha256(args.spec), goal_contract=COMMAND_GOAL_CONTRACT,
        source_checkpoint_sha256=sha256(checkpoint), glutamate_sign=payload["glutamate_sign"],
        dataset_manifest_sha256=sha256(root / "dataset.json"), optimizer_probe_updates=5,
        actor_qualified=False), output_checkpoint)
    write_json(run / "result.json", dict(status="completed", conversion=equivalence,
        before_commands=before_commands, after_commands=after_commands,
        before_feedback=before_feedback, after_feedback=after_feedback,
        optimizer_updates=5, optimizer_training_prefix_observations=320, losses=losses,
        maximum_body_change=body_error, maximum_readout_change=readout_error,
        maximum_trainable_change=trainable_change, original_graph_unchanged=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint=str(output_checkpoint), checkpoint_sha256=sha256(output_checkpoint),
        command_responses_passed=all(after_commands["left_right_speed_response_passed"]),
        feedback_signs_passed=all(all(r["signs_passed"]) for r in after_feedback),
        navigation_qualified=False, physical_vehicle_plant_tested=False,
        physiological_stabilization_verified=False, actor_qualified=False,
        actuator_messages_sent=0, goal_achieved=False,
        termination_reason="split routing, full histories and optimizer protection measured"))
    print(json.dumps(dict(body_change=body_error, readout_change=readout_error,
                         trainable_change=trainable_change)))


if __name__ == "__main__":
    main()
