"""Separate fixed flight command/readout capacity and gradients from visual learning."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    source, data_root = Path(plan["training"]), Path(plan["data"])
    receipt = read(source / "result.json")
    checkpoint_hash = sha256(receipt["checkpoint"])
    assert checkpoint_hash == receipt["checkpoint_sha256"]
    assert torch.cuda.is_available(), "GPU required; no silent fallback"
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_sha256=sha256(args.plan),
            program_sha256=sha256(__file__),
            checkpoint_sha256=checkpoint_hash,
            dataset_sha256=sha256(data_root / "dataset.json"),
            spec_sha256=sha256(plan["spec"]),
            device=torch.cuda.get_device_name(0),
            evaluation_or_development_arrays_opened=False,
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    bodies, valids, targets, masks = [], [], [], []
    for entry in read(data_root / "dataset.json")["episodes"]:
        if entry["role"] != "train":
            continue
        assert sha256(data_root / entry["file"]) == entry["sha256"]
        with load_array(data_root / entry["file"], allow_pickle=False) as a:
            bodies.append(a["body"].copy())
            valids.append(a["body_valid"].copy())
            targets.append(a["target"].copy())
            masks.append(a["target_valid"].copy())
    body, valid = np.concatenate(bodies), np.concatenate(valids)
    median = np.array(
        [np.median(body[valid[:, i], i]) if valid[:, i].any() else 0 for i in range(16)], np.float32
    )
    present = valid.any(0)
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"])
    model.load_state_dict(payload["model"], strict=True)
    model.to("cuda").eval()
    original_parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
    dt = model.spec["dynamics"]["dt"]
    sample_steps = [int(round(t / dt)) for t in plan["sample_times_seconds"]]
    assert all(
        abs(step * dt - t) < 1e-10
        for step, t in zip(sample_steps, plan["sample_times_seconds"], strict=True)
    )
    with torch.no_grad():
        baseline = model._drive_tensor(
            torch.zeros((1, 6), device="cuda"),
            torch.zeros((1, 6), device="cuda", dtype=torch.bool),
            torch.tensor(median[None], device="cuda"),
            torch.tensor(present[None], device="cuda"),
            torch.tensor([[0.0, 1.0]], device="cuda"),
            model.scene_command.mean[None],
        )
        warm = model.initial_state(1)
        for _ in range(int(round(plan["warmup_seconds"] / dt))):
            warm = model._advance_elapsed_tensor(baseline, warm, dt)

    def simulate(commands, power=None, *, grad=False):
        with torch.set_grad_enabled(grad):
            drive = baseline.repeat(len(commands), 1)
            drive[:, model.steering_ports] = commands
            if power is not None:
                drive[:, model.goal_ports] = power[:, None]
            state = warm.repeat(len(commands), 1)
            snapshots = []
            for step in range(max(sample_steps) + 1):
                if step:
                    state = model._advance_elapsed_tensor(drive, state, dt)
                if step in sample_steps:
                    snapshots.append(model._read_state_tensor(state)[0])
            return torch.stack(snapshots), model._read_state_tensor(state)[1]

    grid = np.array(list(itertools.product(plan["steering_current_levels"], repeat=4)), np.float32)
    steering, muscles = simulate(torch.tensor(grid, device="cuda"))
    corners = list(itertools.product([0.0, 1.0], repeat=4))
    neutral = baseline[0, model.steering_ports].detach().cpu().numpy()
    common_grid = np.array(
        [(u, p) for p in plan["common_power_current_levels"] for u in [*corners, tuple(neutral)]],
        dtype=object,
    )
    common_commands = np.array([x[0] for x in common_grid], np.float32)
    power = np.array([x[1] for x in common_grid], np.float32)
    joint, joint_muscles = simulate(
        torch.tensor(common_commands, device="cuda"), torch.tensor(power, device="cuda")
    )
    current = torch.full((1, 4), 0.5, device="cuda", requires_grad=True)
    response, _ = simulate(current, grad=True)
    jacobian = torch.stack(
        [
            torch.autograd.grad(response[-1, 0, head], current, retain_graph=head == 0)[0][0]
            for head in range(2)
        ]
    )
    epsilon = plan["gradient_epsilon"]
    finite_commands = np.full((8, 4), 0.5, np.float32)
    for axis in range(4):
        finite_commands[axis * 2, axis] += epsilon
        finite_commands[axis * 2 + 1, axis] -= epsilon
    finite, _ = simulate(torch.tensor(finite_commands, device="cuda"))
    finite_jacobian = (finite[-1, ::2] - finite[-1, 1::2]).T / (2 * epsilon)
    gradient_ok = torch.allclose(
        jacobian,
        finite_jacobian,
        atol=plan["gradient_absolute_tolerance"],
        rtol=plan["gradient_relative_tolerance"],
    )
    steering_np, joint_np = steering.cpu().numpy(), joint.cpu().numpy()
    target, target_valid = np.concatenate(targets), np.concatenate(masks)
    reports = {}
    for name, values in (
        ("steering_grid_original_power", steering_np),
        ("common_power_corner_grid", joint_np),
    ):
        snapshots = []
        for time, prediction in zip(plan["sample_times_seconds"], values, strict=True):
            lower, upper = prediction.min(0), prediction.max(0)
            heads = []
            for head in range(2):
                y = target[target_valid[:, head], head]
                projected = np.clip(y, lower[head], upper[head])
                heads.append(
                    dict(
                        head=("left_turn", "forward")[head],
                        observed_minimum=float(lower[head]),
                        observed_maximum=float(upper[head]),
                        train_targets_below=int((y < lower[head]).sum()),
                        train_targets_above=int((y > upper[head]).sum()),
                        conditional_interval_projection_rmse=float(
                            np.sqrt(np.mean((projected - y) ** 2))
                        ),
                    )
                )
            snapshots.append(dict(command_elapsed_seconds=time, heads=heads))
        reports[name] = snapshots
    np.savez_compressed(
        run / "component-responses.npz",
        steering_commands=grid,
        steering_intents=steering_np,
        steering_muscles=muscles.cpu().numpy(),
        common_commands=common_commands,
        common_power=power,
        joint_intents=joint_np,
        joint_muscles=joint_muscles.cpu().numpy(),
        baseline_drive=baseline.cpu().numpy(),
        warm_state=warm.cpu().numpy(),
        baseline_body=median,
        baseline_body_valid=present,
        analytical_jacobian=jacobian.detach().cpu().numpy(),
        numerical_jacobian=finite_jacobian.cpu().numpy(),
    )
    model.assert_body_boundary_protected()
    assert all(torch.equal(p, original_parameters[n]) for n, p in model.named_parameters())
    assert sha256(receipt["checkpoint"]) == checkpoint_hash
    result = dict(
        status="completed_component_command_capacity_diagnosis",
        reports=reports,
        steering_current_conditions=len(grid),
        common_power_conditions=len(power),
        gradient_check_passed=gradient_ok,
        analytical_jacobian=jacobian.detach().cpu().tolist(),
        numerical_jacobian=finite_jacobian.cpu().tolist(),
        maximum_gradient_difference=float((jacobian - finite_jacobian).abs().max()),
        checkpoint_and_parameters_unchanged=True,
        protected_body_unchanged=True,
        global_reachability_or_physiological_claim=False,
        evaluation_or_development_arrays_opened=False,
        labels_modified=False,
        optimizer_updates=0,
        actuator_messages_sent=0,
        ui_actions_sent=0,
        goal_achieved=False,
    )
    write_json(run / "result.json", result)
    print(
        json.dumps(
            dict(
                gradient_check_passed=gradient_ok,
                maximum_gradient_difference=result["maximum_gradient_difference"],
                steering_at2s=reports["steering_grid_original_power"][-1],
                joint_at2s=reports["common_power_corner_grid"][-1],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
