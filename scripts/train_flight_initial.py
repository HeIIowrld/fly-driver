"""Initial offline training of the measured flight candidate on a preserved AC episode."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.graph import sha256


def substeps(elapsed, nominal=0.01):
    if not np.isfinite(elapsed) or not 0 < elapsed <= 0.08:
        raise ValueError("Original positive interval at most 80ms required")
    count = int(elapsed // nominal)
    steps = [nominal] * count
    remainder = elapsed - count * nominal
    if remainder > 1e-12:
        steps.append(remainder)
    return steps


def drives_for(model, data, start, stop, *, lesion=None):
    return model._drive_tensor(
        data["visual"][start:stop],
        data["visual_valid"][start:stop],
        data["body"][start:stop],
        data["body_valid"][start:stop],
        data["goals"][start:stop],
        data["scene_features"][start:stop],
        lesion=lesion,
    )


@torch.no_grad()
def replay(model, data, schedule, *, stop=None, lesion=None):
    stop = len(schedule) if stop is None else stop
    drive = drives_for(model, data, 0, stop, lesion=lesion)
    state = model.initial_state(1)
    prediction, muscles = [], []
    for index in range(stop):
        if index:
            for dt in schedule[index]:
                state = model._advance_elapsed_tensor(drive[index - 1 : index], state, dt)
        intent, motor = model._read_state_tensor(state)
        prediction.append(intent[0])
        muscles.append(motor[0])
    prediction, muscles = torch.stack(prediction), torch.stack(muscles)
    if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
        raise ValueError("Nonfinite recurrent replay")
    return prediction, muscles


def metrics(prediction, data):
    out = {}
    for name, mask in [("train", data["train_mask"]), ("development", data["development_mask"])]:
        valid = data["target_valid"] & mask[:, None]
        error = prediction - data["target"]
        out[name] = [
            float(torch.sqrt(error[:, axis][valid[:, axis]].square().mean()).item())
            for axis in range(2)
        ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plan", default="configs/flight-initial-training-20261029.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
    root = Path(args.data)
    receipt = json.loads((root / "result.json").read_text(encoding="utf8"))
    data_plan_path = plan.get("data_plan", args.plan)
    if (
        receipt["plan_sha256"] != sha256(data_plan_path)
        or sha256(root / "episode.npz") != receipt["dataset_sha256"]
    ):
        raise ValueError("Sealed training data/plan changed")
    data_plan = json.loads(Path(data_plan_path).read_text(encoding="utf8"))
    for key in (
        "spec",
        "scene_replay",
        "body_replay",
        "teacher_dataset",
        "raw_recording",
        "train_label_observations",
        "validation_label_observations",
        "task_goal",
        "training_replay_start",
        "training_replay_stop",
        "validation_replay_start",
        "validation_replay_stop",
        "scene_motion_estimate_valid",
    ):
        if plan[key] != data_plan[key]:
            raise ValueError("Revised optimizer plan changed the sealed data contract: " + key)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            plan=plan,
            seed=plan["seed"],
            dataset_sha256=receipt["dataset_sha256"],
            spec_sha256=sha256(plan["spec"]),
            body_contract_sha256=receipt["body_contract_sha256"],
            actuator_messages_sent=0,
        ),
    )
    result = dict(
        status="failed",
        actuator_messages_sent=0,
        actual_actor_integration=False,
        independent_driving_validated=False,
        physiological_reproduction_verified=False,
        goal_achieved=False,
    )
    updates = 0
    try:
        with load_array(root / "episode.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        clock = arrays["observation_time"]
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(clock)]
        data = {
            key: torch.tensor(value, device="cuda")
            for key, value in arrays.items()
            if value.dtype.kind in "fbiu" and key != "observation_time"
        }
        if (
            data["visual_valid"].any()
            or not data["body_valid"][:, [0, 1, 4]].all()
            or data["target_valid"][:807].any()
        ):
            raise ValueError("Unregistered motion/body or student target contract differs")
        model = FlightCircuit(plan["spec"], glutamate_sign=plan["glutamate_sign"]).to("cuda")
        if not torch.all(model.circuit.state_mask == 1):
            raise ValueError("Elapsed-time training requires intact state mask")
        train_stop = plan["training_replay_stop"]
        # Calibration is a separate neutral model experiment, never prepended to original time.
        with torch.no_grad():
            neutral_goal = torch.zeros((1, 2), device="cuda")
            neutral_drive = model._drive_tensor(
                data["visual"][:1],
                data["visual_valid"][:1],
                data["body"][:1],
                data["body_valid"][:1],
                neutral_goal,
                data["scene_features"][:1],
            )
            neutral_state = model.initial_state(1)
            for _ in range(plan["neutral_calibration_steps"]):
                neutral_state = model._advance_elapsed_tensor(neutral_drive, neutral_state, 0.01)
            _, neutral = model._read_state_tensor(neutral_state)
            _, uncalibrated = replay(model, data, schedule, stop=train_stop)
            lateral = 0.5 * (
                (uncalibrated[:, 1] - uncalibrated[:, 0])
                + (uncalibrated[:, 3] - uncalibrated[:, 2])
            )
            neutral_lateral = 0.5 * (
                (neutral[0, 1] - neutral[0, 0]) + (neutral[0, 3] - neutral[0, 2])
            )
            common = 0.5 * (uncalibrated[:, 4] + uncalibrated[:, 5])
            neutral_common = 0.5 * (neutral[0, 4] + neutral[0, 5])
            turn_scale = float(
                torch.quantile(
                    (lateral - neutral_lateral).abs(), plan["motor_scale_quantile"]
                ).item()
            )
            forward_scale = float(
                torch.quantile((common - neutral_common).abs(), plan["motor_scale_quantile"]).item()
            )
            floor = plan.get("motor_scale_neutral_fraction", 0.0)
            turn_scale = max(turn_scale, floor * float(neutral[0, :4].mean().item()))
            forward_scale = max(forward_scale, floor * float(neutral_common.item()))
            model.readout.calibrate_reference(
                neutral[0], turn_scale=turn_scale, forward_scale=forward_scale
            )
            train_rows = data["train_mask"] & data["target_valid"].all(dim=1)
            target_scale = torch.maximum(
                data["target"][train_rows].std(dim=0, correction=0),
                torch.tensor([0.02, 0.01], device="cuda"),
            )
            mean_target = data["target"][train_rows].mean(dim=0)
            constant = mean_target[None].expand_as(data["target"])
            constant_metrics = metrics(constant, data)
            prediction, _ = replay(model, data, schedule)
            initial_metrics = metrics(prediction, data)
        write_json(
            run / "calibration.json",
            dict(
                neutral_muscles=neutral[0].cpu().tolist(),
                turn_scale=turn_scale,
                forward_scale=forward_scale,
                input_stop=train_stop,
                neutral_and_activity_scale_use_labels=False,
                neutral_and_activity_scale_use_validation_inputs=False,
                target_scale_uses_training_labels=True,
                metrics_use_declared_targets=True,
                original_episode_restarted_from_zero_after_calibration=True,
                target_scale=target_scale.cpu().tolist(),
                constant_baseline=constant_metrics,
                initial_metrics=initial_metrics,
            ),
        )
        objective = plan.get("objective", "intent_mse")
        if objective not in ("intent_mse", "inverse_tanh_mse"):
            raise ValueError("Explicit supported training objective required")
        if objective == "inverse_tanh_mse" and torch.any(data["target"].abs() >= 1):
            raise ValueError("Inverse-tanh targets must be strictly interior, never clipped")
        supervised_target = (
            data["target"].atanh() if objective == "inverse_tanh_mse" else data["target"]
        )
        readout = list(model.readout.parameters())
        other = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("readout.")
        ]
        optimizer = torch.optim.Adam(
            [
                dict(params=other, lr=plan["adapter_learning_rate"]),
                dict(params=readout, lr=plan["readout_learning_rate"]),
            ]
        )
        history = []
        started = time.perf_counter()
        original_matrix = model.circuit.matrix.detach().clone()
        for epoch in range(plan["epochs"]):
            state = model.initial_state(1)
            held = None
            for start in range(0, train_stop, plan["chunk_observations"]):
                stop = min(start + plan["chunk_observations"], train_stop)
                optimizer.zero_grad(set_to_none=True)
                drive = drives_for(model, data, start, stop)
                losses = []
                for local, index in enumerate(range(start, stop)):
                    if index:
                        for dt in schedule[index]:
                            state = model._advance_elapsed_tensor(held, state, dt)
                    if objective == "inverse_tanh_mse":
                        intent, _ = model._read_state_preactivation_tensor(state)
                    else:
                        intent, _ = model._read_state_tensor(state)
                    valid = data["target_valid"][index] & data["train_mask"][index]
                    if valid.any():
                        losses.append(
                            (((intent[0] - supervised_target[index]) / target_scale)[valid])
                            .square()
                            .mean()
                        )
                    held = drive[local : local + 1]
                if losses:
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training objective")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), plan["gradient_norm_limit"], error_if_nonfinite=True
                    )
                    optimizer.step()
                    updates += 1
                # Detach preserves both numerical neural state and last encoded input across chunks.
                state, held = state.detach(), held.detach()
                write_json(
                    run / "progress.json",
                    dict(
                        epoch=epoch + 1,
                        chunk_stop=stop,
                        total_epochs=plan["epochs"],
                        optimizer_updates=updates,
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
            prediction, _ = replay(model, data, schedule)
            values = metrics(prediction, data)
            history.append(dict(epoch=epoch + 1, metrics=values, optimizer_updates=updates))
            write_json(run / "history.json", history)
            if epoch + 1 in (1, 3, plan["epochs"]):
                torch.save(
                    dict(
                        schema="flybrain.flight.initial_training.v1",
                        model=model.state_dict(),
                        spec_sha256=sha256(plan["spec"]),
                        dataset_sha256=receipt["dataset_sha256"],
                        epoch=epoch + 1,
                        glutamate_sign=plan["glutamate_sign"],
                        actor_qualified=False,
                    ),
                    run / f"epoch-{epoch + 1:02d}.pt",
                )
        if not torch.equal(
            model.circuit.matrix.coalesce().values(), original_matrix.coalesce().values()
        ):
            raise ValueError("Fixed measured graph weights changed during training")
        final_prediction, _ = replay(model, data, schedule)
        lesion_results = {}
        for lesion in ("vision", "body"):
            altered, _ = replay(model, data, schedule, lesion=lesion)
            lesion_results[lesion] = metrics(altered, data)
        np.savez_compressed(
            run / "predictions.npz",
            intent=final_prediction.cpu().numpy(),
            target=arrays["target"],
            target_valid=arrays["target_valid"],
            observation_time=clock,
            train_mask=arrays["train_mask"],
            development_mask=arrays["development_mask"],
        )
        result.update(
            status="completed_offline_initialization",
            termination_reason="all fixed epochs and full-history evaluations completed",
            optimizer_updates=updates,
            epochs=plan["epochs"],
            initial_metrics=initial_metrics,
            final_metrics=metrics(final_prediction, data),
            constant_baseline=constant_metrics,
            fixed_model_input_lesions=lesion_results,
            graph_weights_unchanged=True,
            checkpoint=str(run / f"epoch-{plan['epochs']:02d}.pt"),
            checkpoint_sha256=sha256(run / f"epoch-{plan['epochs']:02d}.pt"),
            seconds=time.perf_counter() - started,
            validation_used_for_checkpoint_selection=False,
            positive_brake_training_labels=0,
            independent_course_test=False,
            actor_qualified=False,
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), optimizer_updates=updates)
    write_json(run / "result.json", result)
    if result["status"] == "failed":
        raise RuntimeError(result["reason"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
