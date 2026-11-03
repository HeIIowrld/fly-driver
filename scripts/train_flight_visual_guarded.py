"""V6 training diagnosis with frozen-pass gradients and full-history acceptance checks."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT
from flybrain_drive.flight_dataset_lineage import visual_training_lineage
from flybrain_drive.flight_learning import bounded_intent_loss, turn_groups
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.audit_flight_yaw_boundary import pulse
from scripts.train_flight_initial import drives_for, substeps
from scripts.train_flight_visual_boundary import evaluate_visual


def guarded_step(optimizer, parameters, objective, before, *, trials, contraction, armijo):
    """Commit only a descending Adam proposal; rejection restores its moments and weights."""
    values = [p.detach().clone() for p in parameters]
    gradients = [p.grad.detach().clone() for p in parameters]
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    reports = []
    accepted = False
    try:
        optimizer.step()
        deltas = [p.detach().clone() - v for p, v in zip(parameters, values, strict=True)]
        slope = sum(float((g * d).sum()) for g, d in zip(gradients, deltas, strict=True))
        if math.isfinite(slope) and slope < 0:
            for number in range(trials):
                fraction = contraction ** number
                with torch.no_grad():
                    for p, v, delta in zip(parameters, values, deltas, strict=True):
                        p.copy_(v + fraction * delta)
                after = float(objective())
                passed = math.isfinite(after) and after <= before + armijo * fraction * slope
                reports.append(dict(fraction=fraction, loss=after if math.isfinite(after) else None,
                                    accepted=passed))
                if passed:
                    accepted = True
                    break
        return dict(accepted=accepted, before=before,
                    slope=slope if math.isfinite(slope) else None, trials=reports)
    finally:
        if not accepted:
            with torch.no_grad():
                for p, value in zip(parameters, values, strict=True):
                    p.copy_(value)
            optimizer.load_state_dict(original_optimizer)


def objective_pass(model, episodes, chunk_size, *, backward=False, turn_group_counts=None):
    """One fixed-parameter pass, weighted per label rather than per variable-sized chunk."""
    count = sum(int(arrays["target_valid"][:, 0].sum()) for _, _, _, arrays in episodes)
    if turn_group_counts is not None and (
            len(turn_group_counts) != 3 or any(type(c) is not int or c <= 0
                                              for c in turn_group_counts)
            or sum(turn_group_counts) != count):
        raise ValueError("Three positive original training turn-group counts required")
    episode_groups = None
    if turn_group_counts is not None:
        episode_groups = [turn_groups(a["target"][:, 0], a["target_valid"][:, 0])
                          for _, _, _, a in episodes]
        actual = [sum(int(np.count_nonzero(g == i)) for g in episode_groups) for i in range(3)]
        if actual != turn_group_counts:
            raise ValueError("Turn-group denominators must match every original valid train label")
    total, disconnected = 0., 0
    for episode_number, (_, data, schedule, arrays) in enumerate(episodes):
        state, held = model.initial_state(1), None
        for start in range(0, len(schedule), chunk_size):
            stop = min(start + chunk_size, len(schedule))
            drive = drives_for(model, data, start, stop)
            outputs, selected = [], []
            for local, index in enumerate(range(start, stop)):
                if index:
                    for dt in schedule[index]:
                        state = model._advance_elapsed_tensor(held, state, dt)
                if arrays["target_valid"][index, 0]:
                    outputs.append(model._read_state_preactivation_tensor(state)[0][0])
                    selected.append(index)
                held = drive[local:local + 1]
            if outputs:
                valid = data["target_valid"][selected].clone()
                valid[:, 1] = False
                prediction, target = torch.stack(outputs), data["target"][selected]
                scale = torch.tensor([.1, .1], device=state.device)
                if turn_group_counts is None:
                    loss = bounded_intent_loss(prediction, target, valid, scale)
                    contribution = loss * len(selected) / count
                    total += float(loss.detach()) * len(selected) / count
                else:
                    groups = episode_groups[episode_number][selected]
                    terms = []
                    for group, denominator in enumerate(turn_group_counts):
                        chosen = torch.tensor(groups == group, device=state.device)
                        number = int(np.count_nonzero(groups == group))
                        if number:
                            # Preserve float32 neural dynamics; accumulate small class-mean
                            # loss differences in float64 for full-history line search.
                            terms.append(bounded_intent_loss(prediction[chosen].double(),
                                target[chosen].double(), valid[chosen], scale.double())
                                * number / (3 * denominator))
                    contribution = torch.stack(terms).sum()
                    total += float(contribution.detach())
                if backward:
                    if contribution.requires_grad:
                        contribution.backward()
                    else:
                        disconnected += len(selected)
            state, held = state.detach(), held.detach()
    if not math.isfinite(total):
        raise ValueError("Finite full-history objective required")
    return total, disconnected


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root, source = Path(plan["data"]), Path(plan["source"])
    source_receipt, source_config = read(source / "result.json"), read(source / "manifest.json")
    dataset, data_receipt = read(root / "dataset.json"), read(root / "result.json")
    if (sha256(source_receipt["checkpoint"]) != source_receipt["checkpoint_sha256"]
            or sha256(root / "dataset.json") != data_receipt["dataset_manifest_sha256"]):
        raise ValueError("Unchanged source and complete data required")
    lineage = visual_training_lineage(plan, source_config, dataset)
    run = begin_run(args.output, dict(
        arguments=vars(args), plan=plan, seed=plan["seed"], plan_sha256=sha256(args.plan),
        spec_sha256=sha256(plan["spec"]), dataset_manifest_sha256=sha256(root / "dataset.json"),
        source_checkpoint_sha256=source_receipt["checkpoint_sha256"], actuator_messages_sent=0))
    write_json(run / 'data-lineage.json', lineage)
    result = dict(status="failed", goal_achieved=False, actor_qualified=False,
                  actuator_messages_sent=0, development_inputs_used=False,
                  development_labels_used=False, source_recordings_aid_absence_verified=False)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")
        torch.set_num_threads(1)
        torch.manual_seed(plan["seed"])
        episodes = []
        for entry in dataset["episodes"]:
            if entry["role"] != "train":
                continue
            if sha256(root / entry["file"]) != entry["sha256"]:
                raise ValueError("Original training episode changed")
            with load_array(root / entry["file"], allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                    if value.dtype.kind in "fbiu" and key != "observation_time"}
            schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
            episodes.append((entry, data, schedule, arrays))
        payload = torch.load(source_receipt["checkpoint"], map_location="cpu", weights_only=True)
        model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"])
        model.import_v5(payload)
        model.to("cuda")
        features = torch.cat([data["scene_features"] for _, data, _, _ in episodes])
        model.scene_command.set_train_normalization(features.mean(0),
            features.std(0, correction=0).clamp_min(plan["normalization_std_floor"]))
        group_counts = None
        objective_kind = plan.get("turn_objective", "per_label_mean")
        if objective_kind == "equal_left_right_neutral":
            groups = np.concatenate([turn_groups(a["target"][:, 0], a["target_valid"][:, 0])
                                     for _, _, _, a in episodes])
            group_counts = [int(np.count_nonzero(groups == group)) for group in range(3)]
            if not all(group_counts):
                raise ValueError("All three observed turn strata required; never synthesize labels")
        elif objective_kind != "per_label_mean":
            raise ValueError("Explicit supported turn objective required")
        write_json(run / "objective.json", dict(kind=objective_kind, turn_group_counts=group_counts,
            threshold=.01, counts_from_training_labels_only=True,
            label_dependent_weighting_is_cost_sensitive=True,
            actor_inputs_changed=False, original_labels_duplicated_or_removed=False))
        inherited = {key: value.detach().clone() for key, value in model.state_dict().items()
                     if not key.startswith("scene_command.")}
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=plan["learning_rate"])

        @torch.no_grad()
        def objective():
            return objective_pass(model, episodes, plan["chunk_observations"],
                                  turn_group_counts=group_counts)[0]

        initial, _ = evaluate_visual(model, episodes)
        history = []
        for epoch in range(plan["epochs"]):
            optimizer.zero_grad(set_to_none=True)
            before, disconnected = objective_pass(model, episodes, plan["chunk_observations"],
                                                   backward=True, turn_group_counts=group_counts)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            step = guarded_step(optimizer, parameters, objective, before,
                trials=plan["maximum_line_search_trials"],
                contraction=plan["line_search_contraction"], armijo=plan["armijo"])
            metrics, _ = evaluate_visual(model, episodes)
            history.append(dict(epoch=epoch + 1, step=step, unclipped_gradient_norm=float(norm),
                                disconnected_target_evaluations=disconnected, metrics=metrics))
            write_json(run / "history.json", history)
            write_json(run / "progress.json", dict(epoch=epoch + 1,
                       accepted=sum(row["step"]["accepted"] for row in history),
                       seconds=time.perf_counter() - started))
            model.assert_body_boundary_protected()
        for key, value in model.state_dict().items():
            if key not in inherited:
                continue
            old = inherited[key]
            if value.is_sparse:
                assert torch.equal(value.coalesce().indices(), old.coalesce().indices())
                assert torch.equal(value.coalesce().values(), old.coalesce().values())
            else:
                assert torch.equal(value, old), key
        final, predictions = evaluate_visual(model, episodes)
        lesions = {name: evaluate_visual(model, episodes, lesion=name)[0]
                   for name in ("scene_command", "vision", "motion_channels", "intensity_channel")}
        feedback = []
        yaw = np.array([0., -.22, -.07, .07, .22], np.float32)
        for number, index in [(1, 100), (2, 300)]:
            entry, _, _, arrays = episodes[number]
            response = pulse(model, arrays, index, entry["desired_speed_kmh"], yaw)
            delta = (response[:, :, 0] - response[:, :1, 0])[65:90].mean(0)
            feedback.append(dict(file=entry["file"], index=index, delta=delta.tolist(),
                                 signs_passed=(-np.sign(yaw[1:]) * delta[1:] > 1e-5).tolist()))
        for (entry, _, _, arrays), prediction in zip(episodes, predictions, strict=True):
            np.savez_compressed(run / entry["file"], prediction=prediction,
                                target=arrays["target"], target_valid=arrays["target_valid"],
                                observation_time=arrays["observation_time"])
        checkpoint = run / "checkpoint.pt"
        torch.save(dict(schema="flybrain.flight.visual_commands.v6", model=model.state_dict(),
                        spec_sha256=sha256(plan["spec"]), goal_contract=COMMAND_GOAL_CONTRACT,
                        source_checkpoint_sha256=source_receipt["checkpoint_sha256"],
                        glutamate_sign=payload["glutamate_sign"], epoch=plan["epochs"],
                        training_method=plan["schema"], actor_qualified=False), checkpoint)
        result.update(status="completed_guarded_optimizer_experiment", epochs=plan["epochs"],
            turn_objective=objective_kind, turn_group_counts=group_counts,
            accepted_steps=sum(row["step"]["accepted"] for row in history),
            initial_metrics=initial, final_metrics=final, input_lesions=lesions, feedback=feedback,
            feedback_signs_passed=all(all(row["signs_passed"]) for row in feedback),
            all_inherited_parameters_and_buffers_unchanged=True, longitudinal_supervised=False,
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            original_clocks_preserved=True, prefix_replayed_from_original_start=True,
            physical_vehicle_plant_tested=False, physiological_recognition_verified=False,
            seconds=time.perf_counter() - started,
            termination_reason="fixed six train-only epochs; no deployment or qualification")
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / "result.json", result)
        raise
    write_json(run / "result.json", result)
    print(json.dumps(dict(status=result["status"], accepted_steps=result["accepted_steps"])))


if __name__ == "__main__":
    main()
