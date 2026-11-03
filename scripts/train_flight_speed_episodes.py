"""Train a fresh speed-goal flight candidate on complete, separately declared episodes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.flight_learning import GOAL_CONTRACT, bounded_intent_loss
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import drives_for, replay, substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


@torch.no_grad()
def evaluate(model, episodes, *, lesion=None):
    reports, predictions = [], []
    for entry, data, schedule, _arrays in episodes:
        prediction, _ = replay(model, data, schedule, lesion=lesion)
        mask, target = data["target_valid"], data["target"]
        reports.append(dict(file=entry["file"], role=entry["role"],
                            speed_kmh=entry["desired_speed_kmh"],
                            counts=mask.sum(0).cpu().tolist(),
                            rmse=[float((prediction[:, a][mask[:, a]]
                                         - target[:, a][mask[:, a]]).square().mean().sqrt())
                                  for a in range(2)],
                            prediction_std=prediction[mask.all(1)].std(0, correction=0)
                            .cpu().tolist(),
                            target_std=target[mask.all(1)].std(0, correction=0).cpu().tolist()))
        predictions.append(prediction.cpu().numpy())
    return reports, predictions


def baselines(episodes):
    all_train = []
    by_speed = {}
    for entry, data, _, _ in episodes:
        if entry["role"] == "train":
            rows = data["target"][data["target_valid"].all(1)]
            all_train.append(rows)
            by_speed.setdefault(entry["desired_speed_kmh"], []).append(rows)
    global_mean = torch.cat(all_train).mean(0)
    task_mean = {speed: torch.cat(rows).mean(0) for speed, rows in by_speed.items()}
    result = []
    for entry, data, _, _ in episodes:
        row = dict(file=entry["file"], role=entry["role"])
        for name, mean in [("global_train_mean", global_mean),
                           ("speed_conditioned_train_mean", task_mean[entry["desired_speed_kmh"]])]:
            row[name] = [float((data["target"][:, a][data["target_valid"][:, a]] - mean[a])
                              .square().mean().sqrt()) for a in range(2)]
        result.append(row)
    scale = torch.maximum(torch.cat(all_train).std(0, correction=0),
                          torch.tensor([.02, .01], device="cuda"))
    return result, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan, root = read(args.plan), Path(args.data)
    receipt, dataset = read(root / "result.json"), read(root / "dataset.json")
    if (receipt["status"] != "completed"
            or sha256(root / "dataset.json") != receipt["dataset_manifest_sha256"]
            or dataset["goal_contract"] != GOAL_CONTRACT
            or read(root / "manifest.json")["config"]["plan_sha256"]
            != sha256(plan["dataset_plan"])
            or dataset["spec_sha256"] != sha256(plan["spec"])
            or dataset["spec"] != plan["spec"]):
        raise ValueError("Unchanged complete v4 task dataset required")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan, seed=plan["seed"],
                                    dataset_manifest_sha256=sha256(root / "dataset.json"),
                                    plan_sha256=sha256(args.plan),
                                    spec_sha256=sha256(plan["spec"]), actuator_messages_sent=0))
    result = dict(status="failed", goal_achieved=False, actor_qualified=False,
                  actuator_messages_sent=0, independent_course_test=False,
                  actual_actor_integration=False)
    updates, started = 0, time.perf_counter()
    try:
        episodes = []
        for entry in dataset["episodes"]:
            if sha256(root / entry["file"]) != entry["sha256"]:
                raise ValueError("Complete episode changed")
            with load_array(root / entry["file"], allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            clock = arrays["observation_time"]
            schedule = [[]] + [substeps(float(dt)) for dt in np.diff(clock)]
            data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                    if value.dtype.kind in "fbiu" and key != "observation_time"}
            episodes.append((entry, data, schedule, arrays))
        training = [episode for episode in episodes if episode[0]["role"] == "train"]
        model = FlightCircuit(plan["spec"], glutamate_sign=plan["glutamate_sign"]).to("cuda")
        if not torch.all(model.circuit.state_mask == 1):
            raise ValueError("Intact circuit required for elapsed-time rule")
        original = model.circuit.matrix.coalesce().detach().clone()
        baseline, scale = baselines(episodes)
        with torch.no_grad():
            first = training[0][1]
            neutral_drive = model._drive_tensor(first["visual"][:1], first["visual_valid"][:1],
                                                first["body"][:1], first["body_valid"][:1],
                                                torch.zeros((1, 2), device="cuda"),
                                                first["scene_features"][:1])
            state = model.initial_state(1)
            for _ in range(plan["neutral_calibration_steps"]):
                state = model._advance_elapsed_tensor(neutral_drive, state, .01)
            _, neutral = model._read_state_tensor(state)
            muscles = torch.cat([replay(model, data, schedule)[1]
                                 for _, data, schedule, _ in training])
            lateral = .5 * ((muscles[:, 1] - muscles[:, 0]) + (muscles[:, 3] - muscles[:, 2]))
            reference = .5 * ((neutral[0, 1] - neutral[0, 0]) + (neutral[0, 3] - neutral[0, 2]))
            common, neutral_common = .5 * (muscles[:, 4] + muscles[:, 5]), neutral[0, 4:].mean()
            turn_scale = max(float((lateral - reference).abs().quantile(.95)),
                             float(neutral[0, :4].mean()))
            forward_scale = max(float((common - neutral_common).abs().quantile(.95)),
                                float(neutral_common))
            model.readout.calibrate_reference(neutral[0], turn_scale=turn_scale,
                                              forward_scale=forward_scale)
        initial, _ = evaluate(model, episodes)
        write_json(run / "calibration.json", dict(neutral_muscles=neutral[0].cpu().tolist(),
                   turn_scale=turn_scale, forward_scale=forward_scale,
                   target_scale=scale.cpu().tolist(), activity_scale_uses_training_inputs_only=True,
                   target_scale_uses_training_labels=True, original_history_restarted=True))
        optimizer = torch.optim.Adam([
            dict(params=[p for n, p in model.named_parameters() if not n.startswith("readout.")],
                 lr=plan["adapter_learning_rate"]),
            dict(params=list(model.readout.parameters()), lr=plan["readout_learning_rate"])])
        history = []
        for epoch in range(plan["epochs"]):
            for entry, data, schedule, arrays in training:
                state, held = model.initial_state(1), None
                has_target = arrays["target_valid"].any(1)
                for start in range(0, len(schedule), plan["chunk_observations"]):
                    stop = min(start + plan["chunk_observations"], len(schedule))
                    optimizer.zero_grad(set_to_none=True)
                    drive = drives_for(model, data, start, stop)
                    outputs, selected = [], []
                    for local, index in enumerate(range(start, stop)):
                        if index:
                            for dt in schedule[index]:
                                state = model._advance_elapsed_tensor(held, state, dt)
                        if has_target[index]:
                            outputs.append(model._read_state_preactivation_tensor(state)[0][0])
                            selected.append(index)
                        held = drive[local:local + 1]
                    if outputs:
                        loss = bounded_intent_loss(torch.stack(outputs), data["target"][selected],
                                                   data["target_valid"][selected], scale)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite proper bounded-mean objective")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                                       error_if_nonfinite=True)
                        optimizer.step()
                        updates += 1
                    state, held = state.detach(), held.detach()
                    write_json(run / "progress.json", dict(epoch=epoch + 1, episode=entry["file"],
                               chunk_stop=stop, optimizer_updates=updates,
                               seconds=time.perf_counter() - started))
            metrics, _ = evaluate(model, episodes)
            history.append(dict(epoch=epoch + 1, metrics=metrics, optimizer_updates=updates))
            write_json(run / "history.json", history)
            if epoch + 1 in (1, 3, plan["epochs"]):
                torch.save(dict(schema="flybrain.flight.speed_training.v1",
                                model=model.state_dict(),
                                goal_contract=GOAL_CONTRACT, spec_sha256=sha256(plan["spec"]),
                                dataset_manifest_sha256=sha256(root / "dataset.json"),
                                glutamate_sign=plan["glutamate_sign"], epoch=epoch + 1,
                                actor_qualified=False), run / f"epoch-{epoch + 1:02d}.pt")
        matrix = model.circuit.matrix.coalesce()
        if (not torch.equal(matrix.indices(), original.indices())
                or not torch.equal(matrix.values(), original.values())):
            raise ValueError("Original measured graph changed")
        final, predictions = evaluate(model, episodes)
        for episode, prediction in zip(episodes, predictions, strict=True):
            entry, _, _, arrays = episode
            np.savez_compressed(run / entry["file"], prediction=prediction,
                                target=arrays["target"], target_valid=arrays["target_valid"],
                                observation_time=arrays["observation_time"])
        lesions = {name: evaluate(model, episodes, lesion=name)[0]
                   for name in ("vision", "body", "goal")}
        checkpoint = run / f"epoch-{plan['epochs']:02d}.pt"
        result.update(status="completed_offline_initialization", epochs=plan["epochs"],
                      termination_reason="fixed epochs and complete original-history evaluations",
                      initial_metrics=initial, final_metrics=final, constant_baselines=baseline,
                      fixed_candidate_input_lesions=lesions, graph_weights_unchanged=True,
                      checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      optimizer_updates=updates, seconds=time.perf_counter() - started,
                      development_used_for_checkpoint_selection=False,
                      development_is_blind=False, physiological_stabilization_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), optimizer_updates=updates)
    write_json(run / "result.json", result)
    if result["status"] == "failed":
        raise RuntimeError(result["reason"])
    print(json.dumps(dict(status=result["status"], optimizer_updates=updates)))


if __name__ == "__main__":
    main()
