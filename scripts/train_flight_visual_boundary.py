"""Isolate a learned scene-to-descending boundary using intact original episode histories."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT
from flybrain_drive.flight_learning import bounded_intent_loss, step_connected_loss
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.probe_separated_flight_commands import feedback_probe
from scripts.train_flight_initial import drives_for, substeps
from scripts.train_flight_speed_episodes import baselines, evaluate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def evaluate_visual(model, episodes, *, lesion=None):
    reports, predictions = evaluate(model, episodes, lesion=lesion)
    for report, prediction, (_, _, _, arrays) in zip(reports, predictions, episodes, strict=True):
        late = arrays["observation_time"] - arrays["observation_time"][0] >= 3.
        selected = late & arrays["target_valid"][:, 0]
        report["late_turn_count"] = int(selected.sum())
        report["late_turn_rmse"] = float(np.sqrt(np.mean(
            (prediction[selected, 0] - arrays["target"][selected, 0]) ** 2)))
    return reports, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    source, root = Path(plan["source"]), Path(plan["data"])
    receipt, data_receipt = read(source / "result.json"), read(root / "result.json")
    source_config = read(source / "manifest.json")["config"]
    dataset = read(root / "dataset.json")
    checkpoint = receipt["checkpoint"]
    if (sha256(checkpoint) != receipt["checkpoint_sha256"]
            or sha256(root / "dataset.json") != data_receipt["dataset_manifest_sha256"]
            or sha256(root / "dataset.json") != source_config["dataset_manifest_sha256"]):
        raise ValueError("Unchanged v5 source and complete source dataset required")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan, seed=plan["seed"],
        source_checkpoint_sha256=sha256(checkpoint), plan_sha256=sha256(args.plan),
        spec_sha256=sha256(plan["spec"]), dataset_manifest_sha256=sha256(root / "dataset.json"),
        actuator_messages_sent=0, scope=__doc__))
    episodes = []
    for entry in dataset["episodes"]:
        if sha256(root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original episode changed")
        with load_array(root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        episodes.append((entry, data, schedule, arrays))
    training = [episode for episode in episodes if episode[0]["role"] == "train"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"])
    model.import_v5(payload)
    model = model.to("cuda")
    with torch.no_grad():
        train_features = torch.cat([data["scene_features"] for _, data, _, _ in training])
        mean = train_features.mean(0)
        scale = train_features.std(0, correction=0).clamp_min(plan["normalization_std_floor"])
        model.scene_command.set_train_normalization(mean, scale)
    inherited = {key: value.detach().clone() for key, value in model.state_dict().items()
                 if not key.startswith("scene_command.")}
    initial, _ = evaluate_visual(model, episodes)
    baseline, _ = baselines(episodes)
    write_json(run / "normalization.json", dict(
        training_recordings=[e["file"] for e, _, _, _ in training],
        observations=len(train_features), mean=mean.cpu().tolist(), scale=scale.cpu().tolist(),
        target_labels_used=False, development_inputs_used=False,
        source_snapshot_sha256=sha256(root / "dataset.json")))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=plan["learning_rate"])
    updates, detached_boundary_targets, history, started = 0, 0, [], time.perf_counter()
    for epoch in range(plan["epochs"]):
        for entry, data, schedule, arrays in training:
            state, held = model.initial_state(1), None
            for start in range(0, len(schedule), plan["chunk_observations"]):
                stop = min(start + plan["chunk_observations"], len(schedule))
                optimizer.zero_grad(set_to_none=True)
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
                    mask = data["target_valid"][selected].clone()
                    mask[:, 1] = False
                    loss = bounded_intent_loss(torch.stack(outputs), data["target"][selected],
                                               mask, torch.tensor([.1, .1], device="cuda"))
                    if step_connected_loss(loss, optimizer, model.parameters()):
                        updates += 1
                    else:
                        # With a frozen readout, the chunk's first output may depend only on
                        # detached past state/held input. Keep the target; no fake gradient or step.
                        detached_boundary_targets += len(selected)
                state, held = state.detach(), held.detach()
                model.assert_body_boundary_protected()
                write_json(run / "progress.json", dict(epoch=epoch + 1, episode=entry["file"],
                           chunk_stop=stop, updates=updates, seconds=time.perf_counter() - started))
        reports, _ = evaluate_visual(model, episodes)
        history.append(dict(epoch=epoch + 1, metrics=reports, updates=updates))
        write_json(run / "history.json", history)
        if epoch + 1 in (1, 3, plan["epochs"]):
            torch.save(dict(schema="flybrain.flight.visual_commands.v6", model=model.state_dict(),
                spec_sha256=sha256(plan["spec"]), goal_contract=COMMAND_GOAL_CONTRACT,
                source_checkpoint_sha256=sha256(checkpoint),
                glutamate_sign=payload["glutamate_sign"],
                epoch=epoch + 1, actor_qualified=False), run / f"epoch-{epoch + 1:02d}.pt")
    for key, value in model.state_dict().items():
        if key not in inherited:
            continue
        if value.is_sparse:
            original, current = inherited[key].coalesce(), value.coalesce()
            if (not torch.equal(current.indices(), original.indices())
                    or not torch.equal(current.values(), original.values())):
                raise ValueError("Inherited measured graph changed")
        elif not torch.equal(value, inherited[key]):
            raise ValueError("Inherited state changed: " + key)
    final, predictions = evaluate_visual(model, episodes)
    for episode, prediction in zip(episodes, predictions, strict=True):
        entry, _, _, arrays = episode
        np.savez_compressed(run / entry["file"], prediction=prediction, target=arrays["target"],
                            target_valid=arrays["target_valid"],
                            observation_time=arrays["observation_time"])
    lesions = {name: evaluate_visual(model, episodes, lesion=name)[0]
               for name in ("scene_command", "vision", "motion_channels", "intensity_channel")}
    feedback = feedback_probe(model, episodes)
    final_checkpoint = run / f"epoch-{plan['epochs']:02d}.pt"
    write_json(run / "result.json", dict(status="completed_offline_visual_experiment",
        epochs=plan["epochs"], optimizer_updates=updates,
        detached_boundary_target_evaluations=detached_boundary_targets,
        initial_metrics=initial, final_metrics=final,
        constant_baselines=baseline, input_lesions=lesions, feedback=feedback,
        feedback_signs_passed=all(all(row["signs_passed"]) for row in feedback),
        checkpoint=str(final_checkpoint), checkpoint_sha256=sha256(final_checkpoint),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        all_inherited_parameters_and_buffers_unchanged=True,
        seconds=time.perf_counter() - started, longitudinal_supervised=False,
        development_used_for_checkpoint_selection=False, development_is_blind=False,
        actor_qualified=False, physical_vehicle_plant_tested=False,
        physiological_recognition_verified=False, actuator_messages_sent=0, goal_achieved=False,
        termination_reason="fixed epochs, original histories and visual interventions measured"))
    print(json.dumps(dict(status="completed", updates=updates)))


if __name__ == "__main__":
    main()
