"""Train new steering/power DN modulation through unchanged recurrent fly motor dynamics."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps
from scripts.train_flight_visual_guarded import guarded_step


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def objective_pass(model, episodes, chunk_size, *, backward=False):
    total, disconnected = 0.0, 0
    for episode in episodes:
        data, schedule = episode["data"], episode["schedule"]
        state, held = model.initial_state(1), None
        for start in range(0, len(schedule), chunk_size):
            stop = min(start + chunk_size, len(schedule))
            selection = slice(start, stop)
            drive = model._modulate_cached_drive(
                episode["baseline"][selection],
                data["body"][selection],
                data["body_valid"][selection],
                data["goals"][selection],
                data["scene_features"][selection],
            )
            outputs, indices = [], []
            for local, index in enumerate(range(start, stop)):
                if index:
                    for dt in schedule[index]:
                        state = model._advance_elapsed_tensor(held, state, dt)
                if episode["valid_any"][index]:
                    outputs.append(model._read_state_preactivation_tensor(state)[0][0])
                    indices.append(index)
                held = drive[local : local + 1]
            if outputs:
                prediction = torch.stack(outputs).double()
                loss = (
                    functional.binary_cross_entropy_with_logits(
                        2 * prediction, (data["target"][indices].double() + 1) / 2, reduction="none"
                    )
                    * episode["weights"][indices]
                ).sum()
                total += float(loss.detach())
                if backward:
                    if loss.requires_grad:
                        loss.backward()
                    else:
                        disconnected += len(indices)
            state, held = state.detach(), held.detach()
    return total, disconnected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    source, root, initial = Path(plan["source"]), Path(plan["data"]), Path(plan["initialization"])
    receipt, dataset = read(source / "result.json"), read(root / "dataset.json")
    assert dataset["schema"] == "flybrain.flight.embodied_navigation.episodes.v1"
    source_plan = read(source / "manifest.json")["config"]["plan"]
    parent = Path(source_plan["data"])
    assert dataset["parent_dataset_sha256"] == sha256(parent / "dataset.json")
    assert sha256(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
    initial_receipt = read(initial / "result.json")
    assert (
        initial_receipt["status"] == "passed_exact_initial_identity_and_real_graph_joint_gradients"
    )
    assert sha256(initial / "initial-model.pt") == initial_receipt["initial_model_sha256"]
    assert read(root / "result.json")["dataset_manifest_sha256"] == sha256(root / "dataset.json")
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    model = EmbodiedVisualFlightCircuit(plan["source_spec"], payload).to("cuda")
    new_payload = torch.load(initial / "initial-model.pt", map_location="cpu", weights_only=True)
    assert new_payload["schema"] == model.schema
    assert new_payload["source_checkpoint_sha256"] == receipt["checkpoint_sha256"]
    assert model.spec == new_payload["spec"]
    assert dataset["body_contract_sha256"] == model.spec["body_contract_sha256"]
    model.load_state_dict(new_payload["model"], strict=True)
    episodes, group_counts, forward_count = [], np.zeros(3, int), 0
    parent_entries = {e["file"]: e for e in read(parent / "dataset.json")["episodes"]}
    for entry in dataset["episodes"]:
        assert entry["role"] == parent_entries[entry["file"]]["role"]
        assert entry["parent_episode_sha256"] == parent_entries[entry["file"]]["sha256"]
        if entry["role"] != "train":
            continue
        assert sha256(root / entry["file"]) == entry["sha256"]
        with load_array(root / entry["file"], allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        assert arrays["body"].shape == (len(arrays["target"]), 21)
        assert arrays["body_valid"][:, [0, 1, 4, 16, 17, 18, 19, 20]].all()
        valid = arrays["target_valid"]
        groups = np.where(
            arrays["target"][:, 0] > 0.01, 0, np.where(arrays["target"][:, 0] < -0.01, 1, 2)
        )
        group_counts += np.bincount(groups[valid[:, 0]], minlength=3)
        forward_count += int(valid[:, 1].sum())
        data = {
            k: torch.tensor(arrays[k], device="cuda")
            for k in (
                "visual",
                "visual_valid",
                "body",
                "body_valid",
                "goals",
                "scene_features",
                "target",
            )
        }
        with torch.no_grad():
            baseline = model.backbone._drive_tensor(
                data["visual"],
                data["visual_valid"],
                data["body"][:, :16],
                data["body_valid"][:, :16],
                data["goals"],
                data["scene_features"],
            )
            drive = model._modulate_cached_drive(
                baseline, data["body"], data["body_valid"], data["goals"], data["scene_features"]
            )
            assert torch.equal(drive, baseline)
        episodes.append(
            dict(
                entry=entry,
                arrays=arrays,
                data=data,
                baseline=baseline,
                groups=groups,
                valid_any=valid.any(1),
                schedule=[[]] + [substeps(float(t)) for t in np.diff(arrays["observation_time"])],
            )
        )
    assert (group_counts > 0).all() and forward_count > 0
    for e in episodes:
        valid = e["arrays"]["target_valid"]
        weights = np.zeros(valid.shape, np.float64)
        weights[:, 0] = valid[:, 0] * 0.5 / (3 * group_counts[e["groups"]])
        weights[:, 1] = valid[:, 1] * 0.5 / forward_count
        e["weights"] = torch.tensor(weights, device="cuda")
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            plan=plan,
            plan_sha256=sha256(args.plan),
            program_sha256=sha256(__file__),
            dataset_manifest_sha256=sha256(root / "dataset.json"),
            source_checkpoint_sha256=receipt["checkpoint_sha256"],
            initial_checkpoint_sha256=initial_receipt["initial_model_sha256"],
            turn_group_counts=group_counts.tolist(),
            forward_supervised_rows=forward_count,
            device=torch.cuda.get_device_name(0),
            development_arrays_opened=False,
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters and all(not p.requires_grad for p in model.backbone.parameters())
    frozen = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    optimizer = torch.optim.Adam(parameters, lr=plan["learning_rate"])
    records = []
    result = dict(
        status="failed",
        goal_achieved=False,
        actor_qualified=False,
        development_arrays_opened=False,
        braking_positive_targets=0,
        actuator_messages_sent=0,
        ui_actions_sent=0,
    )
    try:
        for epoch in range(plan["epochs"]):
            optimizer.zero_grad(set_to_none=True)
            before, disconnected = objective_pass(
                model, episodes, plan["chunk_observations"], backward=True
            )
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)

            @torch.no_grad()
            def objective():
                return objective_pass(model, episodes, plan["chunk_observations"])[0]

            step = guarded_step(
                optimizer,
                parameters,
                objective,
                before,
                trials=plan["maximum_line_search_trials"],
                contraction=plan["line_search_contraction"],
                armijo=plan["armijo"],
            )
            assert all(torch.equal(p, frozen[n]) for n, p in model.backbone.named_parameters())
            model.backbone.assert_body_boundary_protected()
            row = dict(epoch=epoch + 1, disconnected_labels=disconnected, **step)
            records.append(row)
            torch.save(
                dict(
                    schema=model.schema,
                    model=model.state_dict(),
                    spec=model.spec,
                    goal_contract=payload["goal_contract"],
                    source_checkpoint=receipt["checkpoint"],
                    source_checkpoint_sha256=receipt["checkpoint_sha256"],
                    glutamate_sign=payload["glutamate_sign"],
                    completed_epochs=epoch + 1,
                    actor_qualified=False,
                ),
                run / "last-checkpoint.partial.pt",
            )
            os.replace(run / "last-checkpoint.partial.pt", run / "last-checkpoint.pt")
            write_json(
                run / "progress.json",
                dict(completed_epochs=epoch + 1, epochs=plan["epochs"], latest=row),
            )
            print(json.dumps(row), flush=True)
        assert sha256(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
        result.update(
            status="completed_fixed_joint_command_training",
            epochs=records,
            accepted_optimizer_updates=sum(r["accepted"] for r in records),
            checkpoint=str(run / "last-checkpoint.pt"),
            checkpoint_sha256=sha256(run / "last-checkpoint.pt"),
            inherited_parameters_unchanged=True,
            longitudinal_scope="throttle/coast only; brake supervision0",
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), epochs=records)
        raise
    finally:
        write_json(run / "result.json", result)


if __name__ == "__main__":
    main()
