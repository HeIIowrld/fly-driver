"""Train-only fixed-circuit command-range and cold/prefix response diagnosis."""

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
from scripts.train_flight_initial import drives_for, substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def cases(orientations):
    rows = [("binary/" + ''.join(map(str, v)), np.array(v, np.float32))
            for v in itertools.product([0, 1], repeat=4)]
    for common in [.003, .01, .03, .1, .3]:
        rows.append((f"common/{common}", np.full(4, common, np.float32)))
    for sign in [-1, 1]:
        for amplitude in [.003, .01, .03, .1, .3]:
            rows.append((f"oriented/{sign}/{amplitude}",
                         np.where(np.array(orientations) == sign, amplitude, 0.)
                         .astype(np.float32)))
    return rows


@torch.no_grad()
def prefix(model, data, schedule, index):
    drives = drives_for(model, data, 0, index + 1)
    state = model.initial_state(1)
    for current in range(1, index + 1):
        for dt in schedule[current]:
            state = model._advance_elapsed_tensor(drives[current - 1:current], state, dt)
    return state, drives[index:index + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    plan, data_root = config["plan"], Path(config["plan"]["data"])
    if (sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
            or sha256(plan["spec"]) != config["spec_sha256"]
            or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]):
        raise ValueError("Fixed checkpoint, spec and original dataset required")
    run = begin_run(args.output, dict(arguments=vars(args), seed=863159,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        context_selection="first valid label per train episode; strongest left and nearest-zero "
                          "valid train label across training episodes; deduplicate before probing",
        stimulus="all16 binary four-port corners, five common currents and ten oriented currents",
        states=["cold_zero", "original_prefix"], dt=.01, ticks=200,
        claim="sampled model envelope, not a global attainable-set bound or biology/plant test",
        target_use="context selection and post-hoc nearest-response diagnosis, never actor input",
        development_arrays_opened=False, fresh_evaluation_arrays_opened=False,
        optimizer_updates=0, actuator_messages_sent=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(1)
    torch.manual_seed(863159)
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.assert_body_boundary_protected()
    episodes, contexts, left_candidates, neutral_candidates = {}, [], [], []
    for entry in read(data_root / "dataset.json")["episodes"]:
        if entry["role"] != "train":
            continue
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original train episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        valid = np.flatnonzero(arrays["target_valid"][:, 0])
        contexts.append((entry["file"], int(valid[0])))
        for index in valid:
            left_candidates.append((float(arrays["target"][index, 0]), entry["file"], int(index)))
            neutral_candidates.append((abs(float(arrays["target"][index, 0])),
                                       entry["file"], int(index)))
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        episodes[entry["file"]] = (data, schedule, arrays)
    left, neutral = max(left_candidates), min(neutral_candidates)
    if left[0] > .01:
        contexts.append((left[1], left[2]))
    contexts.append((neutral[1], neutral[2]))
    contexts = list(dict.fromkeys(contexts))
    stimuli = cases(model.spec["steering_orientations"])
    labels = ["learned_current"] + [name for name, _ in stimuli]
    write_json(run / "contexts.json", dict(contexts=contexts, stimulus_labels=labels))
    reports = []
    with torch.no_grad():
        for number, (file, index) in enumerate(contexts):
            data, schedule, arrays = episodes[file]
            source_state, base = prefix(model, data, schedule, index)
            with load_array(root / file, allow_pickle=False) as saved:
                expected = saved["prediction"][index]
            actual = model._read_state_tensor(source_state)[0][0].cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-5)
            currents = torch.cat([base[:, model.steering_ports], torch.tensor(
                np.stack([v for _, v in stimuli]), device="cuda")])
            drive = base.repeat(len(currents), 1)
            drive[:, model.steering_ports] = currents
            states = {"cold_zero": model.initial_state(len(currents)),
                      "original_prefix": source_state.repeat(len(currents), 1)}
            response = {name: [] for name in states}
            for _tick in range(200):
                for name, state in states.items():
                    states[name] = model._advance_elapsed_tensor(drive, state, .01)
                    response[name].append(model._read_state_tensor(states[name])[0])
            responses = {name: torch.stack(values).cpu().numpy()
                         for name, values in response.items()}
            target = float(arrays["target"][index, 0])
            summary = {}
            for name, values in responses.items():
                samples = []
                for tick in [4, 9, 19, 49, 99, 199]:
                    turn = values[tick, :, 0]
                    nearest = int(np.argmin(abs(turn - target)))
                    samples.append(dict(seconds=(tick + 1) * .01,
                        turn_min=float(turn.min()), turn_max=float(turn.max()),
                        learned_turn=float(turn[0]), nearest_sampled_error=float(abs(
                            turn[nearest] - target)), nearest_sampled_case=labels[nearest],
                        most_left_case=labels[int(turn.argmax())],
                        most_right_case=labels[int(turn.argmin())]))
                summary[name] = samples
            np.savez_compressed(run / f"response-{number:02d}.npz", **responses,
                currents=currents.cpu().numpy(), stimulus_labels=np.array(labels),
                model_probe_elapsed_seconds=np.arange(1, 201) * .01)
            reports.append(dict(file=file, index=index, target=target,
                original_observation_time=float(arrays["observation_time"][index]),
                original_elapsed_seconds=float(arrays["observation_time"][index]
                                               - arrays["observation_time"][0]),
                prefix_prediction_error=float(abs(actual - expected).max()),
                current=currents[0].cpu().tolist(), response=summary))
            write_json(run / "progress.json", dict(contexts_completed=len(reports)))
    assert sha256(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
    write_json(run / "result.json", dict(status="completed_model_capacity_diagnostic",
        contexts=reports, sampled_current_cases=len(labels),
        checkpoint_modified=False, optimizer_updates=0, source_arrays_modified=False,
        prefix_replayed_from_original_start=True, development_arrays_opened=False,
        artificial_probe_time_is_not_recorded_data=True, global_attainability_bound_proved=False,
        actor_qualified=False, physiological_recognition_verified=False,
        physical_vehicle_plant_tested=False, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", contexts=len(reports), cases=len(labels))))


if __name__ == "__main__":
    main()
