"""Test a model-only yaw calibration at withheld contexts; retain full original driving replays."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.flight_learning import speed_goals
from flybrain_drive.graph import sha256
from scripts.audit_flight_feedback_paths import encode, read, settle
from scripts.train_flight_initial import substeps
from scripts.train_flight_speed_episodes import evaluate


@torch.no_grad()
def pulse(model, arrays, index, speed, yaw, *, body_lesion=False):
    count = len(yaw)
    inputs = {key: torch.tensor(np.repeat(arrays[key][index:index + 1], count, axis=0),
                                device="cuda")
              for key in ("visual", "visual_valid", "body", "body_valid", "scene_features")}
    inputs["goals"] = torch.tensor(speed_goals(speed, count), device="cuda")

    def drive():
        if not body_lesion:
            return encode(model, inputs)
        return model._drive_tensor(inputs["visual"], inputs["visual_valid"], inputs["body"],
                                   inputs["body_valid"], inputs["goals"], inputs["scene_features"],
                                   lesion="body")

    base = drive()
    inputs["body"][:, 4] += torch.tensor(yaw, device="cuda")
    stimulus = drive()
    state, held = settle(model, base), base
    outputs = []
    for tick in range(150):
        state = model._advance_elapsed_tensor(held, state, .01)
        outputs.append(model._read_state_tensor(state)[0])
        held = stimulus if 50 <= tick < 90 else base
    return torch.stack(outputs).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibration = Path(args.calibration)
    receipt = read(calibration / "result.json")
    original_root = Path(receipt["source_training"])
    original_result = read(original_root / "result.json")
    original_config = read(original_root / "manifest.json")["config"]
    data_root = Path(original_config["arguments"]["data"])
    spec = original_config["plan"]["spec"]
    if (receipt["status"] != "completed_model_calibration"
            or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
            or sha256(data_root / "dataset.json") != original_config["dataset_manifest_sha256"]
            or sha256(spec) != original_config["spec_sha256"]
            or sha256(original_result["checkpoint"]) != original_result["checkpoint_sha256"]):
        raise ValueError("Unchanged calibrated/original model and complete dataset required")
    dataset = read(data_root / "dataset.json")
    # Entire new contexts/amplitudes are declared here, before their responses are evaluated.
    contexts = [(1, 100), (2, 300), (3, 200), (4, 400)]
    yaw = np.array([0., -.22, -.07, .07, .22], np.float32)
    run = begin_run(args.output, dict(arguments=vars(args), seed=863121,
        calibrated_checkpoint_sha256=receipt["checkpoint_sha256"],
        original_checkpoint_sha256=original_result["checkpoint_sha256"],
        contexts=contexts, added_yaw_rad_s=yaw.tolist(), pulse_indices=[50, 90],
        measurement_indices=[65, 90], dt=.01, model_neutral_steps=200,
        expected="all added yaw pulses have opposing delta left-turn intent",
        response_epsilon=1e-5, lesion_delta_limit=1e-6,
        scope="heldout model inputs; no real perturbation, plant, blind course or fly physiology",
        actuator_messages_sent=0, optimizer_updates=0))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    torch.set_num_threads(1)
    original = torch.load(original_result["checkpoint"], map_location="cpu", weights_only=True)
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    for key, value in original["model"].items():
        changed = payload["model"][key]
        if key == "body.weight":
            torch.testing.assert_close(value[:, [0, 1, 3, 4, 5]], changed[:, [0, 1, 3, 4, 5]],
                                       atol=0, rtol=0)
        elif value.is_sparse:
            if (not torch.equal(value.coalesce().indices(), changed.coalesce().indices())
                    or not torch.equal(value.coalesce().values(), changed.coalesce().values())):
                raise ValueError("Original graph changed")
        elif not torch.equal(value, changed):
            raise ValueError("Unrelated fitted parameter/buffer changed: " + key)
    model = FlightCircuit(spec, glutamate_sign=original_config["plan"]["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    episodes = []
    for entry in dataset["episodes"]:
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original complete episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        episodes.append((entry, data, schedule, arrays))
    reports, responses = [], []
    for number, index in contexts:
        entry, _, _, arrays = episodes[number]
        full = pulse(model, arrays, index, entry["desired_speed_kmh"], yaw)
        removed = pulse(model, arrays, index, entry["desired_speed_kmh"], yaw, body_lesion=True)
        delta = full[:, :, 0] - full[:, :1, 0]
        means = delta[65:90].mean(0)
        flags = (-np.sign(yaw[1:]) * means[1:] > 1e-5).tolist()
        lesion = float(np.abs(removed - removed[:, :1]).max())
        reports.append(dict(file=entry["file"], index=index, role=entry["role"],
                            speed_kmh=entry["desired_speed_kmh"], mean_delta=means.tolist(),
                            signs_passed=flags, body_lesion_delta_max=lesion))
        responses.append(full)
    # Independent functional-zero-edges counterexample, without exporting a modified graph.
    intact = model.circuit.matrix
    model.circuit.matrix = intact * 0
    entry, _, _, arrays = episodes[contexts[0][0]]
    zero = pulse(model, arrays, contexts[0][1], entry["desired_speed_kmh"], yaw)
    zero_delta = float(np.abs(zero - zero[:, :1]).max())
    model.circuit.matrix = intact
    real_metrics, _ = evaluate(model, episodes)
    np.savez_compressed(run / "withheld-pulse-responses.npz", responses=np.stack(responses),
                        yaw=yaw)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for panel, report, response in zip(axes.flat, reports, responses, strict=True):
        for index in range(1, len(yaw)):
            panel.plot(np.arange(150) * .01, response[:, index, 0] - response[:, 0, 0],
                       label=f"added yaw {yaw[index]:+.2f}")
        panel.axvspan(.5, .9, color="grey", alpha=.12)
        panel.set(xlabel="Model time (s)", ylabel="Delta left-turn intent",
                  title=f"{report['file']} index{report['index']} / {report['speed_kmh']}km/h")
        panel.legend(fontsize=7)
    figure.savefig(run / "withheld-yaw-feedback.png", dpi=150)
    plt.close(figure)
    write_json(run / "result.json", dict(status="completed", contexts=reports,
        all_withheld_signs_passed=all(all(r["signs_passed"]) for r in reports),
        body_lesion_invariance_passed=all(r["body_lesion_delta_max"] <= 1e-6 for r in reports),
        zero_graph_delta_max=zero_delta, zero_graph_invariance_passed=zero_delta <= 1e-6,
        only_yaw_column_changed=True, full_original_episode_metrics=real_metrics,
        source_metrics=original_result["final_metrics"], navigation_failure_not_reclassified=True,
        physiological_stabilization_verified=False, physical_vehicle_plant_tested=False,
        actor_qualified=False, actuator_messages_sent=0, optimizer_updates=0, goal_achieved=False,
        termination_reason="withheld model signs and original episode tradeoffs measured"))
    print(json.dumps(dict(signs=[r["signs_passed"] for r in reports], zero_graph=zero_delta)))


if __name__ == "__main__":
    main()
