"""Counterfactual yaw pulses at a fixed scene: test engineering feedback sign, never drive."""

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


@torch.no_grad()
def experiment(model, arrays, yaw, lesion):
    count = len(yaw)
    inputs = {key: torch.tensor(np.repeat(arrays[key][:1], count, axis=0), device="cuda")
              for key in ("visual", "visual_valid", "body", "body_valid", "scene_features")}
    inputs["body"][:, :5] = 0
    goals = torch.tensor(speed_goals(15, count), device="cuda")

    def drive():
        return model._drive_tensor(inputs["visual"], inputs["visual_valid"], inputs["body"],
                                   inputs["body_valid"], goals, inputs["scene_features"],
                                   lesion=lesion)

    baseline = drive()
    inputs["body"][:, 4] = torch.tensor(yaw, device="cuda")
    pulse = drive()
    state = model.initial_state(count)
    for _ in range(200):
        state = model._advance_elapsed_tensor(baseline, state, .01)
    outputs, held = [], baseline
    for index in range(150):
        state = model._advance_elapsed_tensor(held, state, .01)
        outputs.append(model._read_state_tensor(state)[0])
        held = pulse if 50 <= index < 90 else baseline
    return torch.stack(outputs).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    result, configuration = read(root / "result.json"), read(root / "manifest.json")["config"]
    checkpoint = Path(result["checkpoint"])
    spec = configuration["plan"]["spec"]
    if (result["status"] != "completed_offline_initialization"
            or sha256(checkpoint) != result["checkpoint_sha256"]
            or sha256(spec) != configuration["spec_sha256"]):
        raise ValueError("Successful unchanged candidate required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["goal_contract"] != GOAL_CONTRACT:
        raise ValueError("Explicit v4 task semantics required")
    data_root = Path(configuration["arguments"]["data"])
    if sha256(data_root / "dataset.json") != configuration["dataset_manifest_sha256"]:
        raise ValueError("Original source dataset changed")
    entry = read(data_root / "dataset.json")["episodes"][0]
    if sha256(data_root / entry["file"]) != entry["sha256"]:
        raise ValueError("Original fixed-scene source changed")
    with load_array(data_root / entry["file"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    # Expectations declared before any intervention. These are car feedback hypotheses,
    # not a claim that fly yaw physiology or the real car plant has been reproduced.
    yaw = np.array([0, -.3, -.1, .1, .3], np.float32)
    run = begin_run(args.output, dict(arguments=vars(args), seed=863110,
        checkpoint_sha256=sha256(checkpoint), yaw_pulses_rad_s=yaw.tolist(),
        dt=.01, model_neutral_steps=200, pulse_indices=[50, 90], measurement_indices=[65, 90],
        task_goal=speed_goals(15, 1)[0].tolist(),
        expected="delta left-turn intent opposes injected yaw for all four signed pulses",
        response_epsilon=1e-5, lesion_delta_limit=1e-6,
        scope="fixed actual-scene snapshot, artificial model-only yaw pulses; no physical plant",
        actuator_messages_sent=0))
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    model = FlightCircuit(spec, glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    full, removed = experiment(model, arrays, yaw, None), experiment(model, arrays, yaw, "body")
    delta = full[:, :, 0] - full[:, :1, 0]
    lesion_delta = removed[:, :, 0] - removed[:, :1, 0]
    average = delta[65:90].mean(0)
    sign_passed = (-np.sign(yaw[1:]) * average[1:] > 1e-5).tolist()
    lesion_error = float(np.abs(lesion_delta).max())
    np.savez_compressed(run / "yaw-pulse-responses.npz", yaw=yaw, full=full, body_removed=removed,
                        delta_left_intent=delta, body_removed_delta=lesion_delta)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for index in range(1, len(yaw)):
        axis.plot(np.arange(150) * .01, delta[:, index], label=f"yaw {yaw[index]:+.1f} rad/s")
    axis.axvspan(.5, .9, alpha=.12, color="grey", label="Artificial pulse")
    axis.axhline(0, linewidth=.5, color="black")
    axis.set(xlabel="Model time (s)", ylabel="Delta left-turn intent",
             title="Fixed-scene body feedback; stabilizing response should oppose yaw")
    axis.legend(fontsize=8)
    figure.savefig(run / "yaw-feedback-sign.png", dpi=150)
    plt.close(figure)
    write_json(run / "result.json", dict(status="completed", yaw_pulses_rad_s=yaw.tolist(),
        mean_delta_left_intent=average.tolist(), stabilizing_sign_passed=sign_passed,
        all_stabilizing_signs_passed=all(sign_passed), body_lesion_delta_max=lesion_error,
        body_lesion_invariance_passed=lesion_error <= 1e-6,
        physiological_stabilization_verified=False, physical_vehicle_plant_tested=False,
        goal_achieved=False, actor_qualified=False, actuator_messages_sent=0,
        termination_reason="predeclared counterfactual body feedback signs measured"))
    print(json.dumps(dict(signs_passed=sign_passed, delta=average.tolist())))


if __name__ == "__main__":
    main()
