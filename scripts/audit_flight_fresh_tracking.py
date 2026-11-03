"""Post-hoc tracking diagnosis beyond aggregate comparator RMSE; no fitting or control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import replay, substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def describe(prediction, target, selected):
    if not selected.any():
        return dict(count=0)
    p, t = prediction[selected], target[selected]
    return dict(count=int(selected.sum()), rmse=float(np.mean((p - t) ** 2) ** .5),
        prediction_mean=float(p.mean()), target_mean=float(t.mean()),
        prediction_std=float(p.std()), target_std=float(t.std()),
        sign_correct_fraction=float((np.sign(p) == np.sign(t)).mean()),
        zero_turn_rmse=float(np.mean(t ** 2) ** .5),
        correlation=float(np.corrcoef(p, t)[0, 1]) if min(p.std(), t.std()) > 1e-8 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.evaluation)
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    plan = config["plan"]
    training = Path(plan["training"])
    training_config = read(training / "manifest.json")["config"]
    trained = read(training / "result.json")
    if (sha256(trained["checkpoint"]) != config["checkpoint_sha256"]
            or sha256(root / receipt["source_record"]["file"])
            != receipt["source_record"]["sha256"]):
        raise ValueError("Unchanged evaluated checkpoint and complete cached episode required")
    run = begin_run(args.output, dict(arguments=vars(args), seed=863158,
        checkpoint_sha256=config["checkpoint_sha256"], evaluation_result_sha256=sha256(
            root / "result.json"), optimizer_updates=0, actuator_messages_sent=0,
        purpose="post-hoc diagnosis, not a new blind or preregistered success gate",
        turn_threshold=.01, original_prefix_and_times_preserved=True,
        first_scene_control="causal hold of first 576 scene features; same body/goals/clocks; "
                            "explicit model counterfactual, never recorded training data"))
    with load_array(root / receipt["source_record"]["file"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    saved = load_array(root / "prediction.npy")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(1)
    payload = torch.load(trained["checkpoint"], map_location="cpu", weights_only=True)
    model = VisualFlightCircuit(training_config["plan"]["spec"],
                                glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
            if value.dtype.kind in "fbiu" and key != "observation_time"}
    schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
    repeated = replay(model, data, schedule)[0].cpu().numpy()
    np.testing.assert_allclose(repeated, saved, atol=3e-6, rtol=2e-5)
    held_data = dict(data)
    held_data["scene_features"] = data["scene_features"][:1].expand_as(data["scene_features"])
    held = replay(model, held_data, schedule)[0].cpu().numpy()
    np.save(run / "first-scene-held-prediction.npy", held)
    target, selected = arrays["target"][:, 0], arrays["target_valid"][:, 0]
    late = arrays["observation_time"] - arrays["observation_time"][0] >= 3.
    reports = {}
    for name, selection in [("full", selected), ("after3s", selected & late)]:
        reports[name] = dict(actual=describe(repeated[:, 0], target, selection),
                            first_scene_held=describe(held[:, 0], target, selection))
        for label, chosen in [("left", target > .01), ("right", target < -.01),
                              ("near_straight", np.abs(target) <= .01)]:
            reports[name][label] = describe(repeated[:, 0], target, selection & chosen)
    write_json(run / "result.json", dict(status="completed_posthoc_tracking_diagnostic",
        aggregate_comparator_gate_passed=receipt["visual_gate_passed"], diagnostics=reports,
        saved_replay_error=float(np.abs(repeated - saved).max()),
        camera_aid_state_verified=True, source_arrays_modified=False,
        source_clocks_preserved=True, optimizer_updates=0, checkpoint_modified=False,
        visual_path_tracking_verified=False, independent_course_test=False,
        physiological_recognition_verified=False, actual_actor_integration=False,
        actor_qualified=False, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", actual=reports["after3s"]["actual"])))


if __name__ == "__main__":
    main()
