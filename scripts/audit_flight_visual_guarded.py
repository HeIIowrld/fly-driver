"""Strict replay of train-only guarded v6 and archived aid-admission consumer evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_csp_vision import validate_frame_aid_state
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.audit_flight_fresh_tracking import describe
from scripts.audit_flight_speed_training import partitioned
from scripts.train_flight_initial import replay, substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--consumer", help="Optional separate archived camera consumer audit")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    consumer = Path(args.consumer) if args.consumer else None
    receipt, config = read(root / "result.json"), read(root / "manifest.json")["config"]
    plan = config["plan"]
    data_root = Path(plan["data"])
    if (receipt["status"] != "completed_guarded_optimizer_experiment"
            or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
            or sha256(data_root / "dataset.json") != config["dataset_manifest_sha256"]
            or sha256(plan["spec"]) != config["spec_sha256"]):
        raise ValueError("Completed unchanged guarded fit and original data required")
    run = begin_run(args.output, dict(arguments=vars(args), seed=863154,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        training_manifest_sha256=sha256(root / "manifest.json"),
        consumer_result_sha256=sha256(consumer / "result.json") if consumer else None,
        development_arrays_opened=False, optimizer_updates=0, actuator_messages_sent=0))
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    parent_receipt = read(Path(plan["source"]) / "result.json")
    if sha256(parent_receipt["checkpoint"]) != config["source_checkpoint_sha256"]:
        raise ValueError("Original parent changed")
    parent = torch.load(parent_receipt["checkpoint"], map_location="cpu", weights_only=True)
    for key, before in parent["model"].items():
        after = payload["model"][key]
        if before.is_sparse:
            assert torch.equal(before.coalesce().indices(), after.coalesce().indices()), key
            assert torch.equal(before.coalesce().values(), after.coalesce().values()), key
        else:
            assert torch.equal(before, after), key
    model = VisualFlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"]).to("cuda")
    model.load_state_dict(payload["model"], strict=True)
    model.assert_body_boundary_protected()
    reports, sources, moments = [], [], []
    for entry in read(data_root / "dataset.json")["episodes"]:
        if entry["role"] != "train":
            continue
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Original training episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        moments.append(arrays["scene_features"].astype(np.float64))
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        prediction = replay(model, data, schedule)[0].cpu().numpy()
        split = partitioned(model, data, schedule)
        np.testing.assert_allclose(prediction, split, atol=3e-6, rtol=2e-5)
        removed = replay(model, data, schedule, lesion="scene_command")[0].cpu().numpy()
        with load_array(root / entry["file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["observation_time"], arrays["observation_time"])
            np.testing.assert_array_equal(saved["target_valid"], arrays["target_valid"])
            np.testing.assert_array_equal(saved["target"], arrays["target"])
            np.testing.assert_allclose(prediction, saved["prediction"], atol=3e-6, rtol=2e-5)
            error = float(np.abs(prediction - saved["prediction"]).max())
        reports.append(dict(file=entry["file"], saved_prediction_error=error,
                            chunk37_error=float(np.abs(split - prediction).max())))
        sources.append((entry, arrays, prediction, removed))
    features = np.concatenate(moments)
    mean, scale = features.mean(0), features.std(0).clip(min=plan["normalization_std_floor"])
    np.testing.assert_allclose(model.scene_command.mean.cpu().numpy(), mean, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(model.scene_command.scale.cpu().numpy(), scale, atol=1e-6, rtol=1e-5)
    history = read(root / "history.json")
    last = history[0]["step"]["before"]
    for row in history:
        step = row["step"]
        assert abs(step["before"] - last) < 1e-5
        if step["accepted"]:
            last = step["trials"][-1]["loss"]
            assert last < step["before"]
    y = np.concatenate([a["target"][a["target_valid"][:, 0], 0] for _, a, _, _ in sources])
    p = np.concatenate([v[a["target_valid"][:, 0], 0] for _, a, v, _ in sources])
    train_strata = {name: describe(p, y, mask) for name, mask in [
        ("left", y > .01), ("right", y < -.01), ("near_neutral", np.abs(y) <= .01)]}
    if receipt.get("turn_group_counts") is not None:
        assert [train_strata[n]["count"] for n in ("left", "right", "near_neutral")] == (
            receipt["turn_group_counts"])
    consumer_report = None
    if consumer is not None:
        consumer_receipt = read(consumer / "result.json")
        for record in read(consumer / "vision/archive.json"):
            path = consumer / "vision" / record["file"]
            assert path.stat().st_size == record["bytes"]
            assert sha256(path) == record["sha256"]
        stream_lines = (consumer / "vision/stream.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in stream_lines]
        for row in rows:
            validate_frame_aid_state(row)
            frame_path = consumer / row["file"]
            assert sha256(frame_path) == row["file_sha256"]
            frame = read(frame_path)
            validate_frame_aid_state(frame)
            assert frame["index"] == row["capture_id"]
            assert {view["view_id"] for view in row["views"]} == {"front", "left", "right", "rear"}
            for view in row["views"]:
                assert sha256(consumer / view["file"]) == view["file_sha256"]
                with Image.open(consumer / view["file"]) as image:
                    rgb = np.array(image.convert("RGB"))
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == view["rgb_sha256"]
        events = [json.loads(line) for line in (consumer / "events.jsonl").read_text().splitlines()]
        for event in events:
            validate_frame_aid_state(event["camera"])
            row = rows[event["camera"]["capture_id"]]
            assert event["camera"]["file_sha256"] == row["file_sha256"]
            assert event["foreground"] is False
        assert len(rows) == consumer_receipt["vision"]["frames"]
        assert len(events) == consumer_receipt["decisions"]
        consumer_report = dict(source_collection_status=consumer_receipt["status"],
            source_collection_error=consumer_receipt.get("reason"), archive_hashes_verified=True,
            admitted_off_frames=len(rows), recorded_decisions=len(events),
            recording_training_admissible=False)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(sources), 1, figsize=(11, 8), layout="constrained")
    for panel, (entry, arrays, prediction, removed) in zip(axes, sources, strict=True):
        time = arrays["observation_time"] - arrays["observation_time"][0]
        target = np.where(arrays["target_valid"][:, 0], arrays["target"][:, 0], np.nan)
        panel.plot(time, target, label="Applied-control target", linewidth=.8)
        panel.plot(time, prediction[:, 0], label="Guarded v6", linewidth=.8)
        panel.plot(time, removed[:, 0], label="New scene information removed", linewidth=.7)
        panel.set(xlabel="Original elapsed time (s)", ylabel="Left-turn intent",
                  title=f"{entry['file']} TRAIN ONLY; no driving/generalization qualification")
        panel.legend(fontsize=7)
    figure.savefig(run / "guarded-train-fit.png", dpi=130)
    plt.close(figure)
    write_json(run / "result.json", dict(status="completed", train_replays=reports,
        all_v5_inherited_state_unchanged=True, train_only_normalization_verified=True,
        full_train_loss_initial=history[0]["step"]["before"], full_train_loss_final=last,
        accepted_steps=sum(row["step"]["accepted"] for row in history),
        original_clocks_and_targets_preserved=True, development_arrays_opened=False,
        train_strata=train_strata, consumer=consumer_report,
        turn_objective=receipt.get("turn_objective", "per_label_mean"),
        actor_qualified=False, physiological_recognition_verified=False,
        physical_vehicle_plant_tested=False, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", train_replays=len(reports))))


if __name__ == "__main__":
    main()
