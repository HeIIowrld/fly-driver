"""Fixed nonlinear information controls; not a biological driver or actor."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def metrics(prediction, target, valid):
    if not valid.any():
        return {"count": 0}
    p, y = prediction[valid], target[valid]
    return dict(
        count=int(valid.sum()),
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        prediction_mean=float(p.mean()),
        target_mean=float(y.mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
    root = Path(plan["data"])
    assert torch.cuda.is_available(), "GPU experiment requires GPU; no silent CPU fallback"
    device = torch.device("cuda:0")
    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_sha256=sha256(args.plan),
            script_sha256=sha256(__file__),
            dataset_sha256=sha256(root / "dataset.json"),
            torch_version=torch.__version__,
            device=torch.cuda.get_device_name(0),
        ),
    )
    episodes = []
    for entry in json.loads((root / "dataset.json").read_text(encoding="utf8"))["episodes"]:
        assert entry["role"] in ("train", "development"), "Evaluation role forbidden"
        assert sha256(root / entry["file"]) == entry["sha256"]
        with load_array(root / entry["file"], allow_pickle=False) as archive:
            a = {
                key: archive[key].copy()
                for key in ("scene_features", "target", "target_valid", "observation_time")
            }
        assert a["scene_features"].shape == (len(a["target"]), 576)
        episodes.append((entry, a))
    train_x = np.concatenate(
        [a["scene_features"][a["target_valid"][:, 0]] for e, a in episodes if e["role"] == "train"]
    ).astype(np.float64)
    train_y = np.concatenate(
        [a["target"][a["target_valid"][:, 0], 0] for e, a in episodes if e["role"] == "train"]
    ).astype(np.float32)
    mean = train_x.mean(0)
    scale = np.maximum(train_x.std(0), plan["normalization_std_floor"])

    def normalize(x):
        return np.clip((x - mean) / scale, -plan["feature_clip"], plan["feature_clip"]).astype(
            np.float32
        )

    x = torch.as_tensor(normalize(train_x), device=device)
    y = torch.as_tensor(train_y, device=device)
    predictions, reports, histories = {}, {}, {}
    for variant in plan["variants"]:
        torch.manual_seed(plan["seed"])
        torch.cuda.manual_seed_all(plan["seed"])
        if variant == "linear_tanh":
            model = nn.Sequential(nn.Linear(576, 1), nn.Tanh())
        elif variant == "hidden64_tanh":
            model = nn.Sequential(
                nn.Linear(576, plan["hidden_units"]),
                nn.Tanh(),
                nn.Linear(plan["hidden_units"], 1),
                nn.Tanh(),
            )
        else:
            raise ValueError(variant)
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"]
        )
        losses = []
        for _epoch in range(plan["epochs"]):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x).squeeze(-1) - y) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        torch.save(model.state_dict(), run / f"{variant}.pt")
        model.eval()
        reports[variant] = []
        with torch.no_grad():
            for entry, a in episodes:
                normalized = normalize(a["scene_features"])
                result = dict(file=entry["file"], role=entry["role"], conditions={})
                valid = a["target_valid"][:, 0]
                late = valid & (a["observation_time"] - a["observation_time"][0] >= 3)
                for condition in plan["counterfactuals"]:
                    inputs = normalized.copy()
                    channels = inputs.reshape(len(inputs), 4, 9, 16)
                    if condition == "motion_train_mean":
                        channels[:, :, 1:] = 0
                    elif condition == "intensity_train_mean":
                        channels[:, :, :1] = 0
                    elif condition == "all_train_mean":
                        inputs[:] = 0
                    elif condition != "intact":
                        raise ValueError(condition)
                    estimate = (
                        model(torch.as_tensor(inputs, device=device)).squeeze(-1).cpu().numpy()
                    )
                    predictions[f"{variant}_{condition}_{entry['file'].removesuffix('.npz')}"] = (
                        estimate
                    )
                    result["conditions"][condition] = dict(
                        full=metrics(estimate, a["target"][:, 0], valid),
                        after3s=metrics(estimate, a["target"][:, 0], late),
                        left=metrics(
                            estimate, a["target"][:, 0], valid & (a["target"][:, 0] > 0.01)
                        ),
                        right=metrics(
                            estimate, a["target"][:, 0], valid & (a["target"][:, 0] < -0.01)
                        ),
                    )
                baseline = np.full(len(a["target"]), train_y.mean())
                result["train_mean_baseline"] = metrics(baseline, a["target"][:, 0], valid)
                reports[variant].append(result)
        histories[variant] = losses
        print(
            json.dumps(
                dict(
                    variant=variant,
                    updates=len(losses),
                    first_loss=losses[0],
                    final_preupdate_loss=losses[-1],
                )
            ),
            flush=True,
        )
    np.savez_compressed(run / "predictions.npz", **predictions)
    np.savez_compressed(run / "normalization.npz", mean=mean, scale=scale)
    for entry, _ in episodes:
        assert sha256(root / entry["file"]) == entry["sha256"]
    write_json(
        run / "result.json",
        dict(
            status="completed_exploratory_information_diagnosis",
            metrics=reports,
            training_loss=histories,
            train_rows=len(train_y),
            train_target_mean=float(train_y.mean()),
            gpu=torch.cuda.get_device_name(0),
            maximum_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            diagnostic_optimizer_updates=plan["epochs"] * len(plan["variants"]),
            fly_circuit_optimizer_updates=0,
            evaluation_role_arrays_opened=False,
            source_arrays_or_times_modified=False,
            diagnostic_only_no_deployment=True,
            ui_actions_sent=0,
            actuator_messages_sent=0,
            goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    main()
