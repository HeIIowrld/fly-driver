"""Audit original timing and cross-history scene coverage without refitting labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return dict(count=0)
    return dict(
        count=len(values),
        minimum=float(values.min()),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(values.max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan["data"])
    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_sha256=sha256(args.plan),
            script_sha256=sha256(__file__),
            dataset_sha256=sha256(root / "dataset.json"),
        ),
    )
    episodes, provenance = [], []
    for entry in read(root / "dataset.json")["episodes"]:
        assert entry["role"] in ("train", "development")
        assert sha256(root / entry["file"]) == entry["sha256"]
        with load_array(root / entry["file"], allow_pickle=False) as archive:
            a = {k: archive[k].copy() for k in archive.files}
        original = entry["original"]
        recording = Path(original["recording"])
        events_file = recording / "events.jsonl"
        assert sha256(events_file) == entry["source_events_sha256"]
        events = [json.loads(line) for line in events_file.read_text(encoding="utf8").splitlines()]
        np.testing.assert_array_equal(
            a["observation_time"], [e["decision_monotonic_seconds"] for e in events]
        )
        seal = read(Path(original["teacher_dataset"]) / "manifest.json")["config"]["configuration"]
        sources = [s for s in seal["sources"] if Path(s["recording"]) == recording]
        assert len(sources) == 1
        targets = Path(sources[0]["targets"]) / "proposal-targets.npz"
        assert sha256(targets) == read(targets.parent / "result.json")["target_sha256"]
        with load_array(targets, allow_pickle=False) as archive:
            np.testing.assert_array_equal(a["actions"], archive["action"])
            np.testing.assert_array_equal(a["action_valid"], archive["action_valid"])
            receipt = archive["label_receipt_time"].copy()
        valid = a["target_valid"][:, 0]
        assert np.isfinite(receipt[valid]).all()
        render = np.array(
            [[v["render_started_lower"] for v in e["camera"]["views"]] for e in events]
        )
        decision = a["observation_time"]
        assert np.all(render <= decision[:, None])
        timing = dict(
            decision_minus_render_lower_seconds=describe((decision[:, None] - render).ravel()),
            label_receipt_minus_decision_seconds=describe(receipt[valid] - decision[valid]),
            label_receipt_minus_render_lower_seconds=describe(
                (receipt[valid, None] - render[valid]).ravel()
            ),
            decision_interval_seconds=describe(np.diff(decision)),
            exact_physical_exposure_or_internal_ai_command_time_known=False,
        )
        selected_target = a["target"][valid, 0]
        strata = dict(
            left=int((selected_target > 0.01).sum()),
            right=int((selected_target < -0.01).sum()),
            neutral=int((np.abs(selected_target) <= 0.01).sum()),
            brake_positive=int(((a["actions"][:, 2] > 0) & a["action_valid"][:, 2]).sum()),
        )
        yaw = []
        for lag in plan["yaw_comparison_lags_seconds"]:
            query = decision + lag
            right = np.searchsorted(decision, query).clip(0, len(query) - 1)
            left = (right - 1).clip(0, len(query) - 1)
            index = np.where(
                abs(decision[right] - query) < abs(decision[left] - query), right, left
            )
            matched = (
                valid
                & a["body_valid"][index, 4]
                & (abs(decision[index] - query) <= plan["yaw_match_tolerance_seconds"])
            )
            x, y = a["target"][matched, 0], a["body"][index[matched], 4]
            correlation = float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-9 else None
            yaw.append(dict(lag_seconds=lag, count=int(matched.sum()), correlation=correlation))
        episodes.append(
            (
                entry,
                a,
                dict(
                    file=entry["file"],
                    role=entry["role"],
                    timing=timing,
                    strata=strata,
                    target_vs_body_yaw=yaw,
                ),
            )
        )
        provenance.append(dict(events_sha256=sha256(events_file), targets_sha256=sha256(targets)))
    train_x = np.concatenate(
        [
            a["scene_features"][a["target_valid"][:, 0]]
            for e, a, _ in episodes
            if e["role"] == "train"
        ]
    ).astype(np.float64)
    train_y = np.concatenate(
        [a["target"][a["target_valid"][:, 0], 0] for e, a, _ in episodes if e["role"] == "train"]
    )
    owners = np.concatenate(
        [
            np.full(int(a["target_valid"][:, 0].sum()), i)
            for i, (e, a, _) in enumerate(episodes)
            if e["role"] == "train"
        ]
    )
    mean, scale = train_x.mean(0), np.maximum(train_x.std(0), plan["normalization_std_floor"])

    def normalize(x):
        return np.clip((x - mean) / scale, -plan["feature_clip"], plan["feature_clip"])

    normalized_train = normalize(train_x)
    stored = {}
    for i, (entry, a, report) in enumerate(episodes):
        valid = a["target_valid"][:, 0]
        x = normalize(a["scene_features"][valid])
        allowed = owners != i if entry["role"] == "train" else np.ones(len(train_y), bool)
        candidates, labels = normalized_train[allowed], train_y[allowed]
        distances, nearest = [], []
        for start in range(0, len(x), 64):
            query = x[start : start + 64]
            squared = (
                np.square(query).sum(1)[:, None]
                + np.square(candidates).sum(1)[None]
                - 2 * query @ candidates.T
            ) / 576
            index = squared.argmin(1)
            distances.extend(squared[np.arange(len(index)), index])
            nearest.extend(labels[index])
        prediction = np.asarray(nearest)
        target = a["target"][valid, 0]
        report["cross_history_neighbor"] = dict(
            normalized_distance_squared=describe(distances),
            rmse=float(np.sqrt(np.mean((prediction - target) ** 2))),
            train_mean_rmse=float(np.sqrt(np.mean((train_y.mean() - target) ** 2))),
            absolute_label_difference=describe(abs(prediction - target)),
            note="Diagnostic only: no temporal context, route identity or causal validity claim",
        )
        stored[entry["file"].removesuffix(".npz") + "_neighbor_prediction"] = prediction
        stored[entry["file"].removesuffix(".npz") + "_neighbor_distance_squared"] = np.asarray(
            distances
        )
        print(json.dumps(report), flush=True)
    np.savez_compressed(run / "neighbor-diagnostics.npz", **stored)
    write_json(
        run / "result.json",
        dict(
            status="completed_original_timing_coverage_diagnosis",
            reports=[r for _, _, r in episodes],
            provenance=provenance,
            normalizer_train_only=True,
            evaluation_role_arrays_opened=False,
            original_labels_or_clocks_modified=False,
            optimizer_updates=0,
            ui_actions_sent=0,
            actuator_messages_sent=0,
            goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    main()
