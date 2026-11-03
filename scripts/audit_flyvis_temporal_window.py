"""Independent CPU audit of actual temporal-window qualifier records; no model execution."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    root = Path(a.run)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "executed-audit.py").write_bytes(Path(__file__).read_bytes())
    result = read(root / "result.json")
    manifest = read(root / "manifest.json")
    plan = manifest["config"]["plan"]
    assert (
        result["status"] == "completed_author_temporal_window_qualifier" and plan["revision"] == 4
    )
    assert (
        sha(root / "executed-program.py")
        == manifest["config"]["program_sha256"]
        == plan["input_sha256"]["scripts/probe_flyvis_temporal_window.py"]
    )
    tree = ast.parse((root / "executed-program.py").read_text(encoding="utf-8-sig"))
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "step"
        for n in ast.walk(tree)
    )
    for key in (
        "optimizer_updates",
        "readout_arrays_opened",
        "target_arrays_opened",
        "development_arrays_opened",
        "evaluation_arrays_opened",
        "runtime_mappings_opened",
        "control_messages_sent",
        "ui_actions_sent",
    ):
        assert result[key] == 0
    sensor = read(Path(plan["sensor_replay"]) / "result.json")
    entry = next(e for e in sensor["episodes"] if e["id"] == plan["train_episode"])
    assert entry["role"] == "train"
    response = Path(entry["output"]) / "responses.npz"
    assert sha(response) == entry["responses_sha256"]
    with load_array(response, allow_pickle=False) as arrays:
        clock = arrays["decision_monotonic_seconds"][:65].copy()
        retina = arrays["retina_intensity"][:65].copy()
        expected = np.concatenate(
            [arrays["additional_activity"][:65], arrays["motion_activity"][:65]], axis=2
        )
    reference = read(root / "original-prefix-reference.json")
    np.testing.assert_array_equal(clock, reference["original_clocks"])
    np.testing.assert_array_equal(np.diff(clock), reference["intervals"])
    assert hashlib.sha256(retina.tobytes()).hexdigest() == reference["retina_prefix_sha256"]
    assert (
        hashlib.sha256(expected.tobytes()).hexdigest()
        == reference["reference_selected_prefix_sha256"]
    )
    arms = {
        name: read(root / (name + ".json"))
        for name in ("detach2_ordinary", "detach2_repeat", "detach8_ordinary")
    }
    flags = read(root / "author-implementation.json")["original_trainable_flags"]
    free = [name for name, value in flags.items() if value]
    comparisons = []
    logical_ticks = 0
    for gap in np.diff(clock):
        count = int(np.floor(gap / 0.005))
        logical_ticks += count + int(gap - count * 0.005 > 1e-12)
    for name, arm in arms.items():
        assert (
            arm["original_intervals"] == 64
            and arm["prefix_observations"] == 65
            and arm["logical_internal_ticks"] == logical_ticks == 576
        )
        assert (
            arm["exact_detach_boundaries"] == 64 // arm["detach_intervals"]
            and arm["recurrent_numeric_resets"] == 0
            and arm["initial_observation_retained"]
        )
        assert (
            not arm["author_parameter_changes"]
            and arm["peak_gpu_tensor_bytes"] <= plan["maximum_peak_gpu_tensor_bytes"]
        )
        assert len(arm["decisions"]) == 64 and len(arm["windows"]) == 8
        max_reference_mean_error = 0.0
        for index, d in enumerate(arm["decisions"], 1):
            assert (
                d["index"] == index
                and d["held_retina_index"] == index - 1
                and d["original_clock"] == clock[index]
                and d["elapsed"] == clock[index] - clock[index - 1]
            )
            assert (
                len(d["state_hashes"]) == 5
                and d["source_neural_max_error"] <= plan["reference_absolute_tolerance"]
            )
            assert (
                max(d["baseline_full_state_max_errors"].values())
                <= (plan["cross_arm_absolute_tolerance"])
            )
            assert (
                d["baseline_selected_activity_max_error"] <= (plan["cross_arm_absolute_tolerance"])
            )
            max_reference_mean_error = max(
                max_reference_mean_error,
                float(abs(np.array(d["view_channel_means"]) - expected[index].mean(-1)).max()),
            )
        assert max_reference_mean_error <= plan["reference_absolute_tolerance"]
        for index, w in enumerate(arm["windows"]):
            assert (
                w["start"] == index * 8
                and w["end"] == index * 8 + 8
                and w["optimizer_updates"] == 0
                and w["detached_state_tensors"] == 5
            )
            assert set(w["gradients"]) == set(free)
            values = np.concatenate([w["gradients"][n] for n in free])
            assert (
                len(values) == w["gradient_elements"] == 734
                and np.isfinite(values).all()
                and np.count_nonzero(values) == w["nonzero_gradients"] > 0
            )
            expected_objective = sum(
                d["diagnostic_scalar"] / 8 for d in arm["decisions"][index * 8 : index * 8 + 8]
            )
            assert expected_objective == w["accumulated_diagnostic_objective"]
        comparisons.append(
            {
                "arm": name,
                "prefix_observations": 65,
                "original_intervals": 64,
                "logical_internal_ticks": logical_ticks,
                "detach_boundaries": arm["exact_detach_boundaries"],
                "peak_gpu_tensor_bytes": arm["peak_gpu_tensor_bytes"],
                "maximum_independent_reference_mean_error": max_reference_mean_error,
                "maximum_original_neural_error": max(
                    d["source_neural_max_error"] for d in arm["decisions"]
                ),
                "window_nonzero_gradient_counts": [w["nonzero_gradients"] for w in arm["windows"]],
            }
        )
    short = arms["detach2_ordinary"]
    long = arms["detach8_ordinary"]
    assert (
        short["initial_state_hashes"] == long["initial_state_hashes"]
        and short["initial_diagnostic_scalar"] == long["initial_diagnostic_scalar"]
    )
    for short_row, long_row in zip(short["decisions"], long["decisions"], strict=True):
        for key in ("original_clock", "held_retina_index", "elapsed"):
            assert short_row[key] == long_row[key]
        assert (
            abs(short_row["diagnostic_scalar"] - long_row["diagnostic_scalar"])
            <= (plan["cross_arm_absolute_tolerance"])
        )
    differences = []
    for short_row, repeat_row, long_row in zip(
        short["windows"], arms["detach2_repeat"]["windows"], long["windows"], strict=True
    ):
        sv = np.concatenate([short_row["gradients"][n] for n in free])
        lv = np.concatenate([long_row["gradients"][n] for n in free])
        repeated = np.concatenate([repeat_row["gradients"][n] for n in free])
        np.testing.assert_allclose(
            sv,
            repeated,
            rtol=plan["gradient_relative_tolerance"],
            atol=plan["gradient_absolute_tolerance"],
        )
        repeat_noise = float(abs(sv - repeated).max())
        credit_difference = float(abs(sv - lv).max())
        assert credit_difference > 10 * (repeat_noise + 1e-12)
        differences.append(
            {
                "start": short_row["start"],
                "end": short_row["end"],
                "maximum_absolute_gradient_difference": credit_difference,
                "repeated_short_gradient_max_difference": repeat_noise,
                "short_norm": float(np.linalg.norm(sv)),
                "long_norm": float(np.linalg.norm(lv)),
            }
        )
    assert any(d["maximum_absolute_gradient_difference"] > 0 for d in differences)
    failures = []
    for path in plan["previous_failed_runs"]:
        p = Path(path)
        old = read(p / "result.json")
        assert old["status"] == "failed"
        failures.append(
            {
                "run": str(p),
                "result_sha256": sha(p / "result.json"),
                "reason": old["reason"],
                "optimizer_updates": old["optimizer_updates"],
            }
        )
    summary = {
        "status": "passed_independent_author_temporal_window_audit",
        "source_run": str(root),
        "source_result_sha256": sha(root / "result.json"),
        "executed_source_sha256": sha(root / "executed-program.py"),
        "source_train_response_sha256": entry["responses_sha256"],
        "observations": 65,
        "intervals": 64,
        "elapsed_seconds": float(clock[-1] - clock[0]),
        "within_arm_detach_numeric_values_bitwise_exact": True,
        "cross_arm_floating_states_within_predeclared_tolerance": True,
        "cross_arm_bitwise_equality": result["numerical_state_output_hashes_exact_across_arms"],
        "maximum_recorded_cross_arm_state_error": max(
            max(d["baseline_full_state_max_errors"].values())
            for arm in arms.values()
            for d in arm["decisions"]
        ),
        "maximum_recorded_cross_arm_selected_activity_error": max(
            d["baseline_selected_activity_max_error"]
            for arm in arms.values()
            for d in arm["decisions"]
        ),
        "arms": comparisons,
        "gradient_credit_differences": differences,
        "original_parameter_values_unchanged": result["original_parameter_values_unchanged"],
        "original_trainable_elements": 734,
        "fixed_parameter_elements": 2959,
        "optimizer_updates": 0,
        "readout_arrays_opened": 0,
        "target_arrays_opened": 0,
        "development_arrays_opened": 0,
        "evaluation_arrays_opened": 0,
        "gpu_used_by_audit": False,
        "control_messages_sent": 0,
        "preserved_failed_recomputation_attempts": failures,
        "limitations": [
            (
                "Qualifies actual author network/local diagnostic on the declared "
                "TRAIN65observation prefix only."
            ),
            (
                "Memory qualification excludes registered feature/readout/actual6target "
                "objective and wholeTRAIN histories."
            ),
            (
                "No optimization, predictive generalization, physiological certification, "
                "motor transfer, livevisual driving, or validlap."
            ),
            (
                "Activation checkpoint remains unqualified; failures retained without "
                "disabled checks/cache hacks."
            ),
        ],
    }
    (out / "result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
