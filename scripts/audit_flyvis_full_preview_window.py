"""CPU audit of full eight-interval every-observation preview graph records."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "executed-audit.py").write_bytes(Path(__file__).read_bytes())
    manifest, result = read(source / "manifest.json"), read(source / "result.json")
    plan = manifest["config"]["plan"]
    assert result["status"] == "completed_full_preview_window_qualifier"
    assert sha(source / "executed-program.py") == manifest["config"]["program_sha256"]
    assert sha(source / "observations.json") == result["observations_sha256"]
    tree = ast.parse((source / "executed-program.py").read_text(encoding="utf-8-sig"))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "step" for n in ast.walk(tree))
    rows = read(source / "observations.json")
    reference = read(source / "original-prefix-reference.json")
    entries = read(Path(plan["registered_data"]) / "dataset.json")["episodes"]
    entry = next(e for e in entries if e["id"] == plan["train_episode"])
    assert entry["role"] == "train"
    assert sha(entry["targets_file"]) == entry["targets_sha256"]
    with load_array(entry["targets_file"], allow_pickle=False) as a:
        clock = a["observation_time"][:65].copy()
        targets = a["preview"][:65].reshape(-1, 6).copy()
        valid = np.repeat(a["preview_valid"][:65], 2, 1)
    body_root = Path(entry["body_replay"])
    proof = read(body_root / "result.json")
    assert sha(body_root / "virtual-body.npz") == proof["virtual_body_sha256"]
    with load_array(body_root / "virtual-body.npz", allow_pickle=False) as a:
        assert hashlib.sha256(a["values"].tobytes()).hexdigest() == (
            reference["body_values_sha256"])
        assert hashlib.sha256(a["valid"].tobytes()).hexdigest() == (
            reference["body_mask_sha256"])
    np.testing.assert_array_equal(clock, reference["original_clocks"])
    np.testing.assert_array_equal(valid, reference["original_target_masks"])
    np.testing.assert_array_equal(np.diff(clock), reference["original_intervals"])
    mean = np.asarray(reference["target_mean"])
    scale = np.asarray(reference["target_scale"])
    weight = np.asarray(reference["per_head_loss_weights"], np.float32)
    max_loss_error = 0.
    for index, row in enumerate(rows):
        assert row["index"] == index and row["original_clock"] == clock[index]
        assert row["held_retina_index"] == (index-1 if index else None)
        assert row["original_elapsed_seconds"] == (
            float(clock[index]-clock[index-1]) if index else None)
        assert row["contributes_gradient"] is (index != 0)
        np.testing.assert_array_equal(valid[index], row["original_target_mask"])
        assert row["neural_max_error"] <= plan["parity_absolute_tolerance"]
        assert row["feature_max_error"] <= plan["parity_absolute_tolerance"]
        inferred = ((np.asarray(row["prediction"])-mean)/scale).astype(np.float32)
        standardized_target = (targets[index].astype(np.float32)-mean.astype(np.float32)) / (
            scale.astype(np.float32))
        loss = (inferred-standardized_target)**2 * valid[index] * weight
        error = float(abs(loss-np.asarray(row["per_head_weighted_loss"])).max())
        max_loss_error = max(max_loss_error, error)
        np.testing.assert_allclose(loss, row["per_head_weighted_loss"], rtol=1e-5, atol=1e-10)
        assert abs(float(loss.sum())-row["summed_weighted_loss"]) <= 1e-9
    assert len(rows) == result["prefix_observations"] == 65
    gradient_counts = np.zeros(6, int)
    for index, window in enumerate(result["windows"]):
        start, end = index*8, index*8+8
        assert window["start"] == start and window["end"] == end
        assert window["original_observation_indices"] == list(range(start+1, end+1))
        counts = valid[start+1:end+1].sum(0)
        np.testing.assert_array_equal(counts, window["gradient_target_counts"])
        gradient_counts += counts
        assert window["exact_detach"] and window["detached_state_tensors"] == 5
        assert window["state_hashes_after_detach"] == rows[end]["numeric_state_hashes"]
        assert abs(sum(r["summed_weighted_loss"] for r in rows[start+1:end+1])
                   - window["summed_weighted_loss"]) <= 1e-9
        assert sum(g["elements"] for g in window["gradients"]) == 734
        assert sum(g["nonzero"] for g in window["gradients"]) == (
            window["nonzero_author_gradient_values"])
        assert all(np.isfinite(g["norm"]) for g in window["gradients"])
        assert window["optimizer_updates"] == 0
        assert window["peak_gpu_tensor_bytes"] < plan["maximum_peak_gpu_tensor_bytes"]
    np.testing.assert_array_equal(gradient_counts, valid[1:].sum(0))
    assert not result["author_parameter_changes"] and not result["readout_parameter_changes"]
    assert result["maximum_total_measured_gpu_bytes"] == max(
        result["initialization_peak_gpu_bytes"], result["every_observation_graph_peak_gpu_bytes"])
    assert result["maximum_total_measured_gpu_bytes"] <= plan["maximum_peak_gpu_tensor_bytes"]
    for key in ("optimizer_updates", "development_arrays_opened", "evaluation_arrays_opened",
                "runtime_mappings_opened", "control_messages_sent", "ui_actions_sent"):
        assert result[key] == 0
    summary = dict(status="passed_independent_full_preview_window_audit", source_run=str(source),
        source_result_sha256=sha(source / "result.json"),
        executed_program_sha256=sha(source / "executed-program.py"),
        original_observations=65, original_intervals=64, logical_internal_ticks=576,
        original_elapsed_seconds=float(clock[-1]-clock[0]),
        all_observation_six_head_losses_independently_recomputed=True,
        per_head_loss_max_recompute_error=max_loss_error,
        original_gradient_target_counts=gradient_counts.tolist(),
        original_clocks_masks_body_exact=True, initial_gray_observation_retained=True,
        exact_detach_boundaries=8, state_tensors_per_boundary=5,
        neural_max_error=result["neural_max_error"], feature_max_error=result["feature_max_error"],
        initialization_peak_gpu_bytes=result["initialization_peak_gpu_bytes"],
        actual_preview_graph_peak_gpu_bytes=result["every_observation_graph_peak_gpu_bytes"],
        measured_total_peak_gpu_bytes=result["maximum_total_measured_gpu_bytes"],
        author_and_frozen_readout_values_unchanged=True, optimizer_updates=0,
        original_trainable_elements=734, original_fixed_elements=2959,
        development_arrays_opened=0, evaluation_arrays_opened=0, gpu_used_by_audit=False,
        control_messages_sent=0, fold_generalization_qualified=False,
        scope=("Actual every-observation registered6-target preview graph prefix qualifier; "
               "original ALLTRAIN readout is diagnostic only; no fitting or heldout claim"))
    (output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n",
                                        encoding="utf8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
