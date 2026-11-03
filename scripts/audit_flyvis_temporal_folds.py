"""Independent checkpoint, whole-history, optimizer cadence and holdout metric audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()
                          ).hexdigest()


def compare_metrics(target, valid, body, body_valid, values, scale):
    current = np.full(len(target), np.nan)
    available = body_valid[:, 4] & body_valid[:, 16] & (body[:, 16] > 1)
    current[available] = -body[available, 4]/body[available, 16]
    target20 = target[:, 5]
    usable = available & valid[:, 5]
    entering = usable & (abs(current) <= .001) & (abs(target20) >= .003)
    exiting = usable & (abs(current) >= .003) & (abs(target20) <= .001)
    opposite = usable & ~(entering | exiting) & (current*target20 < 0)
    opposite &= (abs(current) >= .001) & (abs(target20) >= .001)
    groups = dict(entering=entering, exiting=exiting, opposite=opposite,
        other_valid=usable & ~(entering | exiting | opposite),
        unavailable_body=valid[:, 5] & ~usable, invalid_target=~valid[:, 5],
        left=valid[:, 5] & (target20 < -.001), right=valid[:, 5] & (target20 > .001),
        near_straight=valid[:, 5] & (abs(target20) <= .001), full=valid[:, 5])
    scores = {}
    for name, prediction in values.items():
        heads = []
        for h in range(6):
            keep = valid[:, h] & np.isfinite(prediction[:, h])
            heads.append(dict(count=int(keep.sum()), rmse=float(np.sqrt(np.mean(
                (target[keep, h]-prediction[keep, h])**2))) if keep.any() else None))
        standardized = [np.mean(((prediction[valid[:, h], h]-target[valid[:, h], h])/(
            scale[h]))**2) for h in range(6)]
        diagnostic = {}
        for group, mask in groups.items():
            keep = mask & valid[:, 5] & np.isfinite(prediction[:, 5])
            diagnostic[group] = dict(rows=int(mask.sum()), scored_rows=int(keep.sum()),
                curvature20m_rmse=float(np.sqrt(np.mean((prediction[keep, 5]-(
                    target[keep, 5]))**2))) if keep.any() else None)
        scores[name] = dict(standardized_mse=float(np.mean(standardized)), heads=heads,
                            bearing_head_indices=[0, 2, 4], groups=diagnostic)
    return scores


def assert_nested_equal(left, right, path="metrics"):
    if isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            assert_nested_equal(left[key], right[key], path+"."+key)
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_nested_equal(a, b, path+"."+str(i))
    elif isinstance(left, float):
        assert abs(left-right) <= 1e-12*max(1., abs(left)), (path, left, right)
    else:
        assert left == right, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "executed-audit.py").write_bytes(Path(__file__).read_bytes())
    manifest, result = read(source / "manifest.json"), read(source / "result.json")
    plan = manifest["config"]["plan"]
    for path, digest in plan["input_sha256"].items():
        assert sha(path) == digest, path
    assert result["status"] == "completed_train_only_author_temporal_credit"
    assert result["author_optimizer_updates"] == 1116 and len(result["records"]) == 6
    assert sha(source / "executed-program.py") == manifest["config"]["program_sha256"]
    assert sha(source / "executed-program.py") == plan["input_sha256"][
        "scripts/train_flyvis_temporal_folds.py"]
    for key in ("readout_optimizer_updates", "known_development_arrays_opened",
                "evaluation_arrays_opened", "original_payloads_copied", "control_messages_sent",
                "ui_actions_sent", "runtime_mappings_opened"):
        assert result[key] == 0
    weights = torch.load(Path(plan["model_dir"]) / "best_chkpt", map_location="cpu",
                         weights_only=True)["network"]
    entries = read(Path(plan["registered_data"]) / "dataset.json")["episodes"]
    entries = [e for e in entries if e["role"] == "train"]
    episodes = {}
    for e in entries:
        assert sha(e["targets_file"]) == e["targets_sha256"]
        with load_array(e["targets_file"], allow_pickle=False) as arrays:
            clock = arrays["observation_time"].copy()
            target = arrays["preview"].reshape(-1, 6).copy()
            valid = np.repeat(arrays["preview_valid"], 2, 1)
        body_root = Path(e["body_replay"])
        proof = read(body_root / "result.json")
        assert sha(body_root / "virtual-body.npz") == proof["virtual_body_sha256"]
        with load_array(body_root / "virtual-body.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(clock, arrays["decision_monotonic_seconds"])
            body, mask = arrays["values"].copy(), arrays["valid"].copy()
        episodes[e["id"]] = dict(clock=clock, y=target, valid=valid, body=body, mask=mask)
    audited, paired = [], []
    total_updates = 0
    for record in result["records"]:
        heldout, arm = record["heldout_history"], record["arm"]
        folder = source / heldout / arm["name"]
        assert_nested_equal(record, read(folder / "result.json"), "armrecord")
        fit_ids = [key for key in plan["train_ids"] if key != heldout]
        assert record["fit_history_ids"] == fit_ids and record["heldout_external_role"] == "train"
        checkpoint = folder / "adapted-author.pt"
        assert sha(checkpoint) == record["final_checkpoint_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["fit_history_ids"] == fit_ids and payload["heldout"] == heldout
        assert payload["arm"] == arm and payload["author_optimizer_updates"] == 186
        flags = payload["original_trainable_flags"]
        digest = hashlib.sha256()
        for name in flags:
            digest.update(name.encode())
            digest.update(weights[name].contiguous().numpy().tobytes())
        assert digest.hexdigest() == record["initial_author_parameter_hash"]
        final_digest = hashlib.sha256()
        for name in flags:
            final_digest.update(name.encode())
            final_digest.update(payload["network"][name].contiguous().numpy().tobytes())
        assert final_digest.hexdigest() == record["final_author_parameter_hash"]
        free_count = sum(weights[name].numel() for name in flags if flags[name])
        fixed_count = sum(weights[name].numel() for name in flags if not flags[name])
        assert (free_count, fixed_count) == (734, 2959)
        changed = {name: int(torch.count_nonzero(payload["network"][name]-weights[name]))
                   for name in flags if flags[name]}
        assert changed == record["author_changed_elements"] and sum(changed.values()) > 0
        assert {name for name, count in changed.items() if count} == set(
            record["author_changed_parameter_names"])
        for name in flags:
            if not flags[name]:
                assert torch.equal(payload["network"][name], weights[name])
        fold = read(Path(plan["fold_setup"]) / heldout / "result.json")
        assert fold["fit_history_ids"] == fit_ids and fold["readout_fit_ids"] == fit_ids
        expected_counter = 0
        prior_hash = record["initial_author_parameter_hash"]
        for e in record["fit_records"]:
            history = episodes[e["episode"]]
            raw = read(folder / (e["episode"]+"-training.json"))
            assert raw["observations"] == 739 and raw["original_intervals"] == 738
            assert raw["recurrent_numeric_resets"] == 0 and raw["initial_gray_state_retained"]
            assert fingerprint(history["clock"]) == raw["original_clock_fingerprint"]
            assert fingerprint(history["valid"]) == raw["original_mask_fingerprint"]
            np.testing.assert_array_equal(history["valid"][1:].sum(0),
                                          raw["gradient_target_counts"])
            np.testing.assert_array_equal(history["valid"][0].astype(int),
                                          raw["initial_gray_detached_target_counts"])
            assert len(raw["windows"]) == 93
            logical = 0
            for gap in np.diff(history["clock"]):
                full = int(np.floor(gap/.005))
                logical += full + int(gap-full*.005 > 1e-12)
            assert raw["logical_internal_ticks"] == logical
            observed, detaches = [], 0
            for index, window in enumerate(raw["windows"]):
                start, end = index*8, min(index*8+8, 738)
                assert window["start"] == start and window["end"] == end
                assert window["original_interval_count"] == end-start
                assert window["regularizer_evaluations"] == 1
                assert window["parameters_constant_inside_window"]
                assert window["parameter_hash_before"] == prior_hash
                prior_hash = window["parameter_hash_after"]
                assert np.isfinite(window["gradient_norm_before_clip"])
                expected_counter += 1
                assert window["author_optimizer_updates"] == expected_counter
                chunks = list(range(start+arm["detach_intervals"], end+1,
                                    arm["detach_intervals"]))
                if not chunks or chunks[-1] != end:
                    chunks.append(end)
                assert [d["index"] for d in window["detach_boundaries"]] == chunks
                for boundary in window["detach_boundaries"]:
                    assert boundary["exact"] and boundary["state_tensors"] == 5
                detaches += len(chunks)
                observed.extend(range(start, end))
            assert observed == list(range(738)) and detaches == raw["detach_boundaries"]
        assert expected_counter == record["author_optimizer_updates"] == 186
        total_updates += expected_counter
        predictions_path = folder / "holdout-predictions.npz"
        assert sha(predictions_path) == record["predictions_sha256"]
        holdout = episodes[heldout]
        with load_array(predictions_path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["observation_time"], holdout["clock"])
            predictions = {key: arrays[key].copy() for key in arrays.files
                           if key != "observation_time"}
        assert all(p.shape == (739, 6) and np.isfinite(p).all() for p in predictions.values())
        with load_array(Path(plan["fold_setup"]) / heldout / "normalization.npz",
                        allow_pickle=False) as arrays:
            target_scale = arrays["target_scale"].copy()
        recomputed = compare_metrics(holdout["y"], holdout["valid"], holdout["body"],
                                    holdout["mask"], predictions, target_scale)
        assert_nested_equal(recomputed, record["holdout_metrics"])
        assert_nested_equal(fold["holdout_metrics"], record["original_fold_metrics"])
        audited.append(dict(heldout=heldout, arm=arm["name"], updates=186,
            changed_author_values=sum(changed.values()), fixed_author_values_exact=2959,
            all_original_fit_intervals_covered=1476,
            holdout_metrics_independently_recomputed=True,
            holdout_standardized_mse=recomputed["adapted__intact"]["standardized_mse"],
            original_standardized_mse=fold["holdout_metrics"]["vision_body__intact"][
                "standardized_mse"],
            body_only_standardized_mse=recomputed["body_only"]["standardized_mse"],
            peak_gpu_tensor_bytes=record["peak_gpu_tensor_bytes"]))
    assert total_updates == 1116
    for heldout in plan["train_ids"]:
        short = next(r for r in audited if r["heldout"] == heldout and r["arm"] == "detach2")
        long = next(r for r in audited if r["heldout"] == heldout and r["arm"] == "detach8")
        paired.append(dict(heldout=heldout, detach2_mse=short["holdout_standardized_mse"],
            detach8_mse=long["holdout_standardized_mse"],
            long_relative_change=(long["holdout_standardized_mse"]/(
                short["holdout_standardized_mse"])-1)))
    summary = dict(status="passed_independent_train_only_author_temporal_audit",
        source_run=str(source), source_result_sha256=sha(source / "result.json"),
        program_sha256=sha(source / "executed-program.py"), author_optimizer_updates=1116,
        all_sealed_inputs_sha256_verified=True,
        readout_optimizer_updates=0, arm_count=6, folds=3, original_unique_train_observations=2217,
        fit_observation_exposures=8868, whole_clock_mask_body_and_tail_coverage_exact=True,
        all_original_fixed_author_values_exact=True, all_fold_readout_fit_ids_exact=True,
        candidate_rule="All6 final candidates retained; no best epoch/holdout tuning",
        records=audited, paired=paired, development_arrays_opened=0, evaluation_arrays_opened=0,
        gpu_used_by_audit=False, control_messages_sent=0, independent_final_validation=False,
        physiological_function_verified=False, visual_driver_qualified=False, valid_student_laps=0)
    (output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n",
                                        encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
