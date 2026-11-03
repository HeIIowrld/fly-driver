"""Matched original-clock CPU/GPU MN decoding; no fitting, native mappings or control."""

import argparse
import dataclasses
import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.actuator_codec import project_pedals
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodiment import decode_embodiment
from flybrain_drive.flight_embodiment_trial import (
    PROOF_CONTRACT_FILE,
    REQUIRED_MODEL_SOURCES,
    _reference,
)
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from scripts.qualify_flight_embodiment import fingerprint, model_digest, tensor_digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def verify_registered_inputs(plan):
    for path, digest in plan["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Registered input changed: " + path)


def whole_model_digest(model):
    """Backbone plus all command tensors and non-tensor command coefficients."""
    return dict(
        state_dict={key: tensor_digest(value) for key, value in model.state_dict().items()},
        command_attributes={
            key: getattr(model.command, key) for key in ("turn_gain", "power_gain", "power_ports")
        },
        command_contract_json=json.dumps(model.contract, sort_keys=True),
    )


def finite_maximum_error(first, second, label):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite or mismatched matched-qualification arrays: " + label)
    return float(np.max(abs(first - second)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (
        plan["schema"] != "flybrain.flight.matched_mn_anchor_registration.v1"
        or plan["fit_allowed"] is not False
        or plan["optimizer_updates"] != 0
        or plan["proof_contract_sha256"] != sha256(PROOF_CONTRACT_FILE)
    ):
        raise ValueError("Predeclared matched CPU/GPU registration required")
    verify_registered_inputs(plan)
    cpu = Path(plan["cpu_run"])
    cpu_result = read(cpu / "result.json")
    if (
        cpu_result["status"] != "completed_cpu_original_clock_mn_anchor_qualification"
        or not cpu_result["cpu_repeat_bitwise_exact"]
        or cpu_result["optimizer_updates"] != 0
    ):
        raise ValueError("Completed full CPU result required before CUDA")
    reference = _reference(read(cpu / "reference.json"))
    quiet_data = read(cpu / "quiet-trajectories.json")
    validation = read(cpu / "validation-trajectories.json")
    reference_sha = sha256(cpu / "reference.json")
    if (
        cpu_result["reference_sha256"] != reference_sha
        or validation["shared_reference_sha256"] != reference_sha
        or reference.provenance.numerical_scope != "cpu_repeat"
        or reference.provenance.tensor_dtype != "float32"
    ):
        raise ValueError("Original completed CPU reference and trajectory links must agree")
    for item in reference.provenance.sources:
        if sha256(item.path) != item.sha256:
            raise ValueError("Original CPU reference source changed: " + item.role)
    mandatory_inputs = (
        *REQUIRED_MODEL_SOURCES,
        PROOF_CONTRACT_FILE,
        plan["spec"],
        plan["checkpoint"],
        plan["original_events"],
        *(
            cpu / name
            for name in (
                "result.json",
                "reference.json",
                "quiet-trajectories.json",
                "validation-trajectories.json",
            )
        ),
    )
    registered_paths = {Path(path).resolve() for path in plan["input_sha256"]}
    if any(Path(path).resolve() not in registered_paths for path in mandatory_inputs):
        raise ValueError("Every model/CPU qualification source must be preregistered")
    if (
        quiet_data["postcalibration_reference"]["power_reference"] != reference.quiet_power
        or quiet_data["repeats"][0][-1]["state_sha256"]
        != reference.provenance.quiet_end_state_sha256
    ):
        raise ValueError("Exact CPU private quiet readout and state must match reference")
    groups = ("cases", "controls", "lesions", "lesion_quiet_baselines", "matched_speed_lesions")
    originals = [c for group in groups for c in validation[group]]
    if len(originals) != 30:
        raise ValueError("All30 original registered trajectories required")
    source = [
        r
        for r in (
            json.loads(line)
            for line in Path(plan["original_events"]).read_text(encoding="utf8").splitlines()
        )
        if r["kind"] == "neural_component"
    ]
    clock = np.array([r["decision_clock"] for r in source], dtype=np.float64)
    intervals = np.diff(clock)
    original_masks = np.asarray([r["body_valid"] for r in source], dtype=bool)
    synthetic_mask = np.zeros((68, 21), dtype=bool)
    synthetic_mask[:, [0, 1, 4, 16]] = True
    if (
        len(source) != 68
        or fingerprint(clock) != reference.provenance.original_clock_sha256
        or fingerprint(original_masks) != reference.provenance.original_body_mask_sha256
        or not np.all((intervals > 0) & (intervals <= 0.08))
    ):
        raise ValueError("Original schedule and explicit synthetic mask required")
    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_path=str(Path(args.plan).resolve()),
            plan_sha256=sha256(args.plan),
            program_sha256=sha256(__file__),
            scope=__doc__,
            torch_version=torch.__version__,
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    (run / "registration.json").write_bytes(Path(args.plan).read_bytes())
    result = dict(
        schema="flybrain.flight.matched_mn_anchor_qualification.v1",
        status="failed",
        fit_allowed=False,
        optimizer_updates=0,
        teacher_labels_used=False,
        live_api_calls=0,
        mmf_opens=0,
        control_messages=0,
        intermediate_state_resets=0,
        gpu_used=False,
        gpu_qualified=False,
        student_deployed=False,
        goal_achieved=False,
    )
    models, quiet_states, states, inputs = [], [], [], []
    traces = [dict(quiet=[], cases=[]) for _ in range(3)]
    maxima = {
        key: 0.0
        for key in (
            "state",
            "raw",
            "muscles",
            "common",
            "v1",
            "decoded",
            "encoded",
            "device_decoder_common",
            "device_decoder_v1",
            "device_decoder_v2",
            "device_decoder_encoded",
            "gpu_repeat_state",
            "gpu_repeat_raw",
            "gpu_repeat_muscles",
            "gpu_repeat_common",
            "gpu_repeat_v1",
            "gpu_repeat_decoded",
            "gpu_repeat_encoded",
        )
    }
    gpu_repeat_exact = True
    last_observation = {}
    completed_cases = 0
    frozen = full_frozen = None
    model = payload = target = None

    def save_progress(status):
        for name, trace in zip(
            ("cpu-recomputed", "gpu-repeat-0", "gpu-repeat-1"), traces, strict=True
        ):
            write_json(run / (name + ".json"), trace)
        write_json(
            run / "progress.json",
            dict(
                status=status,
                completed_cases=completed_cases,
                maximum_errors=maxima,
                last_observation=last_observation,
                gpu_repeat_bitwise_exact_so_far=gpu_repeat_exact,
            ),
        )

    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.set_default_device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback qualification")
        torch.cuda.reset_peak_memory_stats()
        result.update(
            gpu_used=True,
            device=torch.cuda.get_device_name(0),
            cuda_version=torch.version.cuda,
            hip_version=torch.version.hip,
            device_index=torch.cuda.current_device(),
            gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            numpy_version=np.__version__,
        )
        payload = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
        private = quiet_data["postcalibration_reference"]
        write_json(
            run / "cpu-readout-state.json",
            dict(
                schema="flybrain.flight.frozen_quiet_readout_state.v1",
                cpu_reference_sha256=sha256(cpu / "reference.json"),
                source_checkpoint_sha256=sha256(plan["checkpoint"]),
                readout_state=private,
            ),
        )
        for device in ("cpu", "cuda", "cuda"):
            model = PhysicalManeuverCircuit(
                plan["spec"],
                payload,
                sparse_backend="cached_csr_cpu" if device == "cpu" else "cached_csr_gpu",
            )
            model = model.to(device).eval()
            old = model_digest(model)
            target = model.backbone.readout.state_dict()
            if set(target) != set(private):
                raise ValueError("Exact complete CPU readout dictionary required")
            model.backbone.readout.load_state_dict(
                {
                    k: torch.as_tensor(private[k], device=target[k].device, dtype=target[k].dtype)
                    for k in target
                },
                strict=True,
            )
            after = model_digest(model)
            changed = {k for k in old if old[k] != after[k]}
            if not changed <= {
                "readout.power_reference",
                "readout.lateral_reference",
                "readout.reference_calibrated",
            }:
                raise ValueError("Only qualified private neutral offsets may change")
            models.append(model)
        frozen = [model_digest(m) for m in models]
        full_frozen = [whole_model_digest(m) for m in models]
        if (
            frozen[0] != frozen[1]
            or frozen[0] != frozen[2]
            or full_frozen[0] != full_frozen[1]
            or full_frozen[0] != full_frozen[2]
        ):
            raise ValueError("Identical fixed model and CPU private readout required")
        write_json(
            run / "frozen-model-digests.json",
            dict(
                before=frozen,
                after=None,
                whole_model_before=full_frozen,
                whole_model_after=None,
                command_buffers_and_scalar_gains_included=True,
            ),
        )
        kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)

        def input_tensors(m, case):
            device = next(m.parameters()).device
            body = torch.zeros(1, 21, device=device)
            body[:, 16], body[:, 4] = case["speed_kmh"] / 3.6, case["actual_yaw_rad_s"]
            mask = torch.tensor(synthetic_mask[:1], device=device)
            command = torch.tensor(
                [[case["desired_yaw_rad_s"], case["desired_speed_m_s"]]], device=device
            )
            return command, body, mask, torch.tensor(racing_goal(100, 1), device=device)

        def observations(current_states, expected, elapsed):
            nonlocal gpu_repeat_exact, last_observation
            rows, arrays, device_arrays = [], [], []
            last_observation = dict(elapsed_seconds=float(elapsed), samples=[])
            for backend, (m, state) in enumerate(zip(models, current_states, strict=True)):
                state_cpu = state.detach().cpu()
                state_array = state_cpu.numpy()
                diagnostic = dict(
                    backend=backend,
                    state_sha256=tensor_digest(state_cpu),
                    nonfinite_state_values=int((~np.isfinite(state_array)).sum()),
                )
                last_observation["samples"].append(diagnostic)
                if diagnostic["nonfinite_state_values"]:
                    raise ValueError("Nonfinite full neural state; no parity gate may ignore NaNs")
                raw, muscles = m.backbone._read_state_tensor(state)
                raw_cpu, muscles_cpu = raw.detach().cpu(), muscles.detach().cpu()
                # Actual subclass route: exact recorded float32 GPU values decoded on CPU.
                recorded = decode_embodiment(raw_cpu, muscles_cpu, reference=reference)
                device_decoded = decode_embodiment(raw, muscles, reference=reference)
                actual_v2 = recorded.canonical_v2.numpy()
                device_v2 = device_decoded.canonical_v2.cpu().numpy()
                row = dict(
                    elapsed_seconds=float(elapsed),
                    state_sha256=diagnostic["state_sha256"],
                    raw_intent=raw_cpu[0].tolist(),
                    muscles=muscles_cpu[0].tolist(),
                    common_power=float(recorded.common_power[0]),
                    canonical_v1=recorded.canonical_v1[0].tolist(),
                    canonical_v2=actual_v2[0].tolist(),
                    decoder_device_for_recorded_control="cpu",
                    device_decoder_common_power=float(device_decoded.common_power[0]),
                    device_decoder_canonical_v1=device_decoded.canonical_v1[0].cpu().tolist(),
                    device_decoder_canonical_v2=device_v2[0].tolist(),
                    pedal_coactivation=bool(recorded.pedal_coactivation[0]),
                )
                row["encoded_v2"] = project_pedals(row["canonical_v2"])["encoded_request"]
                row["device_decoder_encoded_v2"] = project_pedals(
                    row["device_decoder_canonical_v2"]
                )["encoded_request"]
                values = dict(
                    state=state_array,
                    raw=raw_cpu.numpy(),
                    muscles=muscles_cpu.numpy(),
                    common=recorded.common_power.numpy(),
                    v1=recorded.canonical_v1.numpy(),
                    decoded=actual_v2,
                    encoded=np.asarray(row["encoded_v2"]),
                )
                device_values = dict(
                    common=device_decoded.common_power.cpu().numpy(),
                    v1=device_decoded.canonical_v1.cpu().numpy(),
                    decoded=device_v2,
                    encoded=np.asarray(row["device_decoder_encoded_v2"]),
                )
                for key, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite recorded qualification value: " + key)
                    row[key + "_array_sha256"] = fingerprint(value)
                diagnostic["recorded_sample"] = row
                for key, device_value in device_values.items():
                    row["device_" + key + "_array_sha256"] = fingerprint(device_value)
                    error = finite_maximum_error(values[key], device_value, "device route " + key)
                    name = "device_decoder_" + ("v2" if key == "decoded" else key)
                    maxima[name] = max(maxima[name], error)
                    maxima[key] = max(maxima[key], error)
                    if error > (1e-4 if key in ("decoded", "encoded") else 2e-6):
                        raise ValueError(
                            "CPU-from-recorded-values vs device decoder failed: " + key
                        )
                arrays.append(values)
                device_arrays.append(device_values)
                rows.append(row)
            if expected is not None:
                if rows[0]["state_sha256"] != expected["state_sha256"]:
                    raise ValueError("Recomputed CPU full state differs from original CPU run")
                for key in (
                    "elapsed_seconds",
                    "raw_intent",
                    "muscles",
                    "common_power",
                    "canonical_v1",
                    "canonical_v2",
                    "encoded_v2",
                    "pedal_coactivation",
                ):
                    if key in expected and rows[0][key] != expected[key]:
                        raise ValueError("Exact original CPU repeat differs: " + key)
            for first, second in itertools.combinations(range(3), 2):
                for key in ("state", "raw", "muscles", "common", "v1", "decoded", "encoded"):
                    error = finite_maximum_error(arrays[first][key], arrays[second][key], key)
                    maxima[key] = max(maxima[key], error)
                    if first == 1 and second == 2:
                        name = "gpu_repeat_" + key
                        maxima[name] = max(maxima[name], error)
                        gpu_repeat_exact = gpu_repeat_exact and (
                            rows[1][key + "_array_sha256"] == rows[2][key + "_array_sha256"]
                        )
                    limit = 1e-4 if key in ("decoded", "encoded") else 2e-6
                    if error > limit:
                        raise ValueError("Registered matched CPU/GPU gate failed: " + key)
            # Include cross-backend/cross-decoder-route extrema, not only paired routes.
            for key in ("common", "v1", "decoded", "encoded"):
                alternatives = [a[key] for a in arrays] + [a[key] for a in device_arrays]
                for first, second in itertools.combinations(alternatives, 2):
                    error = finite_maximum_error(first, second, "all routes " + key)
                    maxima[key] = max(maxima[key], error)
                    if error > (1e-4 if key in ("decoded", "encoded") else 2e-6):
                        raise ValueError("Cross-backend decoder-route parity gate failed: " + key)
                gpu_repeat_exact = gpu_repeat_exact and (
                    rows[1]["device_" + key + "_array_sha256"]
                    == rows[2]["device_" + key + "_array_sha256"]
                )
            return rows

        with torch.no_grad():
            quiet_case = dict(
                speed_kmh=0, actual_yaw_rad_s=0.0, desired_yaw_rad_s=0.0, desired_speed_m_s=0.0
            )
            inputs = [input_tensors(m, quiet_case) for m in models]
            states = [m.initial_state(1) for m in models]
            for index in range(101):
                if index:
                    states = [
                        m.advance(*values, s, 0.01, **kwargs)[2]
                        for m, values, s in zip(models, inputs, states, strict=True)
                    ]
                # Original quiet raw transient used the old reference.
                # State and MN summary are unaffected by this private output offset.
                expected = quiet_data["repeats"][0][index]
                rows = observations(
                    states,
                    dict(
                        state_sha256=expected["state_sha256"],
                        muscles=expected["muscles"],
                        common_power=expected["common_power"],
                    ),
                    index * 0.01,
                )
                for trace, row in zip(traces, rows, strict=True):
                    trace["quiet"].append(row)
            quiet_states = [s.clone() for s in states]
            save_progress("quiet_complete")
            quiet_raw = max(max(abs(v) for v in t["quiet"][-1]["raw_intent"]) for t in traces)
            quiet_brake = max(t["quiet"][-1]["canonical_v2"][2] for t in traces)
            quiet_pedals_zero = all(t["quiet"][-1]["encoded_v2"][1:] == [0.0, 0.0] for t in traces)
            if quiet_raw > 2e-6 or quiet_brake > 1e-4 or not quiet_pedals_zero:
                raise ValueError("Qualified shared-reference quiet endpoint not neutral")
            for ordinal, case in enumerate(originals):
                inputs = [input_tensors(m, case) for m in models]
                states = [s.clone() for s in quiet_states]  # Independent case boundary only.
                case_traces = [
                    dict(
                        ordinal=ordinal,
                        condition={k: v for k, v in case.items() if k != "observations"},
                        observations=[],
                    )
                    for _ in models
                ]
                for trace, case_trace in zip(traces, case_traces, strict=True):
                    trace["cases"].append(case_trace)
                for index in range(68):
                    if index:
                        states = [
                            m.advance(
                                *values,
                                s,
                                float(intervals[index - 1]),
                                lesion=case["lesion"],
                                **kwargs,
                            )[2]
                            for m, values, s in zip(models, inputs, states, strict=True)
                        ]
                    rows = observations(
                        states, case["observations"][index], clock[index] - clock[0]
                    )
                    for trace, row in zip(case_traces, rows, strict=True):
                        trace["observations"].append(row)
                completed_cases = ordinal + 1
                save_progress("case_complete")
                print(
                    json.dumps(dict(completed_case=ordinal + 1, maximum_errors=maxima)), flush=True
                )
        after = [model_digest(m) for m in models]
        full_after = [whole_model_digest(m) for m in models]
        if after != frozen or full_after != full_frozen:
            raise ValueError("Fixed graph/body/readout changed during matched qualification")
        write_json(
            run / "frozen-model-digests.json",
            dict(
                before=frozen,
                after=after,
                whole_model_before=full_frozen,
                whole_model_after=full_after,
                command_buffers_and_scalar_gains_included=True,
            ),
        )
        for name, trace in zip(
            ("cpu-recomputed", "gpu-repeat-0", "gpu-repeat-1"), traces, strict=True
        ):
            write_json(run / (name + ".json"), trace)
        # Preserve numerical evidence before constructing a matched-scope reference.
        write_json(
            run / "numerical-comparison.json",
            dict(
                maximum_errors=maxima,
                cpu_reference_sha256=sha256(cpu / "reference.json"),
                original_clock_sha256=fingerprint(clock),
                synthetic_mask_sha256=fingerprint(synthetic_mask),
                original_cases=30,
                cpu_recomputed_all_original_state_hashes_exact=True,
                recorded_control_decoder_device="cpu",
                gpu_tensor_decoder_also_compared=True,
                frozen_model_artifact=artifact(run / "frozen-model-digests.json"),
                trace_artifacts={
                    name: artifact(run / (name + ".json"))
                    for name in ("cpu-recomputed", "gpu-repeat-0", "gpu-repeat-1")
                },
            ),
        )
        matched_data = dataclasses.asdict(reference)
        matched_data.update(
            common_power_numerical_uncertainty=maxima["common"],
            maximum_decoded_brake_parity_error=maxima["decoded"],
        )
        matched_data["provenance"]["numerical_scope"] = "matched_cpu_gpu"
        for item in matched_data["provenance"]["sources"]:
            if item["role"] in ("registration", "trajectory"):
                path = run / (
                    "registration.json"
                    if item["role"] == "registration"
                    else "numerical-comparison.json"
                )
                item.update(path=str(path.resolve()), sha256=sha256(path))
        matched = _reference(matched_data)  # Revalidates absolute/relative/100x uncertainty span.
        if (
            matched.quiet_power != reference.quiet_power
            or matched.brake_anchor_power != reference.brake_anchor_power
        ):
            raise ValueError("No per-backend reanchoring allowed")
        write_json(run / "matched-reference.json", matched_data)
        artifacts = dict(
            cpu_reference=artifact(cpu / "reference.json"),
            cpu_readout_state=artifact(run / "cpu-readout-state.json"),
            matched_reference=artifact(run / "matched-reference.json"),
            cpu_repeat_0=artifact(cpu / "validation-trajectories.json"),
            cpu_repeat_1=artifact(run / "cpu-recomputed.json"),
            gpu_repeat_0=artifact(run / "gpu-repeat-0.json"),
            gpu_repeat_1=artifact(run / "gpu-repeat-1.json"),
            qualification_manifest=artifact(run / "manifest.json"),
            registration=artifact(run / "registration.json"),
        )
        required = (*REQUIRED_MODEL_SOURCES, plan["spec"], plan["checkpoint"])
        result.update(
            status="completed_matched_cpu_gpu_mn_anchor_qualification",
            numerical_scope="matched_cpu_gpu",
            gpu_qualified=True,
            contract_file=PROOF_CONTRACT_FILE,
            contract_sha256=sha256(PROOF_CONTRACT_FILE),
            cases=30,
            observations_per_case=68,
            original_intervals_per_case=67,
            artifacts=artifacts,
            current_source_sha256={p: sha256(p) for p in required},
            qualification_program=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
            cpu_repeat_bitwise_exact=True,
            gpu_repeat_passed=True,
            gpu_repeat_bitwise_exact=gpu_repeat_exact,
            recorded_control_decoder_device="cpu",
            gpu_tensor_decoder_also_compared=True,
            all_command_parameters_buffers_and_scalar_gains_unchanged=True,
            all_fixed_parameters_unchanged=True,
            cpu_private_readout_installed_on_gpu=True,
            all_original_clocks_and_synthetic_masks_preserved=True,
            quiet_encoded_pedals_zero=quiet_pedals_zero,
            maximum_raw_parity_error=maxima["raw"],
            maximum_state_parity_error=maxima["state"],
            maximum_decoded_brake_parity_error=maxima["decoded"],
            maximum_common_power_numerical_uncertainty=maxima["common"],
            maximum_encoded_parity_error=maxima["encoded"],
            maximum_canonical_v1_parity_error=maxima["v1"],
            maximum_errors=maxima,
            quiet_raw_max_abs=quiet_raw,
            quiet_v2_brake_max=quiet_brake,
            historical_clock_sha256=fingerprint(clock),
            synthetic_mask_sha256=fingerprint(synthetic_mask),
            original_historical_body_mask_sha256=reference.provenance.original_body_mask_sha256,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        try:
            if models and frozen is not None:
                write_json(
                    run / "frozen-model-digests.json",
                    dict(
                        before=frozen,
                        after=[model_digest(m) for m in models],
                        whole_model_before=full_frozen,
                        whole_model_after=[whole_model_digest(m) for m in models],
                        command_buffers_and_scalar_gains_included=True,
                        preserved_after_failed_qualification=True,
                    ),
                )
        except Exception as digest_error:
            result["failure_model_digest_error"] = str(digest_error)
        try:
            save_progress("failed_partial_evidence_preserved")
        except Exception as persistence_error:
            result["partial_trace_persistence_error"] = str(persistence_error)
        result.update(
            maximum_errors=maxima,
            completed_cases=completed_cases,
            error=type(error).__name__,
            reason=str(error),
            wall_seconds=time.perf_counter() - started,
        )
    finally:
        try:
            models.clear()
            quiet_states.clear()
            states.clear()
            inputs.clear()
            model = payload = target = None
            torch.set_default_device("cpu")
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.empty_cache()
                result["gpu_allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated()
        except Exception as cleanup_error:
            result.update(
                status="failed",
                gpu_qualified=False,
                cleanup_error=type(cleanup_error).__name__ + ": " + str(cleanup_error),
                reason="GPU cleanup failed; partial evidence retained",
            )
        finally:
            try:
                verify_registered_inputs(plan)
                result["all_registered_inputs_reverified_at_exit"] = True
            except Exception as source_error:
                result.update(
                    status="failed",
                    gpu_qualified=False,
                    source_reverification_error=str(source_error),
                    reason="Preregistered source changed during qualification",
                )
            write_json(run / "result.json", result)
    print(json.dumps(result))
    if result["status"] == "failed":
        raise RuntimeError(result["reason"])


if __name__ == "__main__":
    main()
