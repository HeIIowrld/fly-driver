"""Offline original-graph pedal fit. Current unsealed draft always refuses execute.

--qualify-gradients and --fit are separately sealed original-process stages.
No simulator/control modules are imported by this trainer.
"""

import argparse
import gc
import hashlib
import importlib
import json
import time
from pathlib import Path

from flybrain_drive.flight_pedal_train_plan_v1 import (
    sha,
    validate_before_model,
    validate_gradient_certificate,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def host_available_bytes():
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.DWORD),
            ("load", wintypes.DWORD),
            *[
                (name, ctypes.c_ulonglong)
                for name in (
                    "physical_total",
                    "physical_available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ],
        ]

    info = Memory()
    info.length = ctypes.sizeof(info)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
        raise OSError("Available host memory query failed")
    return info.physical_available


class OriginalGraphAdapter:
    """Only the unchanged original graph; frozen private CPU reference, no rebase."""

    original_graph = True

    def __init__(self, plan, budget):
        import numpy as np
        import torch

        from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
        from flybrain_drive.flight_maneuver_learning import TrainablePhysicalManeuverCircuit
        from flybrain_drive.flight_pedal_learning_v2 import (
            bind_registered_reference,
            install_registered_private_offsets,
            pedal_only_parameters,
        )
        from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT
        from scripts.qualify_flight_pedal_graph_v2 import model_digest, tensor_digest

        self.torch, self.plan, self.budget = torch, plan, budget
        self.tensor_digest, self.model_digest = tensor_digest, model_digest
        protocol = plan["protocol"]
        limits = protocol["resource_limits_to_reseal_with_trainer"]
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(protocol["optimizer"]["seed"])
        if (
            torch.__version__ != limits["expected_torch"]
            or torch.version.hip != limits["expected_hip"]
        ):
            raise ValueError("Exact registered torch/HIP required; no CPU fallback")
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_name(0) != limits["actual_device"]
        ):
            raise ValueError("Exact registered actual GPU required; no fallback")
        torch.cuda.reset_peak_memory_stats()
        self.binding = bind_registered_reference(
            Path(plan["model"]["cpu_reference"]).read_bytes(),
            Path(plan["model"]["cpu_readout"]).read_bytes(),
        )
        payload = torch.load(plan["model"]["checkpoint"], map_location="cpu", weights_only=False)
        model = TrainablePhysicalManeuverCircuit(
            plan["model"]["spec"], payload, sparse_backend="cached_csr_cpu"
        ).eval()
        del payload
        if (
            type(model).advance is not PhysicalManeuverCircuit.advance
            or type(model).drive is not PhysicalManeuverCircuit.drive
        ):
            raise ValueError("Unchanged original physical drive/advance required")
        model.command.turn_delta.requires_grad_(False)
        pedal_only_parameters(model)
        if (
            model.command.power_delta.item() != 0
            or model.command.speed_feedforward_delta.item() != 0
        ):
            raise ValueError("Fresh original zero DNg delta initialization required")
        self.kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        protected = model_digest(model)
        all_before = {key: tensor_digest(value) for key, value in model.state_dict().items()}
        with torch.no_grad():
            state = model.initial_state(1)
            values = self.quiet_values("cpu")
            for _ in range(100):
                budget()
                state = model.advance(*values[:4], state, 0.01, **self.kwargs)[2]
            CPU_hash = tensor_digest(state)
            installation = install_registered_private_offsets(
                model.backbone.readout, self.binding, quiet_state_sha256=CPU_hash
            )
            private = model_digest(model)
            changed = [k for k in protected if protected[k] != private[k]]
            if not set(changed) <= {
                "readout.power_reference",
                "readout.lateral_reference",
                "readout.reference_calibrated",
            }:
                raise ValueError("Only3 fixed private neutral buffers may change")
            all_private = {key: tensor_digest(value) for key, value in model.state_dict().items()}
            if not {key for key in all_before if all_before[key] != all_private[key]} <= {
                "backbone.readout.power_reference",
                "backbone.readout.lateral_reference",
                "backbone.readout.reference_calibrated",
            }:
                raise ValueError("A fixed full-model state outside private neutral buffers changed")
            self.CPU_quiet = np.array(state.cpu(), copy=True)
        self.CPU_staging = dict(
            actual_CPU_quiet_sha256=CPU_hash,
            private_offset_installation=installation,
            changed_backbone_keys=changed,
        )
        del state, values
        model.backbone.circuit.sparse_backend = "cached_csr_gpu"
        self.model = model.to("cuda")  # same CPU-staged instance; no GPU offset installation
        moved = {key: tensor_digest(value) for key, value in self.model.state_dict().items()}
        if self.model_digest(self.model) != private or moved != all_private:
            raise ValueError("Original frozen graph/readout values changed on device move")
        self.quiet_receipt = None

    def quiet_values(self, device):
        torch = self.torch
        valid = torch.zeros(1, 21, device=device, dtype=torch.bool)
        valid[:, [0, 1, 4, 16]] = True
        return (
            torch.zeros(1, 2, device=device),
            torch.zeros(1, 21, device=device),
            valid,
            torch.tensor([[0.0, 1.0]], device=device),
        )

    def parameters(self):
        from flybrain_drive.flight_pedal_learning_v2 import pedal_only_parameters

        values = pedal_only_parameters(self.model)
        return dict(
            zip(("command.power_delta", "command.speed_feedforward_delta"), values, strict=True)
        )

    def fixed_fingerprint(self):
        return {
            key: self.tensor_digest(value)
            for key, value in self.model.state_dict().items()
            if key not in {"command.power_delta", "command.speed_feedforward_delta"}
        }

    def state_fingerprint(self, state):
        return self.tensor_digest(state)

    def initialize(self):
        import numpy as np

        torch = self.torch
        with torch.no_grad():
            state = self.model.initial_state(1)
            values = self.quiet_values("cuda")
            for _ in range(100):
                self.budget()
                state = self.model.advance(*values, state, 0.01, **self.kwargs)[2]
            actual = self.tensor_digest(state)
            error = float(np.max(np.abs(state.cpu().numpy() - self.CPU_quiet)))
            prediction = self.predict(state)
            if (
                actual != self.plan["model"]["expected_GPU_quiet_sha256"]
                or error > 1e-6
                or prediction["original_intent"].abs().max().item() > 2e-6
                or prediction["canonical"][0, 2].item() > 1e-4
                or prediction["encoded"][0, 1:].tolist() != [0.0, 0.0]
            ):
                raise ValueError(
                    "Actual GPU quiet differs; never relabel with CPU hash or reanchor"
                )
            self.quiet_receipt = dict(
                actual_GPU_quiet_sha256=actual,
                actual_CPU_quiet_sha256=self.CPU_staging["actual_CPU_quiet_sha256"],
                maximum_CPU_GPU_state_error=error,
                quiet_steps=100,
                quiet_dt_seconds=0.01,
                original_intent=prediction["original_intent"].cpu().tolist(),
                encoded=prediction["encoded"].cpu().tolist(),
                no_GPU_rebase=True,
            )
            return state.detach()

    def advance(self, history, held, dt, state):
        torch = self.torch
        a = history.arrays
        values = [
            torch.tensor(a[key][held : held + 1], device="cuda")
            for key in ("physical_command", "body", "body_valid", "racing_goals")
        ]
        return self.model.advance(*values, state, dt, **self.kwargs)[2]

    def predict(self, state):
        from flybrain_drive.flight_pedal_learning_v2 import v2_pedal_tensors

        raw, mn = self.model.backbone._read_state_tensor(state)
        return v2_pedal_tensors(raw, mn, binding=self.binding)

    def graph_proof(self, loss):
        from scripts.qualify_flight_pedal_graph_gpu_v2 import graph_functions

        circuit = self.model.backbone.circuit
        value = graph_functions(loss)
        value.update(
            backend=circuit.sparse_backend,
            matrix_device=str(circuit._cached_matrix.device),
            transpose_device=str(circuit._cached_transpose.device),
        )
        if (
            value["backend"] != "cached_csr_gpu"
            or value["matrix_device"] != "cuda:0"
            or value["transpose_device"] != "cuda:0"
        ):
            raise ValueError("Actual original GPU CSR/transpose adjoint required")
        return value

    def record(self, prediction):
        return {
            key: prediction[key].detach().cpu().tolist()
            for key in (
                "original_intent",
                "original_motor_summary",
                "common_power",
                "canonical_v1",
                "canonical",
                "encoded",
            )
        }


def execute(plan, plan_path, *, stage):
    started = time.perf_counter()
    result = dict(
        status="failed",
        stage=stage,
        fit_allowed=False,
        optimizer_updates=0,
        student_deployed=False,
        source_payloads_copied=0,
        live_actions=0,
        actual_process_terminal_observed=False,
    )
    out = None
    adapter = None
    torch = None
    wall_limit = (
        plan.get("protocol", {})
        .get("resource_limits_to_reseal_with_trainer", {})
        .get("wall_budget_seconds", 1800)
    )

    def budget():
        if time.perf_counter() - started > wall_limit:
            raise TimeoutError("Original1800-second total wall budget exhausted; no extension")

    try:
        validated = validate_before_model(plan, stage=stage)
        budget()
        result["pre_model_validation_wall_seconds"] = time.perf_counter() - started
        out = Path(plan["output"])
        out.mkdir(parents=True, exist_ok=False)
        write(
            out / "registration-before-data-and-model.json",
            dict(
                plan=plan,
                plan_sha256=sha(plan_path),
                program_sha256=sha(__file__),
                source_validation_complete=True,
            ),
        )
        (out / "executed-program.py").write_bytes(Path(__file__).read_bytes())
        from flybrain_drive.pedal_tbptt_engine_v1 import (
            fit_after_qualification,
            run_pass,
            whole_history,
        )

        load_started = time.perf_counter()
        loader = importlib.import_module(plan["loader_binding"]["module"])
        calibration = Path(plan["evidence"]["nonce_pedal_calibration"]["path"]).read_bytes()
        loaded = getattr(loader, plan["loader_binding"]["function"])(
            plan["capture_root"],
            expected_capture_manifest_sha256=plan["capture_manifest_sha256"],
            pedal_review_bytes=calibration,
        )
        if (
            loaded.get("fit_allowed") is not False
            or loaded.get("all_states_retained") is not True
            or loaded.get("state_frames_dropped") != 0
            or loaded.get("source_clock_schedule_valid") is not True
            or loaded.get("physical_body_interface_valid") is not True
        ):
            raise ValueError("Exact futureV4 whole-history result required; no admission inference")
        history = whole_history(
            loaded["arrays"], expected_fingerprints=plan["history"]["fingerprints"]
        )
        if (
            history.observations != plan["history"]["observations"]
            or len(history.windows) != plan["history"]["windows"]
            or history.nonempty_windows != plan["history"]["windows_with_supervised_cells"]
        ):
            raise ValueError("Actual full history/nonempty window counts differ from seal")
        result["data_load_wall_seconds"] = time.perf_counter() - load_started
        result["host_available_bytes_before_model"] = host_available_bytes()
        if result["host_available_bytes_before_model"] < 8589934592:
            raise MemoryError("Original8GiB host free-memory gate failed; no model start")
        budget()
        model_started = time.perf_counter()
        import torch as actual_torch

        torch = actual_torch
        adapter = OriginalGraphAdapter(plan, budget)
        result["model_preparation_wall_seconds"] = time.perf_counter() - model_started
        result["CPU_staging"] = adapter.CPU_staging
        bindings = dict(
            capture_manifest_sha256=plan["capture_manifest_sha256"],
            capture_nonce=plan["capture_nonce"],
            source_closure_sha256=plan["source_closure"]["sha256"],
            loader_source_sha256=plan["loader_binding"]["source_sha256"],
            history=plan["history"],
            model_binding=plan["model"],
            protocol_sha256=plan["protocol_sha256"],
        )
        result["bindings"] = bindings
        phases = {}

        def progress(event):
            if event["stage"] == "train":
                result["optimizer_updates"] = event["optimizer_updates"]
            event = dict(event, elapsed_total_seconds=time.perf_counter() - started)
            phases[event["stage"]] = event
            write(out / "progress.json", event)
            print(json.dumps(event), flush=True)

        phase_started = time.perf_counter()
        if stage == "qualify_gradients":
            qualification = run_pass(
                adapter, history, kind="qualify", budget=budget, progress=progress
            )
            result.update(
                status="completed_new_train_window_gradient_qualification",
                qualification=qualification,
                optimizer_updates=0,
                fit_allowed=False,
                rank2_passed=qualification["rank2_passed"],
                phase_wall_seconds=dict(qualification=qualification["wall_seconds"]),
            )
            if not qualification["rank2_passed"]:
                result["status"] = "failed_new_train_window_gradient_rank_gate"
        else:
            certificate = validate_gradient_certificate(plan)
            fitted = fit_after_qualification(
                adapter, history, certificate["qualification"], budget=budget, progress=progress
            )
            result.update(
                status=fitted["status"],
                fit_allowed=True,
                fit=fitted,
                optimizer_updates=fitted["online_training"]["optimizer_updates"],
                phase_wall_seconds={
                    k: fitted[v]["wall_seconds"]
                    for k, v in [
                        ("baseline", "baseline_fixed_rollout"),
                        ("training", "online_training"),
                        ("final", "final_fixed_rollout"),
                    ]
                },
            )
            import numpy as np

            before = np.asarray(
                [
                    x["prediction"]["canonical"][0][0]
                    for x in fitted["baseline_fixed_rollout"]["rows"]
                ],
                dtype=np.float64,
            )
            after = np.asarray(
                [x["prediction"]["canonical"][0][0] for x in fitted["final_fixed_rollout"]["rows"]],
                dtype=np.float64,
            )
            result["steering_diagnostic_only"] = dict(
                loss_applied=False,
                invariant_claimed=False,
                full_calibration_claimed=False,
                baseline_maximum_absolute=float(np.abs(before).max()),
                final_maximum_absolute=float(np.abs(after).max()),
                maximum_absolute_change=float(np.abs(after - before).max()),
                RMS_change=float(np.sqrt(np.mean((after - before) ** 2))),
            )
            write(
                out / "final-two-DNg-scalars.json",
                dict(
                    schema="flybrain.two_DNg_final_candidate.v1",
                    values=fitted["final_two_scalar_checkpoint"],
                    bindings=bindings,
                    baseline_and_final_TRAIN_metrics_are_generalization=False,
                    student_deployed=False,
                    optimizer_updates=result["optimizer_updates"],
                ),
            )
        result["history_phase_wall_seconds"] = time.perf_counter() - phase_started
        result["phase_progress"] = phases
        budget()
        # Full original source/payload bindings must also remain unchanged at exit.
        for path, expected in validated["source_closure"]["source_sha256"].items():
            if sha(path) != expected:
                raise ValueError("Source drift during offline stage: " + path)
        result["source_drift_count"] = 0
    except BaseException as error:
        result.update(status="failed", error=type(error).__name__, reason=str(error))
        raise
    finally:
        if torch is not None and torch.cuda.is_initialized():
            adapter = None
            gc.collect()
            try:
                torch.cuda.synchronize()
                result["GPU_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["GPU_allocated_before_empty_cache_bytes"] = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                result["GPU_allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated()
                result["GPU_reserved_after_cleanup_bytes"] = torch.cuda.memory_reserved()
                result["owned_allocator_cleanup_passed"] = (
                    result["GPU_allocated_after_cleanup_bytes"] == 0
                )
            except BaseException as error:
                result["cleanup_error"] = str(error)
                result["owned_allocator_cleanup_passed"] = False
        result["wall_seconds"] = time.perf_counter() - started
        result["original_process_terminal_receipt_required"] = True
        if out is not None:
            result["program_sha256"] = sha(__file__)
            result["plan_sha256"] = sha(plan_path)
            write(out / "result.json", result)
            print(
                json.dumps(
                    dict(
                        status=result["status"],
                        result_sha256=sha(out / "result.json"),
                        process_terminal_not_yet_observed=True,
                    )
                ),
                flush=True,
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--execute", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--qualify-gradients", action="store_true")
    group.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    raw = Path(args.plan).read_bytes()
    plan = json.loads(raw)
    if not args.execute:
        print(
            json.dumps(
                dict(
                    status="inventory_only_no_data_or_model",
                    sealed=plan.get("sealed", False),
                    model_loaded=False,
                    device_queries=0,
                )
            )
        )
        return
    if hashlib.sha256(raw).hexdigest() != args.expected_plan_sha256:
        raise ValueError("Explicit exact sealed execution-plan SHA required")
    if not (args.qualify_gradients or args.fit):
        raise ValueError("Separate --qualify-gradients or --fit stage required")
    result = execute(plan, args.plan, stage="fit" if args.fit else "qualify_gradients")
    if result["status"].startswith("failed"):
        raise RuntimeError(result["status"])


if __name__ == "__main__":
    main()
