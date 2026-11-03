"""Whole-history pedal TBPTT engine; no dataset/model/device is loaded on import.

Only a separately admitted caller may use the original-graph adapter. Tests use
small CPU mocks, which are never original-graph or dataset qualification.
"""

import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np

PARAMETER_NAMES = ("command.power_delta", "command.speed_feedforward_delta")
WINDOW = 8
ARRAY_NAMES = frozenset(
    {
        "decision_clock",
        "native_receipt_clock",
        "native_packet",
        "body",
        "body_valid",
        "physical_command",
        "racing_goals",
        "native_pedals",
        "pedal_loss_valid",
        "original_elapsed_seconds",
        "previous_held_row_index",
    }
)


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class WholeHistory:
    arrays: dict
    fingerprints: dict
    windows: tuple
    effective_loss_valid: np.ndarray

    @property
    def observations(self):
        return len(self.arrays["decision_clock"])

    @property
    def nonempty_windows(self):
        return sum(bool(self.effective_loss_valid[a:b].any()) for a, b in self.windows)


def whole_history(arrays, *, expected_fingerprints=None):
    """Validate original arrays without slicing, interpolation or dtype repair."""
    if set(arrays) != ARRAY_NAMES:
        raise ValueError("Exact complete whole-history array interface required")
    if any(not isinstance(a, np.ndarray) for a in arrays.values()):
        raise ValueError("Original numpy arrays required")
    n = len(arrays["decision_clock"])
    if not 2 <= n <= 1200:
        raise ValueError("Whole history must contain2..1200 observations")
    shapes = dict(
        decision_clock=(n,),
        native_receipt_clock=(n,),
        native_packet=(n,),
        body=(n, 21),
        body_valid=(n, 21),
        physical_command=(n, 2),
        racing_goals=(n, 2),
        native_pedals=(n, 2),
        pedal_loss_valid=(n, 2),
        original_elapsed_seconds=(n,),
        previous_held_row_index=(n,),
    )
    for key, shape in shapes.items():
        a = arrays[key]
        expected = (
            np.bool_
            if key in {"body_valid", "pedal_loss_valid"}
            else np.float64
            if key in {"decision_clock", "native_receipt_clock", "original_elapsed_seconds"}
            else None
            if key in {"native_packet", "previous_held_row_index"}
            else np.float32
        )
        if a.shape != shape or expected is not None and a.dtype != expected:
            raise ValueError("Original shape/dtype differs: " + key)
        if expected is None and a.dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise ValueError("Original integer index/packet dtype required")
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite original history: " + key)
    dt = np.r_[0.0, np.diff(arrays["decision_clock"])]
    if (
        not np.all((dt[1:] > 0) & (dt[1:] <= 0.08))
        or not np.array_equal(dt, arrays["original_elapsed_seconds"])
        or not np.array_equal(arrays["previous_held_row_index"], np.r_[-1, np.arange(n - 1)])
        or not np.all(np.diff(arrays["native_packet"]) > 0)
        or not np.all(np.diff(arrays["native_receipt_clock"]) > 0)
    ):
        raise ValueError("Original clocks/packets/previous-held schedule required; no repair")
    if (
        np.any(arrays["physical_command"][:, 0] != 0)
        or np.any(arrays["physical_command"][:, 1] < 0)
        or not arrays["body_valid"][:, [4, 16]].all()
        or np.any(arrays["body"][:, 16] < 0)
        or np.any((arrays["native_pedals"] < 0) | (arrays["native_pedals"] > 1))
    ):
        raise ValueError("Declared zero-yaw pedal-only body/task/target interface required")
    fingerprints = {key: fingerprint(value) for key, value in arrays.items()}
    if expected_fingerprints is not None and fingerprints != expected_fingerprints:
        raise ValueError("Every admitted whole-history array fingerprint must match")
    native = arrays["native_pedals"]
    effective = (
        arrays["pedal_loss_valid"]
        & ~np.all(native > 0, axis=1)[:, None]
        & ~((native > 0) & (native < 0.02))
    )
    effective.setflags(write=False)
    # Do not mutate caller arrays, source masks or writeability flags.
    return WholeHistory(
        dict(arrays),
        fingerprints,
        tuple((a, min(a + WINDOW, n)) for a in range(0, n, WINDOW)),
        effective,
    )


def parameter_values(adapter):
    values = {k: float(v.detach().cpu().item()) for k, v in adapter.parameters().items()}
    if tuple(values) != PARAMETER_NAMES or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Exactly two finite existing DNg scalar parameters required")
    return values


def make_adam(adapter):
    import torch

    parameter_values(adapter)
    return torch.optim.Adam(
        list(adapter.parameters().values()),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        foreach=False,
        fused=False,
        maximize=False,
    )


def gradient_rank(rows):
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.size == 0:
        matrix = np.empty((0, 2), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or not np.isfinite(matrix).all():
        raise ValueError("Complete finite two-column actual window-gradient matrix required")
    singular = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.zeros(2)
    threshold = float(max(matrix.shape) * np.finfo(np.float32).eps * singular[0])
    return dict(
        matrix=matrix.tolist(),
        shape=list(matrix.shape),
        singular_values=singular.tolist(),
        threshold=threshold,
        rank=int(np.count_nonzero(singular > threshold)),
        finite=True,
        threshold_rule="max(matrix.shape)*float32_eps*largest_singular_value",
    )


def window_loss(predictions, history, start, end):
    import torch

    from .flight_pedal_learning_v2 import native_pedal_supervision, v2_pedal_imitation_loss

    canonical = torch.cat([p["canonical"] for p in predictions], dim=0)
    native = torch.tensor(
        history.arrays["native_pedals"][start:end], dtype=canonical.dtype, device=canonical.device
    )
    valid = torch.tensor(
        history.arrays["pedal_loss_valid"][start:end], dtype=torch.bool, device=canonical.device
    )
    supervision = native_pedal_supervision(native, valid, projection_policy="reject_unreachable")
    effective = supervision["supervision_valid"].detach().cpu().numpy()
    if not np.array_equal(effective, history.effective_loss_valid[start:end]):
        raise ValueError("Unchanged helper masks disagree with predeclared window counts")
    result = v2_pedal_imitation_loss(dict(canonical=canonical), supervision)
    pedals = canonical[:, 1:]
    error = (supervision["canonical_lower"] - pedals).relu() + (
        pedals - supervision["canonical_upper"]
    ).relu()
    squared = error.square().detach().cpu().numpy().astype(np.float64)
    counts = effective.sum(0)
    sums = np.where(effective, squared, 0.0).sum(0, dtype=np.float64)
    fullbrake = (history.arrays["native_pedals"][start:end, 1] == 1) & effective[:, 1]
    return result["loss"], dict(
        counts=counts,
        sums=sums,
        fullbrake_one_count=int(fullbrake.sum()),
        fullbrake_one_squared_error_sum=float(squared[fullbrake, 1].sum()),
    )


def _gradients(loss, parameters):
    import torch

    # A nonempty window may have a mathematically constant loss, e.g. only the
    # untouched first quiet row. Retain its actual zero derivative and window.
    constant = not loss.requires_grad
    raw = (
        (None,) * len(parameters)
        if constant
        else torch.autograd.grad(loss, parameters, allow_unused=True)
    )
    gradients = [
        torch.zeros_like(p) if g is None else g.detach()
        for p, g in zip(parameters, raw, strict=True)
    ]
    if not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("Nonfinite actual original-graph window gradient")
    return gradients, dict(
        loss_graph_constant=constant, unused_parameter_derivatives=[g is None for g in raw]
    )


def run_pass(adapter, history, *, kind, optimizer=None, budget=lambda: None, progress=None):
    """One complete pass. Adapter initialization occurs once, never mid-history."""
    import torch

    started = time.perf_counter()
    if kind not in {"qualify", "train", "evaluate"}:
        raise ValueError("Explicit fixed qualification/train/frozen evaluation stage required")
    if (kind == "train") != (optimizer is not None):
        raise ValueError("Only training has an optimizer; qualification/evaluation have none")
    initial = parameter_values(adapter)
    parameters = list(adapter.parameters().values())
    if optimizer is not None:
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        if [id(p) for p in actual] != [id(p) for p in parameters]:
            raise ValueError("Optimizer may contain only the exact two DNg parameters")
    frozen = adapter.fixed_fingerprint()
    rows, windows, gradients, grad_window_indices = [], [], [], []
    counts, sums = np.zeros(2, np.int64), np.zeros(2, np.float64)
    fullbrake_count, fullbrake_sum, updates = 0, 0.0, 0
    state = adapter.initialize()
    for window_index, (start, end) in enumerate(history.windows):
        budget()
        held_parameters = parameter_values(adapter)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        predictions = []
        with torch.set_grad_enabled(kind != "evaluate"):
            for index in range(start, end):
                budget()
                if parameter_values(adapter) != held_parameters:
                    raise ValueError("Coefficients changed inside an8-observation window")
                if index:
                    # Row i observes the state after holding ORIGINAL row i-1.
                    state = adapter.advance(
                        history,
                        index - 1,
                        float(history.arrays["original_elapsed_seconds"][index]),
                        state,
                    )
                prediction = adapter.predict(state)
                predictions.append(prediction)
                rows.append(
                    dict(
                        index=index,
                        decision_clock=float(history.arrays["decision_clock"][index]),
                        original_elapsed_seconds=float(
                            history.arrays["original_elapsed_seconds"][index]
                        ),
                        previous_held_row_index=int(
                            history.arrays["previous_held_row_index"][index]
                        ),
                        prediction=adapter.record(prediction),
                    )
                )
            loss, metric = window_loss(predictions, history, start, end)
            nonempty = bool(metric["counts"].sum())
            window = dict(
                index=window_index,
                start=start,
                end=end,
                eligible_cells=metric["counts"].tolist(),
                loss=float(loss.detach().cpu()),
                coefficients=held_parameters,
                optimizer_step=False,
            )
            if nonempty and kind in {"qualify", "train"}:
                if getattr(adapter, "original_graph", False) and loss.requires_grad:
                    window["actual_autograd"] = adapter.graph_proof(loss)
                    if not window["actual_autograd"]["original_CSR_adjoint_present"]:
                        raise ValueError("Actual original chemical CSR adjoint required")
                grad, description = _gradients(loss, parameters)
                window.update(description, gradients=[float(g.cpu()) for g in grad])
                if kind == "qualify":
                    gradients.append(window["gradients"])
                    grad_window_indices.append(window_index)
                else:
                    before_update_state = adapter.state_fingerprint(state)
                    for p, g in zip(parameters, grad, strict=True):
                        p.grad = g.clone()
                    optimizer.step()
                    updates += 1
                    window["optimizer_step"] = True
                    if progress:
                        progress(
                            dict(
                                stage=kind,
                                event="optimizer_step_returned",
                                window=window_index,
                                observations=end,
                                optimizer_updates=updates,
                                coefficients=parameter_values(adapter),
                            )
                        )
                    if adapter.state_fingerprint(state) != before_update_state:
                        raise ValueError("Optimizer rewrote previously computed numerical state")
            boundary = adapter.state_fingerprint(state)
            state = state.detach()
            if adapter.state_fingerprint(state) != boundary:
                raise ValueError("Detach must preserve numerical state exactly")
            window.update(state_before_detach=boundary, state_after_detach=boundary)
        if adapter.fixed_fingerprint() != frozen:
            raise ValueError("Frozen graph/body/readout/turn parameter changed")
        counts += metric["counts"]
        sums += metric["sums"]
        fullbrake_count += metric["fullbrake_one_count"]
        fullbrake_sum += metric["fullbrake_one_squared_error_sum"]
        windows.append(window)
        if progress:
            progress(
                dict(
                    stage=kind,
                    event="window_completed",
                    window=window_index,
                    observations=end,
                    optimizer_updates=updates,
                )
            )
        del predictions, prediction, loss
    expected_updates = history.nonempty_windows if kind == "train" else 0
    if updates != expected_updates or len(rows) != history.observations:
        raise ValueError("Exact nonempty-window update/all-state count violated")
    final = parameter_values(adapter)
    if kind != "train" and initial != final:
        raise ValueError("Fixed-parameter qualification/evaluation changed coefficients")
    for p in parameters:
        p.grad = None
    available = counts > 0
    means = [float(sums[i] / counts[i]) if counts[i] else None for i in range(2)]
    result = dict(
        stage=kind,
        original_graph=bool(getattr(adapter, "original_graph", False)),
        observations=history.observations,
        intervals=history.observations - 1,
        original_array_fingerprints=history.fingerprints,
        windows=windows,
        rows=rows,
        optimizer_updates=updates,
        expected_optimizer_updates=expected_updates,
        state_frames_dropped=0,
        intermediate_numerical_state_resets=0,
        first_observation_advanced=False,
        coefficients_initial=initial,
        coefficients_final=final,
        fixed_fingerprint=frozen,
        quiet_receipt=adapter.quiet_receipt,
        wall_seconds=time.perf_counter() - started,
        metrics=dict(
            per_axis_squared_error_sum=sums.tolist(),
            per_axis_eligible_cells=counts.tolist(),
            per_axis_mean=means,
            mean_over_available_axes=float(np.mean([m for m in means if m is not None]))
            if available.any()
            else None,
            fullbrake_one_cells=fullbrake_count,
            fullbrake_one_squared_error_sum=fullbrake_sum,
            fullbrake_one_mean=fullbrake_sum / fullbrake_count if fullbrake_count else None,
        ),
    )
    if kind == "qualify":
        result.update(
            window_gradient_matrix=gradient_rank(gradients),
            gradient_window_indices=grad_window_indices,
            rank2_passed=gradient_rank(gradients)["rank"] == 2,
        )
    return result


def fit_after_qualification(
    adapter,
    history,
    certificate,
    *,
    budget=lambda: None,
    optimizer_factory=make_adam,
    progress=None,
):
    """Caller must first verify the external sealed artifact/terminal admission."""
    recomputed = gradient_rank(certificate.get("window_gradient_matrix", {}).get("matrix", []))
    if (
        certificate.get("stage") != "qualify"
        or certificate.get("original_graph") != bool(getattr(adapter, "original_graph", False))
        or certificate.get("observations") != history.observations
        or certificate.get("optimizer_updates") != 0
        or certificate.get("rank2_passed") is not True
        or recomputed["rank"] != 2
        or certificate.get("original_array_fingerprints") != history.fingerprints
        or certificate.get("coefficients_initial") != parameter_values(adapter)
        or certificate.get("coefficients_final") != parameter_values(adapter)
        or certificate.get("fixed_fingerprint") != adapter.fixed_fingerprint()
        or certificate.get("gradient_window_indices")
        != [
            i for i, (a, b) in enumerate(history.windows) if history.effective_loss_valid[a:b].any()
        ]
        or len(recomputed["matrix"]) != history.nonempty_windows
        or recomputed["matrix"]
        != [
            certificate["windows"][i]["gradients"]
            for i in certificate.get("gradient_window_indices", [])
        ]
    ):
        raise ValueError(
            "Exact same-model/data complete-window rank2 certificate required before optimizer"
        )
    baseline = run_pass(adapter, history, kind="evaluate", budget=budget, progress=progress)
    optimizer = optimizer_factory(adapter)
    training = run_pass(
        adapter, history, kind="train", optimizer=optimizer, budget=budget, progress=progress
    )
    del optimizer
    candidate = run_pass(adapter, history, kind="evaluate", budget=budget, progress=progress)
    return dict(
        status="completed_one_pass_pedal_component_fit",
        baseline_fixed_rollout=baseline,
        online_training=training,
        final_fixed_rollout=candidate,
        best_checkpoint_selection=False,
        intermediate_checkpoints_written=0,
        changed_DNg_scalar_values=sum(
            training["coefficients_initial"][k] != training["coefficients_final"][k]
            for k in PARAMETER_NAMES
        ),
        final_two_scalar_checkpoint=training["coefficients_final"],
    )
