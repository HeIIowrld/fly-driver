"""Small two-scalar CPU mocks only; no original graph, dataset or GPU query."""

import copy

import numpy as np
import pytest
import torch

from flybrain_drive.pedal_tbptt_engine_v1 import (
    PARAMETER_NAMES,
    fingerprint,
    fit_after_qualification,
    gradient_rank,
    make_adam,
    run_pass,
    whole_history,
)


def history(n=24, *, mask=None):
    clock = 10.0 + np.arange(n, dtype=np.float64) * 0.04
    body = np.zeros((n, 21), np.float32)
    body[:, 16] = 1 + (np.arange(n) % 5)
    commands = np.zeros((n, 2), np.float32)
    commands[:, 1] = 1 + (np.arange(n) % 3)
    native = np.zeros((n, 2), np.float32)
    native[::2, 0] = 0.8
    native[1::2, 1] = 1
    values = dict(
        decision_clock=clock,
        native_receipt_clock=clock - 0.0001,
        native_packet=np.arange(100, 100 + n, dtype=np.int64),
        body=body,
        body_valid=np.ones((n, 21), bool),
        physical_command=commands,
        racing_goals=np.tile(np.asarray([0.0, 1.0], np.float32), (n, 1)),
        native_pedals=native,
        pedal_loss_valid=np.ones((n, 2), bool) if mask is None else mask,
        original_elapsed_seconds=np.r_[0.0, np.diff(clock)],
        previous_held_row_index=np.r_[-1, np.arange(n - 1)],
    )
    return whole_history(values)


class MockAdapter:
    original_graph = False

    def __init__(self, *, deficient=False):
        self.a = torch.nn.Parameter(torch.tensor(0.1))
        self.b = torch.nn.Parameter(torch.tensor(0.2))
        self.deficient = deficient
        self.initializations = 0
        self.advances = []
        self.fixed = "unchanged-protected-mock"
        self.quiet_receipt = None

    def parameters(self):
        return dict(zip(PARAMETER_NAMES, (self.a, self.b), strict=True))

    def fixed_fingerprint(self):
        return self.fixed

    def initialize(self):
        self.initializations += 1
        self.quiet_receipt = dict(mock=True, original_graph=False)
        return torch.zeros(1, 2)

    def advance(self, whole, held, dt, state):
        command = float(whole.arrays["physical_command"][held, 1])
        speed = float(whole.arrays["body"][held, 16])
        self.advances.append(
            dict(
                held=held, dt=dt, command=command, speed=speed, state_before=state.detach().clone()
            )
        )
        right = self.a if self.deficient else self.b
        return state + torch.stack((self.a * command * dt, right * speed * dt))[None]

    def predict(self, state):
        pedals = state.sigmoid()
        steer = 0.01 * (state[:, :1] - state[:, 1:])
        return dict(canonical=torch.cat((steer, pedals), -1))

    def record(self, prediction):
        return dict(canonical=prediction["canonical"].detach().tolist())

    def state_fingerprint(self, state):
        return fingerprint(state.detach().numpy())


def test_previous_held_inputs_no_zero_advance_no_hidden_reset_even_all_masked():
    whole = history(10, mask=np.zeros((10, 2), bool))
    before = {k: a.copy() for k, a in whole.arrays.items()}
    adapter = MockAdapter()
    result = run_pass(adapter, whole, kind="qualify")
    assert adapter.initializations == 1
    assert [x["held"] for x in adapter.advances] == list(range(9))
    assert [x["dt"] for x in adapter.advances] == whole.arrays["original_elapsed_seconds"][
        1:
    ].tolist()
    row1 = result["rows"][1]["prediction"]["canonical"][0][1]
    expected = torch.sigmoid(
        torch.tensor(0.1)
        * float(whole.arrays["physical_command"][0, 1])
        * float(whole.arrays["original_elapsed_seconds"][1])
    ).item()
    wrong_current = torch.sigmoid(
        torch.tensor(0.1)
        * float(whole.arrays["physical_command"][1, 1])
        * float(whole.arrays["original_elapsed_seconds"][1])
    ).item()
    assert row1 == expected and row1 != wrong_current
    assert result["optimizer_updates"] == 0 and result["window_gradient_matrix"]["shape"] == [0, 2]
    assert [w["end"] for w in result["windows"]] == [8, 10]
    assert all(w["state_before_detach"] == w["state_after_detach"] for w in result["windows"])
    assert all(np.array_equal(before[k], a) for k, a in whole.arrays.items())


def test_empty_middle_window_advances_all_states_short_final_steps_exactly_twice():
    mask = np.ones((19, 2), bool)
    mask[8:16] = False
    whole = history(19, mask=mask)
    adapter = MockAdapter()
    result = run_pass(adapter, whole, kind="train", optimizer=make_adam(adapter))
    assert len(result["rows"]) == 19 and len(adapter.advances) == 18
    assert [w["optimizer_step"] for w in result["windows"]] == [True, False, True]
    assert [(w["start"], w["end"]) for w in result["windows"]] == [(0, 8), (8, 16), (16, 19)]
    assert result["optimizer_updates"] == whole.nonempty_windows == 2
    assert (
        result["windows"][0]["state_before_detach"] != result["windows"][1]["state_before_detach"]
    )
    assert result["windows"][1]["coefficients"] == result["windows"][2]["coefficients"]


def test_complete_rank2_gate_precedes_optimizer_and_fixed_rollouts_are_separate():
    whole = history()
    adapter = MockAdapter()
    certificate = run_pass(adapter, whole, kind="qualify")
    assert certificate["rank2_passed"]
    before = dict(certificate["coefficients_initial"])
    assert certificate["coefficients_final"] == before
    fitted = fit_after_qualification(adapter, whole, certificate)
    baseline, training, final = (
        fitted[k] for k in ("baseline_fixed_rollout", "online_training", "final_fixed_rollout")
    )
    assert adapter.initializations == 4  # qualification plus three independent full passes
    assert baseline["coefficients_initial"] == baseline["coefficients_final"] == before
    assert training["optimizer_updates"] == 3
    assert (
        final["coefficients_initial"]
        == final["coefficients_final"]
        == training["coefficients_final"]
    )
    assert fitted["changed_DNg_scalar_values"] == 2
    assert baseline["rows"] == certificate["rows"]
    assert baseline["rows"] != final["rows"]
    # A coupled steering output can move even though only two pedal coefficients learn.
    assert (
        baseline["rows"][-1]["prediction"]["canonical"][0][0]
        != (final["rows"][-1]["prediction"]["canonical"][0][0])
    )


def test_rank_deficient_all_windows_preserved_and_no_optimizer_factory_called():
    whole = history()
    adapter = MockAdapter(deficient=True)
    cert = run_pass(adapter, whole, kind="qualify")
    assert cert["window_gradient_matrix"]["rank"] == 1
    assert len(cert["window_gradient_matrix"]["matrix"]) == 3
    assert all(row[1] == 0 for row in cert["window_gradient_matrix"]["matrix"])
    calls = []
    with pytest.raises(ValueError, match="before optimizer"):
        fit_after_qualification(
            adapter, whole, cert, optimizer_factory=lambda model: calls.append(model)
        )
    assert calls == []


def test_whole_history_mismatch_refuses_certificate_before_baseline_or_optimizer():
    adapter = MockAdapter()
    whole = history()
    cert = run_pass(adapter, whole, kind="qualify")
    other = history()
    other.arrays["body"][3, 16] += 0.01
    other = whole_history(other.arrays)
    count = adapter.initializations
    with pytest.raises(ValueError, match="same-model/data"):
        fit_after_qualification(adapter, other, cert)
    assert adapter.initializations == count


def test_fullbrake_one_retained_coactivation_only_masks_both_pedal_cells():
    whole = history(16)
    whole.arrays["native_pedals"][4] = [0.5, 1.0]
    whole = whole_history(whole.arrays)
    assert not whole.effective_loss_valid[4].any()
    assert whole.effective_loss_valid[1, 1]
    result = run_pass(MockAdapter(), whole, kind="evaluate")
    assert result["metrics"]["fullbrake_one_cells"] == 8
    assert result["metrics"]["fullbrake_one_squared_error_sum"] > 0
    assert len(result["rows"]) == 16


def test_nonempty_zero_gradient_window_retained_in_complete_matrix():
    whole = history(8, mask=np.zeros((8, 2), bool))
    whole.arrays["pedal_loss_valid"][0] = True
    whole = whole_history(whole.arrays)
    cert = run_pass(MockAdapter(), whole, kind="qualify")
    assert cert["gradient_window_indices"] == [0]
    assert cert["window_gradient_matrix"]["matrix"] == [[0.0, 0.0]]
    assert not cert["rank2_passed"]


def test_clock_repair_current_held_indices_and_missing_tail_rejected():
    for key in ["original_elapsed_seconds", "previous_held_row_index"]:
        a = {k: v.copy() for k, v in history(10).arrays.items()}
        a[key][2] += 0.001 if key == "original_elapsed_seconds" else 1
        with pytest.raises(ValueError, match="no repair"):
            whole_history(a)
    a = {k: v.copy() for k, v in history(10).arrays.items()}
    a["body"] = a["body"][:-1]
    with pytest.raises(ValueError, match="shape/dtype"):
        whole_history(a)


def test_corrupt_frozen_value_is_not_absorbed_into_training_state():
    class Corrupt(MockAdapter):
        def advance(self, whole, held, dt, state):
            result = super().advance(whole, held, dt, state)
            if held == 5:
                self.fixed = "corrupt"
            return result

    with pytest.raises(ValueError, match="Frozen graph"):
        run_pass(Corrupt(), history(), kind="qualify")


def test_metrics_are_global_axis_sum_counts_and_rank_has_no_row_selection():
    whole = history()
    result = run_pass(MockAdapter(), whole, kind="evaluate")
    canonical = np.asarray(
        [r["prediction"]["canonical"][0][1:] for r in result["rows"]], np.float32
    )
    native = whole.arrays["native_pedals"]
    upper = np.where(native == 0, np.float32(0.01), native)
    error = np.maximum(native - canonical, 0) + np.maximum(canonical - upper, 0)
    expected = np.where(whole.effective_loss_valid, error * error, 0).astype(np.float64).sum(0)
    assert result["metrics"]["per_axis_squared_error_sum"] == expected.tolist()
    assert gradient_rank([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])["shape"] == [3, 2]
    deficient = copy.deepcopy([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
    assert gradient_rank(deficient)["rank"] == 1


def test_full_fixed_fingerprint_covers_command_buffers_excludes_only_two_scalars():
    from types import SimpleNamespace

    from scripts.train_flight_pedal_first_history_v1 import OriginalGraphAdapter

    values = {
        "command.power_delta": torch.tensor(0.1),
        "command.speed_feedforward_delta": torch.tensor(0.2),
        "command.turn_delta": torch.zeros(2),
        "command.orientations": torch.tensor([1.0, -1.0]),
        "backbone.readout.power_reference": torch.tensor(0.001),
    }
    adapter = object.__new__(OriginalGraphAdapter)  # no original constructor/model/device
    adapter.model = SimpleNamespace(state_dict=lambda: values)
    adapter.tensor_digest = lambda value: fingerprint(value.numpy())
    fixed = adapter.fixed_fingerprint()
    values["command.power_delta"].add_(0.1)
    assert adapter.fixed_fingerprint() == fixed
    values["command.orientations"].add_(1)
    assert adapter.fixed_fingerprint() != fixed
