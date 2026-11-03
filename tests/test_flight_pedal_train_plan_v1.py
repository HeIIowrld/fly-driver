"""No model/device/data imports needed by source preparation and seal rejection."""

import builtins
import json
from pathlib import Path

import pytest

from flybrain_drive.flight_pedal_train_plan_v1 import (
    DRAFT_PATH,
    DRAFT_SHA256,
    preparation_template,
    sha,
    validate_before_model,
)


def test_current_two_stage_draft_remains_unsealed_and_unknown_loader():
    raw = Path(DRAFT_PATH).read_bytes()
    assert sha(DRAFT_PATH) == DRAFT_SHA256
    plan = preparation_template(raw)
    assert plan["sealed"] is False and plan["model_execution_authorized"] is False
    assert all(v is None for v in plan["loader_binding"].values())
    assert all(v is None for v in plan["gradient_qualification"].values())
    assert plan["fit_allowed"] is False
    protocol = json.loads(raw)
    assert (
        "new_train_tbptt_gradient_qualification_sha256"
        not in (protocol["bindings_to_seal_before_any_model"])
    )
    assert (
        protocol["additional_bindings_to_seal_before_any_optimizer"][
            "new_train_tbptt_gradient_qualification_sha256"
        ]
        is None
    )


def test_unsealed_refusal_occurs_before_any_hash_or_import(monkeypatch):
    plan = preparation_template(Path(DRAFT_PATH).read_bytes())
    import flybrain_drive.flight_pedal_train_plan_v1 as gate

    monkeypatch.setattr(
        gate,
        "sha",
        lambda path: (_ for _ in ()).throw(
            AssertionError("No source/data hash should precede seal refusal")
        ),
    )
    original = builtins.__import__

    def forbid_model(name, *args, **kwargs):
        if name.startswith(("torch", "flybrain_drive.pedal_train_loader")):
            raise AssertionError("No model or future loader import before seal")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_model)
    for stage in ["qualify_gradients", "fit"]:
        with pytest.raises(ValueError, match="UNSEALED"):
            validate_before_model(plan, stage=stage)


def test_old_pre_fix_draft_or_hyperparameter_change_cannot_prepare():
    raw = Path(DRAFT_PATH).read_bytes()
    with pytest.raises(ValueError, match="Exact reviewed"):
        preparation_template(raw + b" ")


def test_stage_permission_does_not_turn_gradient_qualification_into_fit():
    plan = preparation_template(Path(DRAFT_PATH).read_bytes())
    plan.update(
        sealed=True,
        model_execution_authorized=True,
        execution_stage="qualify_gradients",
        fit_allowed=True,
    )
    with pytest.raises(ValueError, match="fitfalse"):
        validate_before_model(plan, stage="qualify_gradients")


def test_execute_unsealed_never_constructs_adapter_or_output(tmp_path, monkeypatch):
    from scripts import train_flight_pedal_first_history_v1 as trainer

    plan = preparation_template(Path(DRAFT_PATH).read_bytes())
    plan["output"] = str(tmp_path / "never-created")
    calls = []
    monkeypatch.setattr(trainer, "OriginalGraphAdapter", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="UNSEALED"):
        trainer.execute(plan, DRAFT_PATH, stage="qualify_gradients")
    assert calls == [] and not Path(plan["output"]).exists()
