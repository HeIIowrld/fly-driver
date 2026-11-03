# ruff: noqa: E501 -- Exact inherited synthetic fixture context is intentional.
"""Synthetic metadata only; no real capture arrays, model, maps, GPU or control."""

import base64
import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.actor_cadence_trace_v1 import ActorCadenceProfile, ActorTrace, ActorTraceSink
from flybrain_drive.pedal_load_b1 import (
    NATIVE_SOURCE_FIELDS,
    api_windows,
    derive_history,
    load_candidate,
)
from flybrain_drive.pedal_train_b1 import (
    CONFIGURATION_BINDINGS,
    R2_INPUTS,
    SOURCE_FILES,
    local_import_closure,
    pedal_gate,
    route_admission,
    seal_declaration,
    sha_bytes,
    sha_file,
    validate_declaration,
)
from scripts.build_pedal_b1 import BASE, TARGET, build, inverse


def declaration():
    value = json.loads(Path("configs/pedal-b1-20261030.json").read_bytes())
    value.update(
        status="sealed_before_capture",
        nonce="a" * 32,
        actual_pipeline_qualification_completed=True,
        native_pedal_calibration_sha256="b" * 64,
        pipeline_qualification_review_sha256="c" * 64,
        native_pedal_continuity_review_sha256="d" * 64,
        route_preflight_sha256="e" * 64,
        expected_start_spline=0.5,
        source_sha256={k: "f" * 64 for k in SOURCE_FILES},
        pipeline_qualification_review_file="SYNTHETIC-pipeline.json",
        route_preflight_file="SYNTHETIC-start.json",
        native_pedal_continuity_review_file="SYNTHETIC-pedal.json",
        controls_ini_sha256=json.loads(Path("configs/ac-axis-probe.json").read_bytes())[
            "profile_sha256"
        ],
    )
    value["source_sha256"].update(CONFIGURATION_BINDINGS)
    value["source_sha256"].update(R2_INPUTS)
    value["bindings"] = dict(
        engine_pid=123,
        generation="fixture-generation",
        epoch=1,
        installed_lua_sha256="f" * 64,
        output="fixture-capture",
    )
    return value


def review(value):
    raw = json.dumps(
        dict(
            schema="flybrain.pedal_calibration_continuity_review.v1",
            original_calibration_sha256=value["native_pedal_calibration_sha256"],
            nonce=value["nonce"],
            current_engine_pid=value["bindings"]["engine_pid"],
            current_generation=value["bindings"]["generation"],
            source_sha256=value["source_sha256"],
            controls_ini_sha256=value["controls_ini_sha256"],
            physical_pedal_input_path_qualified=True,
            steering_qualification_claimed=False,
            fit_allowed=False,
        )
    ).encode()
    value["native_pedal_continuity_review_sha256"] = sha_bytes(raw)
    return raw


def receipt():
    def state(p):
        return dict(
            diagnostic_route=dict(spline_position=p, lap_count=0, actor_input=False),
            wheels_outside=0,
        )

    def call(name, a, b):
        return dict(
            name=name, started_clock=a, finished_clock=b, script_clock=b + 0.001, returned=True
        )

    return dict(
        before=state(0.5),
        after=state(0.53),
        telemetry=[dict(state(0.51), phase="driving"), dict(state(0.52), phase="stopping")],
        api_calls=[
            call("enable_ai", 0.2, 0.21),
            call("set_ai_steer_multiplier", 0.22, 0.23),
            call("set_ai_speed_limit", 0.24, 0.25),
            call("resume_owned_ai", 0.26, 0.27),
            call("pace_set100", 5.01, 5.11),
            call("pace_force_stop8", 15.01, 15.11),
            call("pace_release_stop", 23.01, 23.02),
            call("pace_set60", 23.03, 23.04),
            call("disable_ai", 30.1, 30.2),
        ],
        pace_transitions=[
            dict(script_clock=t, phase=p, desired_speed_m_s=g)
            for t, p, g in zip(
                (0.0, 5.0, 15.0, 23.0),
                ("accelerate60", "accelerate100", "stop0", "accelerate60"),
                (60 / 3.6, 100 / 3.6, 0.0, 60 / 3.6),
                strict=True,
            )
        ],
    )


def clock():
    return dict(
        generation="fixture-generation",
        host_clock_quantization_seconds=0.0,
        offset_lower_seconds=999.999,
        offset_upper_seconds=1000.001,
        exchanges=[
            dict(sent=1000.0, received=1000.002, producer_clock=0.001),
            dict(sent=1040.0, received=1040.002, producer_clock=40.001),
        ],
    )


class Body:
    def __init__(self):
        self.rows = []

    def encode(self, sample, **kwargs):
        values, valid = np.zeros(21), np.ones(21, bool)
        valid[19] = False  # Preserve an unavailable organ, not an all-valid replacement.
        values[16] = sample.speed_magnitude_m_s
        self.rows.append((sample.legacy.packet_id, kwargs))
        return SimpleNamespace(values=values, valid=valid)


def events(
    pedals=((0.0, 1.0), (0.2, 0.3), (0.005, 0.0)), times=(1.0, 1.04, 1.08), phase="accelerate60"
):
    rows = []
    for index, ((gas, brake), source) in enumerate(zip(pedals, times, strict=True)):
        raw = bytearray(420)
        struct.pack_into("<iff", raw, 0, index + 1, gas, brake)
        struct.pack_into("<i", raw, 416, 1)
        sample = decode_teacher_physics(bytes(raw), 1000 + source + 0.005)
        authority = sample.authority_record()
        authority["teacher_authority_eligible"] = True
        native = dict(
            packet_id=index + 1,
            received_monotonic_seconds=1000 + source + 0.005,
            prefix_base64=base64.b64encode(bytes(raw[:308])).decode(),
            prefix_sha256=sha_bytes(bytes(raw[:308])),
        )
        goal = 0.0 if phase == "stop0" else 60 / 3.6
        rows.append(
            dict(
                index=index,
                decision_monotonic_seconds=1000 + source + 0.01,
                native=native,
                native_teacher_authority=authority,
                camera=dict(generation="fixture-generation", capture_id=index),
                external_task_at_receipt=dict(
                    schema="flybrain.external_speed_task.v1",
                    actor_input=True,
                    held_external_instruction=True,
                    desired_speed_m_s=goal,
                    phase=phase,
                    source_script_clock=source - 0.01,
                    observed_at_host_seconds=1000 + source - 0.001,
                ),
            )
        )
    return rows


def frame_reader(rows):
    def get(event):
        source = event["decision_monotonic_seconds"] - 1000 - 0.01
        return dict(
            generation="fixture-generation",
            index=event["index"],
            views=[
                dict(
                    view_id=name,
                    render_started_clock=source,
                    render_call_finished_clock=source + 0.001,
                )
                for name in ("front", "left", "right", "rear")
            ],
        )

    return get


def derive(rows, *, value=None, api=None, with_review=True):
    value = declaration() if value is None else value
    raw = review(value) if with_review else None
    body = Body()
    output = derive_history(
        rows,
        declaration=value,
        receipt=api or receipt(),
        clock_record=clock(),
        frame_reader=frame_reader(rows),
        body_encoder=body,
        episode_id="fixture",
        pedal_review_bytes=raw,
    )
    return output, body


def test_draft_never_seals_missing_identity_or_upgrades_source_role():
    draft = json.loads(Path("configs/pedal-b1-20261030.json").read_bytes())
    validate_declaration(draft, sealed=False)
    with pytest.raises(ValueError, match="Fresh|Actual reviewed hostV4|external"):
        validate_declaration(draft)
    valid = declaration()
    validate_declaration(valid)
    for role in ("diagnostic", "development", "evaluation", "transit"):
        with pytest.raises(ValueError):
            validate_declaration(dict(valid, original_role=role))


def test_collector_exact_derivative_binds_manifest_before_any_transport_activation():
    base, candidate = BASE.read_bytes(), TARGET.read_bytes()
    assert candidate == build(base) and inverse(candidate) == base
    text = candidate.decode()
    assert text.index("role='train', fit_allowed=False") < text.index("context.prepare()")
    assert text.index("stream.write(sealed_train['raw'])") < text.index("context.prepare()")
    assert "train_declaration_sha256" in text and "train_nonce=sealed_train['nonce']" in text


def test_old_role_refused_before_opening_any_events(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(config=dict(role="diagnostic"))), encoding="utf8")
    with pytest.raises(ValueError, match="no role promotion"):
        load_candidate(tmp_path, expected_capture_manifest_sha256=sha_file(manifest))
    assert not (tmp_path / "recording").exists()


def test_all_rows_times_masks_held_inputs_retained_and_only_pedal_cells_masked():
    rows = events()
    original = copy.deepcopy(rows)
    output, body = derive(rows)
    a = output["arrays"]
    assert output["fit_allowed"] is False and rows == original
    assert len(body.rows) == output["observations"] == 3 and output["state_frames_dropped"] == 0
    assert a["previous_held_row_index"].tolist() == [-1, 0, 1]
    assert a["original_elapsed_seconds"][0] == 0
    np.testing.assert_array_equal(a["original_elapsed_seconds"][1:], np.diff(a["decision_clock"]))
    assert not a["body_valid"][:, 19].any() and a["body"].shape == (3, 21)
    assert a["pedal_loss_valid"].tolist() == [[True, True], [False, False], [False, True]]
    assert a["native_pedals"][0, 1] == 1 and not a["native_pedals"].flags.writeable
    assert output["phase_ids"] == ["initial60"] * 3


def test_missing_pedal_review_masks_loss_without_fabricating_full_steering_calibration():
    output, _ = derive(events(), with_review=False)
    assert not output["arrays"]["pedal_loss_valid"].any()
    assert output["full_brake_one_observations"] == 1 and output["observations"] == 3
    value = declaration()
    raw = review(value)
    value["bindings"]["generation"] = "another-generation"
    with pytest.raises(ValueError, match="continuity"):
        pedal_gate(raw, value)


def test_api_postcall_points_are_unknown_and_transition_uncertainty_never_retimes_task():
    api = receipt()
    del api["api_calls"][0]["started_clock"]
    output, _ = derive(events(), api=api)
    assert not output["arrays"]["pedal_loss_valid"].any()
    assert output["arrays"]["physical_command"][0, 1] == np.float32(60 / 3.6)
    activity, phases, reasons = api_windows(receipt(), "fixture-generation")
    assert not reasons and activity.enable_call_bounds.upper == 0.27
    assert [p.phase_id for p in phases] == ["initial60", "accelerate100", "stop0", "restart60"]
    assert phases[1].interval.lower == 5.11


def test_historical_packet_prefix_mismatch_rejected_not_repaired():
    rows = events()
    rows[1]["native"]["prefix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="same original420"):
        derive(rows)


def test_large_original_interval_retained_but_whole_history_not_admitted():
    output, _ = derive(events(times=(1.0, 1.09, 1.13)))
    assert output["observations"] == 3 and not output["source_clock_schedule_valid"]
    assert not output["arrays"]["pedal_loss_valid"].any()
    assert output["arrays"]["original_elapsed_seconds"][1] > 0.08


def test_stop_full_brake_and_zero_task_alias_are_retained_without_a_model_loss_claim():
    output, _ = derive(events(pedals=((0.0, 1.0),) * 3, times=(16.0, 16.04, 16.08), phase="stop0"))
    assert output["phase_ids"] == ["stop0"] * 3
    assert output["qualified_full_brake_one_cells"] == 3
    assert output["zero_task_stationary_fullbrake_rows"] == 3
    assert output["stationary_hold_alias_limit_retained"] and output["model_loss_not_evaluated"]


def test_whole_route_protection_no_telemetry_slicing_or_actor_pose():
    value, api = declaration(), receipt()
    assert route_admission(api, value)["passed"]
    api["telemetry"][0]["diagnostic_route"]["spline_position"] = 0.7
    result = route_admission(api, value)
    assert not result["passed"] and "protected_DEV_EVAL_overlap" in result["reasons"]
    assert result["entire_history_retained"] and result["actor_input"] is False


def test_transition_ambiguity_keeps_original_held_command_instead_of_oracle_replacement():
    output, _ = derive(events(times=(5.05, 5.09, 5.13)))
    assert not output["arrays"]["pedal_loss_valid"].any()
    np.testing.assert_array_equal(
        output["arrays"]["physical_command"][:, 1], np.full(3, np.float32(60 / 3.6))
    )
    assert output["observations"] == 3
    assert output["held_task_never_replaced_by_actual_phase_or_oracle"]


def test_complete_synthetic_file_loader_uses_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "flybrain_drive.pedal_load_b1.external_binding_gate",
        lambda value: dict(SYNTHETIC_prequalified_metadata_fixture=True, fit_allowed=False),
    )
    owned_cleanup = cleanup_fixture()
    root = tmp_path / "future-train-candidate"
    recording, camera = root / "recording", root / "recording/vision"
    camera.mkdir(parents=True)
    (recording / "control").mkdir()
    value = declaration()
    value["bindings"]["output"] = str(root)
    value["source_sha256"] = {p: sha_file(p) for p in SOURCE_FILES}
    value["bindings"]["installed_lua_sha256"] = value["source_sha256"][
        "integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua"
    ]
    pedal_review = review(value)
    role_raw = json.dumps(value).encode()
    (root / "original-train-declaration.json").write_bytes(role_raw)
    (root / "executed-program.py").write_bytes(TARGET.read_bytes())
    manifest = dict(
        config=dict(
            role="train",
            original_train_declaration=value,
            program_sha256=sha_file(TARGET),
            train_nonce=value["nonce"],
            train_declaration_sha256=sha_bytes(role_raw),
            actor_cadence_profile=ActorCadenceProfile().record(),
            actor_cadence_profile_sha256=value["source_sha256"][value["actor_profile"]],
        )
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    (root / "result.json").write_text(
        json.dumps(
            dict(
                status="completed_train_candidate_b1_transport_collection",
                role="train",
                fit_allowed=False,
                train_declaration_sha256=sha_bytes(role_raw),
                transport_state="closed",
                owned_threads_still_live=[],
                lease_retained=False,
                heartbeat_evidence=dict(status="complete"),
                actor_trace=dict(
                    status="complete", index_file=str(root / "actor-trace/index.json")
                ),
            )
        ),
        encoding="utf8",
    )
    child = Path("scripts/record_ac_engine_ai_probe_actor_trace_v1.py")
    (recording / "executed-program.py").write_bytes(child.read_bytes())
    proof = dict(
        schema="flybrain.native.same_copy_recorder_source_proof.v1",
        before_camera_activation=True,
        before_control_activation=True,
        executed_program_sha256=sha_file(child),
        source_files={p: dict(sha256=sha_file(p)) for p in NATIVE_SOURCE_FIELDS},
    )
    (recording / "native-authority-source-proof.json").write_text(
        json.dumps(proof), encoding="utf8"
    )
    args = {
        k: True
        for k in (
            "pace_task",
            "same_copy_native_authority",
            "mmf_metadata",
            "batched_shared_copy",
            "async_frame_persistence",
            "source_timing",
        )
    }
    from flybrain_drive.frame_persistence_profile_v4 import HostCopyProfile

    args.update(
        flight_shadow_config=None,
        duration_seconds=30,
        frame_persistence_profile="configs/ac-frame-host-v4-20261030.json",
        actor_cadence_profile="configs/ac-actor-cadence-trace-30ms-20261030.json",
    )
    (recording / "manifest.json").write_text(
        json.dumps(
            dict(
                config=dict(
                    arguments=args,
                    frame_persistence_profile=HostCopyProfile().record(),
                    actor_cadence_profile=ActorCadenceProfile().record(),
                    actor_cadence_profile_sha256=value["source_sha256"][value["actor_profile"]],
                )
            )
        ),
        encoding="utf8",
    )
    (recording / "result.json").write_text(
        json.dumps(
            dict(
                status="completed",
                decisions=3,
                frame_persistence_profile=HostCopyProfile().record(),
                camera_wait_misses=0,
                native_baseline_packet_id=0,
                owned_cleanup=owned_cleanup,
                actor_cadence_profile=ActorCadenceProfile().record(),
                actor_trace_decisions=3,
            )
        ),
        encoding="utf8",
    )
    (recording / "identity.json").write_text("{}", encoding="utf8")
    api = receipt()
    api.update(
        explicit_pace_task=True,
        pace_task_passed=True,
        pace_contract="flybrain.external_speed_task.v1",
        motion_hard_speed_kmh=120,
        maximum_displacement_guard_m=1100,
        maximum_speed_kmh=100,
        pace_brake_peak=0.2,
        pace_moving_brake_seen=True,
        pace_before_stop_speed_kmh=70,
        pace_min_speed_kmh=0,
    )
    for key in ("before", "after"):
        api[key].update(generation=value["bindings"]["generation"], epoch=1)
    api["telemetry"][0].update(external_task_phase="stop0", speed_kmh=30, brake=0.2)
    (recording / "control/api-receipt.json").write_text(json.dumps(api), encoding="utf8")
    archive, rows = [], events()
    for row in rows:
        name = f"frame-{row['index']:04d}.json"
        raw = json.dumps(frame_reader(rows)(row)).encode()
        row["camera"]["file_sha256"] = sha_bytes(raw)
        (camera / name).write_bytes(raw)
        archive.append(dict(file=name, bytes=len(raw), sha256=sha_bytes(raw)))
    raw = json.dumps(clock()).encode()
    (camera / "clock-final.json").write_bytes(raw)
    archive.append(dict(file="clock-final.json", bytes=len(raw), sha256=sha_bytes(raw)))
    (camera / "archive.json").write_text(json.dumps(archive), encoding="utf8")
    (recording / "events.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf8"
    )
    save_trace_fixture(root / "actor-trace", rows, owned_cleanup)
    monkeypatch.setattr(
        "flybrain_drive.pedal_load_b1.NavigationBodyEncoder", lambda *a, **k: Body()
    )
    loaded = load_candidate(
        root,
        expected_capture_manifest_sha256=sha_file(root / "manifest.json"),
        pedal_review_bytes=pedal_review,
    )
    assert loaded["observations"] == 3 and loaded["fit_allowed"] is False
    assert not (recording / "observations.npz").exists()
    assert loaded["source_references"]["source_payloads_copied"] == 0


def test_invalid_required_body_mask_retains_all_rows_but_cannot_admit_state_advance():
    class MissingYaw(Body):
        def encode(self, sample, **kwargs):
            value = super().encode(sample, **kwargs)
            value.valid[4] = False
            return value

    value, rows = declaration(), events()
    raw = review(value)
    loaded = derive_history(
        rows,
        declaration=value,
        receipt=receipt(),
        clock_record=clock(),
        frame_reader=frame_reader(rows),
        body_encoder=MissingYaw(),
        episode_id="fixture",
        pedal_review_bytes=raw,
    )
    assert loaded["observations"] == 3 and loaded["state_frames_dropped"] == 0
    assert not loaded["physical_body_interface_valid"]
    assert not loaded["arrays"]["body_valid"][:, 4].any()
    assert not loaded["arrays"]["pedal_loss_valid"].any()


def test_missing_real_external_reviews_refuse_even_a_structural_role_before_write(tmp_path):
    value = declaration()
    with pytest.raises(FileNotFoundError):
        seal_declaration(value, tmp_path / "must-not-seal.json")
    assert not (tmp_path / "must-not-seal.json").exists()


def test_mandatory_effective_source_closure_covers_all_consumed_dependencies():
    minimum = {
        "configs/ac-csp-exterior-front-v3-retina.json",
        "configs/ac-axis-probe.json",
        "src/flybrain_drive/immutable_frame_writer.py",
        "src/flybrain_drive/ac_capture.py",
        "src/flybrain_drive/ac_vjoy.py",
        "src/flybrain_drive/evidence.py",
        "src/flybrain_drive/ac_camera_archive.py",
        "src/flybrain_drive/ac_camera_mount.py",
        "src/flybrain_drive/ac_shared_texture_host_v4.py",
        "src/flybrain_drive/ac_pixel_retina.py",
        "src/flybrain_drive/ac_metadata_mmf.py",
        "src/flybrain_drive/ac_csp_geometry.py",
        "src/flybrain_drive/session_heartbeat_candidate.py",
    }
    assert minimum <= set(SOURCE_FILES)
    assert set(CONFIGURATION_BINDINGS) <= set(SOURCE_FILES)
    valid = declaration()
    for name in sorted(minimum):
        missing = copy.deepcopy(valid)
        del missing["source_sha256"][name]
        with pytest.raises(ValueError, match="source qualification"):
            validate_declaration(missing)
    reference = next(iter(CONFIGURATION_BINDINGS))
    valid["source_sha256"][reference] = "0" * 64
    with pytest.raises(ValueError, match="reference binding"):
        validate_declaration(valid)


def test_ast_closure_follows_nested_relative_imports_and_cycles_without_execution(tmp_path):
    package = tmp_path / "src/flybrain_drive"
    package.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (package / "source_closure_inventory_v1.py").write_bytes(
        Path("src/flybrain_drive/source_closure_inventory_v1.py").read_bytes()
    )
    (package / "__init__.py").write_text('raise RuntimeError("must not execute")\n')
    (package / "first.py").write_text("def later():\n    from .second import helper\n")
    (package / "second.py").write_text("from .first import later\ndef helper(): pass\n")
    assert set(local_import_closure(["src/flybrain_drive/first.py"], project_root=tmp_path)) == {
        "src/flybrain_drive/first.py",
        "src/flybrain_drive/second.py",
        "src/flybrain_drive/__init__.py",
    }


def cleanup_fixture():
    return dict(
        schema="flybrain.collection.owned_cleanup.v4",
        complete=True,
        control_process_terminal=True,
        camera={
            k: True
            for k in (
                "camera_thread_joined",
                "producer_terminal_known",
                "persistence_joined",
                "precompute_joined",
                "shared_reader_closed",
            )
        },
    )


def save_trace_fixture(directory, rows, cleanup):
    value = [0]

    def tick():
        value[0] += 100
        return value[0]

    trace = ActorTrace(wall=tick, cpu=tick)
    for i, event in enumerate(rows):
        trace.begin_iteration()
        trace.budget_started(float(i))
        for phase in (
            "child_poll",
            "status",
            "candidate_wait",
            "native_receive",
            "candidate_select",
        ):
            with trace.span(phase):
                pass
        trace.decision(i, event["decision_monotonic_seconds"])
        for phase in ("retinal_pack", "event_encode", "event_write", "event_flush"):
            with trace.span(phase):
                pass
        with trace.span("sleep"):
            trace.sleep_requested(0.0)
        trace.finish_iteration()
    trace.begin_iteration()
    trace.budget_started(float(len(rows)))
    with trace.span("child_poll"):
        pass
    trace.finish_iteration()
    trace.seal()
    sink = ActorTraceSink()
    sink.accept(trace, cleanup)
    assert sink.persist(directory, context=None)["status"] == "complete"
