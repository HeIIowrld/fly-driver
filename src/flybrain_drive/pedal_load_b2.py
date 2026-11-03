"""Whole prospective TRAIN candidate -> RAM only, always fitfalse; no live interfaces.

Original states/held inputs are never sliced, interpolated, reset or repaired.
This loader does not independently qualify pixel replay, pedal calibration,
exclusive authority, continuous route coverage or an optimizer/data admission.
"""

import base64
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .ac_teacher_physics import decode_teacher_physics
from .actor_cadence_trace_v1 import ActorCadenceProfile, verify_actor_trace
from .camera_status_policy_v1 import ECHO_KEY, from_record, validate_producer_echo
from .frame_persistence_profile_v4 import HostCopyProfile
from .pedal_train_b2 import (
    ACTOR_PROFILE,
    CAMERA_STATUS_PROFILE,
    external_binding_gate,
    is_sha,
    pedal_gate,
    route_admission,
    sha_bytes,
    sha_file,
    validate_declaration,
)
from .teacher_loss_eligibility import (
    SourceInterval,
    TaskPhase,
    TeacherActivity,
    project_host_receipt_interval,
)
from .teacher_pace import PACE_CONTRACT, goal_matches, validate_pace_receipt
from .virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix

PHASE_IDS = ("initial60", "accelerate100", "stop0", "restart60")
PHASE_NAMES = ("accelerate60", "accelerate100", "stop0", "accelerate60")
GOALS = (60 / 3.6, 100 / 3.6, 0.0, 60 / 3.6)
NATIVE_SOURCE_FIELDS = frozenset(
    {
        "src/flybrain_drive/ac_teacher_physics.py",
        "src/flybrain_drive/ac_native_physics.py",
        "src/flybrain_drive/ac_teacher_status.py",
        "src/flybrain_drive/ac_image_buffer.py",
        "src/flybrain_drive/ac_csp_vision_lifecycle_status_v1.py",
        "src/flybrain_drive/camera_status_policy_v1.py",
        "integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua",
        "src/flybrain_drive/ac_csp_retina.py",
        "scripts/record_ac_wgc.py",
        "src/flybrain_drive/frame_persistence_profile_v4.py",
        "src/flybrain_drive/ac_shared_texture_host_v4.py",
        "src/flybrain_drive/collection_cleanup_v4.py",
        "src/flybrain_drive/actor_cadence_trace_v1.py",
    }
)


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return sha_bytes(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes())


def api_windows(receipt, generation):
    """Conservative callback bounds; post-call points never replace START metadata."""
    calls = receipt.get("api_calls", [])
    required = (
        "enable_ai",
        "set_ai_steer_multiplier",
        "set_ai_speed_limit",
        "resume_owned_ai",
        "pace_set100",
        "pace_force_stop8",
        "pace_release_stop",
        "pace_set60",
        "disable_ai",
    )
    selected = {}
    for name in required:
        rows = [r for r in calls if r.get("name") == name]
        if len(rows) != 1:
            return None, [], ["API_call_missing_duplicate_or_unknown"]
        row = rows[0]
        bounds = [row.get(k) for k in ("started_clock", "finished_clock", "script_clock")]
        if (
            row.get("returned") is not True
            or any(type(x) not in (int, float) or not np.isfinite(x) for x in bounds)
            or not 0 <= bounds[0] < bounds[1] <= bounds[2]
        ):
            return None, [], ["API_original_start_finish_bounds_unqualified"]
        selected[name] = row
    transitions = receipt.get("pace_transitions", [])
    if (
        len(transitions) != 4
        or [r.get("phase") for r in transitions] != list(PHASE_NAMES)
        or not all(
            goal_matches(r.get("desired_speed_m_s"), g)
            for r, g in zip(transitions, GOALS, strict=True)
        )
    ):
        return None, [], ["actual_task_transition_declaration_incomplete"]
    transition = [r.get("script_clock") for r in transitions]
    if any(type(t) not in (int, float) or not np.isfinite(t) for t in transition):
        return None, [], ["actual_task_transition_clock_unknown"]
    setup_finish = max(selected[k]["finished_clock"] for k in required[:4])
    disable = selected["disable_ai"]
    lower = [
        setup_finish,
        max(transition[1], selected["pace_set100"]["finished_clock"]),
        max(transition[2], selected["pace_force_stop8"]["finished_clock"]),
        max(
            transition[3],
            selected["pace_release_stop"]["finished_clock"],
            selected["pace_set60"]["finished_clock"],
        ),
    ]
    upper = transition[1:] + [disable["started_clock"]]
    if (
        any(not a < b for a, b in zip(lower, upper, strict=True))
        or any(a >= b for a, b in zip(transition, transition[1:], strict=False))
        or not selected["enable_ai"]["started_clock"] < setup_finish
    ):
        return None, [], ["actual_task_API_windows_overlap_or_reverse"]
    activity = TeacherActivity(
        SourceInterval(generation, selected["enable_ai"]["started_clock"], setup_finish),
        SourceInterval(generation, disable["started_clock"], disable["finished_clock"]),
    )
    return (
        activity,
        [
            TaskPhase(name, SourceInterval(generation, a, b))
            for name, a, b in zip(PHASE_IDS, lower, upper, strict=True)
        ],
        [],
    )


def temporal_reasons(sample, cameras, projected, activity, phases, held):
    reasons = []
    if not sample.builtin_ai_controlled:
        reasons.append("native_AI_false")
    if activity is None or projected is None or len(cameras) != 4:
        return reasons + ["whole_camera_HOST_activity_bounds_unavailable"], None
    intervals = [*cameras, projected.source_interval]
    generation = activity.enable_call_bounds.generation
    active = (activity.enable_call_bounds.upper, activity.disable_call_bounds.lower)
    if any(
        i.generation != generation or not active[0] < i.lower <= i.upper < active[1]
        for i in intervals
    ):
        reasons.append("whole_camera_or_HOST_receipt_outside_active_teacher_window")
    # Identity comes from original HELD heartbeat source point, not desired speed alone.
    point = held["source_script_clock"]
    possible = [
        (i, p)
        for i, p in enumerate(phases)
        if p.interval.lower < point < p.interval.upper
        and held["phase"] == PHASE_NAMES[i]
        and goal_matches(held["desired_speed_m_s"], GOALS[i])
    ]
    if len(possible) != 1:
        return reasons + ["held_task_phase_identity_ambiguous"], None
    _, phase = possible[0]
    if any(
        i.generation != generation
        or not phase.interval.lower < i.lower <= i.upper < phase.interval.upper
        for i in intervals
    ):
        reasons.append("whole_camera_or_HOST_receipt_phase_ambiguous")
    return reasons, phase.phase_id


class MetadataCamera:
    """Read explicit flat JSON members in-place; never stage/copy pixels or ZIPs."""

    def __init__(self, directory):
        self.root = Path(directory)
        rows = json.loads((self.root / "archive.json").read_bytes())
        self.records = {r["file"]: r for r in rows}
        if len(rows) != len(self.records):
            raise ValueError("Duplicate original camera archive member")
        self.bundle = None
        descriptor = self.root / "bundle.json"
        if descriptor.exists():
            info = json.loads(descriptor.read_bytes())
            if (
                info.get("file") != "camera.zip"
                or sha_file(self.root / "archive.json") != info["archive_manifest_sha256"]
            ):
                raise ValueError("Original camera descriptor differs")
            self.bundle = zipfile.ZipFile(self.root / "camera.zip")
            names = self.bundle.namelist()
            if len(names) != len(set(names)) or set(names) != set(self.records):
                self.bundle.close()
                raise ValueError("Original camera bundle member set differs")

    def read(self, name):
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"(frame-\d{4}|clock-final|result)\.json", name)
            or name not in self.records
        ):
            raise ValueError("Only explicit original frame/clock JSON metadata may be read")
        record = self.records[name]
        if not 0 < record["bytes"] <= 256 * 1024:
            raise ValueError("Bounded source metadata member required")
        if self.bundle:
            value = self.bundle.read(name)
        else:
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("Camera metadata escaped original root")
            value = path.read_bytes()
        if len(value) != record["bytes"] or sha_bytes(value) != record["sha256"]:
            raise ValueError("Original camera metadata bytes differ")
        return value

    def close(self):
        if self.bundle:
            self.bundle.close()


def camera_intervals(event, frame, generation):
    candidate = event["camera"]
    if (
        candidate["generation"] != generation
        or frame["generation"] != generation
        or frame["index"] != candidate["capture_id"]
        or {v["view_id"] for v in frame["views"]} != {"front", "left", "right", "rear"}
        or len(frame["views"]) != 4
    ):
        raise ValueError("Same original four-view frame/generation required")
    result = []
    for view in frame["views"]:
        lower, upper = view["render_started_clock"], view["render_call_finished_clock"]
        for key in ("save_finished_clock", "reference_finished_clock"):
            if view.get(key) is not None:
                upper = max(upper, view[key])
        result.append(SourceInterval(generation, lower, upper))
    return result


def derive_history(
    events,
    *,
    declaration,
    receipt,
    clock_record,
    frame_reader,
    body_encoder,
    episode_id,
    pedal_review_bytes=None,
):
    """Pure original-event traversal; caller supplies saved metadata/body encoder only."""
    generation = declaration["bindings"]["generation"]
    calibration = pedal_gate(pedal_review_bytes, declaration)
    activity, phases, window_reasons = api_windows(receipt, generation)
    route = route_admission(receipt, declaration)
    n = len(events)
    if n < 2:
        raise ValueError("Whole nonempty multi-observation history required")
    values, masks, clocks, receipts, packets, commands, native, loss_masks = ([] for _ in range(8))
    raw_held, reasons, phase_ids, raw_prefix_sha, camera_refs = ([] for _ in range(5))
    for index, event in enumerate(events):
        if event.get("index") != index:
            raise ValueError("Every original ordered observation required; no slicing")
        decision, old = event["decision_monotonic_seconds"], event["native"]
        authority = event["native_teacher_authority"]
        raw = base64.b64decode(authority["prefix_base64"], validate=True)
        sample = decode_teacher_physics(raw, old["received_monotonic_seconds"])
        if (
            sha_bytes(raw) != authority["prefix_sha256"]
            or sample.physics.prefix_bytes != base64.b64decode(old["prefix_base64"], validate=True)
            or sha_bytes(sample.physics.prefix_bytes) != old["prefix_sha256"]
            or authority["body_prefix_sha256"] != old["prefix_sha256"]
            or sample.physics.packet_id != old["packet_id"]
            or authority["packet_id"] != old["packet_id"]
            or authority["received_monotonic_seconds"] != old["received_monotonic_seconds"]
            or authority["builtin_ai_controlled"] is not sample.builtin_ai_controlled
            or authority.get("actor_input") is not False
            or authority.get("same_native_copy") is not True
        ):
            raise ValueError("Native body/control/AI must match same original420-byte copy")
        body = body_encoder.encode(
            decode_navigation_body_prefix(
                sample.physics.prefix_bytes, sample.physics.received_monotonic_seconds
            ),
            episode_id=episode_id,
            decision_monotonic_seconds=decision,
        )
        held = event["external_task_at_receipt"]
        if (
            held.get("schema") != PACE_CONTRACT
            or held.get("actor_input") is not True
            or held.get("held_external_instruction") is not True
            or held.get("phase") not in set(PHASE_NAMES)
            or not 0
            <= held["observed_at_host_seconds"]
            <= old["received_monotonic_seconds"]
            <= decision
            or not np.isfinite(held["source_script_clock"])
            or not any(goal_matches(held["desired_speed_m_s"], g) for g in GOALS)
        ):
            raise ValueError("Original held external task required; never oracle replace it")
        frame = frame_reader(event)
        cameras = camera_intervals(event, frame, generation)
        try:
            projected = project_host_receipt_interval(
                old["received_monotonic_seconds"], clock_record, generation=generation
            )
        except ValueError:
            projected = None
        why, phase_id = temporal_reasons(sample, cameras, projected, activity, phases, held)
        why += window_reasons
        if authority.get("teacher_authority_eligible") is not True:
            why.append("recorded_samecopy_necessary_authority_gate_false")
        if not 0 <= decision - old["received_monotonic_seconds"] <= 0.05:
            why.append("original_body_receipt_age_outside50ms")
        if not route["passed"]:
            why.append("whole_route_gate_failed")
        if not calibration["qualified"]:
            why.append("pedal_calibration_source_continuity_external_gate_pending")
        pedal = [sample.physics.fields[k] for k in ("gas", "brake")]
        if not np.isfinite(pedal).all() or not all(0 <= p <= 1 for p in pedal):
            raise ValueError("Original native pedal range invalid; never clip")
        cell = [not why and (p == 0 or p >= 0.02) for p in pedal]
        per_axis = [list(why) for _ in pedal]
        for axis, p in enumerate(pedal):
            if 0 < p < 0.02:
                per_axis[axis].append("native_subfloor_unreachable")
        if all(p > 0 for p in pedal):
            cell = [False, False]
            for row in per_axis:
                row.append("native_pedal_coactivation_unrepresentable")
        values.append(body.values)
        masks.append(body.valid)
        clocks.append(decision)
        receipts.append(old["received_monotonic_seconds"])
        packets.append(old["packet_id"])
        commands.append([0.0, held["desired_speed_m_s"]])  # declared zero component, not fake yaw
        raw_held.append(held)
        native.append(pedal)
        loss_masks.append(cell)
        reasons.append(per_axis)
        phase_ids.append(phase_id)
        raw_prefix_sha.append(authority["prefix_sha256"])
        camera_refs.append(event["camera"])
    arrays = dict(
        decision_clock=np.asarray(clocks, np.float64),
        native_receipt_clock=np.asarray(receipts, np.float64),
        native_packet=np.asarray(packets),
        body=np.asarray(values, np.float32),
        body_valid=np.asarray(masks, bool),
        physical_command=np.asarray(commands, np.float32),
        racing_goals=np.tile([0.0, 1.0], (n, 1)).astype(np.float32),
        native_pedals=np.asarray(native, np.float32),
        pedal_loss_valid=np.asarray(loss_masks, bool),
    )
    if arrays["body"].shape != (n, 21) or arrays["body_valid"].shape != (n, 21):
        raise ValueError("Original full body21/masks required")
    intervals = np.diff(arrays["decision_clock"])
    schedule_ok = (
        np.isfinite(intervals).all()
        and np.all((intervals > 0) & (intervals <= 0.08))
        and np.all(np.diff(arrays["native_packet"]) > 0)
    )
    body_ok = (
        np.isfinite(arrays["body"]).all()
        and arrays["body_valid"][:, [4, 16]].all()
        and np.all(arrays["body"][:, 16] >= 0)
    )
    arrays["original_elapsed_seconds"] = np.r_[0.0, intervals]
    arrays["previous_held_row_index"] = np.r_[-1, np.arange(n - 1)]
    if not schedule_ok or not body_ok:
        arrays["pedal_loss_valid"][:] = False
        for row in reasons:
            for axis in row:
                axis.append("whole_history_clock_packet_or_body_gate_failed_no_repair")
    for a in arrays.values():
        a.setflags(write=False)
    return dict(
        arrays=arrays,
        original_held_tasks=tuple(raw_held),
        original_camera_references=tuple(camera_refs),
        original_native420_sha256=raw_prefix_sha,
        phase_ids=phase_ids,
        per_pedal_loss_reasons=reasons,
        route=route,
        fingerprints={k: fingerprint(v) for k, v in arrays.items()},
        observations=n,
        intervals=n - 1,
        source_clock_schedule_valid=bool(schedule_ok),
        physical_body_interface_valid=bool(body_ok),
        neural_states_computed=False,
        all_states_retained=True,
        state_frames_dropped=0,
        first_advance_seconds=0.0,
        hold_rule="previous original command/body/mask/goals over each original positive dt<=80ms",
        held_task_never_replaced_by_actual_phase_or_oracle=True,
        full_brake_one_observations=int(np.count_nonzero(arrays["native_pedals"][:, 1] == 1)),
        qualified_full_brake_one_cells=int(
            np.count_nonzero(
                (arrays["native_pedals"][:, 1] == 1) & arrays["pedal_loss_valid"][:, 1]
            )
        ),
        stationary_hold_alias_limit_retained=True,
        external_pedal_gate=calibration,
        zero_task_stationary_fullbrake_rows=int(
            np.count_nonzero(
                (arrays["physical_command"][:, 1] == 0)
                & (arrays["body"][:, 16] == 0)
                & (arrays["native_pedals"][:, 1] == 1)
            )
        ),
        model_loss_not_evaluated=True,
        exact_native_physical_clock_verified=False,
        exclusive_teacher_control_verified=False,
        pixel_replay_and_observer_coverage_external_gate_required=True,
        fit_allowed=False,
        optimizer_updates=0,
    )


def load_candidate(root, *, expected_capture_manifest_sha256, pedal_review_bytes=None):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not is_sha(expected_capture_manifest_sha256) or sha_file(manifest_path) != (
        expected_capture_manifest_sha256
    ):
        raise ValueError("Exact original capture manifest binding required")
    manifest = json.loads(manifest_path.read_bytes())
    config = manifest["config"]
    # Refuse old roles BEFORE opening any event/NPZ/pixel payload.
    if config.get("role") != "train" or "original_train_declaration" not in config:
        raise ValueError("Only original prospectively bound TRAIN captures; no role promotion")
    declaration = config["original_train_declaration"]
    validate_declaration(declaration)
    external_gates = external_binding_gate(declaration)
    source_pin = declaration["source_sha256"]["scripts/record_pedal_b2.py"]
    if config.get("program_sha256") != source_pin or sha_file(root / "executed-program.py") != (
        source_pin
    ):
        raise ValueError("Actual TRAIN capture source differs from predeclared parent")
    raw = (root / "original-train-declaration.json").read_bytes()
    if (
        sha_bytes(raw) != config["train_declaration_sha256"]
        or json.loads(raw) != declaration
        or config["train_nonce"] != declaration["nonce"]
        or root.resolve() != Path(declaration["bindings"]["output"]).resolve()
    ):
        raise ValueError("Original role/nonce/declaration bytes/capture output differ")
    parent_result = json.loads((root / "result.json").read_bytes())
    recording = root / "recording"
    if not recording.resolve().is_relative_to(root.resolve()):
        raise ValueError("Original recording escaped prospectively named capture root")
    result = json.loads((recording / "result.json").read_bytes())
    if (
        parent_result.get("status") != "completed_train_candidate_b2_transport_collection"
        or parent_result.get("role") != "train"
        or parent_result.get("fit_allowed") is not False
        or parent_result.get("train_declaration_sha256") != sha_bytes(raw)
        or result.get("status") != "completed"
        or result.get("camera_wait_misses") != 0
        or parent_result.get("transport_state") != "closed"
        or parent_result.get("owned_threads_still_live") != []
        or parent_result.get("lease_retained") is not False
        or parent_result.get("heartbeat_evidence", {}).get("status") != "complete"
        or parent_result.get("actor_trace", {}).get("status") != "complete"
    ):
        raise ValueError("Completed whole original candidate/raw wait gate required; failed unfit")
    if type(result.get("decisions")) is not int or not 1 <= result["decisions"] <= 2048:
        raise ValueError("Original recorder decisions count required; no observations alias")
    for path, digest in declaration["source_sha256"].items():
        if sha_file(path) != digest:
            raise ValueError("Original collector/loader/body source continuity differs: " + path)
    source_proof = json.loads((recording / "native-authority-source-proof.json").read_bytes())
    child_manifest = json.loads((recording / "manifest.json").read_bytes())
    HostCopyProfile.from_record(child_manifest["config"].get("frame_persistence_profile"))
    HostCopyProfile.from_record(result.get("frame_persistence_profile"))
    ActorCadenceProfile.from_record(config.get("actor_cadence_profile"))
    ActorCadenceProfile.from_record(child_manifest["config"].get("actor_cadence_profile"))
    ActorCadenceProfile.from_record(result.get("actor_cadence_profile"))
    for attachment in (config, child_manifest["config"], parent_result, result):
        from_record(attachment.get("camera_status_profile"))
    for attachment in (config, child_manifest["config"]):
        if (
            attachment.get("camera_status_profile_sha256")
            != declaration["source_sha256"][CAMERA_STATUS_PROFILE]
        ):
            raise ValueError("Actual lifecycle profile attachment differs")
    policy = result["vision"]["camera_status_policy_validation"]
    from_record(policy.get("profile"))
    if policy.get("requested_policy") != declaration["camera_status_policy"] or any(
        policy.get(k) is not True
        for k in (
            "first_frame_echo_verified",
            "terminal_echo_verified",
            "checked_after_owned_cleanup",
        )
    ):
        raise ValueError("Actual first-MMF/post-cleanup terminal lifecycle echo proof required")
    args = child_manifest["config"]["arguments"]
    if (
        not isinstance(args.get("camera_status_profile"), str)
        or sha_file(args["camera_status_profile"])
        != declaration["source_sha256"][CAMERA_STATUS_PROFILE]
    ):
        raise ValueError("Exact actual child lifecycle profile file required")
    if (
        not isinstance(args.get("frame_persistence_profile"), str)
        or sha_file(args["frame_persistence_profile"])
        != declaration["source_sha256"]["configs/ac-frame-host-v4-20261030.json"]
    ):
        raise ValueError("Actual original child hostV4 profile attachment differs")
    if (
        not isinstance(args.get("actor_cadence_profile"), str)
        or sha_file(args["actor_cadence_profile"]) != declaration["source_sha256"][ACTOR_PROFILE]
        or config.get("actor_cadence_profile_sha256") != declaration["source_sha256"][ACTOR_PROFILE]
        or child_manifest["config"].get("actor_cadence_profile_sha256")
        != declaration["source_sha256"][ACTOR_PROFILE]
    ):
        raise ValueError("Actual original R2 actor30ms profile attachment differs")
    if (
        not all(
            args.get(k) is True
            for k in (
                "pace_task",
                "same_copy_native_authority",
                "mmf_metadata",
                "batched_shared_copy",
                "async_frame_persistence",
                "source_timing",
            )
        )
        or args.get("flight_shadow_config") is not None
        or args.get("duration_seconds") != 30
    ):
        raise ValueError("Original samecopy bounded pace recording contract required")
    expected_child = declaration["source_sha256"][
        "scripts/record_ac_engine_ai_probe_lifecycle_status_v1.py"
    ]
    if (
        source_proof.get("schema") != "flybrain.native.same_copy_recorder_source_proof.v1"
        or source_proof.get("before_camera_activation") is not True
        or source_proof.get("before_control_activation") is not True
        or source_proof.get("executed_program_sha256") != expected_child
        or sha_file(recording / "executed-program.py") != expected_child
        or set(source_proof["source_files"]) != NATIVE_SOURCE_FIELDS
        or any(
            declaration["source_sha256"].get(k) != v["sha256"]
            for k, v in source_proof["source_files"].items()
        )
    ):
        raise ValueError("Actual samecopy camera/body recorder source proof differs")
    receipt = json.loads((recording / "control/api-receipt.json").read_bytes())
    if any(
        receipt.get(k, {}).get("generation") != declaration["bindings"]["generation"]
        or receipt.get(k, {}).get("epoch") != declaration["bindings"]["epoch"]
        for k in ("before", "after")
    ):
        raise ValueError("Original whole control history generation differs")
    validate_pace_receipt(receipt)
    events_path = recording / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf8").splitlines()]
    if len(events) != result["decisions"] or len(events) > 2048:
        raise ValueError("Every original observation/event required")
    if (
        result.get("actor_trace_decisions") != len(events)
        or Path(parent_result["actor_trace"].get("index_file", "")).resolve()
        != (root / "actor-trace/index.json").resolve()
    ):
        raise ValueError("Every original event must attach to the same terminal actor trace")
    actor_trace = verify_actor_trace(
        root / "actor-trace", events, expected_cleanup=result.get("owned_cleanup")
    )
    encoder = NavigationBodyEncoder(
        declaration["body_contract"],
        native_identity=json.loads((recording / "identity.json").read_bytes()),
    )
    camera = MetadataCamera(recording / "vision")
    try:
        for name in ("frame-0000.json", "result.json"):
            original = json.loads(camera.read(name))
            validate_producer_echo(
                original.get(ECHO_KEY),
                policy=declaration["camera_status_policy"],
                request_id=result["camera_request_id"],
                generation=declaration["bindings"]["generation"],
            )
            if name == "result.json" and original != result["vision"]["producer_terminal"]:
                raise ValueError("Original terminal lifecycle metadata differs")
        clock = json.loads(camera.read("clock-final.json"))

        def frame_reader(event):
            name = f"frame-{event['camera']['capture_id']:04d}.json"
            data = camera.read(name)
            if sha_bytes(data) != event["camera"]["file_sha256"]:
                raise ValueError("Original event/frame metadata attachment differs")
            frame = json.loads(data)
            validate_producer_echo(
                frame.get(ECHO_KEY),
                policy=declaration["camera_status_policy"],
                request_id=result["camera_request_id"],
                generation=declaration["bindings"]["generation"],
            )
            return frame

        derived = derive_history(
            events,
            declaration=declaration,
            receipt=receipt,
            clock_record=clock,
            frame_reader=frame_reader,
            body_encoder=encoder,
            episode_id=declaration["nonce"],
            pedal_review_bytes=pedal_review_bytes,
        )
    finally:
        camera.close()
    derived.update(
        schema="flybrain.pedal_load_b2.whole_history.v1",
        status="complete_original_B2_history_derivation_unadmitted",
        data_or_fit_permission=False,
        actor_trace_verified=actor_trace,
        external_source_pipeline_gates=external_gates,
    )
    derived["source_references"] = dict(
        capture_manifest_sha256=expected_capture_manifest_sha256,
        declaration_sha256=sha_bytes(raw),
        events_file=str(events_path),
        events_sha256=sha_file(events_path),
        original_observations_file=str(recording / "observations.npz"),
        observations_opened=False,
        camera_pixels_opened=False,
        source_payloads_copied=0,
        parent_result_sha256=sha_file(root / "result.json"),
        recording_result_sha256=sha_file(recording / "result.json"),
        api_receipt_sha256=sha_file(recording / "control/api-receipt.json"),
    )
    baseline = result.get("native_baseline_packet_id")
    if type(baseline) is not int or np.any(derived["arrays"]["native_packet"] <= baseline):
        raise ValueError("Every original packet must advance past original paused baseline")
    return derived
