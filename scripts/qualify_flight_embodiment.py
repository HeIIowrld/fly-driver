"""Original-clock synthetic MN anchor qualification; no driving fit or live interfaces."""

import argparse
import dataclasses
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.actuator_codec import project_pedals
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.flight_embodiment import (
    REFERENCE_SCHEMA,
    AnchorProvenance,
    EmbodimentReference,
    HashedSource,
    decode_embodiment,
)
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


def tensor_digest(value):
    value = value.detach().cpu()
    if value.layout == torch.sparse_coo:
        return tensor_digest(value._indices()) + tensor_digest(value._values())
    return fingerprint(value.numpy())


def model_digest(model):
    return {k: tensor_digest(v) for k, v in model.backbone.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (
        plan["schema"] != "flybrain.flight.mn_anchor_qualification.v1"
        or plan["device"] != "cpu"
        or plan["fit_allowed"] is not False
        or plan["optimizer_updates"] != 0
    ):
        raise ValueError("Separate CPU diagnostic registration required")
    for path, digest in plan["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("Registered source differs: " + path)
    protocol = read(plan["independent_protocol"])
    if (
        protocol["quiet"]["steps"] != 100
        or protocol["quiet"]["elapsed_seconds_per_step"] != 0.01
        or protocol["validation_grid"]["cases"] != 18
    ):
        raise ValueError("Exact preregistered original-clock protocol required")
    source = [
        r
        for r in (
            json.loads(line)
            for line in Path(plan["original_events"]).read_text(encoding="utf8").splitlines()
        )
        if r["kind"] == "neural_component"
    ]
    clocks = np.array([r["decision_clock"] for r in source], dtype=np.float64)
    original_masks = np.array([r["body_valid"] for r in source], dtype=bool)
    intervals = np.diff(clocks)
    declared = protocol["calibration_anchor"]
    if (
        len(source) != 68
        or not np.array_equal(clocks, declared["original_decision_clocks"])
        or not np.array_equal(intervals, declared["original_elapsed_seconds"])
        or not np.all((intervals > 0) & (intervals <= 0.08))
    ):
        raise ValueError("Original full historical schedule required; never retime")
    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_sha256=sha256(args.plan),
            program_sha256=sha256(__file__),
            role="synthetic_neural_component_diagnostic",
            original_events_sha256=sha256(plan["original_events"]),
            torch_version=torch.__version__,
            original_clock_fingerprint=fingerprint(clocks),
            original_mask_fingerprint=fingerprint(original_masks),
            source_clocks_are_historical_schedule_not_new_native_observations=True,
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    write_json(run / "registration-before-trajectory.json", plan)
    result = dict(
        status="failed",
        fit_allowed=False,
        optimizer_updates=0,
        teacher_labels_used=False,
        gpu_used=False,
        live_api_calls=0,
        mmf_opens=0,
        student_deployed=False,
        original_payloads_copied=0,
        goal_achieved=False,
    )
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.set_default_device("cpu")
        payload = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
        kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        goals = torch.tensor(racing_goal(100, 1))
        valid = torch.zeros(1, 21, dtype=torch.bool)
        valid[:, [0, 1, 4, 16]] = True
        zero = torch.zeros(1, 2)
        quiet_body = torch.zeros(1, 21)
        models, quiet_states, quiet_rows, frozen = [], [], [], []

        def sample(model, state, elapsed):
            raw, muscles = model.backbone._read_state_tensor(state)
            p = 0.5 * (muscles[:, 4] + muscles[:, 5])
            return dict(
                elapsed_seconds=float(elapsed),
                state_sha256=tensor_digest(state),
                raw_intent=raw[0].tolist(),
                muscles=muscles[0].tolist(),
                common_power=float(p[0]),
            )

        with torch.no_grad():
            for _repeat in range(2):
                model = PhysicalManeuverCircuit(
                    plan["spec"], payload, sparse_backend="cached_csr_cpu"
                ).eval()
                original = model_digest(model)
                state = model.initial_state(1)
                rows = [sample(model, state, 0.0)]
                for index in range(100):
                    _, _, state = model.advance(
                        zero, quiet_body, valid, goals, state, 0.01, **kwargs
                    )
                    rows.append(sample(model, state, (index + 1) * 0.01))
                _, muscles = model.backbone._read_state_tensor(state)
                readout = model.backbone.readout
                turn, forward = readout.turn_log_gain.clone(), readout.forward_log_gain.clone()
                readout.calibrate_reference(muscles[0], turn_scale=1.0, forward_scale=1.0)
                readout.turn_log_gain.copy_(turn)
                readout.forward_log_gain.copy_(forward)
                after = model_digest(model)
                allowed = {
                    "readout.power_reference",
                    "readout.lateral_reference",
                    "readout.reference_calibrated",
                }
                changed = {k for k in original if original[k] != after[k]}
                if not changed <= allowed:
                    raise ValueError("Only private quiet-neutral offsets may differ")
                raw, _ = model.backbone._read_state_tensor(state)
                if not torch.equal(raw, torch.zeros_like(raw)):
                    raise ValueError("Same-state quiet neural neutral must be exact")
                models.append(model)
                quiet_states.append(state.detach().clone())
                quiet_rows.append(rows)
                frozen.append(after)
            if quiet_rows[0] != quiet_rows[1] or frozen[0] != frozen[1]:
                raise ValueError("Matched CPU quiet repeat differs")
            write_json(
                run / "quiet-trajectories.json",
                dict(
                    repeats=quiet_rows,
                    quiet_precalibration_readout="original checkpoint reference",
                    postcalibration_reference={
                        k: v.tolist() for k, v in models[0].backbone.readout.state_dict().items()
                    },
                ),
            )

            def trajectory(
                model,
                initial,
                *,
                speed_kmh,
                desired_yaw=0.0,
                actual_yaw=0.0,
                desired_speed=0.0,
                lesion=None,
            ):
                body = quiet_body.clone()
                body[:, 16], body[:, 4] = speed_kmh / 3.6, actual_yaw
                command = torch.tensor([[desired_yaw, desired_speed]])
                state = initial.clone()  # Independent case boundary only.
                model.drive(command, body, valid, goals, lesion=lesion, **kwargs)
                rows = [sample(model, state, 0.0)]
                for index, elapsed in enumerate(intervals):
                    _, _, state = model.advance(
                        command, body, valid, goals, state, float(elapsed), lesion=lesion, **kwargs
                    )
                    rows.append(sample(model, state, clocks[index + 1] - clocks[0]))
                return dict(
                    speed_kmh=speed_kmh,
                    desired_yaw_rad_s=desired_yaw,
                    actual_yaw_rad_s=actual_yaw,
                    desired_speed_m_s=desired_speed,
                    lesion=lesion,
                    synthetic_body_mask_indices=[0, 1, 4, 16],
                    observations=rows,
                    original_intervals_retained=len(intervals),
                    intermediate_state_resets=0,
                    initial_elapsed_seconds=0.0,
                )

            cal = [
                trajectory(m, s, speed_kmh=100) for m, s in zip(models, quiet_states, strict=True)
            ]
            if cal[0] != cal[1]:
                raise ValueError("All matched CPU anchor state hashes/outputs must be exact")
            write_json(
                run / "calibration-trajectory.json",
                dict(
                    registered_endpoint_index=67,
                    endpoint_is_not_time_or_case_minimum=True,
                    repeats=cal,
                ),
            )
            p0 = float(models[0].backbone.readout.power_reference)
            anchor = cal[0]["observations"][-1]["common_power"]
            sources = dict(
                graph_spec=plan["spec"],
                graph_checkpoint=plan["checkpoint"],
                readout_source="src/flybrain_drive/flight_circuit.py",
                decoder_source="src/flybrain_drive/flight_embodiment.py",
                registration=str(run / "registration-before-trajectory.json"),
                trajectory=str(run / "calibration-trajectory.json"),
            )
            provenance = AnchorProvenance(
                sources=tuple(HashedSource(k, str(p), sha256(p)) for k, p in sources.items()),
                tensor_dtype="float32",
                numerical_scope="cpu_repeat",
                original_clock_sha256=fingerprint(clocks),
                original_body_mask_sha256=fingerprint(original_masks),
                quiet_end_state_sha256=tensor_digest(quiet_states[0]),
                anchor_state_sha256=cal[0]["observations"][-1]["state_sha256"],
                quiet_duration_seconds=1.0,
                step_speed_error_m_s=-100 / 3.6,
                anchor_elapsed_seconds=float(clocks[-1] - clocks[0]),
                fixed_time_registered_before_trajectory=True,
                quiet_to_step_state_continuous=True,
                original_graph_body_readout_frozen=True,
                all_original_samples_retained=True,
                anchor_power_observed=True,
                optimizer_updates_before_anchor=0,
                teacher_labels_used=False,
            )
            reference = EmbodimentReference(
                schema=REFERENCE_SCHEMA,
                quiet_power=p0,
                brake_anchor_power=anchor,
                original_readout_power_reference=p0,
                quiet_raw_neural_intent=(0.0, 0.0),
                common_power_numerical_uncertainty=0.0,
                maximum_decoded_brake_parity_error=0.0,
                quiet_decoded_brake_error=0.0,
                provenance=provenance,
            )
            write_json(run / "reference.json", dataclasses.asdict(reference))
            print(json.dumps(dict(anchor=anchor, quiet=p0, span=reference.span)), flush=True)

            def decode_case(case):
                previous = None
                for row in case["observations"]:
                    raw = torch.tensor([row["raw_intent"]])
                    muscles = torch.tensor([row["muscles"]])
                    out = decode_embodiment(raw, muscles, reference=reference)
                    v1, v2 = out.canonical_v1[0].numpy(), out.canonical_v2[0].numpy()
                    if not np.array_equal(v1, canonical_controls(raw[0].numpy())):
                        raise ValueError("Original canonical path differs")
                    if not np.array_equal(v1[:2], v2[:2]):
                        raise ValueError("Steering/throttle changed")
                    row.update(
                        canonical_v1=v1.tolist(),
                        canonical_v2=v2.tolist(),
                        encoded_v2=project_pedals(v2)["encoded_request"],
                        pedal_coactivation=bool(out.pedal_coactivation[0]),
                        delta_v2_brake=None if previous is None else float(v2[2] - previous),
                    )
                    previous = float(v2[2])
                return case

            cal = [decode_case(c) for c in cal]
            if cal[0] != cal[1]:
                raise ValueError("Matched CPU decoded trajectory differs")
            if cal[0]["observations"][0]["canonical_v2"][2] != 0.0:
                raise ValueError("Quiet decoder neutral differs")
            if cal[0]["observations"][-1]["canonical_v2"][2] != 1.0:
                raise ValueError("Observed endpoint decoder anchor differs")
            write_json(run / "calibration-decoded.json", dict(repeats=cal))
            conditions = list(itertools.product([60, 100], [-0.3, 0.0, 0.3], [-0.3, 0.0, 0.3]))
            cases = [
                decode_case(
                    trajectory(
                        models[0],
                        quiet_states[0],
                        speed_kmh=speed,
                        desired_yaw=desired,
                        actual_yaw=actual,
                    )
                )
                for speed, desired, actual in conditions
            ]
            controls = [
                decode_case(trajectory(models[0], quiet_states[0], speed_kmh=0)),
                *[
                    decode_case(
                        trajectory(
                            models[0], quiet_states[0], speed_kmh=speed, desired_speed=speed / 3.6
                        )
                    )
                    for speed in (60, 100)
                ],
            ]
            lesions = [
                decode_case(
                    trajectory(
                        models[0],
                        quiet_states[0],
                        speed_kmh=100,
                        desired_yaw=0.3,
                        actual_yaw=0.3,
                        lesion=lesion,
                    )
                )
                for lesion in ("speed_error", "desired_yaw", "yaw_feedback")
            ]
            repeat_cases = [
                decode_case(
                    trajectory(
                        models[1],
                        quiet_states[1],
                        speed_kmh=speed,
                        desired_yaw=desired,
                        actual_yaw=actual,
                    )
                )
                for speed, desired, actual in conditions
            ]
            repeat_controls = [
                decode_case(trajectory(models[1], quiet_states[1], speed_kmh=0)),
                *[
                    decode_case(
                        trajectory(
                            models[1], quiet_states[1], speed_kmh=speed, desired_speed=speed / 3.6
                        )
                    )
                    for speed in (60, 100)
                ],
            ]
            repeat_lesions = [
                decode_case(
                    trajectory(
                        models[1],
                        quiet_states[1],
                        speed_kmh=100,
                        desired_yaw=0.3,
                        actual_yaw=0.3,
                        lesion=lesion,
                    )
                )
                for lesion in ("speed_error", "desired_yaw", "yaw_feedback")
            ]
            if cases != repeat_cases or controls != repeat_controls or lesions != repeat_lesions:
                raise ValueError("All grid/control/lesion CPU repeats must be bitwise exact")
            lesion_baselines = [
                decode_case(trajectory(models[0], quiet_states[0], speed_kmh=0, lesion=lesion))
                for lesion in ("speed_error", "desired_yaw", "yaw_feedback")
            ]
            matched_lesions = [
                decode_case(
                    trajectory(
                        models[0],
                        quiet_states[0],
                        speed_kmh=100,
                        desired_yaw=0.3,
                        actual_yaw=0.3,
                        desired_speed=100 / 3.6,
                        lesion=lesion,
                    )
                )
                for lesion in ("speed_error", "desired_yaw", "yaw_feedback")
            ]
            write_json(
                run / "validation-trajectories.json",
                dict(
                    cases=cases,
                    controls=controls,
                    lesions=lesions,
                    lesion_quiet_baselines=lesion_baselines,
                    matched_speed_lesions=matched_lesions,
                    all_grid_control_and_lesion_repeats_bitwise_exact=True,
                    shared_reference_sha256=sha256(run / "reference.json"),
                ),
            )
            for model, expected in zip(models, frozen, strict=True):
                if model_digest(model) != expected:
                    raise ValueError("Frozen graph/body/gain/mix/private-reference changed")

            def summary(case):
                rows = case["observations"]
                brake = np.array([r["canonical_v2"][2] for r in rows])
                power = np.array([r["common_power"] for r in rows])
                crossing = {}
                for name, values, threshold in [
                    ("encoded_positive", np.array([r["encoded_v2"][2] for r in rows]), 0.0),
                    ("half", brake, 0.5),
                    ("eight_tenths", brake, 0.8),
                ]:
                    found = np.flatnonzero(
                        values > threshold if threshold == 0 else values >= threshold
                    )
                    crossing[name] = (
                        None if not len(found) else rows[int(found[0])]["elapsed_seconds"]
                    )
                return {k: v for k, v in case.items() if k != "observations"} | dict(
                    final_brake=float(brake[-1]),
                    max_brake=float(brake.max()),
                    min_common=float(power.min()),
                    max_common=float(power.max()),
                    coactivation_samples=sum(r["pedal_coactivation"] for r in rows),
                    decreasing_brake_intervals=int((np.diff(brake) < 0).sum()),
                    crossings=crossing,
                )

            latent = []
            for case in cases:
                baseline = next(
                    c
                    for c in cases
                    if c["speed_kmh"] == case["speed_kmh"]
                    and c["desired_yaw_rad_s"] == c["actual_yaw_rad_s"] == 0
                )
                latent.append(
                    dict(
                        speed_kmh=case["speed_kmh"],
                        desired_yaw_rad_s=case["desired_yaw_rad_s"],
                        actual_yaw_rad_s=case["actual_yaw_rad_s"],
                        max_common_deviation_over_span=max(
                            abs(a["common_power"] - b["common_power"])
                            for a, b in zip(
                                case["observations"], baseline["observations"], strict=True
                            )
                        )
                        / reference.span,
                        max_decoded_brake_deviation=max(
                            abs(a["canonical_v2"][2] - b["canonical_v2"][2])
                            for a, b in zip(
                                case["observations"], baseline["observations"], strict=True
                            )
                        ),
                    )
                )
            speed_steer = max(
                abs(a["canonical_v1"][0] - b["canonical_v1"][0])
                for c in cases
                if c["speed_kmh"] == 60
                for d in cases
                if d["speed_kmh"] == 100
                and c["desired_yaw_rad_s"] == d["desired_yaw_rad_s"]
                and c["actual_yaw_rad_s"] == d["actual_yaw_rad_s"]
                for a, b in zip(c["observations"], d["observations"], strict=True)
            )
            result.update(
                status="completed_cpu_original_clock_mn_anchor_qualification",
                observations_per_case=68,
                original_intervals_per_case=67,
                anchor_elapsed_seconds=reference.provenance.anchor_elapsed_seconds,
                quiet_power=p0,
                observed_anchor_power=anchor,
                observed_span=reference.span,
                reference_sha256=sha256(run / "reference.json"),
                cpu_repeat_bitwise_exact=True,
                numerical_scope="cpu_repeat_only_not_gpu",
                gpu_qualified=False,
                default_readout_graph_body_unchanged_except_declared_private_neutral=True,
                cases=[summary(c) for c in cases],
                controls=[summary(c) for c in controls],
                lesions=[summary(c) for c in lesions],
                latent_yaw_power_coupling=latent,
                lesion_quiet_baselines=[summary(c) for c in lesion_baselines],
                matched_speed_lesions=[summary(c) for c in matched_lesions],
                maximum_latent_speed_steering_difference=speed_steer,
                fullrange_anchor_not_allcondition_policy_or_fastbraking_proof=True,
                wall_seconds=time.perf_counter() - started,
            )
    except Exception as error:
        result.update(
            error=type(error).__name__,
            reason=str(error),
            wall_seconds=time.perf_counter() - started,
        )
    finally:
        write_json(run / "result.json", result)
    print(json.dumps(result))
    if result["status"] == "failed":
        raise RuntimeError(result["reason"])


if __name__ == "__main__":
    main()
