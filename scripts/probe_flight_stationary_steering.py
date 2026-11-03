"""Finite neural-task -> vJoy steering probe with full brake hold; never navigation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_static_identity_prefix
from flybrain_drive.ac_native_receipts import NativeReceiptCollector
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_actuator import stationary_steering_request
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.record_ac_engine_ai_probe import verified_shadow
from scripts.record_ac_wgc import physics_record


def read(path):
    refuse_leased_transport('probe_flight_stationary_steering')
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_plan(plan):
    expected = [("visual_baseline", 0.), ("left_task_command", .5),
                ("right_task_command", -.5), ("return_to_visual", 0.)]
    if (plan["schema"] != "flybrain.flight.stationary_steering_probe.v1"
            or [(s["name"], s["task_turn_bias"]) for s in plan["stages"]] != expected
            or plan["samples_per_stage"] != 25 or plan["period_seconds"] != .04
            or plan["steering_request_scale"] != .6 or plan["maximum_steering_request"] != .1
            or plan["maximum_speed_kmh"] != .1
            or plan["maximum_output_source_age_seconds"] != .15):
        raise ValueError("Exact bounded stationary component profile required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    refuse_leased_transport('probe_flight_stationary_steering')
    plan = read(args.plan)
    validate_plan(plan)
    axis = read(plan["axis_config"])
    inference, certificate = verified_shadow(
        plan["inference_config"], plan["inference_verification"])
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan, seed=certificate[
        "initialization_seed"], plan_sha256=sha256(args.plan), inference_certificate=certificate,
        axis_configuration=axis, scope=__doc__, actor_qualified=False))
    result = dict(status="failed", goal_achieved=False, actor_qualified=False,
                  navigation_verified=False, learned_longitudinal_control_applied=False,
                  model_outputs_applied_to_steering=False, ui_actions_sent=0)
    session, vision, reader, ai, collector, child = None, None, None, None, None, None
    resumed, records, cleanup_errors = False, [], []
    spool = Path(os.environ["LOCALAPPDATA"]) / "flybrain/event-spool" / uuid.uuid4().hex
    spool.mkdir(parents=True, exist_ok=False)
    log = (spool / "events.jsonl").open("x", encoding="utf8")
    api_root = Path(os.environ["LOCALAPPDATA"]) / "flybrain/csp-session"

    def emit(event):
        log.write(json.dumps(event, allow_nan=False) + "\n")
        log.flush()

    def lifecycle(command):
        nonlocal child
        refuse_leased_transport('probe_flight_stationary_steering')
        if child is not None and child.poll() is None:
            raise RuntimeError("Same lifecycle client still live; never send another command")
        output = run / ("api-" + command)
        with (run / ("api-" + command + ".log")).open("x", encoding="utf8") as pipe:
            child = subprocess.Popen([sys.executable, "-m", "scripts.control_ac_session",
                "--command", command, "--execute", "--output", str(output)],
                stdout=pipe, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                result["live_lifecycle_client_pid"] = child.pid
                raise TimeoutError(
                    "Lifecycle outcome unknown; inspect the same owned client") from error
        if child.returncode != 0 or read(output / "result.json")["status"] != "completed":
            raise RuntimeError("Owned lifecycle command failed: " + command)

    def stationary(sample):
        f = sample.fields
        if abs(f["speedKmh"]) >= .1 or f["gas"] >= .02 or f["brake"] <= .98:
            raise ValueError("Native stationary full-brake hold violated")

    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError("Exactly one original engine required")
        window = windows[0]
        state = read(api_root / "status.json")
        if (time.time() - (api_root / "status.json").stat().st_mtime > 2
                or state["paused"] is not True or state["ai_controlled"] is not False
                or state["camera_active"] is not False or state["ideal_line_shown"] is not False
                or state.get("pending_receipt_id") or state.get("error")
                or native.user.GetForegroundWindow() == window.hwnd
                or sha256(axis["profile_path"]) != axis["profile_sha256"]):
            raise ValueError("Idle background paused source/profile required")
        refuse_leased_transport('probe_flight_stationary_steering')
        controllers = inspect_vjoy_controllers()
        if (controllers["candidate_ac_joy_index"] != 0 or len(controllers["vjoy_devices"]) != 1
                or controllers["vjoy_devices"][0]["instance_guid"] != axis["instance_guid"]):
            raise ValueError("Reviewed controller identity/index required")
        refuse_leased_transport('probe_flight_stationary_steering')
        sdk = load_vjoy_sdk(axis["dll_path"], axis["dll_sha256"])
        if sdk.GetVJDStatus(axis["device_id"]) != 1:
            raise ValueError("Controller must be free; never steal ownership")
        refuse_leased_transport('probe_flight_stationary_steering')
        with ExistingNamedMapping("Local\\acpmf_static", 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / "identity.json", identity)
        reader = NativePhysicsReader(timeout=.15, clock=time.perf_counter)
        ai = NativeTeacherStatusReader()
        before = reader.receive()
        stationary(before)
        if ai.snapshot()["builtin_ai_controlled"]:
            raise ValueError("Independent native AI ownership must be off")
        write_json(run / "preflight.json", dict(session=state, native=physics_record(before),
            native_identity=identity, controller=controllers, actuator_messages_sent=0))
        if not args.execute:
            result.update(status="read_only_preflight", termination_reason="no execute option")
        else:
            torch.set_num_threads(1)
            torch.manual_seed(certificate["initialization_seed"])
            stream = FlightStream(plan["inference_config"], native_identity=identity,
                                  episode_id=run.name)
            # Lifecycle API requires a free controller and inactive camera. Resume first
            # from verified brake hold, then acquire this one explicit joystick.
            lifecycle("resume")
            resumed = True
            stationary(reader.receive())
            collector = NativeReceiptCollector()
            collector.start()
            refuse_leased_transport('probe_flight_stationary_steering')
            session = VJoySession(sdk, axis["device_id"],
                                  [AxisBinding(**b) for b in axis["bindings"]])
            emit(dict(kind="driver_open", request=session.open()))
            vision = CSPVisionRecorder(window.hwnd, read(inference["camera_config"]),
                max_frames=300, period=.04, max_bytes=512 * 1024**2,
                image_format="bmp", shared_transport=True)
            vision.wait_ready()
            reader.receive()  # fresh native anchor; never reused as a dated actor sample

            def guard():
                if (run / "stop").exists():
                    raise RuntimeError("Owned stop file received")
                if (native.probe(window.hwnd, require_foreground=False) != window
                        or native.user.GetForegroundWindow() == window.hwnd
                        or sha256(axis["profile_path"]) != axis["profile_sha256"]
                        or ai.snapshot()["builtin_ai_controlled"]):
                    raise ValueError("Game/profile/background/AI identity changed")
                current = read(api_root / "status.json")
                if (time.time() - (api_root / "status.json").stat().st_mtime > 2
                        or current["generation"] != state["generation"]
                        or current["epoch"] != state["epoch"] or current["paused"]
                        or current["ai_controlled"] or current["ideal_line_shown"] is not False):
                    raise ValueError("Live AI-off session identity changed")
                collector.check()

            for phase_index, phase in enumerate(plan["stages"]):
                stream.goals[0, 0] = phase["task_turn_bias"]
                for index in range(25):
                    started = time.perf_counter()
                    guard()
                    vision.wait_for_candidate(timeout=.04)
                    sample = reader.receive()
                    stationary(sample)
                    candidate = vision.candidate_before(sample.received_monotonic_seconds)
                    pixels = {name: value["rgb"]
                              for name, value in vision.samples_for(candidate).items()}
                    decision = vision.host_clock()
                    # Persist the original actor observation before inference or request can fail.
                    emit(dict(kind="actor_input", phase_index=phase_index, phase_sample=index,
                              task_turn_bias=phase["task_turn_bias"], camera=candidate,
                              native=physics_record(sample), decision_monotonic_seconds=decision))
                    response = stream.advance(clock=decision, camera=candidate, pixels=pixels,
                        native_sample=decode_virtual_body_prefix(sample.prefix_bytes,
                            sample.received_monotonic_seconds))
                    codec = stationary_steering_request(response["intent"], steering_scale=.6,
                                                        maximum_steering=.1)
                    guard()
                    latest = reader.receive()
                    stationary(latest)
                    age = time.perf_counter() - min(v["render_started_lower"]
                                                    for v in candidate["views"])
                    if age > .15:
                        raise ValueError("Output source too old before driver request")
                    request = session.submit(**codec["requested_control"])
                    row = dict(kind="flight_axis_observation", phase_index=phase_index,
                        phase=phase["name"], phase_sample=index,
                        task_turn_bias=phase["task_turn_bias"],
                        camera=candidate, native=physics_record(sample),
                        native_before_request=physics_record(latest),
                        decision_monotonic_seconds=decision, inference=response, codec=codec,
                        request=request, source_age_at_request_upper_seconds=age,
                        was_foreground=False)
                    records.append(row)
                    emit(row)
                    time.sleep(max(0., .04 - (time.perf_counter() - started)))
            result.update(status="completed_component_samples",
                          model_outputs_applied_to_steering=True,
                          termination_reason="four predetermined stationary task phases completed")
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason=str(error))
    finally:
        if session is not None and session.owned:
            try:
                for index in range(5):
                    emit(dict(kind="cleanup_neutral", index=index,
                              request=session.submit(0., 0., 1.)))
                    time.sleep(.04)
            except Exception as error:
                cleanup_errors.append("driver: " + str(error))
            finally:
                try:
                    emit(dict(kind="driver_close", request=session.close()))
                except Exception as error:
                    cleanup_errors.append("driver close: " + str(error))
        if collector is not None:
            try:
                report = collector.close()
                samples = [physics_record(s) for s in collector.samples]
                write_json(run / "native-receipts.json", samples)
                write_json(run / "native-receipts-result.json", dict(report,
                    file_sha256=sha256(run / "native-receipts.json")))
                result["native_receipts"] = report
            except Exception as error:
                cleanup_errors.append("native collector: " + str(error))
        if vision is not None:
            try:
                result["vision"] = vision.close()
            except Exception as error:
                cleanup_errors.append("camera: " + str(error))
        if resumed:
            try:
                post = reader.receive()
                stationary(post)
                if (sdk.GetVJDStatus(axis["device_id"]) != 1
                        or abs(post.fields["steerAngle"]) > .005):
                    raise ValueError("Released centered brake hold not independently observed")
                write_json(run / "native-after-release.json", physics_record(post))
                lifecycle("pause")
                result["released_neutral_and_pause_verified"] = True
            except Exception as error:
                cleanup_errors.append("final state: " + str(error))
        if ai is not None:
            ai.close()
        if reader is not None:
            reader.close()
        log.close()
        if (spool / "events.jsonl").exists():
            shutil.copyfile(spool / "events.jsonl", run / "events.jsonl")
            assert sha256(spool / "events.jsonl") == sha256(run / "events.jsonl")
            write_json(run / "event-spool.json", dict(source=str(spool / "events.jsonl"),
                source_retained=True, sha256=sha256(run / "events.jsonl")))
        if vision is not None and not vision.thread.is_alive():
            try:
                result["vision_archive"] = vision.archive_to(run / "vision", bundle=True)
            except Exception as error:
                cleanup_errors.append("archive: " + str(error))
        result.update(observations=len(records), actuator_messages_sent=session.submit_attempts
                      if session else 0, steering_response_verified=False,
                      cleanup_errors=cleanup_errors)
        if cleanup_errors:
            result["status"] = "failed"
        write_json(run / "result.json", result)
    if result["status"] == "failed":
        raise RuntimeError(result.get("reason", str(cleanup_errors)))
    print(json.dumps(dict(status=result["status"], observations=len(records),
                          requests=result["actuator_messages_sent"])))


if __name__ == "__main__":
    main()
