"""Finite moving flight-steering/body-feedback probe; engineering pedals, no navigation claim."""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_diagnostic_admission import verified_diagnostic_transport
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import (
    PHYSICS_LAYOUT,
    PHYSICS_PREFIX,
    NativePhysicsReader,
    decode_static_identity_prefix,
)
from flybrain_drive.ac_native_receipts import NativeReceiptCollector
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_timed_pulse import TimedSteeringPulse
from flybrain_drive.ac_vjoy import AxisBinding, VJoySession, load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_fast_stream import FastFlightStream, verified_fast_plan
from flybrain_drive.flight_moving_probe import moving_component_request
from flybrain_drive.flight_perturbation import perturb_request, validate_perturbation_plan
from flybrain_drive.flight_scratch_warmup import prime_flight_operators
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.flight_yaw_trial import apply_yaw_mode
from flybrain_drive.flyvis_frozen_tail import install_frozen_visual_tail
from flybrain_drive.flyvis_graphed_tail import install_graphed_visual_tail
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.audit_ac_timed_steering import verified_timed_steering
from scripts.record_ac_engine_ai_probe import verified_shadow
from scripts.record_ac_wgc import physics_record
from scripts.verify_flight_yaw_trial import verified_yaw_trial


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def validate_plan(plan):
    expected = [("stationary_prefix", 0, 25), ("launch", 0, 75),
                ("left_task_command", .5, 25), ("right_task_command", -.5, 25),
                ("return_to_visual", 0, 25), ("braking", 0, 50)]
    if (plan["schema"] != "flybrain.flight.moving_body_probe.v1"
            or [(s["name"], s["task_turn_bias"], s["samples"])
                for s in plan["stages"]] != expected
            or plan["maximum_speed_kmh"] != 10 or plan["maximum_displacement_m"] != 20
            or plan["period_seconds"] != .04):
        raise ValueError("Exact bounded moving component profile required")
    audit = read(Path(plan["stationary_response_audit"]) / "result.json")
    if (audit["status"] != "completed_stationary_response_audit"
            or audit["steering_sign_and_stationary_response_verified"] is not True):
        raise ValueError("Actual stationary neural steering response required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--perturbation-trial-index", type=int)
    args = parser.parse_args()
    plan = read(args.plan)
    trial, yaw_verification = None, None
    if args.perturbation_trial_index is None:
        validate_plan(plan)
    else:
        trial = validate_perturbation_plan(plan, args.perturbation_trial_index)
        yaw_verification = verified_yaw_trial(plan)
    timed_trial = plan['schema'].endswith('.v2') and trial is not None
    timed_verification = verified_timed_steering(plan) if timed_trial else None
    diagnostic_verification = verified_diagnostic_transport(plan)
    axis = read(plan["axis_config"])
    inference, certificate = verified_shadow(
        plan["inference_config"], plan["inference_verification"])
    fast_certificate = verified_fast_plan(plan)
    run = begin_run(args.output, dict(arguments=vars(args), plan=plan, seed=certificate[
        "initialization_seed"], plan_sha256=sha256(args.plan), inference_certificate=certificate,
        axis_configuration=axis, scope=__doc__, actor_qualified=False,
        perturbation_trial=trial, yaw_verification=yaw_verification,
        timed_verification=timed_verification, diagnostic_verification=diagnostic_verification,
        fast_inference_certificate=fast_certificate))
    result = dict(status="failed", goal_achieved=False, actor_qualified=False,
                  navigation_verified=False, learned_longitudinal_control_applied=False,
                  model_outputs_applied_to_steering=False, ui_actions_sent=0)
    session, vision, reader, ai, collector, child = None, None, None, None, None, None
    resumed, records, cleanup_errors = False, [], []
    client, origin, timed = None, None, None
    failure_diagnostic = None
    gc_events = []

    def gc_event(phase, info):
        gc_events.append(dict(phase=phase, clock=time.perf_counter(),
                              generation=info['generation']))

    gc.callbacks.append(gc_event)
    spool = Path(os.environ["LOCALAPPDATA"]) / "flybrain/event-spool" / uuid.uuid4().hex
    spool.mkdir(parents=True, exist_ok=False)
    log = (spool / "events.jsonl").open("x", encoding="utf8")
    log_lock = threading.Lock()
    api_root = Path(os.environ["LOCALAPPDATA"]) / "flybrain/csp-session"

    def emit(event):
        with log_lock:
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

    def lifecycle(command):
        nonlocal child
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

    def diagnostic(sample):
        packet = client.snapshot()
        if abs(packet.received_monotonic_seconds-sample.received_monotonic_seconds) > .05:
            raise ValueError("Fresh native/diagnostic receipt alignment required")
        namespace = packet.raw["privileged_teacher"]
        if namespace["actor_input"] is not False:
            raise ValueError("Diagnostic world position must never enter actor")
        truth = namespace["raw"]
        xy = np.array([truth["world_position_x"], truth["world_position_y"]], dtype=float)
        fields, offset = {}, 0
        values = PHYSICS_PREFIX.unpack(sample.prefix_bytes)
        for name, _, width in PHYSICS_LAYOUT:
            if width == 1:
                fields[name] = values[offset]
            offset += width
        distance = float(np.linalg.norm(xy-origin)) if origin is not None else 0.
        item = dict(sdk=record_packet(packet), native=physics_record(sample),
                    gear=fields["gear"], tyres_out=fields["numberOfTyresOut"],
                    auto_shifter=fields["autoShifterOn"], position=xy.tolist(),
                    displacement_m=distance, actor_input=False)
        emit(dict(kind="diagnostic", **item))
        if (not np.isfinite(xy).all() or distance >= 20
                or abs(sample.fields["speedKmh"]) >= 10 or item["tyres_out"] != 0
                or item["gear"] not in (1, 2) or item["auto_shifter"] != 1):
            raise ValueError("Moving component speed/displacement/gear/tyres bound violated")
        return item

    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError("Exactly one original engine required")
        window = windows[0]
        state, modified = read_published_json(api_root / "status.json")
        if (not 0 <= time.time()-modified <= 2
                or state["paused"] is not True or state["ai_controlled"] is not False
                or state["camera_active"] is not False or state["ideal_line_shown"] is not False
                or state.get("pending_receipt_id") or state.get("error")
                or native.user.GetForegroundWindow() == window.hwnd
                or sha256(axis["profile_path"]) != axis["profile_sha256"]):
            raise ValueError("Idle background paused source/profile required")
        controllers = inspect_vjoy_controllers()
        if (controllers["candidate_ac_joy_index"] != 0 or len(controllers["vjoy_devices"]) != 1
                or controllers["vjoy_devices"][0]["instance_guid"] != axis["instance_guid"]):
            raise ValueError("Reviewed controller identity/index required")
        sdk = load_vjoy_sdk(axis["dll_path"], axis["dll_sha256"])
        if sdk.GetVJDStatus(axis["device_id"]) != 1:
            raise ValueError("Controller must be free; never steal ownership")
        with ExistingNamedMapping("Local\\acpmf_static", 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / "identity.json", identity)
        reader = NativePhysicsReader(timeout=.15, clock=time.perf_counter)
        ai = NativeTeacherStatusReader()
        client = FileTelemetryClient(Path(os.environ["LOCALAPPDATA"]) / "flybrain/telemetry",
                                     timeout=.15, clock=time.perf_counter)
        before = reader.receive()
        stationary(before)
        first_diagnostic = diagnostic(before)
        origin = np.array(first_diagnostic["position"])
        if ai.snapshot()["builtin_ai_controlled"]:
            raise ValueError("Independent native AI ownership must be off")
        write_json(run / "preflight.json", dict(session=state, native=physics_record(before),
            native_identity=identity, controller=controllers, actuator_messages_sent=0))
        if not args.execute:
            result.update(status="read_only_preflight", termination_reason="no execute option")
        else:
            torch.set_num_threads(1)
            torch.manual_seed(certificate["initialization_seed"])
            stream = (FastFlightStream(plan['fast_stream_config'], native_identity=identity,
                                       episode_id=run.name) if fast_certificate else
                      FlightStream(plan["inference_config"], native_identity=identity,
                                   episode_id=run.name))
            if trial is not None:
                apply_yaw_mode(stream, trial['mode'])
            if not fast_certificate and plan.get('visual_backend') == 'frozen_visual_tail':
                install_frozen_visual_tail(stream)
            if not fast_certificate and plan.get('visual_backend') == 'graphed_visual_tail':
                install_graphed_visual_tail(stream)
            stream.goals[0, 1] = 6./30.  # Explicit6kmh task, original speed-goal units.
            # Lifecycle API requires a free controller and inactive camera. Resume first
            # from verified brake hold, then acquire this one explicit joystick.
            lifecycle("resume")
            resumed = True
            stationary(reader.receive())
            session = VJoySession(sdk, axis["device_id"],
                                  [AxisBinding(**b) for b in axis["bindings"]])
            emit(dict(kind="driver_open", request=session.open()))
            if timed_trial:
                if plan.get('preparation_engine_settle_seconds'):
                    if plan['preparation_engine_settle_seconds'] != 2.:
                        raise ValueError('Exact pre-observation2s stationary preparation required')
                    preparation_started = time.perf_counter()
                    preparation_samples = []
                    while time.perf_counter()-preparation_started < 2.:
                        sample = reader.receive()
                        stationary(sample)
                        if ai.snapshot()['builtin_ai_controlled']:
                            raise ValueError('AI ownership changed during engine preparation')
                        preparation_samples.append(physics_record(sample))
                        time.sleep(.02)
                    assert stream.steps == 0 and stream.order.clock is None
                    write_json(run/'engine-preparation.json', dict(
                        started=preparation_started, finished=time.perf_counter(),
                        native_samples=preparation_samples, actor_observations=0,
                        hidden_resets=0, initial_episode_state_preserved=True))
                collected = time.perf_counter()
                gc.collect()
                torch.cuda.empty_cache()
                write_json(run/'preparation-gc.json', dict(seconds=time.perf_counter()-collected,
                                                          actor_observations=0))
                write_json(run/'scratch-warmup.json', prime_flight_operators(stream))
                write_json(run/'prepared-memory.json', dict(
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(), actor_observations=0,
                    visual_backend=plan.get('visual_backend', 'upstream')))
                timed = TimedSteeringPulse(session, emit)
            collector = NativeReceiptCollector()
            collector.start()
            vision = CSPVisionRecorder(window.hwnd, read(inference["camera_config"]),
                max_frames=400, period=.04, max_bytes=512 * 1024**2,
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
                current, modified = read_published_json(api_root / "status.json")
                if (not 0 <= time.time()-modified <= 2
                        or current["generation"] != state["generation"]
                        or current["epoch"] != state["epoch"] or current["paused"]
                        or current["ai_controlled"] or current["ideal_line_shown"] is not False):
                    raise ValueError("Live AI-off session identity changed")
                collector.check()
                if timed is not None:
                    timed.check()

            for phase_index, phase in enumerate(plan["stages"]):
                stream.goals[0, 0] = phase["task_turn_bias"]
                for index in range(phase["samples"]):
                    started = time.perf_counter()
                    guard()
                    emit(dict(kind='guard_timing', stage='before_camera',
                              started=started, finished=time.perf_counter()))
                    vision.wait_for_candidate(timeout=.04)
                    sample = reader.receive()
                    measured = diagnostic(sample)
                    if phase["name"] == "stationary_prefix":
                        stationary(sample)
                    elif measured["gear"] != 2:
                        raise ValueError("First gear not independently observed after prefix")
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
                    emit(dict(kind='inference_complete', phase=phase['name'], phase_sample=index,
                              inference=response, body_values=stream.last_body.values.tolist(),
                              body_valid=stream.last_body.valid.tolist()))
                    harness_phase = ('launch' if phase["name"] in
                                     ('baseline', 'perturbation', 'recovery') else phase["name"])
                    codec = moving_component_request(response["intent"], phase=harness_phase,
                                                     speed_kmh=sample.fields["speedKmh"])
                    disturbance = (perturb_request(codec['requested_control'], phase=phase['name'],
                                                  sign=trial['sign'])
                                   if trial is not None else None)
                    guard_started = time.perf_counter()
                    guard()
                    emit(dict(kind='guard_timing', stage='before_request', started=guard_started,
                              finished=time.perf_counter()))
                    receive_started = time.perf_counter()
                    latest = reader.receive()
                    latest_diagnostic = diagnostic(latest)
                    emit(dict(kind='native_pre_request_timing', started=receive_started,
                              finished=time.perf_counter()))
                    if phase["name"] == "stationary_prefix":
                        stationary(latest)
                    age = time.perf_counter() - min(v["render_started_lower"]
                                                    for v in candidate["views"])
                    if age > .15:
                        emit(dict(kind='stale_request_rejected', age=age, inference=response,
                                  observation_clock=decision, request_clock=time.perf_counter()))
                        raise ValueError("Output source too old before driver request")
                    buttons = ((axis["gear_buttons_zero_based"]["up"],)
                               if phase_index == 0 and index < 3
                               and first_diagnostic["gear"] == 1 else ())
                    control = (disturbance['requested_control'] if disturbance is not None
                               else codec["requested_control"])
                    if timed is not None:
                        if phase['name'] == 'perturbation' and index == 0:
                            timed.begin(trial['sign'])
                        event = timed.submit(codec['requested_control'],
                            source_render_started_lower=min(v['render_started_lower']
                                                           for v in candidate['views']),
                            pressed_buttons=buttons)
                        request = event['request']
                        disturbance = dict(base_control=codec['requested_control'],
                            disturbance=event['disturbance'], clipped=event['clipped'],
                            requested_control=request['requested_control'], timed=True)
                    else:
                        request = session.submit(**control, pressed_buttons=buttons)
                    if not codec["steering_override"]:
                        result["model_outputs_applied_to_steering"] = True
                    row = dict(kind="flight_axis_observation", phase_index=phase_index,
                        phase=phase["name"], phase_sample=index,
                        perturbation=disturbance, yaw_trial=trial,
                        task_turn_bias=phase["task_turn_bias"],
                        camera=candidate, native=physics_record(sample),
                        native_before_request=physics_record(latest),
                        diagnostic_before_request=latest_diagnostic,
                        body_values=stream.last_body.values.tolist(),
                        body_valid=stream.last_body.valid.tolist(),
                        decision_monotonic_seconds=decision, inference=response, codec=codec,
                        request=request, source_age_at_request_upper_seconds=age,
                        was_foreground=False)
                    records.append(row)
                    emit(row)
                    time.sleep(max(0., .04 - (time.perf_counter() - started)))
            result.update(status="completed_physical_yaw_trial" if trial is not None
                          else "completed_moving_component_samples",
                          model_outputs_applied_to_steering=True,
                          termination_reason="continuous moving component phases completed")
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason=str(error))
        if client is not None:
            failure_diagnostic = dict(exception_clock=time.perf_counter(),
                last_sequence=client.last_sequence,
                last_original_packet=record_packet(client.last_packet)
                if client.last_packet is not None else None,
                transient_read_retries=client.transient_read_retries,
                stale_snapshot_retries=client.stale_snapshot_retries,
                diagnostic_age_limit_seconds=.05, bounded_wait_limit_seconds=.01)
    finally:
        if timed is not None:
            try:
                result['timed_worker'] = timed.stop()
            except Exception as error:
                cleanup_errors.append('timed actuator: '+str(error))
        timed_alive = timed is not None and timed.thread.is_alive()
        result['timed_worker_alive'] = timed_alive
        if session is not None and session.owned and not timed_alive:
            try:
                stopped, stationary_samples = False, 0
                for index in range(75):
                    request = session.submit(0., 0., 1.)
                    time.sleep(.04)
                    sample = reader.receive()
                    emit(dict(kind="cleanup_neutral", index=index, request=request,
                              native=physics_record(sample)))
                    if (abs(sample.fields["speedKmh"]) < .1 and sample.fields["gas"] < .02
                            and sample.fields["brake"] > .98):
                        stationary_samples += 1
                    else:
                        stationary_samples = 0
                    if stationary_samples >= 5:
                        stopped = True
                        break
                if not stopped:
                    raise ValueError("Full-brake cleanup did not independently settle")
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
        gc.callbacks.remove(gc_event)
        if client is not None:
            result['diagnostic_transport'] = dict(
                transient_read_retries=client.transient_read_retries,
                stale_snapshot_retries=client.stale_snapshot_retries)
        if failure_diagnostic is not None:
            write_json(run/'failure-diagnostic.json', failure_diagnostic)
        write_json(run / 'gc-events.json', gc_events)
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
                      if session else 0, moving_balance_verified=False,
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
