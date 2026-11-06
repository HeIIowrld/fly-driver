"""Same-prefix current/connection ablations of an archived live student trial."""
import argparse
import base64
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.launch_pedal_candidate import load_control_actor_runtime
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder, NavigationBodyObservation, decode_navigation_body_prefix,
)
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.sac_steering_experiment import snapshot


ARMS = ("normal", "visual_currents_off", "fine_motion_off", "body_current_off",
        "descending_balance_off", "descending_power_off", "chemical_transmission_off")


def disable(runtime, arm):
    def zero(_module, _args, output):
        return torch.zeros_like(output)

    handles = []
    warm = None
    if arm == "visual_currents_off":
        handles.extend(module.register_forward_hook(zero) for module in
                       (runtime.model.visual, runtime.current_calibration))
    elif arm == "fine_motion_off":
        handles.extend(module.register_forward_hook(zero) for module in
                       (runtime.motion_port, runtime.current_calibration.motion_projection))
    elif arm == "body_current_off":
        handles.append(runtime.model.body_feedback.register_forward_hook(zero))
    elif arm in ("descending_balance_off", "descending_power_off"):
        axis = int(arm == "descending_power_off")

        def remove_axis(_module, _args, output):
            return tuple(torch.zeros_like(x) if i == axis else x for i, x in enumerate(output))

        handles.append(runtime.model.motor.command.register_forward_hook(remove_axis))
    elif arm == "chemical_transmission_off":
        circuit = runtime.model.motor.backbone.circuit
        matrix = circuit.matrix.coalesce()
        circuit.matrix = torch.sparse_coo_tensor(
            matrix.indices(), torch.zeros_like(matrix.values()), matrix.shape).coalesce()
        # Warm the new sparse cache on isolated constant tensors, as for the
        # intact backend. It must not advance the live recurrent model state.
        warm = prepare_fixed_cpu_backend(runtime.model)
    return handles, warm


def archive(root):
    result = read(root / "result.json")
    for name, digest in result["artifact_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError("Original artifact changed: " + name)
    args = read(root / "manifest.json")["config"]["arguments"]
    identity = read(root / "ready-preflight.json")["identity"]
    episode = read(root / "resume-identity.json")["request_id"]
    scenes = {e["capture_id"]: e for e in read(root / "camera-events.json")
              if e.get("status") == "scene_published"}
    with np.load(root / "published-visual-inputs.npz") as data:
        arrays = {k: data[k].copy() for k in data.files}
    positions = {int(v): i for i, v in enumerate(arrays["capture_ids"])}
    if set(positions) != set(scenes) or len(positions) != len(arrays["capture_ids"]):
        raise ValueError("Published scene coverage differs")
    packets = {}
    for row in read(root / "actor/owner-native-copy.json"):
        auth = row["authority"]
        raw = base64.b64decode(auth["prefix_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != auth["prefix_sha256"]:
            raise ValueError("Original native bytes changed")
        packets.setdefault(auth["packet_id"],
                           (raw[:PHYSICS_PREFIX.size], auth["received_monotonic_seconds"]))
    events = read(root / "actor/owner-ticker-events.json")
    terminal = read(root / "actuator/actuator-terminal.json")
    sdk = [e["decision"]["decision_id"] for e in terminal["events"]
           if e["kind"] == "student_request" and (e.get("receipt") or {}).get("driver_accepted")]
    if not sdk:
        raise ValueError("No actual student SDK decision for branch alignment")
    outputs = [e for e in events if e["status"] == "shadow_output"
               and e["decision_envelope"]["decision_id"] <= sdk[-1]]
    if len(outputs) < 24:
        raise ValueError("At least eight prefix and sixteen branch observations required")
    return args, identity, episode, scenes, arrays, positions, packets, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    out = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        source_result_sha256=sha256(root / "result.json"), seed=None, branches=ARMS,
        scope="Same recorded input, sixteen neural outputs ending at the last SDK decision; offline only"))
    result = dict(status="failed", arms={}, vehicle_control_sent=False, actor_qualified=False)
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    gc.disable()
    try:
        cfg, identity, episode, scenes, arrays, positions, packets, outputs = archive(root)
        branch = len(outputs) - 16
        result.update(prefix_decisions=branch, branch_decisions=16,
                      first_branch_decision=outputs[branch]["decision_envelope"]["decision_id"],
                      last_branch_decision=outputs[-1]["decision_envelope"]["decision_id"])
        predictions, muscles = {}, {}
        reference = None
        for arm in ARMS:
            runtime = None
            handles = []
            p, m = [], []
            detail = result["arms"][arm] = dict(status="failed")
            try:
                runtime, _ = load_control_actor_runtime(
                    cfg["actor_kind"], cfg["checkpoint"], cfg["checkpoint_sha256"])
                runtime.begin_episode(episode)
                encoder = NavigationBodyEncoder(
                    "configs/ac-virtual-navigation-body-v2.json", native_identity=identity)
                for idx, event in enumerate(outputs):
                    if idx == branch:
                        digest = state_digest(snapshot(runtime.model))
                        if arm == "normal":
                            reference = digest
                        if digest != reference:
                            raise ValueError("The recurrent prefix state differs")
                        handles, warm = disable(runtime, arm)
                        if state_digest(snapshot(runtime.model)) != digest:
                            raise ValueError("The intervention altered the prefix state")
                        detail["branch_state_sha256"] = digest
                        if warm is not None:
                            detail["isolated_sparse_backend_warmup"] = warm
                    scene = scenes[event["capture_id"]]
                    j = positions[event["capture_id"]]
                    raw, received = packets[event["body_packet"]]
                    if (received != event["body_first_receipt"] or
                            hashlib.sha256(arrays["motion"][j].tobytes()).hexdigest() != event["motion_features_sha256"]):
                        raise ValueError("Body/motion original binding differs")
                    clock = event["decision_time"]
                    body = encoder.encode(decode_navigation_body_prefix(raw, received),
                                          episode_id=episode, decision_monotonic_seconds=clock)
                    views = tuple(NavigationBodyObservation(
                        episode, v["packet"], v["receipt"], scene["camera_received"],
                        np.asarray(v["values"]), np.asarray(v["valid"]), encoder.contract_sha256)
                        for v in scene["view_body"])
                    lowers = tuple(scene["render_start_lowers"])
                    context = per_view_temporal_body_context(
                        views, body, render_start_lowers=lowers,
                        camera_receipt_time=scene["camera_received"], decision_time=clock)
                    motion = MotionSpatialObservation(episode, scene["camera_received"], lowers,
                                                      arrays["coarse"][j], arrays["motion"][j])
                    decision = runtime.decide(
                        decision_time=clock, scene_features=arrays["scene"][j],
                        static_features=arrays["static"][j], body_context=context,
                        native_sample=decode_native_extension(raw, received), motion_observation=motion)
                    p.append(decision["prediction"])
                    m.append(decision["muscles"])
                    if arm == "normal" or idx < branch:
                        if np.max(np.abs(p[-1] - event["prediction"])) > 2e-6:
                            raise ValueError("Recorded prediction did not replay")
                p, m = np.asarray(p), np.asarray(m)
                if arm != "normal":
                    if not (np.array_equal(p[:branch], predictions["normal"][:branch]) and
                            np.array_equal(m[:branch], muscles["normal"][:branch])):
                        raise ValueError("Prefix predictions or muscles differ")
                    delta = p[branch:] - predictions["normal"][branch:]
                    muscle_delta = m[branch:] - muscles["normal"][branch:]
                    detail.update(output_rmse=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                                  max_abs_output_delta=np.max(np.abs(delta), axis=0).tolist(),
                                  muscle_rmse=np.sqrt(np.mean(muscle_delta**2, axis=0)).tolist())
                else:
                    detail["max_recorded_output_delta"] = float(np.max(np.abs(
                        p - np.asarray([e["prediction"] for e in outputs]))))
                predictions[arm], muscles[arm] = p, m
                detail["status"] = "completed"
            except Exception as exc:
                detail.update(error=type(exc).__name__, reason=str(exc), decisions=len(p))
                if runtime is not None and getattr(runtime, "last_rejection", None):
                    detail["runtime_rejection"] = runtime.last_rejection
                if arm == "normal":
                    raise
            finally:
                for handle in handles:
                    handle.remove()
                if runtime is not None:
                    runtime.close()
        np.savez_compressed(out / "response.npz",
                            **{f"{arm}_{key}": values[arm] for key, values in
                               (("outputs", predictions), ("muscles", muscles)) for arm in values})
        result["status"] = ("completed" if len(predictions) == len(ARMS)
                            else "completed_with_failed_interventions")
    except Exception as exc:
        result.update(error=type(exc).__name__, reason=str(exc))
    finally:
        write_json(out / "result.json", result)
    if result["status"] == "failed":
        raise SystemExit(result["reason"])


if __name__ == "__main__":
    main()
