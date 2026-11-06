"""Fork the actual accepted moving-student input trace at one recurrent state."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.launch_pedal_candidate import load_control_actor_runtime
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder, NavigationBodyObservation, decode_navigation_body_prefix,
)
from scripts.audit_balanced_flight_components import ARMS, disable
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.sac_steering_experiment import restore, snapshot


def archive(run, *, failed_magione_short_trace=False):
    outcome = read(run/"result.json")
    teacher = read(run/"teacher/result.json")
    pins = {"events.json": outcome["events_sha256"],
            "owner-native-copy.json": outcome["native_sha256"],
            "owner-ticker-events.json": outcome["ticker_events_sha256"],
            "published-visual-inputs.npz": outcome["published_visual_inputs_sha256"],
            "teacher/events.jsonl": teacher["events_sha256"]}
    cleanup_verified = outcome["cleanup_verified"]
    if failed_magione_short_trace:
        cfg = read(run/"manifest.json")["config"]
        pause = read(run/"final-pause/result.json")
        exit_receipt = read(run.parent/"owned.exit.json")
        stopped = teacher["moving_student_trial"]["cleanup"]
        cleanup_verified = (cfg["identity"]["track"]=="magione" and outcome["status"]=="failed"
            and outcome["native_sampler_stopped"] is True and outcome["pause_returncode"]==0
            and sha256(run/"teacher/result.json")==outcome["teacher_result_sha256"]
            and stopped["stationary_verified"] is True and stopped["maximum_tyres_out"]==0
            and stopped["new_damage"] is False and pause["status"]=="completed"
            and pause["target_state_observed"] is True
            and exit_receipt["original_child_handle"] is True and not exit_receipt["forced_termination"])
    if not cleanup_verified or any(sha256(run/name)!=digest for name,digest in pins.items()):
        raise ValueError("Sealed moving trial with verified physical cleanup required")
    with (run/"teacher/events.jsonl").open(encoding="utf8") as handle:
        records = [json.loads(line) for line in handle]
    events = [v["event"] for v in records if v["kind"]=="moving_actuator_event"]
    accepted = [v for v in events if v["kind"]=="student_request" and v["receipt"]["driver_accepted"]]
    minimum = 4 if failed_magione_short_trace else 16
    if len(accepted)<minimum or len(accepted)!=teacher["student_requests"]:
        raise ValueError(f"At least {minimum} accepted actual student requests required")
    outputs = [v for v in read(run/"owner-ticker-events.json") if v["status"]=="shadow_output"]
    by_id = {v["decision_envelope"]["decision_id"]:v["decision_envelope"] for v in outputs}
    for event in accepted:
        decision=event["decision"]
        if by_id[decision["decision_id"]]!=decision:
            raise ValueError("Original SDK decision differs from actor output")
    first,last=[accepted[i]["decision"]["decision_id"] for i in (0,-1)]
    outputs=[v for v in outputs if v["decision_envelope"]["decision_id"]<=last]
    branch=next(i for i,v in enumerate(outputs) if v["decision_envelope"]["decision_id"]==first)
    scenes={v["capture_id"]:v for v in read(run/"events.json") if v["status"]=="scene_published"}
    with np.load(run/"published-visual-inputs.npz") as data:
        arrays={key:data[key].copy() for key in data.files}
    positions={int(value):i for i,value in enumerate(arrays["capture_ids"])}
    packets={}
    for row in read(run/"owner-native-copy.json"):
        native=row["authority"]
        raw=base64.b64decode(native["prefix_base64"],validate=True)
        if hashlib.sha256(raw).hexdigest()!=native["prefix_sha256"]:
            raise ValueError("Original native hash differs")
        packets.setdefault(native["packet_id"],(raw[:PHYSICS_PREFIX.size],native["received_monotonic_seconds"]))
    return pins, scenes, arrays, positions, packets, outputs, branch, accepted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True)
    parser.add_argument("--compare-compact",type=Path)
    parser.add_argument("--export-decision-muscles",action="store_true",
                        help="Archive the full-prefix clock and muscle traces from the normal and intervention replays")
    parser.add_argument("--failed-magione-short-trace",action="store_true",
                        help="Use a verified safe-stop trace from an explicitly failed Magione interval")
    args=parser.parse_args()
    run=args.run
    out=begin_run(args.output,dict(run=str(run),source_result_sha256=sha256(run/"result.json"),
        program_sha256=sha256(__file__),seed=None,vehicle_control_sent=False,
        failed_magione_short_trace=args.failed_magione_short_trace,export_decision_muscles=args.export_decision_muscles))
    result=dict(status="failed",vehicle_control_sent=False,arms={})
    runtime=None
    comparison=None
    try:
        pins,scenes,arrays,positions,packets,events,branch,accepted=archive(run,failed_magione_short_trace=args.failed_magione_short_trace)
        config=read(run/"manifest.json")["config"]
        cfg=config["arguments"]
        episode=read(run/"resume-identity.json")["request_id"]
        torch.set_num_threads(cfg.get("moving_body_threads",1))
        torch.set_flush_denormal(cfg["flush_body_denormal"])
        runtime,_=load_control_actor_runtime(cfg["actor_kind"],cfg["current_checkpoint"],cfg["current_checkpoint_sha256"])
        runtime.begin_episode(episode)
        runtime.monotonic=lambda:0.
        encoder=NavigationBodyEncoder("configs/ac-virtual-navigation-body-v2.json",native_identity=config["identity"])
        arguments=[]
        for event in events:
            scene=scenes[event["capture_id"]]
            j=positions[event["capture_id"]]
            raw,received=packets[event["body_packet"]]
            if (received!=event["body_first_receipt"] or
                    hashlib.sha256(arrays["motion"][j].tobytes()).hexdigest()!=event["motion_features_sha256"]):
                raise ValueError("Original packet and scene receipts differ")
            clock=event["decision_time"]
            body=encoder.encode(decode_navigation_body_prefix(raw,received),
                episode_id=episode,decision_monotonic_seconds=clock)
            views=tuple(NavigationBodyObservation(
                episode,v["packet"],v["receipt"],scene["camera_received"],
                np.asarray(v["values"]),np.asarray(v["valid"]),encoder.contract_sha256)
                for v in scene["view_body"])
            lowers=tuple(scene["render_start_lowers"])
            context=per_view_temporal_body_context(views,body,render_start_lowers=lowers,
                camera_receipt_time=scene["camera_received"],decision_time=clock)
            motion=MotionSpatialObservation(episode,scene["camera_received"],lowers,arrays["coarse"][j],arrays["motion"][j])
            arguments.append(dict(decision_time=clock,scene_features=arrays["scene"][j],static_features=arrays["static"][j],
                body_context=context,native_sample=decode_native_extension(raw,received),motion_observation=motion))
        normal=[]
        full_muscles=[]
        accepted_muscles=[]
        for index,argument in enumerate(arguments):
            if index==branch:
                prefix=snapshot(runtime.model)
                boundary=(runtime.last_time,runtime.encoder.last,runtime.encoder.last_decision)
            value=runtime.decide(**argument)
            normal.append(value["prediction"])
            if args.export_decision_muscles:
                full_muscles.append(value["muscles"])
            if index>=branch:
                accepted_muscles.append(value["muscles"])
        normal=np.asarray(normal)
        parity=float(np.max(np.abs(normal-np.asarray([e["prediction"] for e in events]))))
        if parity>2e-6:
            raise ValueError(f"Original moving output replay differs: {parity}")
        digest=state_digest(prefix)
        result.update(prefix_decisions=branch,branch_decisions=len(events)-branch,
            accepted_requests=len(accepted),maximum_original_output_difference=parity,
            prefix_sha256=digest,source_sha256=pins)
        if args.compare_compact:
            checkpoint=args.compare_compact/"pedals.pt"
            expected=read(args.compare_compact/"result.json")["checkpoint_sha256"]
            comparison,_=load_control_actor_runtime(cfg["actor_kind"],checkpoint,expected)
            with torch.no_grad():
                old=runtime.head(torch.tensor(np.asarray(accepted_muscles))).numpy()
                new=comparison.head(torch.tensor(np.asarray(accepted_muscles))).numpy()
            if not np.allclose(old,normal[branch:],atol=2e-6,rtol=0) or not np.array_equal(old[:,0],new[:,0]):
                raise ValueError("Sealed high-speed readout comparison differs in original or steering")
            idx=[e["decision"]["decision_id"]-events[branch]["decision_envelope"]["decision_id"] for e in accepted]
            result["comparison"]=dict(checkpoint_sha256=expected,actual_student_requests_replayed=len(idx),
                observed_head_mean_pedals=np.mean(old[idx,1:],axis=0).tolist(),
                comparison_head_mean_pedals=np.mean(new[idx,1:],axis=0).tolist(),
                observed_head_simultaneous=int(np.sum(np.all(old[idx,1:]>.02,axis=1))),
                comparison_head_simultaneous=int(np.sum(np.all(new[idx,1:]>.02,axis=1))),
                max_abs_delta=float(np.max(np.abs(new[idx]-old[idx]))),comparison_head_closed_loop_on_this_trace=False)
        matrix=runtime.model.motor.backbone.circuit.matrix
        arrays_out={}
        if args.export_decision_muscles:
            arrays_out.update(full_muscles=np.asarray(full_muscles), full_prediction=normal,
                full_decision_time=np.asarray([arg["decision_time"] for arg in arguments]),
                full_decision_ids=np.asarray([e["decision_envelope"]["decision_id"] for e in events]),
                accepted_decision_ids=np.asarray([e["decision"]["decision_id"] for e in accepted]))
        failed_trace=args.failed_magione_short_trace
        for arm in ARMS:
            restore(runtime.model,prefix)
            runtime.last_time,runtime.encoder.last,runtime.encoder.last_decision=boundary
            if state_digest(snapshot(runtime.model))!=digest:
                raise ValueError("Original branch state changed")
            handles,_=disable(runtime,arm)
            try:
                branch_values=[runtime.decide(**a) for a in arguments[branch:]]
                values=np.asarray([v["prediction"] for v in branch_values])
                if comparison is not None:
                    with torch.no_grad():
                        compared=comparison.head(torch.tensor(np.asarray([v["muscles"] for v in branch_values]))).numpy()
                    if arm=="normal" and not np.array_equal(compared,new):
                        raise ValueError("Restored comparison head branch is not exact")
                    arrays_out["comparison_"+arm]=compared
                    result["comparison"].setdefault("arms",{})[arm]=dict(
                        accepted_output_rmse=np.sqrt(np.mean((compared[idx]-new[idx])**2,axis=0)).tolist())
                if arm=="normal" and not np.array_equal(values,normal[branch:]):
                    raise ValueError("Restored normal branch is not exact")
                arrays_out[arm]=values
                if args.export_decision_muscles or (arm=="normal" and (failed_trace or comparison is not None)):
                    arrays_out[arm+"_muscles"]=np.asarray([v["muscles"] for v in branch_values])
                result["arms"][arm]=dict(output_rmse=np.sqrt(np.mean((values-normal[branch:])**2,axis=0)).tolist())
            finally:
                for handle in handles:handle.remove()
                circuit=runtime.model.motor.backbone.circuit
                if circuit.matrix is not matrix:
                    circuit.matrix=matrix
                    prepare_fixed_cpu_backend(runtime.model)
        np.savez_compressed(out/"branches.npz",**arrays_out)
        result.update(status="completed",branches_sha256=sha256(out/"branches.npz"))
    except Exception as error:
        result["reason"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        if comparison is not None:comparison.close()
        if runtime is not None:runtime.close()
        write_json(out/"result.json",result)


if __name__=="__main__":
    main()
