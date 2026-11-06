"""Replay real archived high-speed onset windows with same-prefix component forks."""
import argparse
import base64
import gc
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
from scripts.audit_balanced_flight_components import ARMS, disable
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.sac_steering_experiment import snapshot, restore
from scripts.train_qualified_sac_unified import context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit",required=True)
    p.add_argument("--phase-audit",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=begin_run(a.output,dict(args=vars(a),program_sha256=sha256(__file__),
        audit_sha256=sha256(a.phase_audit), seed=None, vehicle_control_sent=False,
        scope="Offline forks of consumed original nominal development; no new corner or wall-time qualification"))
    result=dict(status="failed",windows=[],vehicle_control_sent=False)
    runtime=None
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    enabled=gc.isenabled()
    gc.disable()
    try:
        fit=Path(a.fit); cfg=read(fit/"manifest.json")["config"]
        fr=read(fit/"result.json")
        pa=read(a.phase_audit)["fits"][str(fit)]
        if fr["predictions_sha256"]!=pa["predictions_sha256"] or sha256(fit/"predictions.npz")!=fr["predictions_sha256"]:
            raise ValueError("Exact phase and audit trajectories required")
        base=read("runs/fixture-adjoint-learning-20261104/plan-01.json")
        old=read(read(base["base_port_plan"])["parent_fit_plan"])
        candidate=read(old["candidate_plan"])
        spec=read(Path(candidate["motion"]["fit"])/"manifest.json")["config"]
        (_,_,entries,items,extras,_,_,_,_,features,maps,coarse,_,_)=load_motion_experiment(spec["plan"],"fine_motion")
        i=2
        if entries[i]["role"]!="development":
            raise ValueError("Explicit fixed development role required")
        d,c,_=items[i]
        clock=d["decision_time"]
        # Fixed first three qualifying, well separated high-speed onset windows.
        windows=[]
        for onset in pa["episodes"][i]["onset_events"]:
            if d["body"][onset,16]<40 or (windows and clock[onset]-clock[windows[-1][1]]<1.):
                continue
            begin=int(np.searchsorted(clock,clock[onset]-.25))
            end=int(np.searchsorted(clock,clock[onset]+.25))
            windows.append((begin,onset,end))
            if len(windows)==3:break
        if len(windows)!=3:
            raise ValueError("Not enough distinct preregistered high-speed onset windows")
        metadata=read(Path(entries[i]["data"])/"manifest.json")["config"]["arguments"]
        native=Path(metadata["recording"])/"camera/native-body.jsonl"
        qualification=read(Path(metadata["qualification"])/"result.json")
        if sha256(native)!=qualification["native_body_sha256"]:
            raise ValueError("Native source differs from qualification")
        needed=set(map(int,d["body_packet"][:windows[-1][2]]))
        samples={}
        with native.open(encoding="utf-8") as handle:
            for line in handle:
                raw=json.loads(line)["authority"]; packet=raw["packet_id"]
                if packet in needed and packet not in samples:
                    prefix=base64.b64decode(raw["prefix_base64"],validate=True)
                    if hashlib.sha256(prefix).hexdigest()!=raw["prefix_sha256"]:
                        raise ValueError("Native prefix hash differs")
                    samples[packet]=decode_native_extension(prefix[:PHYSICS_PREFIX.size],raw["received_monotonic_seconds"])
                if len(samples)==len(needed):break
        if len(samples)!=len(needed):
            raise ValueError("Native source coverage incomplete")
        runtime,_=load_control_actor_runtime("balanced_"+cfg["actor_kind"],fit/"pedals.pt",fr["checkpoint_sha256"])
        runtime.begin_episode(c["episode_id"])
        # Offline virtual compute clock only; original sensor and decision times are
        # still validated. This is not a real-time qualification.
        runtime.monotonic=lambda:0.
        def decide(j):
            frame=int(maps[i][j])
            motion=MotionSpatialObservation(c["episode_id"],float(d["camera_receipt"][j]),
                tuple(map(float,d["render_lowers"][j])),coarse[i][frame],features[i][frame].numpy())
            return runtime.decide(decision_time=float(clock[j]),scene_features=d["scene"][j],
                static_features=extras[i][j].numpy(),body_context=context(d,c,j),
                native_sample=samples[int(d["body_packet"][j])],motion_observation=motion)["prediction"]
        def state():
            return (snapshot(runtime.model),runtime.last_time,runtime.encoder.last,runtime.encoder.last_decision)
        def return_to(s):
            model,when,last,decision=s
            restore(runtime.model,model)
            runtime.last_time=when
            runtime.encoder.last=last
            runtime.encoder.last_decision=decision
        with np.load(fit/"predictions.npz") as z:expected=z[f"prediction_{i}"].copy()
        prefixes={}
        index={w[0] for w in windows}
        intact=[]
        for j in range(windows[-1][2]):
            if j in index:prefixes[j]=state()
            intact.append(decide(j))
            if j%1000==0:write_json(out/"progress.json",dict(phase="chronological_replay",decisions=j))
        intact=np.asarray(intact)
        parity=float(np.max(abs(intact-expected[:len(intact)])))
        if parity>2e-6:
            raise ValueError(f"Original high-speed output parity differs: {parity}")
        arrays={}
        for k,(begin,onset,end) in enumerate(windows):
            prefix=prefixes[begin]
            digest=state_digest(prefix[0])
            normal=intact[begin:end]
            label=d["labels"][begin:end]
            mask=d["fit_mask"][begin:end]
            row=dict(begin=begin,onset=onset,end=end,prefix_sha256=digest,
                     speed_at_onset_m_s=float(d["body"][onset,16]),arms={})
            result["windows"].append(row)
            for arm in ARMS:
                return_to(prefix)
                circuit=runtime.model.motor.backbone.circuit
                original_matrix=circuit.matrix
                handles=[]
                try:
                    handles,_=disable(runtime,arm)
                    if state_digest(snapshot(runtime.model))!=digest:
                        raise ValueError("Fork changed its recurrent prefix")
                    values=np.asarray([decide(j) for j in range(begin,end)])
                    if arm=="normal" and not np.array_equal(values,normal):
                        raise ValueError("Normal fork differs from the chronological history")
                    arrays[f"window{k}_{arm}"]=values
                    row["arms"][arm]=dict(output_rmse=np.sqrt(np.mean((values-normal)**2,axis=0)).tolist(),
                       reference_rmse=np.sqrt(np.mean((values[mask]-label[mask])**2,axis=0)).tolist(),
                       reference_labels=int(mask.sum()))
                finally:
                    for h in handles:h.remove()
                    if circuit.matrix is not original_matrix:
                        circuit.matrix=original_matrix
                        prepare_fixed_cpu_backend(runtime.model)
        np.savez_compressed(out/"branches.npz",**arrays)
        result.update(status="completed",prefix_decisions=len(intact),maximum_original_output_difference=parity,
            native_sha256=qualification["native_body_sha256"],branches_sha256=sha256(out/"branches.npz"))
    except Exception as exc:
        result["reason"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if runtime is not None:runtime.close()
        if enabled:gc.enable()
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
