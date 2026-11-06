"""Own one fixed track switch, stationary shutdown or Magione geometry observation."""
import argparse
import math
import os
import time
import uuid
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.win_process_receipt import ProcessReceipt, engine_inventory, require_single_engine
from scripts.control_ac_session import read_session_heartbeat,terminal_receipt,validate_heartbeat


def native_identity():
    with ExistingNamedMapping(r"Local\acpmf_static",200) as mapping:
        return decode_static_identity_prefix(mapping.read())


def verify_geometry(receipt, request, before):
    sample_session=receipt.get("before") or {}
    if (receipt.get("status")!="heldout_geometry_sampled"
            or receipt.get("command")!=request["command"]
            or receipt.get("geometry_schema")!="flybrain.csp.heldout_geometry.v1"
            or receipt.get("actor_input") is not False
            or any(sample_session.get(k)!=before.get(k) for k in
                   ("generation","epoch","track_id","window_handle","paused","in_menu"))
            or sample_session.get("ai_controlled") is not False
            or sample_session.get("camera_active") is not False):
        raise ValueError("Exact stationary Magione geometry receipt required")
    rows=receipt.get("rows")
    if not isinstance(rows,list) or len(rows)!=512:
        raise ValueError("Complete bounded Magione spline sample required")
    for index,row in enumerate(rows):
        if not isinstance(row,dict):
            raise ValueError("Original geometry sample object required")
        xyz=row.get("native_xyz")
        if (type(row.get("progress")) not in (int,float) or row["progress"]!=index/512 or not isinstance(xyz,list) or len(xyz)!=3
                or any(type(value) not in (float,int) or not math.isfinite(value) or abs(value)>=1e6
                       for value in xyz)
                or any(type(row.get(side)) not in (int,float) or not 0<row[side]<100
                       for side in ("left_m","right_m"))):
            raise ValueError("Finite ordered spline sample and positive side distances required")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",required=True)
    operation=parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--target",choices=("monza","magione"))
    operation.add_argument("--geometry",action="store_true")
    operation.add_argument("--shutdown",choices=("monza","magione"),
        help="Shut down the stationary track engine without requesting a replacement")
    operation.add_argument("--activate",action="store_true")
    operation.add_argument("--restore",action="store_true")
    parser.add_argument("--upgrade-from-magione", type=Path,
        help="Exact already activated candidate for a stationary Magione hot reload")
    parser.add_argument("--relaunch-magione",action="store_true",
        help="Replace the stationary original Magione engine with a fresh Magione engine")
    parser.add_argument("--axis-config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.relaunch_magione and args.target!="magione":
        raise ValueError("Explicit Magione destination required for same-track engine relaunch")
    candidate=Path(args.candidate)
    proof=read(candidate/"result.json")
    digest=sha256(candidate/"FlyBrainCamera.lua")
    if proof["status"]!="completed_compiled_heldout_csp_candidate" or digest!=proof["candidate_sha256"]:
        raise ValueError("Exact compiled two-track candidate required")
    prior=args.upgrade_from_magione
    prior_hash=None
    if prior is not None:
        if not args.activate:
            raise ValueError("Magione hot upgrade requires explicit activation")
        prior_proof=read(prior/"result.json")
        prior_hash=sha256(prior/"FlyBrainCamera.lua")
        if (prior_proof["status"]!="completed_compiled_heldout_csp_candidate"
                or prior_proof["candidate_sha256"]!=prior_hash or prior_hash==digest):
            raise ValueError("Exact distinct prior compiled Magione source required")
    original=candidate/"installed-base.lua"
    baseline="0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e"
    reload_source=args.activate or args.restore
    if reload_source and sha256(original)!=baseline:
        raise ValueError("Exact original CSP rollback bytes required")
    out=begin_run(args.output,dict(args={**vars(args),"upgrade_from_magione":str(prior) if prior else None},
        program_sha256=sha256(__file__),seed=None,
        candidate_sha256=digest,actor_input=False,ui_actions_sent=0,actuator_messages_sent=0))
    result=dict(status="failed",request_publications=0,ui_actions_sent=0,actuator_messages_sent=0,
                live_student_corner_verified=False)
    handle=None
    staged=None
    try:
        api=WindowsACWindow()
        windows=api.discover()
        if len(windows)!=1:raise ValueError("Exactly one background AC engine required")
        window=windows[0]

        def check_engine_inventory(filename):
            inventory=engine_inventory()
            write_json(out/filename,inventory)
            require_single_engine(window.pid,inventory)

        check_engine_inventory("engines-before.json")
        installed=Path(window.executable).parent/"apps/lua/FlyBrainCamera/FlyBrainCamera.lua"
        axis=read(args.axis_config)
        sdk=load_vjoy_sdk(axis["dll_path"],axis["dll_sha256"])
        root=Path(os.environ["LOCALAPPDATA"])/"flybrain/csp-session"
        origin=args.shutdown or ("magione" if args.geometry or args.target=="monza" or args.relaunch_magione or prior else "monza")
        destination=origin if reload_source else args.target
        source_hash=(prior_hash or baseline) if args.activate else digest
        destination_hash=(baseline if args.restore else digest)

        def state_for_request():
            state,modified,_=read_session_heartbeat(root/"status.json")
            validate_heartbeat(state,time.time()-modified,mutating=True)
            paused_track=state["paused"] is True and state["in_menu"] is False
            return_from_menu=not (prior or args.relaunch_magione) and (origin=="magione" or args.restore or args.shutdown) and state["in_menu"] is True
            if (state["track_id"]!=origin or state["car_id"]!="bmw_z4_gt3"
                    or int(state["window_handle"].removesuffix("ULL"))!=window.hwnd
                    or not (paused_track or return_from_menu) or state["camera_active"]
                    or state.get("camera_request_id") or api.user.GetForegroundWindow()==window.hwnd
                    or sdk.GetVJDStatus(axis["device_id"])!=1 or sha256(axis["profile_path"])!=axis["profile_sha256"]
                    or api.probe(window.hwnd,require_foreground=False)!=window):
                raise ValueError("Stationary released source track and background authority required")
            with NativeTeacherPhysicsReader() as reader:
                sample=reader.receive()
            fields=sample.physics.fields
            if (sample.builtin_ai_controlled or abs(fields["speedKmh"])>=.1
                    or fields["gas"]>=.02 or fields["brake"]<.98):
                raise ValueError("Actual stationary brake hold required")
            identity=native_identity()
            if identity["track"]!=origin or identity["car_model"]!="bmw_z4_gt3":
                raise ValueError("Native original track or car differs")
            return state,sample.authority_record()

        before,native=state_for_request()
        installed_hash=sha256(installed)
        write_json(out/"before.json",dict(session=before,native=native,installed_sha256=installed_hash))
        capability=("restart_engine" if reload_source else
                    "stationary_engine_shutdown_v1" if args.shutdown else
                    "heldout_track_geometry_v1" if args.geometry else
                    "heldout_magione_engine_relaunch_v1" if args.relaunch_magione else "restart_heldout_track_v1")
        if (installed_hash!=source_hash or before.get("capabilities",{}).get(capability) is not True):
            if args.execute:raise ValueError("Live pinned heldout capability is missing")
            result.update(status="read_only_candidate_not_active",observed_track=origin)
            return
        if not args.execute:
            result.update(status="read_only_ready",observed_track=origin)
            return

        handle=ProcessReceipt(window.pid)
        if not handle.snapshot()["active"]:raise ValueError("Original engine has no live process handle")
        request=dict(schema="flybrain.csp.session.request.v1",id=uuid.uuid4().hex,
            command=("restart-engine" if reload_source else
                     "shutdown-engine" if args.shutdown else
                     "sample-heldout-geometry" if args.geometry else "restart-heldout-track"),
            generation=before["generation"],epoch=before["epoch"],expected_track_id=origin,
            native_ai_off_verified=True,released_controller_verified=True)
        if not args.geometry:
            request["old_engine_pid"]=window.pid
        if not args.geometry and not reload_source and not args.shutdown:
            request["target_track_id"]=args.target
            if args.relaunch_magione:
                request["same_track_engine_relaunch"]=True
        if reload_source:
            backup=out/"installed-before.lua"
            backup.write_bytes(installed.read_bytes())
            if sha256(backup)!=source_hash:
                raise ValueError("Exact pre-transaction source backup required")
            replacement=original if args.restore else candidate/"FlyBrainCamera.lua"
            staged=installed.with_name(installed.name+".heldout-"+request["id"]+".partial")
            with staged.open("xb") as handle_file:
                handle_file.write(replacement.read_bytes())
            if sha256(staged)!=destination_hash:
                raise ValueError("Staged candidate bytes differ")
        check_engine_inventory("engines-before-request.json")
        fresh,native=state_for_request()
        if sha256(installed)!=source_hash:raise ValueError("Pinned installed source changed")
        request=fresh_unpublished_request(before,fresh,request,publication_started=False)
        if reload_source:
            write_json(out/"source-transition-intent.json",dict(original_sha256=source_hash,
                destination_sha256=destination_hash,original_pid=window.pid,request_id=request["id"]))
            if sha256(installed)!=source_hash or not handle.snapshot()["active"]:
                raise ValueError("Source or original process changed before replacement")
            os.replace(staged,installed)
            staged=None
            result["installed_source_changed"]=True
            if sha256(installed)!=destination_hash:
                raise ValueError("Installed candidate hash differs")
            hot_deadline=time.monotonic()+8
            while time.monotonic()<hot_deadline:
                observed,_=state_for_request()
                if observed["generation"]!=before["generation"]:
                    active=observed.get("capabilities",{}).get("restart_heldout_track_v1") is True
                    if (active is not args.activate or not handle.snapshot()["active"]
                            or sha256(installed)!=destination_hash):
                        raise ValueError("Observed reload does not match staged source")
                    write_json(out/"observed-hot-reload.json",dict(session=observed,pid=window.pid))
                    result.update(status="verified_hot_reloaded_csp_source",installed_sha256=destination_hash,
                                  observed_track=origin,old_pid=window.pid,new_pid=window.pid)
                    return
                time.sleep(.05)
            if prior:
                raise TimeoutError("Magione hot reload was not observed; no engine restart published")
            check_engine_inventory("engines-before-restart-fallback.json")
            fresh,native=state_for_request()
            if not fresh["paused"]:
                raise ValueError("No hot reload in menu; no engine restart published")
            request=fresh_unpublished_request(before,fresh,request,publication_started=False)
        journal=root/"intent-journal"
        journal.mkdir(exist_ok=True)
        write_json(journal/(request["id"]+".json"),dict(request=request,before=fresh,native=native,
                     publication_attempts_before_journal=0))
        write_json(out/"request.json",request)
        result.update(status="geometry_outcome_unknown" if args.geometry else
                      "shutdown_outcome_unknown" if args.shutdown else "restart_outcome_unknown",
                      request_publications=1,request_id=request["id"])
        write_json(out/"result.json",result)
        write_json(root/"request.json",request)
        intent=None
        deadline=time.monotonic()+(5 if args.geometry else 90)
        while time.monotonic()<deadline:
            if intent is None:
                intent=terminal_receipt(root,request)
                if intent is not None:
                    if args.geometry:
                        write_json(out/"geometry-receipt.json",intent)
                        rows=verify_geometry(intent,request,fresh)
                        after,_=state_for_request()
                        if (not handle.snapshot()["active"]
                                or after["generation"]!=fresh["generation"] or after["epoch"]!=fresh["epoch"]):
                            raise ValueError("Geometry session changed after sampling")
                        write_json(out/"geometry.json",dict(track_id=origin,actor_input=False,rows=rows))
                        result.update(status="verified_stationary_heldout_geometry",sample_count=len(rows),
                            geometry_sha256=sha256(out/"geometry.json"),observed_track=origin)
                        return
                    write_json(out/("shutdown-intent.json" if args.shutdown else "restart-intent.json"),intent)
                    expected_intent=("engine_shutdown_intent_committed" if args.shutdown else "engine_restart_intent_committed")
                    if (intent["status"]!=expected_intent
                            or intent.get("command")!=request["command"]
                            or (not reload_source and intent.get("target_track_id")!=args.target)
                            or intent.get("old_engine_pid")!=window.pid):
                        raise ValueError("Exact immutable engine transition intent is missing: "+str(intent.get("status")))
                    if args.shutdown and any(intent.get("before",{}).get(key)!=fresh[key]
                            for key in ("generation","epoch","track_id","window_handle","paused","in_menu")):
                        raise ValueError("Shutdown intent differs from the authorized stationary session")
            old=handle.snapshot()
            if args.shutdown:
                if intent is not None and old["termination_signaled"]:
                    write_json(out/"original-engine-exit.json",old)
                    inventory=engine_inventory()
                    write_json(out/"engines-after.json",inventory)
                    if inventory:
                        raise ValueError("Original engine exited but native engine absence was not observed")
                    result.update(status="verified_stationary_engine_shutdown",original_pid=window.pid)
                    return
                time.sleep(.05)
                continue
            engines=api.discover()
            if intent is not None and not old["active"] and len(engines)==1 and engines[0].pid!=window.pid:
                new=engines[0]
                try:
                    state,modified,_=read_session_heartbeat(root/"status.json")
                    validate_heartbeat(state,time.time()-modified,mutating=False)
                    ident=native_identity()
                    if (Path(new.executable)==Path(window.executable) and state["generation"]!=before["generation"]
                            and int(state["window_handle"].removesuffix("ULL"))==new.hwnd
                            and state["track_id"]==ident["track"]==destination and ident["car_model"]=="bmw_z4_gt3"
                            and not state["is_online"] and not state["ai_controlled"] and not state["camera_active"]
                            and sha256(installed)==destination_hash
                            and (not reload_source or bool(state.get("capabilities",{}).get("restart_heldout_track_v1"))
                                 is args.activate)):
                        inventory=engine_inventory()
                        write_json(out/"engines-after.json",inventory)
                        require_single_engine(new.pid,inventory)
                        write_json(out/"observed-new-engine.json",dict(native_identity=ident,session=state,
                            pid=new.pid,foreground=api.user.GetForegroundWindow()==new.hwnd,original_process=old))
                        result.update(status="verified_reloaded_csp_source" if reload_source else "verified_heldout_engine_track",
                                      observed_track=ident["track"],installed_sha256=destination_hash,
                                      original_pid=window.pid,new_pid=new.pid)
                        return
                except (OSError,ValueError,TimeoutError):
                    pass
            time.sleep(.05)
        if args.geometry:raise TimeoutError("Same-request Magione geometry outcome was not observed")
        if args.shutdown:
            raise TimeoutError("Shutdown intent and original engine termination were not jointly observed")
        raise TimeoutError("Exact new track and old engine exit were not jointly observed")
    except Exception as error:
        result["reason"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        if staged is not None and staged.exists():staged.unlink()
        if handle is not None:
            try:
                result["final_original_process"]=handle.snapshot()
            except (OSError,RuntimeError) as error:
                result["final_original_process_observation_error"]=str(error)
            finally:
                handle.close()
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
