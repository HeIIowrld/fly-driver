"""Read-only readiness audit for moving brake and held-out corner trials."""
import argparse
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.preflight_current_flight_control import inspect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    repo_lua=Path("integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua")
    legacy=Path("scripts/pilot_mailbox_flight_control.py")
    install=Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa")
    installed_lua=install/"apps/lua/FlyBrainCamera/FlyBrainCamera.lua"
    root=install/"content/tracks"
    split=Path("runs/thermal-control-improvement-20261104/newhot-cohort-01.json")
    out=begin_run(a.output,dict(program_sha256=sha256(__file__),seed=None,read_only=True,
        existing_pilot_sha256=sha256(legacy),source_split_sha256=sha256(split),
        vehicle_control_sent=False))
    result=dict(status="failed",vehicle_control_sent=False)
    try:
        state=inspect()
        write_json(out/"observed-current-session.json",state)
        candidate=root/"magione"/"models.ini"
        pilot=legacy.read_text(encoding="utf-8")
        lua=repo_lua.read_text(encoding="utf-8")
        if sha256(installed_lua)!=state["installed_lua_sha256"]:
            raise ValueError("Installed lifecycle source changed after preflight")
        active_source=installed_lua.read_text(encoding="utf-8")
        track_restriction="before.track_id == 'monza'" in lua
        original_launch_bounds="maximum_speed_kmh=10." in pilot and "maximum_travel_m=1." in pilot
        source_tracks=[]
        for e in read(split)["episodes"]:
            p=Path(e["data"])
            cfg=read(p/"manifest.json")["config"]
            ident=read(cfg["arguments"]["identity"])["identity"]
            source_tracks.append(dict(role=e["role"],episode_id=e["episode_id"],track=ident["track"]))
        result.update(status="completed_readonly_moving_trial_admission_audit",
            active_track=state["identity"]["track"],current_session_paused=state["session"]["paused"],
            direct_pedal_fit_source_tracks=source_tracks,existing_launch_bounds_10kmh_1m=original_launch_bounds,
            repository_lifecycle_source_sha256=sha256(repo_lua),
            repo_source_matches_installed=sha256(repo_lua)==state["installed_lua_sha256"],
            repo_lifecycle_fixed_monza=track_restriction,
            actual_lifecycle_source_sha256=state["installed_lua_sha256"],
            actual_lifecycle_fixed_monza="before.track_id == 'monza'" in active_source,
            actual_restart_excludes_race_ini="request.race_ini == nil" in active_source,
            prospective_unused_geometry=dict(track="magione",installed_model_ini_present=candidate.is_file(),
              models_ini_sha256=sha256(candidate) if candidate.is_file() else None,
              used_by_direct_pedal_fit=any(v["track"]=="magione" for v in source_tracks),
              all_parent_training_lineage_confirmed=False,authorized_live_entry_measured=False),
            student_high_speed_authority_transfer_admitted=False,
            student_heldout_corner_actuation_admitted=False,
            outstanding=[
                "One SDK owner must prove mutually exclusive pacing and student grant while rolling",
                "Real matched native entry pose, direction and speed; source-qualified continuous prefix",
                "Independent abort corridor and measured brake-to-stationary travel at entry speed",
                "Verified non-desktop lifecycle and diagnostic geometry for prospective held-out track",
                "No teacher commands counted after the student grant, and no cleanup commands counted as learned braking"])
    except Exception as exc:
        result["reason"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
