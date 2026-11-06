"""Summarize owned launch trials using inner source/SDK records and outer exit receipts."""
import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.summarize_current_flight_control import collect


def events_for(path, row):
    root = path / "run"
    result = read(root / "result.json")
    row["termination_reason"] = result["termination_reason"]
    row["sequence_sha256"] = sha256(path / "result.json")
    for key in ("pause_action", "pause_process_terminal"):
        if key in result:
            row[key] = result[key]
    gate = root / "final-pause/camera-idle-admission.json"
    if gate.is_file():
        camera = read(gate)
        row["camera_gate"] = dict(observations=len(camera),
                                  saw_old_active_heartbeat=any(v["active"] for v in camera),
                                  ended_idle=camera[-1]["active"] is False)
    if not (root / "actuator/actuator-terminal.json").is_file():
        return
    actuator = read(root / "actuator/actuator-terminal.json")
    requests = [e for e in actuator["events"] if e["kind"] == "student_request"
                and (e.get("receipt") or {}).get("driver_accepted")]
    if not requests:
        return
    source = {e["decision_envelope"]["decision_id"]: e
              for e in read(root / "actor/owner-ticker-events.json") if e["status"] == "shadow_output"}
    ids, latency = [], []
    for request in requests:
        decision = request["decision"]
        i = decision["decision_id"]
        event = source[i]
        if (event["decision_envelope"] != decision or request["conversion"]["pedal_override"]
                or request["conversion"]["clipping_applied"]):
            raise ValueError("Original student payload or conversion changed")
        delay = request["receipt"]["driver_call_started_perf_counter_seconds"] - event["output_host_time"]
        if delay < 0 or (ids and i <= ids[-1]):
            raise ValueError("Source publication/SDK order invalid")
        ids.append(i)
        latency.append(delay)
    row["source_to_sdk_seconds"] = dict(min=float(min(latency)), median=float(np.median(latency)),
                                        max=float(max(latency)))
    row["student_decision_ids_skipped_between_first_and_last"] = ids[-1] - ids[0] + 1 - len(ids)
    row["unmodified_student_requests_verified"] = len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(args.base)
    out = Path(args.output)
    if out.exists():
        raise ValueError("Unique comparison summary required")
    entries = [row for row in collect(base) if Path(row["directory"]).name == "run"]
    for row in entries:
        path = Path(row["directory"]).parent
        events_for(path, row)
    owners = {}
    for name in ("comparison-owned-01", "comparison-owned-02", "phase-comparison-owned-01"):
        result = base / name / "result.json"
        if result.is_file():
            owners[name] = dict(result=read(result), sha256=sha256(result))
    write_json(out, dict(schema="flybrain.balanced_launch_comparison.v1", runs=entries,
                         comparisons=owners, total_student_sdk_accepted=sum(
                             row["student_sdk_accepted"] for row in entries),
                         actor_qualified=False, full_lap_verified=False, ui_actions_sent=0))


if __name__ == "__main__":
    main()
