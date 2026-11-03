"""Record collector lineage and heartbeat freshness, including unavailable sessions.

This read-only audit never starts the simulator, camera, controls or collection.
An old source seal and a live heartbeat are not a qualification to collect data.
"""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def heartbeat_summary(path, now):
    """Classify freshness without promoting stale fields to current simulator state."""
    try:
        status, modified = read_published_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"state": "unavailable", "error_type": type(exc).__name__}
    age = now - modified
    if not 0 <= age <= 2:
        return {"state": "stale", "age_seconds": age, "current_fields": None}
    fields = ("generation", "epoch", "paused", "ai_controlled", "camera_active",
              "speed_kmh", "gas", "brake", "is_online", "expert_probe_active",
              "physics_api_allowed")
    return {"state": "fresh", "age_seconds": age,
            "current_fields": {key: status.get(key) for key in fields}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-declaration", "installed-camera", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    previous, installed = Path(args.previous_declaration), Path(args.installed_camera)
    declaration = json.loads(previous.read_text(encoding="utf8"))
    sources = declaration["source_sha256"]
    if not isinstance(sources, dict) or len(sources) != 105:
        raise ValueError("Exact previously sealed 105-source inventory required")
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "previous_declaration_sha256": sha256(previous), "read_only": True})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    comparison = []
    for name, expected in sorted(sources.items()):
        path = Path(name)
        actual = sha256(path) if path.is_file() else None
        comparison.append({"path": name, "previous": expected, "current": actual,
                           "matches": actual == expected})
    camera_paths = {
        "installed": installed,
        "candidate": Path(
            "integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua"
        ),
        "repository_main": Path("integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua"),
    }
    camera = {name: sha256(path) if path.is_file() else None for name, path in camera_paths.items()}
    status_path = Path(os.environ["LOCALAPPDATA"]) / "flybrain/csp-session/status.json"
    heartbeat = heartbeat_summary(status_path, time.time())
    result = {"status": "completed_read_only_lineage_and_heartbeat_audit",
        "source_count": len(comparison), "source_comparisons": comparison,
        "camera_sha256": camera, "heartbeat": heartbeat,
        "old_qualification_is_not_current_qualification": True,
        "collector_qualified": False, "capture_started": False, "actor_deployed": False}
    write_json(run / "result.json", result)
    print(json.dumps({"mismatches": [c for c in comparison if not c["matches"]],
                      "camera": camera, "heartbeat": heartbeat}), flush=True)


if __name__ == "__main__":
    main()
