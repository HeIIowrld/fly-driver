"""Read-only source and lifecycle comparison for a new pedal collection."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-declaration", required=True)
    parser.add_argument("--installed-camera", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    previous = Path(args.previous_declaration)
    declaration = json.loads(previous.read_text(encoding="utf8"))
    sources = declaration["source_sha256"]
    if not isinstance(sources, dict) or len(sources) != 105:
        raise ValueError("Exact previously sealed source inventory required")
    comparisons = []
    for name, expected in sorted(sources.items()):
        path = Path(name)
        actual = sha256(path) if path.is_file() else None
        comparisons.append({"path": name, "previous": expected, "current": actual,
                            "matches": actual == expected})
    installed = Path(args.installed_camera)
    candidate = Path("integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua")
    main = Path("integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua")
    hashes = {"installed": sha256(installed), "candidate": sha256(candidate),
              "repository_main": sha256(main)}
    status_path = Path(os.environ["LOCALAPPDATA"]) / "flybrain/csp-session/status.json"
    status, modified = read_published_json(status_path)
    age = time.time() - modified
    if not 0 <= age <= 2:
        raise ValueError("Fresh lifecycle status required")
    snapshot = {k: status.get(k) for k in (
        "generation", "epoch", "paused", "ai_controlled", "camera_active", "speed_kmh",
        "gas", "brake", "is_online", "expert_probe_active", "physics_api_allowed",
    )}
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "previous_declaration_sha256": sha256(previous), "installed_path": str(installed),
        "read_only": True})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = {"status": "completed_read_only_comparison", "source_count": len(comparisons),
        "source_comparisons": comparisons, "current_camera_hashes": hashes,
        "status_snapshot": snapshot, "status_age_seconds_at_read": age,
        "old_qualification_is_not_current_qualification": True,
        "collector_qualified": False, "capture_started": False, "actor_deployed": False,
        "goal_achieved": False}
    write_json(run / "result.json", result)
    print(json.dumps({"mismatches": [c for c in comparisons if not c["matches"]],
                      "camera": hashes, "status": snapshot, "age": age}), flush=True)


if __name__ == "__main__":
    main()
