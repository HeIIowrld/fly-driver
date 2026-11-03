import json
import os

from scripts.audit_pedal_collection_offline import heartbeat_summary


def test_stale_heartbeat_is_not_current_state(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"generation": "old", "paused": True}), encoding="utf8")
    os.utime(path, (1000, 1000))
    stale = heartbeat_summary(path, 1010)
    assert stale["state"] == "stale"
    assert stale["current_fields"] is None
    fresh = heartbeat_summary(path, 1001)
    assert fresh["state"] == "fresh"
    assert fresh["current_fields"]["generation"] == "old"


def test_missing_heartbeat_is_explicit(tmp_path):
    assert heartbeat_summary(tmp_path / "missing.json", 1000)["state"] == "unavailable"
