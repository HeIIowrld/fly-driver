"""Rerun the frozen policy on the three original sensor timelines."""
import configparser
from pathlib import Path
import os
import subprocess
import time

import left_policy
from fixed_left import EVAL, old
from replay_independent import check_extraction, load_fixed_policy, validate_transfer
from scripts.capture_independent_corner import check_session

OUT = Path("runs/vision-corner-frozen-transfer-20261113-01").resolve()
receipt, frozen, *_ = load_fixed_policy()
check_extraction(frozen)
reference = receipt.with_name("comparison.json")
spline = validate_transfer("ks_highlands", "layout_short")
ini = Path(subprocess.check_output(
    ["powershell", "-NoProfile", "-Command",
     "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; [Environment]::GetFolderPath('MyDocuments')"],
    text=True, encoding="utf-8").strip()) / "Assetto Corsa/cfg/race.ini"
cfg = configparser.ConfigParser(interpolation=None)
cfg.read(ini, encoding="utf-8-sig")
status = Path(os.environ["LOCALAPPDATA"]) / "flybrain/csp-session/status.json"
session = dict(
    reserved_track="ks_highlands", reserved_layout="layout_short",
    configured_track=cfg.get("RACE", "TRACK", fallback=None),
    configured_layout=cfg.get("RACE", "CONFIG_TRACK", fallback=None),
    race_ini_sha256=old.digest(ini), frozen_sha256=old.digest(receipt),
    spline_sha256=old.digest(spline), heartbeat_age_s=time.time() - status.stat().st_mtime,
    lifecycle_requests_sent=0, ui_actions_sent=0, actuator_messages_sent=0,
)
protocol = OUT.parent / "vision-vallelunga-local-corner-protocol-20261112-01/capture.json"
try:
    _, window, _, readiness = check_session("ks_highlands", old.read(protocol)["capture"]["csp_sha256"])
    session.update(engine_pid=window.pid, readiness=readiness, ready=all(readiness.values()))
except (ValueError, OSError, TimeoutError) as exc:
    session.update(ready=False, reason=str(exc), error=type(exc).__name__)
old.write(OUT / "independent-preflight.json", session)
print("Independent preflight", session["ready"], session.get("reason", ""), flush=True)

replay = OUT / "three-circuit-replay"
replay.mkdir(exist_ok=False)
for item in (receipt, receipt.with_name("policy.joblib")):
    (replay / item.name).write_bytes(item.read_bytes())
left_policy.OUT = replay
left_policy.compare()
actual = old.read(replay / "comparison.json")
expected = old.read(reference)
assert actual == expected, "Frozen comparison changed"
for track, _ in EVAL:
    filename = f"{track}-forecasts.jsonl"
    assert old.digest(replay / filename) == old.digest(receipt.with_name(filename))
old.write(OUT / "reproduction.json", dict(
    frozen_sha256=old.digest(receipt), reference_comparison_sha256=old.digest(reference),
    new_comparison_sha256=old.digest(replay / "comparison.json"),
    three_forecast_digests_identical=True, tracks=[track for track, _ in EVAL],
    preflight_sha256=old.digest(OUT / "independent-preflight.json"),
    source_sha256=old.digest(OUT / "rerun.py"), source_revision=frozen["source_revision"],
    seed=frozen["seed"], termination="frozen_sensor_replay_complete"))
print("Identical metrics and forecast digests across all three recordings", flush=True)
