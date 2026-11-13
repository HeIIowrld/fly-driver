"""Capture a reserved Highlands Short four-camera archive using the existing CSP recorder."""
import argparse
import configparser
from pathlib import Path
import sys

from replay_independent import AC_TRACKS, OUT, load_fixed_policy, validate_transfer
from fixed_left import digests, old
from scripts.capture_independent_corner import run as capture

TRACK = "ks_highlands"
LAYOUT = "layout_short"
PROTOCOL = OUT.parent / "vision-vallelunga-local-corner-protocol-20261112-01/capture.json"


def verify_layout(path):
    data = path.read_bytes()
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    cfg.read_string(data.decode("utf-8-sig"))
    expected = {("RACE", "TRACK"): TRACK, ("RACE", "CONFIG_TRACK"): LAYOUT,
                ("RACE", "MODEL"): "bmw_z4_gt3", ("REMOTE", "ACTIVE"): "0", ("REPLAY", "ACTIVE"): "0"}
    if any(cfg.get(section, key, fallback=None) != value for (section, key), value in expected.items()):
        raise ValueError("Expected active offline BMW Z4 GT3 Highlands Short configuration")
    return data


def prepare(destination, observer_receipt):
    acquisition = old.read(PROTOCOL)
    warning, fusion, base = (destination / f"{name}.json" for name in ("warning", "fusion", "base"))
    reference = old.read(Path(acquisition["warning_end_protocol"]))
    reference_fusion = old.read(Path(reference["fusion_protocol"]))
    camera_config = Path(reference_fusion["camera_config"])
    program = Path(capture.__code__.co_filename)
    old.write(base, dict(proposed_track=TRACK, hashes=digests((validate_transfer(TRACK, LAYOUT),))))
    old.write(fusion, dict(base_protocol=str(base), development_rehearsal=False, camera_config=str(camera_config),
                           hashes=digests((base, camera_config))))
    old.write(warning, dict(fusion_protocol=str(fusion), hashes=digests((fusion,))))
    protocol = destination / "capture.json"
    old.write(protocol, dict(warning_end_protocol=str(warning), capture=acquisition["capture"],
                             hashes=digests((warning, PROTOCOL, program, Path(__file__), observer_receipt))))
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-ini", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.race_ini = args.race_ini.resolve()
    destination = args.output.resolve()
    observer_receipt = load_fixed_policy()[0]
    initial = verify_layout(args.race_ini)
    destination.mkdir(parents=True, exist_ok=False)
    args.protocol = prepare(destination, observer_receipt)
    args.output = destination / "capture"
    capture(args)
    receipt = args.output / "result.json"
    outcome = old.read(receipt)
    if verify_layout(args.race_ini) != initial:
        raise ValueError("Active race configuration changed during capture")
    outcome.update(layout=LAYOUT, race_ini_sha256=old.digest(args.race_ini),
                   layout_source="unchanged_active_race_ini")
    archives = args.output / "archive-map.json"
    if outcome["status"] == "captured":
        outcome["archives_sha256"] = old.digest(archives)
    old.write(receipt, outcome)


if __name__ == "__main__":
    main()
