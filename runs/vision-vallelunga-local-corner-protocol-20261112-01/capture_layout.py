"""Record the requested layout before and after the existing CSP capture workflow."""

import argparse
from pathlib import Path

from scripts.capture_independent_corner import run
from evaluate import OUT, TRACK, verify_layout, verify
from train_vision_teacher_line import digest, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-ini",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    verify(read(OUT/"lock.json")["hashes"])
    args.protocol = OUT/"capture.json"
    verify_layout(args.race_ini)
    race_ini = args.race_ini.read_bytes()
    before = digest(args.race_ini)
    run(args)
    receipt = args.output/"result.json"
    result = read(receipt)
    if result["track"] != TRACK:
        raise ValueError("Capture track differs from the frozen route")
    if args.race_ini.read_bytes() != race_ini:
        raise ValueError("Race configuration changed during the capture")
    result.update(race_ini_sha256=before,layout_source="unchanged_active_race_ini",
                  capture_wrapper_sha256=digest(Path(__file__)))
    write(receipt,result)


if __name__ == "__main__":
    main()
