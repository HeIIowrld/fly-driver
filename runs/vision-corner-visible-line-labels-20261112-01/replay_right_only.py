"""Hold left-edge extraction fixed while testing the right-edge contrast change."""
from pathlib import Path

import replay_signed as variant

absolute = variant.supported.directional_evidence
signed = variant.signed_evidence


def right_only(image, side):
    return signed(image, side) if side == 1 else absolute(image, side)


if __name__ == "__main__":
    variant.check_polarity()
    variant.OUT = Path(__file__).resolve().parent / "right-only"
    variant.OUT.mkdir(exist_ok=True)
    variant.check_polarity = lambda: None
    variant.signed_evidence = right_only
    variant.main()
    path = variant.OUT / "result.json"
    result = variant.replay.read(path)
    result["source_sha256"][str(Path(__file__).resolve())] = variant.replay.digest(Path(__file__))
    result["termination"] = "right_only_signed_contrast_and_fixed_control_replays_complete"
    variant.replay.write(path, result)
