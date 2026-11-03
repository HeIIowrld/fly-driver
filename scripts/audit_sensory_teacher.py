"""Check the privileged teacher on the sensory steering pilot's evaluation cases."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", help="Evaluation-only scenario JSON")
    parser.add_argument("--no-render", action="store_true",
                        help="Oracle teacher only: skip images using O observations; same physics")
    args = parser.parse_args()
    from flybrain_drive.sensory_audit import evaluate_teacher
    from scripts.evaluate_routed_controller import read_suite

    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    print(json.dumps(evaluate_teacher(config, read_suite(config, args.suite), args.output,
                                      render=not args.no_render)))


if __name__ == "__main__":
    main()
