"""Bounded fixed-speed sensory BC/DAgger; no PPO or Assetto Corsa control."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from flybrain_drive.sensory_imitation import train_sensory

    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    print(train_sensory(config, args.output))


if __name__ == "__main__":
    main()
