"""Record a synthetic SDK-free publisher through actual localhost TCP, never the game."""

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = begin_run(args.output, {"scope": "synthetic TCP interoperability fixture",
                                 "game_runtime_verified": False, "actuator_messages_sent": 0})
    app_path = (Path(__file__).resolve().parents[1] / "integrations" / "ac"
                / "flybrain_telemetry" / "flybrain_telemetry.py")
    spec = importlib.util.spec_from_file_location("flybrain_tcp_fixture", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    producer = app.Publisher(port=0)
    stopped = threading.Event()

    def publish():
        while not stopped.is_set():
            producer.publish({"fixture_kind": "synthetic_no_game_sdk",
                              "angular_velocity_x": 1., "angular_velocity_y": 2.,
                              "angular_velocity_z": 3., "simulation_clock_verified": False})
            stopped.wait(.02)

    worker = threading.Thread(target=publish)
    worker.start()
    try:
        command = [sys.executable, "-m", "scripts.record_ac_tcp", "--port", str(producer.port),
                   "--samples", "20", "--output", str(run / "recording")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        (run / "recorder.log").write_text(completed.stdout + completed.stderr, encoding="utf8")
        if completed.returncode:
            raise RuntimeError("Synthetic TCP recorder failed; see recorder.log")
        result = json.loads((run / "recording" / "result.json").read_text(encoding="utf8"))
        if result["samples"] != 20:
            raise ValueError("Unexpected fixture sample count")
        write_json(run / "result.json", {"status": "completed", "samples": 20,
            "fixture_kind": "synthetic_no_game_sdk", "game_runtime_verified": False,
            "goal_achieved": False, "actuator_messages_sent": 0})
    finally:
        stopped.set()
        worker.join(2)
        producer.close()


if __name__ == "__main__":
    main()
