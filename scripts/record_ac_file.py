"""Bounded raw recording from AC's read-only local file publisher."""

import argparse
import json
import os
from pathlib import Path

from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.evidence import begin_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(Path(os.environ.get("LOCALAPPDATA", "~"))
                                                 / "flybrain" / "telemetry"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=2.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000_000:
        raise ValueError("A positive finite sample budget is required")
    run = begin_run(args.output, {"arguments": vars(args), "scope": "read-only raw file telemetry",
        "actor_stream_created": False, "actuator_messages_sent": 0,
        "dropped_intermediate_snapshots_possible": True})
    count = 0
    try:
        client = FileTelemetryClient(args.directory, timeout=args.timeout)
        write_json(run / "producer.json", json.loads(client.hello.wire_bytes))
        with (run / "telemetry.jsonl").open("x", encoding="utf8") as stream:
            for _ in range(args.samples):
                stream.write(json.dumps(record_packet(client.receive())) + "\n")
                count += 1
        write_json(run / "result.json", {"status": "completed", "samples": count,
            "transient_read_retries": client.transient_read_retries,
            "game_runtime_verified": False, "actor_stream_created": False,
            "actuator_messages_sent": 0, "goal_achieved": False})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "samples": count,
            "error": type(error).__name__, "message": str(error), "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
