"""Bounded read-only recording from the Flybrain AC TCP telemetry app."""

import argparse
import json

from flybrain_drive.ac_tcp import TcpTelemetryClient, record_packet
from flybrain_drive.evidence import begin_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2348)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=2.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000_000:
        raise ValueError("A finite sample budget is required")
    run = begin_run(args.output, {"arguments": vars(args), "scope": "read-only raw telemetry",
                                 "actor_stream_created": False, "actuator_messages_sent": 0})
    count = 0
    try:
        with TcpTelemetryClient(args.host, args.port, args.timeout) as client:
            client.subscribe()
            write_json(run / "producer.json", json.loads(client.hello.wire_bytes))
            with (run / "telemetry.jsonl").open("x", encoding="utf8") as stream:
                for _ in range(args.samples):
                    stream.write(json.dumps(record_packet(client.receive())) + "\n")
                    count += 1
        write_json(run / "result.json", {"status": "completed", "samples": count,
                                         "goal_achieved": False, "actor_stream_created": False,
                                         "game_runtime_verified": False,
                                         "actuator_messages_sent": 0})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "samples": count,
                                         "error": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
