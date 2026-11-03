"""Record bounded native AC physics without plugins, controls or networking."""

import argparse
import base64
import hashlib
import json
import time

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import (
    PHYSICS_PREFIX,
    NativePhysicsReader,
    decode_static_identity_prefix,
)
from flybrain_drive.evidence import begin_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--period", type=float, default=.02)
    parser.add_argument("--timeout", type=float, default=2.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000_000 or not .001 <= args.period <= 10:
        raise ValueError("Finite native sample budget and recording period required")
    run = begin_run(args.output, {"arguments": vars(args),
        "physics_prefix_bytes": PHYSICS_PREFIX.size,
        "access": "OpenFileMapping FILE_MAP_READ; never create or write mappings",
        "actuator_messages_sent": 0, "native_units_and_axes_calibrated": False})
    count, first_id, last_id = 0, None, None
    try:
        with ExistingNamedMapping("Local\\acpmf_static", 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / "identity.json", identity)
        with NativePhysicsReader(timeout=args.timeout) as reader:
            with (run / "physics.jsonl").open("x", encoding="utf8") as stream:
                for _ in range(args.samples):
                    sample = reader.receive()
                    row = {"packet_id": sample.packet_id,
                        "received_monotonic_seconds": sample.received_monotonic_seconds,
                        "fields": dict(sample.fields),
                        "prefix_base64": base64.b64encode(sample.prefix_bytes).decode(),
                        "prefix_sha256": hashlib.sha256(sample.prefix_bytes).hexdigest(),
                        "source_time_seconds": None, "simulation_clock_verified": False,
                        "repeated_copies_matched": True, "atomicity_verified": False}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    first_id = sample.packet_id if first_id is None else first_id
                    last_id, count = sample.packet_id, count + 1
                    time.sleep(args.period)
        write_json(run / "result.json", {"status": "completed", "samples": count,
            "first_packet_id": first_id, "last_packet_id": last_id,
            "identity": identity, "actual_motion_response_verified": False,
            "native_units_and_axes_calibrated": False, "image_synchronization_verified": False,
            "actor_stream_created": False, "actuator_messages_sent": 0, "goal_achieved": False})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "samples": count,
            "error": type(error).__name__, "message": str(error), "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
