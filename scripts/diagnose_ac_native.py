"""Read-only existing-map discovery and optional bounded localhost socket diagnostics."""

import argparse
import json
import socket

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_probe import inspect_installation
from flybrain_drive.evidence import begin_run, write_json


def loopback(kind):
    server = socket.socket(socket.AF_INET, kind)
    client = socket.socket(socket.AF_INET, kind)
    accepted = None
    for stream in (server, client):
        stream.settimeout(1.5)
    try:
        server.bind(("127.0.0.1", 0))
        if kind == socket.SOCK_STREAM:
            server.listen(1)
        client.connect(server.getsockname())
        if kind == socket.SOCK_STREAM:
            accepted, _ = server.accept()
            accepted.settimeout(1.5)
        sent = client.send(b"flybrain-local-probe")
        receiver = accepted if accepted is not None else server
        value = receiver.recv(128)
        return {"status": "received", "sent": sent, "received": len(value),
                "fixture_only": True}
    except OSError as error:
        return {"status": "error", "error": type(error).__name__,
                "message": str(error), "fixture_only": True}
    finally:
        if accepted is not None:
            accepted.close()
        client.close()
        server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-loopback", action="store_true")
    args = parser.parse_args()
    run = begin_run(args.output, {"scope": "native presence and optional localhost fixture",
                                 "game_started": False, "actuator_messages_sent": 0})
    mappings = {}
    for name in ("Local\\acpmf_physics", "Local\\acpmf_graphics", "Local\\DualBufferImage"):
        try:
            with ExistingNamedMapping(name, 4) as mapping:
                mappings[name] = {"status": "exists_source_unverified",
                                  "header_hex": mapping.read().hex()}
        except FileNotFoundError:
            mappings[name] = {"status": "absent_not_created"}
        except OSError as error:
            mappings[name] = {"status": "error", "error": type(error).__name__}
    result = {"mappings": mappings, "installation": inspect_installation(),
              "game_runtime_verified": False, "actuator_messages_sent": 0}
    if args.check_loopback:
        result["localhost_fixtures"] = {"udp": loopback(socket.SOCK_DGRAM),
                                          "tcp": loopback(socket.SOCK_STREAM)}
    write_json(run / "native-probe.json", result)
    write_json(run / "result.json", {"status": "completed", "goal_achieved": False})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
