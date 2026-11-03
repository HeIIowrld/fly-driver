"""Read-only50-sample diagnostic transport receipt; no game or actuator mutation."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.evidence import begin_run, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run = begin_run(a.output, dict(scope=__doc__, actuator_messages_sent=0))
    rows = []
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0)
    try:
        client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry',
                                     timeout=.15, clock=time.perf_counter)
        for index in range(50):
            started = time.perf_counter()
            packet = client.snapshot()
            finished = time.perf_counter()
            rows.append(dict(index=index, started=started, finished=finished,
                original_age=finished-packet.received_monotonic_seconds,
                publisher_errors=packet.raw['publisher_errors'], packet=record_packet(packet)))
            time.sleep(.02)
        result.update(status='completed_readonly_diagnostic_probe', samples=len(rows),
            maximum_original_age=max(r['original_age'] for r in rows),
            maximum_snapshot_seconds=max(r['finished']-r['started'] for r in rows),
            stale_snapshot_retries=client.stale_snapshot_retries,
            transient_read_retries=client.transient_read_retries,
            publisher_error_count_change=rows[-1]['publisher_errors']-rows[0]['publisher_errors'])
    except Exception as error:
        result.update(reason=str(error), error=type(error).__name__, samples=len(rows))
        write_json(run/'result.json', result)
        raise
    finally:
        write_json(run/'samples.json', rows)
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
