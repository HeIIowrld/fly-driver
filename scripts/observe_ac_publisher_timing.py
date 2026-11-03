"""Bounded read-only host-stage observation; retain packets in RAM until observation ends."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=75.)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if not 0 < a.seconds <= 90:
        raise ValueError('Observation must be bounded to90seconds')
    run = begin_run(a.output, dict(arguments=vars(a), source_sha256=sha256(__file__),
        read_only=True, actor_input=False, hot_loop_disk_writes=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA'])/'flybrain/telemetry',
        timeout=.02, clock=time.perf_counter)
    rows, errors, maxima = [], [], {}
    started = time.perf_counter()
    previous = None
    while time.perf_counter()-started < a.seconds:
        try:
            packet = client.snapshot()
            if packet.sequence != previous:
                if len(rows) >= 5000:
                    raise RuntimeError('Bounded packet budget exceeded')
                rows.append(record_packet(packet))
                previous = packet.sequence
                for group in ('publisher_update_timing', 'publisher_previous_io_timing'):
                    for stage, values in packet.raw.get(group, {}).get('stages', {}).items():
                        maxima[group+'/'+stage] = dict(values['maximum'])
        except (OSError, ValueError, TimeoutError) as error:
            if len(errors) < 1000:
                errors.append(dict(received_monotonic_seconds=time.perf_counter(),
                                   error=type(error).__name__, reason=str(error)))
        time.sleep(.01)
    ended = time.perf_counter()
    with (run/'packets.jsonl').open('w', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    write_json(run/'result.json', dict(status='completed_read_only_timing_observation',
        started_host_seconds=started, ended_host_seconds=ended, packets=len(rows),
        producer_session=client.session_id, errors=errors, latched_maxima=maxima,
        packets_sha256=sha256(run/'packets.jsonl'), disk_persistence_only_after_observation=True,
        actor_input=False, source_times_not_cross_clock_aligned=True,
        student_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
