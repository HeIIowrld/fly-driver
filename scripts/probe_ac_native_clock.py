"""Measure native AC timing progression; no actor clock or input is created."""

import argparse
import json
import time

from flybrain_drive.ac_native_clock import NativeGraphicsTimingReader
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 2 <= args.samples <= 500:
        raise ValueError('Bounded sample count required')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__))
    rows = []
    try:
        with NativeGraphicsTimingReader() as graphics, NativePhysicsReader(timeout=2) as physics:
            with (run / 'samples.jsonl').open('x', encoding='utf8') as stream:
                for _ in range(args.samples):
                    before = graphics.snapshot()
                    body = physics.receive()
                    after = graphics.snapshot()
                    row = dict(graphics_before=before.metadata(),
                               physics=physics_record(body), graphics_after=after.metadata())
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    rows.append(row)
                    time.sleep(.1)
        times = [r['graphics_after']['current_time_raw'] for r in rows]
        write_json(run / 'result.json', dict(
            status='completed', samples=len(rows),
            graphics_statuses=sorted({r['graphics_after']['status'] for r in rows}),
            distinct_graphics_packets=len({r['graphics_after']['packet_id'] for r in rows}),
            distinct_physics_packets=len({r['physics']['packet_id'] for r in rows}),
            current_time_raw_first=times[0], current_time_raw_last=times[-1],
            current_time_raw_nondecreasing=all(
                a <= b for a, b in zip(times, times[1:], strict=False)),
            raw_time_units_verified=False, lap_transition_verified=False,
            simulation_clock_verified=False, actor_stream_created=False,
            ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', samples=len(rows),
            error=type(error).__name__, message=str(error), goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
