"""Audit raw AC window/physics/graphics records; never bless them as actor samples."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_clock import decode_graphics_timing
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import write_json


def checked_prefix(row):
    raw = base64.b64decode(row['prefix_base64'], validate=True)
    if hashlib.sha256(raw).hexdigest() != row['prefix_sha256']:
        raise ValueError('Native prefix hash differs')
    return raw


def audit(run):
    run = Path(run).resolve()
    if json.loads((run / 'result.json').read_text())['status'] != 'completed':
        raise ValueError('Completed acquisition required; partial records remain separate')
    rows = [json.loads(line) for line in (run / 'captures.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('No actual image records')
    previous_physics, previous_graphics, previous_capture = -1, -1, -1
    temperatures, loads, brackets, statuses = [], [], [], set()
    for row in rows:
        file = (run / row['file']).resolve()
        if not file.is_relative_to(run):
            raise ValueError('Image path escapes recording')
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['file_sha256']:
            raise ValueError('Image file hash differs')
        rgb = load_array(file, allow_pickle=False)
        if (rgb.dtype != np.uint8
                or rgb.shape != (row['pixel_height'], row['pixel_width'], 3)
                or hashlib.sha256(rgb.tobytes()).hexdigest() != row['rgb_sha256']):
            raise ValueError('Image content differs')
        if row['capture_timespan_native'] <= previous_capture:
            raise ValueError('Capture timestamp reused/reset')
        previous_capture = row['capture_timespan_native']
        for name in ('physics_before', 'physics_after'):
            recorded = row[name]
            sample = decode_physics_prefix(checked_prefix(recorded),
                                          recorded['received_monotonic_seconds'])
            # JSON normalizes tuple/list, but never permits extra pose or route fields.
            decoded = json.loads(json.dumps(dict(sample.fields)))
            if sample.packet_id != recorded['packet_id'] or decoded != recorded['fields']:
                raise ValueError('Decoded physics differs from recorded fields')
            if sample.packet_id <= previous_physics:
                raise ValueError('Physics packet reused/reset')
            previous_physics = sample.packet_id
            temperatures.append(sample.fields['tyreCoreTemperature'])
            loads.append(sample.fields['wheelLoad'])
        for name in ('graphics_before', 'graphics_after'):
            recorded = row[name]
            sample = decode_graphics_timing(checked_prefix(recorded),
                                           recorded['received_monotonic_seconds'])
            if sample.metadata() != recorded or sample.packet_id < previous_graphics:
                raise ValueError('Graphics fields differ or sequence reset')
            previous_graphics = sample.packet_id
            statuses.add(sample.status)
        ordered = [row['graphics_before']['received_monotonic_seconds'],
                   row['physics_before']['received_monotonic_seconds'],
                   row['started_monotonic_seconds'], row['finished_monotonic_seconds'],
                   row['physics_after']['received_monotonic_seconds'],
                   row['graphics_after']['received_monotonic_seconds']]
        if not np.isfinite(ordered).all() or any(
                a > b for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError('Host receipt order differs')
        brackets.append(ordered[-1] - ordered[0])
    return dict(
        scope=__doc__, frames=len(rows), raw_native_prefixes_verified=4 * len(rows),
        unique_rgb_hashes=len({r['rgb_sha256'] for r in rows}),
        foreground_frames=sum(r['was_foreground_at_receipt'] for r in rows),
        graphics_statuses=sorted(statuses),
        host_group_span_seconds=dict(median=float(np.median(brackets)), max=max(brackets)),
        raw_tyre_temperature_min=np.min(temperatures, axis=0).tolist(),
        raw_tyre_temperature_max=np.max(temperatures, axis=0).tolist(),
        raw_wheel_load_min=np.min(loads, axis=0).tolist(),
        raw_wheel_load_max=np.max(loads, axis=0).tolist(),
        source_time_units_axes_verified=False, exposure_bounds_verified=False,
        actor_stream_created=False, training_admissible=False, goal_achieved=False,
        note='Host group span includes acquisition wait; not exposure error or latency bound')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Audit output already exists')
    write_json(args.output, audit(args.run))


if __name__ == '__main__':
    main()
