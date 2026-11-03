"""Fit a teacher-only steering/curvature relation from actual AC motion, not fly physiology."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--steering-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source_run)
    events = source / 'observations-and-requests.jsonl'
    config = json.loads((source / 'manifest.json').read_text(encoding='utf8'))['config']
    result = json.loads((source / 'result.json').read_text(encoding='utf8'))
    audit = json.loads(Path(args.steering_audit).read_text(encoding='utf8'))
    if result['status'] != 'segment_completed':
        raise ValueError('Completed actual motion run required')
    gain = audit['native_steering_calibration_gain'] * config['arguments']['steering_request_scale']
    xy, yaw, steering, speeds = [], [], [], []
    for row in (json.loads(s) for s in events.read_text(encoding='utf8').splitlines()):
        if row['kind'] != 'raw_pair':
            continue
        for key, tag in [('sdk', 'wire'), ('native', 'prefix')]:
            raw = base64.b64decode(row[key][tag+'_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != row[key][tag+'_sha256']:
                raise ValueError('Actual motion raw hash differs')
        state = json.loads(base64.b64decode(row['sdk']['wire_base64']))['privileged_teacher']['raw']
        xy.append([state['world_position_x'], state['world_position_y']])
        values = PHYSICS_PREFIX.unpack(base64.b64decode(row['native']['prefix_base64']))
        offset = 0
        for name, _, count in PHYSICS_LAYOUT:
            if name == 'heading':
                yaw.append(-values[offset])
            offset += count
        steering.append(row['native']['fields']['steerAngle']/gain)
        speeds.append(row['native']['fields']['speedKmh'])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    angles = np.unwrap(yaw)
    ratios, accepted_speeds = [], []
    for index in range(5, len(xy)-5):
        distance = arc[index+5]-arc[index-5]
        if distance > 2 and abs(steering[index]) > .07:
            ratio = (angles[index+5]-angles[index-5])/distance/steering[index]
            if np.isfinite(ratio) and ratio < 0:
                ratios.append(ratio)
                accepted_speeds.append(speeds[index])
    if len(ratios) < 50:
        raise ValueError('Insufficient moving samples for teacher curvature fit')
    quantiles = np.percentile(ratios, [10, 25, 50, 75, 90])
    if not -.5 <= quantiles[2] <= -.005:
        raise ValueError('Unexpected empirical steering direction or scale')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=861701,
        source_events_sha256=sha256(events), steering_audit_sha256=sha256(args.steering_audit),
        method='native yaw change / SDK planar path; +/-5 samples, abs steer>.07, path>2m',
        source_configuration=config))
    write_json(run / 'result.json', dict(status='completed', accepted_samples=len(ratios),
        curvature_per_canonical_steering=float(quantiles[2]),
        ratio_quantiles_10_25_50_75_90=quantiles.tolist(),
        observed_speed_kmh_range=[min(accepted_speeds), max(accepted_speeds)],
        profile_sha256=config['axis_configuration']['profile_sha256'],
        producer_session_id=json.loads((source / 'preflight.json').read_text(encoding='utf8'))[
            'producer_session_id'],
        teacher_only=True, low_speed_extrapolation_verified=False,
        physical_roadwheel_units_verified=False, actor_input=False, goal_achieved=False))


if __name__ == '__main__':
    main()
