"""Compare original host-receipt availability before each view's render bound."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(root):
    raw = (root / 'events.jsonl').read_bytes()
    rows = [r for r in map(json.loads, raw.splitlines()) if r['kind'] == 'native']
    clocks = [float(r['native']['received_monotonic_seconds']) for r in rows]
    if any(b <= a for a, b in zip(clocks, clocks[1:])):
        raise ValueError('Original native receipts must advance')
    no_prior = aged = intrinsically_old = eligible = 0
    camera_groups = set()
    intrinsic = []
    unique_views = unique_missing = 0
    for i, row in enumerate(rows):
        camera = row.get('camera')
        if camera is None:
            continue
        first_capture = camera['capture_id'] not in camera_groups
        camera_groups.add(camera['capture_id'])
        for view in camera['views']:
            lower = float(view['render_started_lower'])
            receipt = float(camera['received'])
            if not lower <= receipt <= clocks[i]:
                raise ValueError('Causal original view bounds required')
            intrinsic.append(1000 * (receipt - lower))
            intrinsically_old += receipt - lower > .05
            j = bisect.bisect_left(clocks, lower, 0, i) - 1
            if j < 0:
                no_prior += 1
            elif receipt - clocks[j] > .05:
                aged += 1
            else:
                eligible += 1
            if first_capture:
                unique_views += 1
                unique_missing += j < 0 or receipt - clocks[j] > .05
    total = no_prior + aged + eligible
    intervals = np.diff(clocks) * 1000
    return dict(events_sha256=hashlib.sha256(raw).hexdigest(), native_packets=len(rows),
                distinct_camera_groups=len(camera_groups), view_decisions=total,
                no_prior=no_prior, previous_body_older_than_50ms=aged, eligible=eligible,
                missing_fraction=(no_prior + aged) / total,
                unique_camera_views=unique_views, unique_camera_views_missing=unique_missing,
                unique_camera_view_missing_fraction=unique_missing / unique_views,
                view_render_lower_to_receipt_above_50ms=intrinsically_old,
                native_interval_median_ms=float(np.median(intervals)),
                native_interval_p95_ms=float(np.percentile(intervals, 95)),
                view_lower_to_receipt_median_ms=float(np.median(intrinsic)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(dict(baseline=analyze(args.baseline), candidate=analyze(args.candidate),
                          scope='host receipts only, temporally correlated observations, not physical synchronization'),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
