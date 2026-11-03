"""Prospective geographic teacher roles; diagnostic spline never becomes actor input."""

import math


def validate_regions(regions):
    previous = 0.
    for region in regions:
        lo, hi = region['start'], region['end']
        if (region['role'] not in ('train', 'development')
                or not all(math.isfinite(v) for v in (lo, hi))
                or not previous <= lo < hi < 1):
            raise ValueError('Ordered disjoint geographic train/development regions required')
        previous = hi
    if {r['role'] for r in regions} != {'train', 'development'}:
        raise ValueError('Both geographic train and development roles required')


def next_lease(spline, regions, *, maximum_advance=.13):
    validate_regions(regions)
    if not math.isfinite(spline) or not 0 <= spline < 1 or maximum_advance != .13:
        raise ValueError('Reviewed spline and declared planning advance estimate required')
    for region in regions:
        if region['start'] <= spline and spline+maximum_advance <= region['end']:
            return dict(role=region['role'], duration_seconds=30,
                        expected_start_spline=spline, region=dict(region))
    # A30s step can jump the entire admissible start window of a narrow region.
    # Compare the approach plus that window with the same planning estimate;
    # whole-history route admission still checks actual positions afterwards.
    short = any(0 < (r['start']-spline) % 1
                and (r['start']-spline) % 1 + r['end']-r['start']-maximum_advance
                <= maximum_advance for r in regions)
    return dict(role='transit', duration_seconds=3 if short else 30,
                expected_start_spline=spline, region=None)


def validate_lease_route(lease, receipt):
    before, after = receipt['before']['diagnostic_route'], receipt['after']['diagnostic_route']
    samples = receipt['telemetry']
    routes = [before, *(s['diagnostic_route'] for s in samples), after]
    if (not samples or any(r['actor_input'] is not False for r in routes)
            or abs(before['spline_position']-lease['expected_start_spline']) > 1e-4):
        raise ValueError('Original prospective start/diagnostic-only route differs')
    positions = [r['spline_position'] for r in routes]
    if any(not math.isfinite(p) or not 0 <= p < 1 for p in positions):
        raise ValueError('Finite original route positions required')
    if lease['role'] == 'transit':
        return dict(role='transit', fit_allowed=False, entire_history_retained=True)
    lo, hi = lease['region']['start'], lease['region']['end']
    if (any(not lo <= p <= hi for p in positions)
            or any(a-b > .001 for a, b in zip(positions[:-1], positions[1:], strict=True))
            or any(s['wheels_outside'] != 0 for s in samples)
            or len({r['lap_count'] for r in routes}) != 1):
        raise ValueError('Whole lease left its declared geographic/on-track region; no slicing')
    return dict(role=lease['role'], fit_allowed=False, entire_history_retained=True,
                minimum_spline=min(positions), maximum_spline=max(positions),
                labels_and_raw_replay_required=True)
