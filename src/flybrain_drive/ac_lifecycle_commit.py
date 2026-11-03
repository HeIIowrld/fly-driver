"""Fresh deadline for a never-published lifecycle intent; issued IDs are never renewed."""

import math

CONTEXT_KEYS = ('schema', 'generation', 'epoch', 'window_handle', 'car_id', 'track_id',
                'is_online', 'paused', 'live', 'in_menu', 'ai_controlled',
                'expert_probe_active', 'camera_active', 'ideal_line_shown', 'physics_api_allowed')


def fresh_unpublished_request(before, fresh, request, *, publication_started):
    if publication_started:
        raise ValueError('Never renew or republish an issued lifecycle ID')
    if (request.get('schema') != 'flybrain.csp.session.request.v1'
            or fresh.get('schema') != 'flybrain.csp.session.status.v1'
            or request.get('generation') != before.get('generation')
            or request.get('epoch') != before.get('epoch')
            or any(before.get(key) != fresh.get(key) for key in CONTEXT_KEYS)):
        raise ValueError('Lifecycle context changed before initial publication')
    if (fresh.get('error') or fresh.get('pending_receipt_id')
            or fresh.get('is_online') is not False or fresh.get('ai_controlled') is not False
            or fresh.get('expert_probe_active') is not False):
        raise ValueError('Same idle offline lifecycle owner required')
    values = [fresh.get(key) for key in ('speed_kmh', 'gas', 'brake', 'script_clock')]
    if (not all(type(value) in (int, float) and math.isfinite(value) for value in values)
            or abs(values[0]) >= .1 or not 0 <= values[1] < .02
            or not .98 < values[2] <= 1 or values[3] < before['script_clock']):
        raise ValueError('Fresh stationary full-brake context and monotonic script clock required')
    return dict(request, expires_clock=values[3]+3)
