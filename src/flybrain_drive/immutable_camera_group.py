"""Copy one verified shared-camera candidate into an immutable four-view snapshot."""

import hashlib
import math

import numpy as np

from .flyvis_camera import VIEW_ORDER


def require_matching_camera_geometry(source, expected):
    """Require the full reviewed mounting, orientation and projection contract."""
    if not isinstance(source, dict) or not isinstance(expected, dict):
        raise ValueError('Explicit camera geometry contracts required')
    for key in ('view_order', 'views', 'camera_mount_contract'):
        if source.get(key) != expected.get(key):
            raise ValueError('Camera geometry contract differs: ' + key)
    if (source.get('view_order') != list(VIEW_ORDER)
            or set(source.get('views', {})) != set(VIEW_ORDER)):
        raise ValueError('Exactly four ordered camera geometries required')


def freeze_camera_group(candidate, samples, *, selected_at, maximum_age=.15):
    """Keep source identity and original receipt/lower bounds; never repair missing views."""
    if (not isinstance(selected_at, (int, float)) or isinstance(selected_at, bool)
            or not math.isfinite(selected_at) or not 0 < maximum_age <= .15
            or candidate.get('source_kind') != 'csp_four_camera'
            or candidate.get('ideal_line_shown') is not False
            or candidate.get('aid_probe_only') is not False
            or not isinstance(candidate.get('generation'), str)
            or type(candidate.get('capture_id')) is not int or candidate['capture_id'] < 0
            or not isinstance(candidate.get('received'), (int, float))
            or not math.isfinite(candidate['received'])
            or candidate['received'] > selected_at):
        raise ValueError('Admissible original camera identity and receipt required')
    views = candidate.get('views', [])
    if ([v.get('view_id') for v in views] != list(VIEW_ORDER)
            or set(samples) != set(VIEW_ORDER)):
        raise ValueError('Exact ordered original four views required')
    frozen, metadata = {}, []
    for view in views:
        name = view['view_id']
        sample = samples[name]
        lower = view.get('render_started_lower')
        if (sample.get('capture_id') != candidate['capture_id']
                or sample.get('generation') != candidate['generation']
                or sample.get('received') != candidate['received']
                or sample.get('render_started_lower') != lower
                or not isinstance(lower, (int, float)) or not math.isfinite(lower)
                or lower > candidate['received']
                or not 0 <= selected_at - lower <= maximum_age):
            raise ValueError('Original view identity or conservative age differs')
        rgb = sample.get('rgb')
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (180, 320, 3):
            raise ValueError('Exact original RGB layout required')
        raw = rgb.tobytes(order='C')
        digest = hashlib.sha256(raw).hexdigest()
        if digest != view.get('rgb_sha256'):
            raise ValueError('Original RGB hash differs')
        # bytes owns storage; consumers cannot re-enable writes or mutate producer memory.
        frozen[name] = np.frombuffer(raw, dtype=np.uint8).reshape(180, 320, 3)
        metadata.append({'view_id': name, 'rgb_sha256': digest,
                         'render_started_lower': lower,
                         'age_upper_at_selection_seconds': selected_at-lower})
    return {'capture_id': candidate['capture_id'], 'generation': candidate['generation'],
            'received': candidate['received'], 'selected_at': float(selected_at),
            'views': metadata, 'pixels': frozen, 'physical_exposure_time_verified': False}


def require_nonzero_camera_views(group):
    """Conservatively reject an entirely black view; this does not certify image validity."""
    pixels = group.get('pixels') if isinstance(group, dict) else None
    if not isinstance(pixels, dict) or set(pixels) != set(VIEW_ORDER):
        raise ValueError('Four immutable views required for blank-frame check')
    for name in VIEW_ORDER:
        rgb = pixels[name]
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (180, 320, 3):
            raise ValueError('Exact original RGB layout required')
        if not np.any(rgb):
            raise ValueError('Entirely black camera view: ' + name)
