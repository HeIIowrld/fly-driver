"""Returned CSP camera pose/projection consistency, independent of any game reader."""

import re

import numpy as np


def matrix(text):
    values = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)
    if len(values) != 16:
        raise ValueError('Exact finite 4x4 camera matrix required')
    value = np.array(values, dtype=float).reshape(4, 4)
    if not np.isfinite(value).all():
        raise ValueError('Exact finite 4x4 camera matrix required')
    return value


def camera_geometry(views):
    if len(views) != 4 or {v['view_id'] for v in views} != {'front', 'left', 'right', 'rear'}:
        raise ValueError('Four distinct camera identities required')
    rows, group_origin = {}, None
    for record in views:
        v = matrix(record['view_matrix_diagnostic'])
        p = matrix(record['projection_matrix_diagnostic'])
        r = v[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=2e-5):
            raise ValueError('Camera rotation is not orthonormal')
        look = np.array(record['renderer_only_world_look'])
        up = np.array(record['renderer_only_world_up'])
        position = np.array(record['renderer_only_world_position'])
        origin_keys = ('renderer_origin_shift_before_diagnostic',
                       'renderer_origin_shift_after_diagnostic')
        marked = ('renderer_origin_contract_diagnostic' in record
                  or any(key in record for key in origin_keys))
        if marked:
            contract = 'getSim.originShift_before_after_v1'
            if record.get('renderer_origin_contract_diagnostic', contract) != contract:
                raise ValueError('Unknown renderer origin contract')
            before, after = (np.asarray(record.get(key), dtype=float) for key in origin_keys)
            if (before.shape != (3,) or after.shape != (3,)
                    or not np.isfinite(before).all() or not np.isfinite(after).all()
                    or not np.array_equal(before, after)):
                raise ValueError('Explicit stable finite same-update renderer origin required')
            origin = before
        else:
            # Legacy sources were checked in unshifted coordinates. They retain
            # the original strict test; never infer an offset from a failed pose.
            origin = np.zeros(3)
        if group_origin is None:
            group_origin = (marked, origin)
        elif group_origin[0] != marked or not np.array_equal(group_origin[1], origin):
            raise ValueError('All four cameras require the same recorded renderer origin')
        if (not np.allclose(-r[:, 2], look, atol=2e-5)
                or not np.allclose(r[:, 1], up, atol=2e-5)
                or not np.allclose(-v[3, :3] @ r.T, position + origin, atol=.002)):
            raise ValueError('Returned view matrix differs from requested camera pose')
        if (p[0, 0] <= 0 or p[1, 1] <= 0 or p[2, 3] != -1 or p[3, 3] != 0
                or not np.isclose(p[1, 1]/p[0, 0], record['width']/record['height'], atol=1e-5)):
            raise ValueError('Unexpected perspective projection or aspect ratio')
        vertical = float(np.degrees(2*np.arctan(1/p[1, 1])))
        horizontal = float(np.degrees(2*np.arctan(1/p[0, 0])))
        if not np.isclose(vertical, record['requested_fov_degrees'], atol=.001):
            raise ValueError('Requested FOV does not match returned vertical projection')
        rows[record['view_id']] = dict(vertical_fov_degrees=vertical,
            horizontal_fov_degrees=horizontal, look=look, screen_right=r[:, 0])
    forward, screen_right = rows['front']['look'], rows['front']['screen_right']
    if (not np.isclose(rows['left']['look'] @ screen_right, -1, atol=2e-5)
            or not np.isclose(rows['right']['look'] @ screen_right, 1, atol=2e-5)
            or not np.isclose(rows['rear']['look'] @ forward, -1, atol=2e-5)):
        raise ValueError('Left/right/rear camera labels disagree with front projection')
    return {key: {k: v for k, v in row.items() if k.endswith('_degrees')}
            for key, row in rows.items()}

