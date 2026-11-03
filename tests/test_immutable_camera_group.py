import hashlib

import numpy as np
import pytest

from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.immutable_camera_group import freeze_camera_group, require_matching_camera_geometry


def group():
    pixels = np.zeros((180, 320, 3), dtype=np.uint8)
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    views = [{'view_id': name, 'rgb_sha256': digest, 'render_started_lower': 1.0}
             for name in VIEW_ORDER]
    candidate = {'source_kind': 'csp_four_camera', 'ideal_line_shown': False,
                 'aid_probe_only': False, 'generation': 'g', 'capture_id': 3,
                 'received': 1.02, 'views': views}
    samples = {name: {'rgb': pixels.copy(), 'capture_id': 3, 'generation': 'g',
                      'received': 1.02, 'render_started_lower': 1.0} for name in VIEW_ORDER}
    return candidate, samples


def test_frozen_pixels_are_detached_and_immutable():
    candidate, samples = group()
    result = freeze_camera_group(candidate, samples, selected_at=1.04)
    samples['front']['rgb'][0, 0, 0] = 1
    assert result['pixels']['front'][0, 0, 0] == 0
    with pytest.raises(ValueError):
        result['pixels']['front'].setflags(write=True)


def test_stale_or_mismatched_original_is_rejected():
    candidate, samples = group()
    with pytest.raises(ValueError, match='age'):
        freeze_camera_group(candidate, samples, selected_at=1.16)
    samples['front']['rgb'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='hash'):
        freeze_camera_group(candidate, samples, selected_at=1.04)
    candidate, samples = group()
    samples['left']['capture_id'] = 4
    with pytest.raises(ValueError, match='identity'):
        freeze_camera_group(candidate, samples, selected_at=1.04)
    del samples['rear']
    with pytest.raises(ValueError, match='four views'):
        freeze_camera_group(candidate, samples, selected_at=1.04)


def test_different_or_missing_mount_does_not_pass_geometry_gate():
    base = {'view_order': list(VIEW_ORDER), 'views': {v: {} for v in VIEW_ORDER}}
    require_matching_camera_geometry(base, dict(base))
    different_mount = dict(base, camera_mount_contract={'layout_id': 'exterior_front_v3'})
    with pytest.raises(ValueError, match='camera_mount_contract'):
        require_matching_camera_geometry(different_mount, base)
    different_origins = dict(base, views={v: {'origin_body_m': [1, 0, 1]} for v in VIEW_ORDER})
    with pytest.raises(ValueError, match='views'):
        require_matching_camera_geometry(different_origins, base)


def test_nonzero_view_guard_rejects_black_without_claiming_valid_scene():
    from flybrain_drive.immutable_camera_group import require_nonzero_camera_views
    pixels = {name: np.ones((180, 320, 3), dtype=np.uint8) for name in ('front', 'left', 'right', 'rear')}
    require_nonzero_camera_views({'pixels': pixels})
    pixels['rear'].fill(0)
    with pytest.raises(ValueError, match='black camera view: rear'):
        require_nonzero_camera_views({'pixels': pixels})
