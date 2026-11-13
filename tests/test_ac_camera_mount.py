import copy

import numpy as np
import pytest

from flybrain_drive.ac_camera_mount import (
    EXTERIOR_FRONT_CONTRACT,
    EXTERIOR_SIDE_120_CONTRACT,
    configured_mount,
    validate_frame_mount,
)


def fixture():
    rows = []
    for name, position, look in [
        ('front', [0, 1, -2.5], [0, 0, -1]),
        ('left', [-1.15, 1, 0], [-1, 0, 0]),
        ('right', [1.15, 1, 0], [1, 0, 0]),
        ('rear', [0, 1, 2.4], [0, 0, 1]),
    ]:
        view = np.eye(4)
        view[:3, 0], view[:3, 1], view[:3, 2] = (
            np.cross(look, [0, 1, 0]), [0, 1, 0], -np.array(look))
        view[3, :3] = -np.array(position)@view[:3, :3]
        projection = np.zeros((4, 4))
        projection[0, 0], projection[1, 1], projection[2, 3] = .5625, 1., -1.
        rows.append(dict(view_id=name, renderer_only_world_position=position,
            renderer_only_world_look=look, renderer_only_world_up=[0, 1, 0],
            width=320, height=180, requested_fov_degrees=90,
            view_matrix_diagnostic=str(view.tolist()),
            projection_matrix_diagnostic=str(projection.tolist())))
    return dict(car_id='bmw_z4_gt3', camera_layout_revision=3,
        camera_mount_contract=copy.deepcopy(EXTERIOR_FRONT_CONTRACT), views=rows)


def test_mount_identity_cannot_be_reinterpreted_by_old_or_new_configuration():
    frame = fixture()
    config = dict(camera_mount_contract=copy.deepcopy(EXTERIOR_FRONT_CONTRACT))
    validate_frame_mount(frame, config)
    with pytest.raises(ValueError, match='Legacy'):
        validate_frame_mount(frame, {})
    legacy = dict(frame, camera_layout_revision=2, camera_mount_contract=None)
    validate_frame_mount(legacy, {})
    with pytest.raises(ValueError, match='differs'):
        validate_frame_mount(legacy, config)
    config['camera_mount_contract']['front_offset_forward_m'] = 2.6
    with pytest.raises(ValueError, match='Exact explicit'):
        configured_mount(config)


def test_mount_checks_actual_returned_centers_not_only_metadata_revision():
    frame = fixture()
    # Consistent camera pose, wrong declared mount: identity/rotation checks alone would pass.
    front = frame['views'][0]
    front['renderer_only_world_position'][2] -= .1
    view = np.eye(4)
    view[3, :3] = -np.array(front['renderer_only_world_position'])
    front['view_matrix_diagnostic'] = str(view.tolist())
    with pytest.raises(ValueError, match='centers'):
        validate_frame_mount(frame, dict(camera_mount_contract=EXTERIOR_FRONT_CONTRACT))


def test_wide_side_mount_requires_actual_120_degree_side_matrices():
    frame = fixture()
    old_config = dict(camera_mount_contract=EXTERIOR_FRONT_CONTRACT)
    wide_config = dict(camera_mount_contract=EXTERIOR_SIDE_120_CONTRACT)
    frame.update(camera_layout_revision=4, camera_mount_contract=EXTERIOR_SIDE_120_CONTRACT)
    with pytest.raises(ValueError, match='side FOV'):
        validate_frame_mount(frame, wide_config)
    scale = 1 / np.sqrt(3)
    for view in frame['views'][1:3]:
        projection = np.zeros((4, 4))
        projection[0, 0], projection[1, 1], projection[2, 3] = .5625 * scale, scale, -1.
        view.update(projection_matrix_diagnostic=str(projection.tolist()), requested_fov_degrees=120)
    validate_frame_mount(frame, wide_config)
    with pytest.raises(ValueError, match='differs'):
        validate_frame_mount(frame, old_config)
