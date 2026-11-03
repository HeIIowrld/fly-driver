import copy

import numpy as np
import pytest

from flybrain_drive.ac_csp_geometry import camera_geometry, matrix


def fixture():
    # Synthetic world with +X screen-right and forward towards -Z.
    records = []
    for name, look in [('front', [0, 0, -1]), ('left', [-1, 0, 0]),
                       ('right', [1, 0, 0]), ('rear', [0, 0, 1])]:
        look = np.array(look, float)
        up = np.array([0, 1, 0], float)
        v = np.eye(4)
        v[:3, 0], v[:3, 1], v[:3, 2] = np.cross(look, up), up, -look
        p = np.zeros((4, 4))
        p[0, 0], p[1, 1], p[2, 3] = .5625, 1, -1
        records.append(dict(view_id=name, view_matrix_diagnostic=str(v.tolist()),
            projection_matrix_diagnostic=str(p.tolist()), width=320, height=180,
            requested_fov_degrees=90, renderer_only_world_position=[0, 0, 0],
            renderer_only_world_look=look.tolist(), renderer_only_world_up=up.tolist()))
    return records


def test_projection_audit_rejects_swapped_side_labels_and_mismatched_fov():
    records = fixture()
    result = camera_geometry(records)
    assert result['front']['horizontal_fov_degrees'] == pytest.approx(121.2844929)
    swapped = copy.deepcopy(records)
    swapped[1]['view_id'], swapped[2]['view_id'] = 'right', 'left'
    with pytest.raises(ValueError, match='labels'):
        camera_geometry(swapped)
    records[0]['requested_fov_degrees'] = 56
    with pytest.raises(ValueError, match='vertical'):
        camera_geometry(records)
    with pytest.raises(ValueError, match='4x4'):
        matrix('1,2,3')


def shifted_fixture(shift=(0, 0, 1024)):
    records = fixture()
    for record in records:
        v = matrix(record['view_matrix_diagnostic'])
        v[3, :3] = -np.asarray(shift) @ v[:3, :3]
        record['view_matrix_diagnostic'] = str(v.tolist())
        record['renderer_origin_contract_diagnostic'] = 'getSim.originShift_before_after_v1'
        record['renderer_origin_shift_before_diagnostic'] = list(shift)
        record['renderer_origin_shift_after_diagnostic'] = list(shift)
    return records


@pytest.mark.parametrize('shift', [(0, 0, 1024), (-1024, 0, 2048), (0, 0, 0)])
def test_independently_recorded_origin_preserves_original_pose_contract(shift):
    assert camera_geometry(shifted_fixture(shift)) == camera_geometry(fixture())


@pytest.mark.parametrize('failure', ['missing', 'changed', 'mixed', 'nan', 'wrong_sign', 'legacy'])
def test_origin_never_inferred_from_camera_disagreement(failure):
    records = shifted_fixture()
    if failure == 'missing':
        del records[0]['renderer_origin_shift_before_diagnostic']
    elif failure == 'changed':
        records[0]['renderer_origin_shift_after_diagnostic'] = [0, 0, 0]
    elif failure == 'mixed':
        records[-1]['renderer_origin_shift_before_diagnostic'] = [0, 0, 0]
        records[-1]['renderer_origin_shift_after_diagnostic'] = [0, 0, 0]
    elif failure == 'nan':
        records[0]['renderer_origin_shift_before_diagnostic'] = [0, 0, float('nan')]
    elif failure == 'wrong_sign':
        for record in records:
            record['renderer_origin_shift_before_diagnostic'] = [0, 0, -1024]
            record['renderer_origin_shift_after_diagnostic'] = [0, 0, -1024]
    else:
        for record in records:
            for key in list(record):
                if key.startswith('renderer_origin_'):
                    del record[key]
    with pytest.raises(ValueError):
        camera_geometry(records)
