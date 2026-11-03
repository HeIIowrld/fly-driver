import pytest

from scripts.project_timed_fourview_retina import check_view


def view():
    return {'view_id': 'front', 'file': '0000-front.png', 'width': 320, 'height': 180,
            'requested_fov_degrees': 56,
            'projection_matrix_diagnostic': '((1.05,0,0,0),(0,1.88,0,0),'
            '(0,0,-1, -1),(0,0,-.1,0))'}


def test_projection_checks_original_geometry_and_file_identity():
    config = {'projection_x': 1.05, 'projection_y': 1.88}
    check_view(view(), 'front', 0, config)
    wrong = dict(view(), projection_matrix_diagnostic='((2,0,0,0),(0,1.88,0,0),'
                 '(0,0,-1,-1),(0,0,-.1,0))')
    with pytest.raises(ValueError, match='projection'):
        check_view(wrong, 'front', 0, config)
    with pytest.raises(ValueError, match='identity'):
        check_view(dict(view(), file='other.png'), 'front', 0, config)
