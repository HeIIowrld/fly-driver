import numpy as np
import pytest

from flybrain_drive.diagnostic_path_projection import FutureTrace, project_reference_points


def telemetry():
    return [dict(script_clock=i*.05, position=[0, .4, -i*.5], phase='driving',
                 ai_controlled=True, paused=False, wheels_outside=0) for i in range(100)]


def test_trace_preserves_camera_clock_horizontal_distance_and_full_height():
    source = telemetry()
    start, points, valid = FutureTrace(source).at(.025)
    np.testing.assert_allclose(start, [0, .4, -.25])
    np.testing.assert_allclose(points, [[0, .4, -5.25], [0, .4, -10.25], [0, .4, -20.25]])
    assert valid.all()
    source[10]['paused'] = True
    assert not FutureTrace(source).at(.025)[2].any()


def test_projection_handedness_and_explicit_origin_cancellation():
    perspective = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -1], [0, 0, -.2, 0]])
    def text(m):
        return ' '.join(map(str, m.ravel()))
    view = dict(view_matrix_diagnostic=text(np.eye(4)),
                projection_matrix_diagnostic=text(perspective))
    points = np.array([[0, 0, -5], [1, 1, -5], [0, 0, 5]])
    uv, front, inside = project_reference_points(points, view)
    np.testing.assert_allclose(uv[:2], [[.5, .5], [.6, .4]])
    assert front.tolist() == [True, True, False]
    assert inside.tolist() == [True, True, False]
    translated = np.eye(4)
    translated[3, :3] = [-100, 0, 0]
    view.update(view_matrix_diagnostic=text(translated),
        renderer_origin_shift_before_diagnostic=[100, 0, 0],
        renderer_origin_shift_after_diagnostic=[100, 0, 0])
    np.testing.assert_allclose(project_reference_points(points, view)[0], uv)
    view['renderer_origin_shift_after_diagnostic'] = [101, 0, 0]
    with pytest.raises(ValueError, match='origin'):
        project_reference_points(points, view)
