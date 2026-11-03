import numpy as np
import pytest

from flybrain_drive.teacher_path_preview import future_path_preview


def rows(x, z):
    return [dict(script_clock=i*.05, position=[a, 0, b], phase='driving',
                 ai_controlled=True, wheels_outside=0, paused=False)
            for i, (a, b) in enumerate(zip(x, z, strict=True))]


def test_straight_interpolation_and_missing_horizon():
    telemetry = rows(np.zeros(81), np.linspace(0, 20, 81))
    y, mask = future_path_preview(telemetry, .025, [0, 0, 1], [1, 0, 0])
    np.testing.assert_array_equal(mask, [True, True, False])
    np.testing.assert_allclose(y, 0)


def test_screen_right_sign_curvature_and_rotation_invariance():
    angle = np.linspace(0, .8, 161)
    telemetry = rows(50*(1-np.cos(angle)), 50*np.sin(angle))
    y, mask = future_path_preview(telemetry, 0, [0, 0, 1], [1, 0, 0])
    assert mask.all() and (y > 0).all()
    np.testing.assert_allclose(y[:, 1], .02, atol=1e-5)
    for row in telemetry:
        x, _, z = row['position']
        row['position'] = [z+1000, 0, -x+2000]
    rotated, _ = future_path_preview(telemetry, 0, [1, 0, 0], [0, 0, -1])
    np.testing.assert_allclose(rotated, y, atol=1e-12)


@pytest.mark.parametrize('fault', ['gap', 'teleport', 'cleanup', 'outside'])
def test_never_cross_gap_reset_or_cleanup(fault):
    telemetry = rows(np.zeros(161), np.linspace(0, 40, 161))
    if fault == 'gap':
        for row in telemetry[10:]:
            row['script_clock'] += .2
    elif fault == 'teleport':
        for row in telemetry[10:]:
            row['position'][0] += 100
    elif fault == 'cleanup':
        telemetry[10]['phase'] = 'recovery'
    else:
        telemetry[10]['wheels_outside'] = 1
    _, mask = future_path_preview(telemetry, 0, [0, 0, 1], [1, 0, 0])
    assert not mask.any()


def test_unsorted_clock_rejected():
    telemetry = rows([0, 0, 0], [0, 1, 2])
    telemetry[1]['script_clock'] = 0
    with pytest.raises(ValueError, match='ordered'):
        future_path_preview(telemetry, 0, [0, 0, 1], [1, 0, 0])
