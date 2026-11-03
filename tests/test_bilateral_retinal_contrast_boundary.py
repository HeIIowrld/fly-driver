import pytest
import torch
from flybrain_drive.bilateral_retinal_contrast_boundary import BilateralRetinalContrastBoundary
from test_unified_flight_sensorimotor import make_model, invoke


def test_reflection_invariance_and_explicit_view_validity():
    torch.manual_seed(42)
    m = BilateralRetinalContrastBoundary(2)
    scene = torch.randn(2, 4, 9, 4, 4)
    reflected = scene.clone()
    reflected[:, 0] = scene[:, 0].flip(-1)
    reflected[:, 3] = scene[:, 3].flip(-1)
    reflected[:, 1] = scene[:, 2].flip(-1)
    reflected[:, 2] = scene[:, 1].flip(-1)
    valid = torch.ones(2, 4, dtype=torch.bool)
    torch.testing.assert_close(m.features(reflected.reshape(2, 576), valid),
                               -m.features(scene.reshape(2, 576), valid))
    valid[0, 2] = False
    with pytest.raises(ValueError):
        m(scene.reshape(2, 576), valid)


def test_zero_initialized_boundary_preserves_single_circuit_behavior():
    torch.manual_seed(9)
    baseline = make_model()
    candidate = make_model()
    candidate.load_state_dict(baseline.state_dict())
    candidate.spatial_contrast = BilateralRetinalContrastBoundary(2)
    scene = torch.randn(1, 576)
    for now in (1., 1.04, 1.08):
        a = invoke(baseline, now, scene)
        b = invoke(candidate, now, scene)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(baseline.state, candidate.state)


def test_constructor_accepts_exact_bilateral_boundary_and_checks_ports():
    import copy
    from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
    from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
    from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
    from test_unified_flight_sensorimotor import SyntheticMotor
    motor = SyntheticMotor()
    visual = copy.deepcopy(motor.backbone)
    temporal = OpponentTemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    context = PerViewVisualBodyContextBoundary(2)
    boundary = BilateralRetinalContrastBoundary(2)
    model = UnifiedFlightSensorimotor(motor, visual, temporal, context, spatial_contrast=boundary)
    assert model.spatial_contrast is boundary
    with pytest.raises(ValueError, match='compatible spatial'):
        UnifiedFlightSensorimotor(motor, visual, temporal, context,
                                 spatial_contrast=BilateralRetinalContrastBoundary(3))
