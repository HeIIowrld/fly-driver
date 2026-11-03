import pytest
import torch

from flybrain_drive.flight_temporal_motion_boundary import TemporalMotionBoundary


def test_prior_hold_steady_decay_and_explicit_reset():
    boundary = TemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    with torch.no_grad():
        boundary.projection.weight.fill_(.01)
    scene, views = torch.zeros((1, 576)), torch.ones((1, 4), dtype=torch.bool)
    first, state = boundary.step(scene, views, None, None)
    assert torch.equal(first, torch.zeros_like(first))
    changed = scene.clone().reshape(1, 4, 9, 16)
    changed[:, 0, 1, :] = 1
    changed = changed.flatten(1)
    # A sample received now cannot affect the interval that just ended.
    still, state = boundary.step(changed, views, state, .04)
    assert torch.equal(still, torch.zeros_like(still))
    pulse, state = boundary.step(changed, views, state, .04)
    assert torch.all(pulse > 0)
    for _ in range(50):
        tail, state = boundary.step(changed, views, state, .04)
    assert torch.all(tail.abs() < pulse.abs() * .001)
    reset, _ = boundary.step(changed, views, None, None)
    assert torch.equal(reset, torch.zeros_like(reset))
    invalid = views.clone()
    invalid[0, 2] = False
    with pytest.raises(ValueError, match='four valid'):
        boundary.step(changed, invalid, state, .04)
    with pytest.raises(ValueError, match='80ms'):
        boundary.step(changed, views, state, .1)


def test_neutral_projection_preserves_other_ports_and_can_receive_gradient():
    boundary = TemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    scene, views = torch.zeros((1, 576)), torch.ones((1, 4), dtype=torch.bool)
    _, state = boundary.step(scene, views, None, None)
    changed = scene.clone().reshape(1, 4, 9, 16)
    changed[:, 1, 2, :] = 1
    changed = changed.flatten(1)
    _, state = boundary.step(changed, views, state, .04)
    delta, _ = boundary.step(changed, views, state, .04)
    drive = torch.tensor([[.2, .3, .4, .5]])
    output = boundary.modulate(drive, torch.tensor([1, 3]), delta)
    torch.testing.assert_close(output, drive, rtol=0, atol=0)
    output.sum().backward()
    assert boundary.projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        boundary.projection.weight.fill_(.1)
    delta, _ = boundary.step(changed, views, state, .04)
    output = boundary.modulate(drive, torch.tensor([1, 3]), delta)
    assert torch.equal(output[:, [0, 2]], drive[:, [0, 2]])
    assert torch.all(output[:, [1, 3]] > drive[:, [1, 3]])


def test_fixed_motion_slice_matches_mask_and_rejects_mutation():
    torch.manual_seed(3)
    mean, scale = torch.randn(576), torch.rand(576) + .1
    boundary = TemporalMotionBoundary(mean, scale, 2)
    scene = torch.randn(2, 576, requires_grad=True)
    views = torch.ones((2, 4), dtype=torch.bool)
    _, state = boundary.step(scene, views, None, None)
    expected = ((scene - mean) / scale).clamp(-5, 5)[:, boundary.motion_mask]
    assert torch.equal(state[2], expected)
    actual_grad = torch.autograd.grad(state[2].sum(), scene, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), scene)[0]
    assert torch.equal(actual_grad, expected_grad)
    boundary.motion_mask[0] = True
    with pytest.raises(ValueError, match='motion-channel mask'):
        boundary.step(scene, views, None, None)


@pytest.mark.parametrize('index', range(3))
def test_each_nonfinite_prior_trace_is_rejected(index):
    boundary = TemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    scene, views = torch.zeros((1, 576)), torch.ones((1, 4), dtype=torch.bool)
    _, state = boundary.step(scene, views, None, None)
    damaged = [x.clone() for x in state]
    damaged[index][0, 9] = float('nan')
    with pytest.raises(ValueError, match='Prior traces'):
        boundary.step(scene, views, tuple(damaged), .01)


def test_nonfinite_scene_is_rejected():
    boundary = TemporalMotionBoundary(torch.zeros(576), torch.ones(576), 2)
    scene, views = torch.zeros((1, 576)), torch.ones((1, 4), dtype=torch.bool)
    scene[0, 4] = float('inf')
    with pytest.raises(ValueError, match='finite scene'):
        boundary.step(scene, views, None, None)
