import torch

from flybrain_drive.flight_centered_motion_boundary import CenteredMotionBoundary


def test_mean_anchor_and_materialization_preserve_original_nonmotion_inputs():
    torch.manual_seed(8)
    original = torch.nn.Linear(588, 5)
    mean = torch.randn(576)
    adapter = CenteredMotionBoundary(original, mean)
    with torch.no_grad():
        adapter.delta.normal_(0, .03)
    x = torch.randn(7, 588)
    x[:, 12:] = mean
    # Luminance remains free while all motion channels remain at their train means.
    channels = x[:, 12:].reshape(-1, 4, 9, 16)
    channels[:, :, 0] = torch.randn(7, 4, 16)
    torch.testing.assert_close(adapter(x), original(x), rtol=0, atol=1e-7)
    x = torch.randn(7, 588)
    weight, bias = adapter.materialize()
    torch.testing.assert_close(adapter(x), torch.nn.functional.linear(x, weight, bias),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight[:, :12], original.weight[:, :12], rtol=0, atol=0)
    torch.testing.assert_close(weight[:, 12:].reshape(5, 4, 9, 16)[:, :, 0],
                               original.weight[:, 12:].reshape(5, 4, 9, 16)[:, :, 0],
                               rtol=0, atol=0)
