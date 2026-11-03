import pytest

torch = pytest.importorskip("torch")

from flybrain_drive.flight_embodied_navigation import NavigationCommandBoundary, modulate_current


def boundary():
    torch.manual_seed(863432)
    return NavigationCommandBoundary(torch.zeros(576), torch.ones(576))


def test_zero_modulation_preserves_original_currents_exactly_and_stays_bounded():
    current = torch.linspace(0, 1, 1001)
    assert torch.equal(modulate_current(current, torch.zeros_like(current)), current)
    for delta in (-8.0, 8.0):
        result = modulate_current(current, torch.full_like(current, delta))
        assert torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    b = boundary()
    delta = b(
        torch.randn(3, 576),
        torch.zeros(3, 21),
        torch.ones(3, 21, dtype=torch.bool),
        torch.tensor([[0.0, 1.0]]).repeat(3, 1),
    )
    assert torch.equal(delta, torch.zeros(3, 5))


def test_turn_command_cannot_shortcut_through_body_yaw():
    b = boundary()
    with torch.no_grad():
        b.turn_output.weight.normal_()
        b.power_output.weight.normal_()
    body = torch.zeros(2, 21)
    body[:, 4] = torch.tensor([-1.0, 1.0])
    body[1, 17:] = torch.tensor([5.0, 10.0, 15.0, 20.0])
    delta = b(
        torch.ones(2, 576),
        body,
        torch.ones(2, 21, dtype=torch.bool),
        torch.tensor([[0.0, 1.0]]).repeat(2, 1),
    )
    assert torch.equal(delta[0, :4], delta[1, :4])


def test_both_descending_command_heads_receive_trainable_gradients():
    b = boundary()
    delta = b(
        torch.randn(4, 576),
        torch.randn(4, 21),
        torch.ones(4, 21, dtype=torch.bool),
        torch.tensor([[0.0, 1.0]]).repeat(4, 1),
    )
    (delta[:, :4].sum() + delta[:, 4].sum()).backward()
    assert b.turn_output.weight.grad.abs().max() > 0
    assert b.power_output.weight.grad.abs().max() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
