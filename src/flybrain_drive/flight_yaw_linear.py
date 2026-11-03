"""Equivalent zero-yaw intervention at the frozen engineering body's linear encoder."""

from torch.nn import functional as F

from .flight_commands import FrozenLinear


def zero_yaw_linear_forward(layer):
    """Keep original parameters/input tensors; use a separate yaw-column-zero weight copy."""
    if (type(layer) is not FrozenLinear or layer.weight.ndim != 2
            or layer.weight.shape[1] != 6 or layer.bias.shape != (layer.weight.shape[0],)
            or layer.training or list(layer.parameters())):
        raise ValueError('Exact frozen six-input linear body encoder required')

    def current_versions():
        return tuple((name, id(value), value._version)
                     for name, value in layer.named_buffers())

    versions = current_versions()
    weight = layer.weight.detach().clone()
    weight[:, 2] = 0.  # tanh([angular_x,angular_y,yaw])*valid, followed by3 validity bits.

    def forward(value):
        if layer.training or list(layer.parameters()) or versions != current_versions():
            raise ValueError('Original frozen body parameters changed')
        return F.linear(value, weight, layer.bias)

    return forward


def apply_linear_yaw_mode(stream, mode):
    if mode not in ('full', 'zero_yaw') or hasattr(stream, 'yaw_trial_mode'):
        raise ValueError('One explicit full/zero_yaw condition per original episode required')
    if stream.steps or stream.order.clock is not None:
        raise ValueError('Yaw intervention must be prepared before the first original observation')
    stream.flight.assert_body_boundary_protected()
    if stream.flight.context_indices.detach().cpu().tolist() != [0, 1, *range(8, 16)]:
        raise ValueError('Other body context must exclude yaw for equivalent intervention')
    if mode == 'zero_yaw':
        stream.flight.body.forward = zero_yaw_linear_forward(stream.flight.body)
    stream.yaw_trial_mode = mode
    stream.flight.assert_body_boundary_protected()
    return stream
