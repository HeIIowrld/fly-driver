"""Existing positive output-neuron throttle codec, with no sensory residual or new ports."""

import torch


def tensors_identical(first, second):
    if first.layout != second.layout or first.shape != second.shape or first.dtype != second.dtype:
        return False
    if first.layout == torch.strided:
        return torch.equal(first, second)
    if first.layout == torch.sparse_coo:
        a, b = first.coalesce(), second.coalesce()
        return torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
    raise ValueError('Unsupported checkpoint tensor layout')


def throttle_from_activity(activity, ports, logits, log_gain):
    if (activity.ndim != 2 or ports.ndim != 1 or logits.shape != ports.shape
            or log_gain.ndim != 0):
        raise ValueError('Output activity, declared forward ports and one log gain required')
    drive = activity[:, ports] @ logits.softmax(0)
    return -torch.expm1(-log_gain.exp()*drive)


def throttle_bank_from_activity(activity, ports, logits, log_gain):
    """Independent existing decoders in parallel; no new sensory or neural inputs."""
    if (activity.ndim != 2 or ports.ndim != 1 or logits.ndim != 2
            or not 1 <= len(logits) <= 8 or logits.shape[1] != len(ports)
            or log_gain.shape != (len(logits),)):
        raise ValueError('One to eight independent decoders on the same output ports required')
    drive = activity[:, ports] @ logits.softmax(-1).T
    return -torch.expm1(-log_gain.exp()[None, :]*drive)
