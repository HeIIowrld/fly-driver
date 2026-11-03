import numpy as np
import torch


def test_parallel_decoder_bank_matches_individual_codec_and_has_no_cross_candidate_gradient():
    from flybrain_drive.throttle_readout import (
        throttle_bank_from_activity,
        throttle_from_activity,
    )

    activity = torch.tensor([[.1, .2, .4], [.2, .5, .1]])
    ports = torch.tensor([0, 2])
    logits = torch.tensor([[.2, -.1], [-.4, .7]], requires_grad=True)
    gain = torch.tensor([.1, -.3], requires_grad=True)
    value = throttle_bank_from_activity(activity, ports, logits, gain)
    expected = torch.stack([throttle_from_activity(activity, ports, logits[i], gain[i])
                            for i in range(2)], dim=1)
    torch.testing.assert_close(value, expected)
    value[:, 0].sum().backward()
    assert logits.grad[0].abs().sum() > 0 and gain.grad[0].abs() > 0
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))
    assert gain.grad[1] == 0

from flybrain_drive.throttle_readout import tensors_identical, throttle_from_activity
from flybrain_drive.vehicle_circuit import VehicleReadout


def test_cached_throttle_value_and_gradients_equal_live_decoder_and_other_heads_stay_fixed():
    graph = dict(ids=np.array(['s1', 's2', 'f1', 'f2', 'halt']), roles=np.array(['output']*5))
    groups = dict(steering=['s1', 's2'], forward=['f1', 'f2'], halt=['halt'])
    readout = VehicleReadout(graph, groups, [-1, 1], pedal_mode='independent',
                             gain_parameterization='log')
    activity = torch.tensor([[.1, .2, .3, .1, .2], [.2, .1, .1, .4, .1]])
    baseline = readout(activity)
    logits = readout.forward_weights.detach().clone().requires_grad_()
    gain = readout.pedal_log_gain[0].detach().clone().requires_grad_()
    cached = throttle_from_activity(activity, readout.forward_ports, logits, gain)
    torch.testing.assert_close(cached, baseline[:, 1], rtol=0, atol=0)
    baseline[:, 1].square().sum().backward()
    cached.square().sum().backward()
    torch.testing.assert_close(logits.grad, readout.forward_weights.grad)
    torch.testing.assert_close(gain.grad, readout.pedal_log_gain.grad[0])
    with torch.no_grad():
        readout.forward_weights.add_(torch.tensor([.2, -.3]))
        readout.pedal_log_gain[0].add_(.5)
    assert torch.equal(readout(activity)[:, [0, 2]], baseline[:, [0, 2]])


def test_frozen_sparse_graph_equality_checks_indices_and_values_without_densifying():
    graph = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1., -2.], (2, 2)).coalesce()
    assert tensors_identical(graph, graph.clone())
    changed = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1., -3.], (2, 2)).coalesce()
    assert not tensors_identical(graph, changed)
