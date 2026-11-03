"""Synthetic circuit tests; neither driving performance nor measured neuronal tuning."""

import copy

import numpy as np
import pytest
import torch

from flybrain_drive.neurotransmitters import join_transmitters
from flybrain_drive.sensory_policy import SensoryFeatureEncoder
from flybrain_drive.vehicle_circuit import (
    SignedRateCircuit,
    VehicleCircuitController,
    VehicleCircuitRunner,
    VehicleReadout,
    _RectifiedTanhSurrogate,
    encode_positive_gain,
    positive_gain,
)


def graph():
    return {"ids": np.array(list(map(str, range(9)))),
            "types": np.array(["eye", "eye", "temp", "load", "relay", "DN", "DN", "DN", "AN"]),
            "roles": np.array(["input"] * 4 + ["internal"] + ["output"] * 4),
            "pre": np.array([0, 1, 2, 3, 4, 4, 4, 2]),
            "post": np.array([4, 4, 4, 4, 5, 6, 7, 8]),
            "weights": np.ones(8, np.float32)}


def constructor():
    angle = np.linspace(-.2, .2, 3)
    directions = np.stack([np.column_stack([np.cos(angle), np.sin(angle), angle * 0])] * 2)
    return dict(visual_ids=["0", "1"], organ_groups={
        "temperature": {"body_ids": ["2"], "feature_indices": [0]},
        "load": {"body_ids": ["3"], "feature_indices": [1]},
    }, directions=directions, available=np.ones((2, 3)), organ_size=2,
                signs=[1, 1, 1, -1, 1, 1, 1, 1, 1],
                incoming_totals=[0, 0, 0, 0, 8, 2, 2, 2, 2],
                output_groups={"steering": ["5", "6"], "forward": ["7"], "halt": ["8"]},
                steering_signs=[-1, 1], organ_calibration_sha256="a" * 64,
                eye_map_hashes={"left": "a" * 64, "right": "b" * 64}, neighbors=2)


def observation(model):
    values = {key: torch.zeros(1, 1) for key in SensoryFeatureEncoder.fields}
    values.update(ray_values=torch.full((1, 2, 3, 3), .4), ray_valid=torch.ones(1, 2, 3),
                  ray_directions=model.directions[None].clone(), organs=torch.tensor([[.2, .3]]),
                  organ_valid=torch.ones(1, 2))
    return values


def test_scipy_cpu_csr_inference_matches_original_and_refreshes_changed_graph():
    pytest.importorskip('scipy.sparse')
    args = constructor()
    net = SignedRateCircuit(graph(), args['signs'], args['incoming_totals'],
                            dt=.01, tau=.05, sparse_backend='scipy_csr_cpu')
    torch.manual_seed(2)
    state = torch.rand(2, 9)
    drive = torch.rand(2, 9) * net.input_mask
    with pytest.raises(ValueError, match='no_grad'):
        net._advance_tensor(drive, state)
    with torch.no_grad():
        _, actual = net._advance_tensor(drive, state)
        net.sparse_backend = 'cached_csr_cpu'
        _, expected = net._advance_tensor(drive, state)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        net.matrix.mul_(.5)
        net.sparse_backend = 'scipy_csr_cpu'
        _, changed = net._advance_tensor(drive, state)
        net.sparse_backend = 'cached_csr_cpu'
        _, expected_changed = net._advance_tensor(drive, state)
        torch.testing.assert_close(changed, expected_changed, atol=1e-7, rtol=1e-6)
        assert not torch.equal(actual, changed)
        net.sparse_backend = 'scipy_csr_cpu'
        with pytest.raises(ValueError, match='no_grad'):
            net._advance_tensor(drive, state.requires_grad_())


def test_independent_pedals_preserve_overlap_without_changing_legacy_or_steering():
    args = constructor()
    legacy = VehicleReadout(graph(), args['output_groups'], args['steering_signs'])
    independent = VehicleReadout(graph(), args['output_groups'], args['steering_signs'],
                                pedal_mode='independent')
    independent.load_state_dict(legacy.state_dict())
    activity = torch.ones(2, 4)
    old, new = legacy(activity), independent(activity)
    assert (old[:, 1:].sum(1) <= 1).all() and (new[:, 1:].sum(1) > 1).all()
    torch.testing.assert_close(old[:, 0], new[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(old[:, 2], new[:, 2], rtol=0, atol=0)
    torch.testing.assert_close(legacy(torch.zeros(1, 4)), torch.zeros(1, 3))


def test_log_gain_migration_preserves_actions_and_adam_can_update_large_relative_gain():
    args = constructor()
    legacy = VehicleReadout(graph(), args['output_groups'], args['steering_signs'],
                            pedal_mode='independent')
    new = VehicleReadout(graph(), args['output_groups'], args['steering_signs'],
                         pedal_mode='independent', gain_parameterization='log')
    with torch.no_grad():
        legacy.steering_log_gain.fill_(1e6)
        legacy.pedal_log_gain.fill_(1e6)
    state = copy.deepcopy(legacy.state_dict())
    for name in ('steering_log_gain', 'pedal_log_gain'):
        state[name] = encode_positive_gain(positive_gain(state[name]), 'log')
    new.load_state_dict(state)
    activity = torch.tensor([[1e-8, 2e-8, 3e-8, 4e-8]])
    torch.testing.assert_close(new(activity), legacy(activity), rtol=2e-6, atol=1e-8)
    gains = []
    for model in (legacy, new):
        optimizer = torch.optim.Adam([model.pedal_log_gain], lr=.03)
        optimizer.zero_grad()
        model(activity)[:, 2].square().sum().backward()
        optimizer.step()
        gains.append(positive_gain(model.pedal_log_gain, model.gain_parameterization).detach())
    assert gains[0][1] == 1e6  # Float32 increment is too small at this magnitude.
    assert gains[1][1] < .99e6


@pytest.mark.parametrize('mode', ['softplus', 'log'])
def test_positive_gain_encoding_rejects_invalid_and_roundtrips_scale_range(mode):
    gains = torch.tensor([1e-8, 1., 1e8])
    torch.testing.assert_close(positive_gain(encode_positive_gain(gains, mode), mode), gains)
    with pytest.raises(ValueError, match='positive'):
        encode_positive_gain(torch.tensor([0., float('nan')]), mode)


def test_silent_neuron_surrogate_changes_only_backward_not_forward_activity():
    exact_net = torch.tensor([-1., 0., .5], requires_grad=True)
    surrogate_net = exact_net.detach().clone().requires_grad_(True)
    exact = torch.relu(torch.tanh(exact_net))
    surrogate = _RectifiedTanhSurrogate.apply(surrogate_net, .05)
    torch.testing.assert_close(surrogate, exact, rtol=0, atol=0)
    exact.sum().backward()
    surrogate.sum().backward()
    assert not exact_net.grad[:2].any() and (surrogate_net.grad[:2] > 0).all()
    torch.testing.assert_close(surrogate_net.grad[2:], exact_net.grad[2:])
    assert (surrogate >= 0).all() and (surrogate <= 1).all()


def test_sensory_pedal_conditioning_reaches_scale_and_rejects_unreachable_drive_transactionally():
    from flybrain_drive.multiview_learning import condition_independent_pedals

    args = constructor()
    readout = VehicleReadout(graph(), args['output_groups'], args['steering_signs'],
                            pedal_mode='independent')
    activity = torch.tensor([[.1, .2, 2e-8, 4e-8]]).repeat(32, 1)
    before_steering = readout(activity)[:, 0].detach().clone()
    report = condition_independent_pedals(readout, activity,
                                         target_rms=[.2, .3], max_gain=1e8)
    actual = readout(activity)
    torch.testing.assert_close(actual[:, 0], before_steering, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 1:], torch.tensor([[.2, .3]]).repeat(32, 1))
    assert not report['teacher_labels_used']
    gains = readout.pedal_log_gain.detach().clone()
    activity[:, 3] = 0
    with pytest.raises(ValueError, match='gain budget'):
        condition_independent_pedals(readout, activity, target_rms=[.2, .3], max_gain=1e8)
    torch.testing.assert_close(gains, readout.pedal_log_gain, rtol=0, atol=0)


def test_signed_nonnegative_dynamics_inhibit_without_inverting_activity_or_missing_mass():
    kwargs = constructor()
    net = SignedRateCircuit(graph(), kwargs["signs"], kwargs["incoming_totals"], dt=.01, tau=.05)
    dense = net.matrix.to_dense()
    assert dense[4, 3] == -.125 and dense[4].abs().sum() == .5
    quiet = torch.zeros(1, 9)
    quiet[0, 0] = 1
    inhibited = quiet.clone()
    inhibited[0, 3] = 1
    a, b = torch.zeros(1, 9), torch.zeros(1, 9)
    for _ in range(40):
        out, a = net(quiet, a)
        suppressed, b = net(inhibited, b)
    assert out[0, 2] > suppressed[0, 2]
    assert (a >= 0).all() and (b >= 0).all()
    bad = quiet.clone()
    bad[:, 4] = .1
    with pytest.raises(ValueError, match="input-port"):
        net(bad, a)


def test_output_only_readout_requests_have_no_bias_and_braking_attenuates_throttle():
    args = constructor()
    readout = VehicleReadout(graph(), args["output_groups"], args["steering_signs"])
    np.testing.assert_array_equal(readout(torch.zeros(1, 4)).detach(), [[0, 0, 0]])
    normal = readout(torch.tensor([[0., 1., 1., 0.]]))
    braking = readout(torch.tensor([[0., 1., 1., 1.]]))
    assert normal[0, 0] > 0 and normal[0, 1] > 0 and normal[0, 2] == 0
    assert braking[0, 2] > 0 and braking[0, 1] < normal[0, 1]
    assert ((braking >= -1) & (braking <= 1)).all()
    with pytest.raises(ValueError, match="outputs"):
        VehicleReadout(graph(), {**args["output_groups"], "forward": ["4"]}, [-1, 1])


def test_multimodal_gradient_crosses_fixed_graph_and_detach_preserves_numerical_state():
    net = VehicleCircuitController(graph(), **constructor())
    with torch.no_grad():
        for parameter in net.adapter.parameters():
            parameter.fill_(.1)
    obs = observation(net)
    state = net.initial_state()
    matrix = net.circuit.matrix.to_dense().clone()
    for _ in range(8):
        action, state = net(obs, state)
    action.square().sum().backward()
    assert net.adapter.visual_adapter.weight.grad.abs().sum() > 0
    assert all(layer.weight.grad.abs().sum() > 0 for layer in net.adapter.organ_adapters)
    same, detached = net(obs, state.detach())
    again, carried = net(obs, state)
    torch.testing.assert_close(same, again, rtol=0, atol=0)
    torch.testing.assert_close(detached, carried, rtol=0, atol=0)
    reset, _ = net(obs, net.initial_state())
    assert not torch.allclose(reset, again)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    optimizer.step()
    torch.testing.assert_close(net.circuit.matrix.to_dense(), matrix, rtol=0, atol=0)


def test_missing_sensing_and_cut_graph_cannot_be_replaced_by_normalization_or_truth(tmp_path):
    net = VehicleCircuitController(graph(), **constructor())
    obs = observation(net)
    missing = copy.deepcopy(obs)
    missing["ray_valid"].zero_()
    missing["organ_valid"].zero_()
    net.feature_mean.fill_(.3)
    assert not net.prepare_drive(missing).any()
    with pytest.raises(ValueError, match="actor"):
        net.prepare_drive({**obs, "route_progress": torch.ones(1)})
    cut = copy.deepcopy(net)
    cut.circuit.matrix = cut.circuit.matrix * 0
    state = cut.initial_state()
    for _ in range(12):
        action, state = cut(obs, state)
    assert not action.any()
    path = tmp_path / "state.pt"
    torch.save(net.state_dict(), path)
    restored = VehicleCircuitController(graph(), **constructor())
    restored.load_state_dict(torch.load(path, weights_only=True))
    a, b = net.initial_state(), restored.initial_state()
    for _ in range(8):
        x, a = net(obs, a)
        y, b = restored(obs, b)
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_consensus_join_retains_missingness_and_individual_uncertainty():
    records = [{"body": 1, "consensus_nt": "acetylcholine", "predicted_nt": "unclear",
                "predicted_nt_confidence": .3},
               {"body": 2, "consensus_nt": "gaba", "predicted_nt": "gaba",
                "predicted_nt_confidence": .9}]
    hypothesis = {"acetylcholine": 1, "gaba": -1, "missing": 0}
    signs, rows, report = join_transmitters([2, 1, 3], records, hypothesis)
    np.testing.assert_array_equal(signs, [-1, 1, 0])
    assert rows[2]["missing"] and report["missing_body_ids"] == ["3"]
    assert report["individual_confidence_below_0_8_body_ids"] == ["1"]
    assert report["consensus_differs_from_individual_body_ids"] == ["1"]
    assert "consensus_confidence" not in rows[1]
    with pytest.raises(ValueError, match="Duplicate"):
        join_transmitters([1], records + [records[0]], hypothesis)
    with pytest.raises(ValueError, match="Unspecified"):
        join_transmitters([1], [{"body": 1, "consensus_nt": "unknown"}], hypothesis)


def test_runner_carries_state_and_explicit_restart_or_episode_change_resets_it():
    from flybrain_drive.body_organs import BodyOrganEncoder, ScalarOrganSpec
    from flybrain_drive.compound_eye import synthetic_eye
    from flybrain_drive.sensor_contract import ActorObservation, BodySample, EyeSample, SampleTime
    from flybrain_drive.sensory_policy import ObservationPacker

    eyes = [synthetic_eye(side, rows=2, columns=3) for side in ("left", "right")]
    organs = BodyOrganEncoder([], scalars=[ScalarOrganSpec("wheel_normal_load", 0, 10000, 1000,
                                                         10000, populations=3)])
    packer = ObservationPacker(eyes, organs, specific_force_scale=20, wheel_speed_scale=50)
    kwargs = constructor()
    kwargs.update(directions=np.stack([eye.directions_body for eye in eyes]),
                  available=np.ones((2, 6)), organ_size=organs.feature_count,
                  organ_calibration_sha256=organs.calibration_sha256,
                  eye_map_hashes={eye.side: eye.sha256 for eye in eyes})
    model = VehicleCircuitController(graph(), **kwargs)
    with torch.no_grad():
        for parameter in model.adapter.parameters():
            parameter.fill_(.1)
    runner = VehicleCircuitRunner(model, packer)

    def obs(seq=0, episode="a"):
        time = SampleTime(seq * .01, 100 + seq * .01)
        views = tuple(EyeSample(e.side, seq, np.full((6, 3), .4), np.ones(6), time, e.sha256)
                      for e in eyes)
        body = BodySample("wheel_normal_load", [2000] * 4, [1] * 4, time, "N",
                          "contact_normal_compression_positive", "declared_sensor_model")
        return ActorObservation(episode, seq, time.simulation_seconds,
                                time.received_monotonic_seconds, views, (body,))

    first, _ = runner.step(obs())
    initial_state = runner.state.clone()
    runner.step(obs(1))
    assert not torch.equal(initial_state, runner.state)
    reset, _ = runner.step(obs(episode="b"))
    np.testing.assert_array_equal(first, reset)
    torch.testing.assert_close(initial_state, runner.state, rtol=0, atol=0)
    runner.reset()
    restarted, _ = runner.step(obs(episode="b"))
    np.testing.assert_array_equal(first, restarted)


def test_compact_scipy_backend_omits_only_exact_zeros_and_preserves_state():
    pytest.importorskip('scipy.sparse')
    args=constructor();signs=np.array(args['signs']);signs[2]=0
    net=SignedRateCircuit(graph(),signs,args['incoming_totals'],dt=.01,tau=.05,sparse_backend='scipy_csr_cpu')
    original=net.matrix.coalesce().values().clone();assert (original==0).any()
    state=torch.rand(1,9);drive=torch.rand(1,9)*net.input_mask
    with torch.no_grad():
        ordinary=state.clone()
        for _ in range(20):_,ordinary=net._advance_tensor(drive,ordinary)
        ordinary_nnz=net._scipy_matrix.nnz
        net.sparse_backend='scipy_csr_compact_cpu';compact=state.clone()
        for _ in range(20):_,compact=net._advance_tensor(drive,compact)
        assert net._scipy_matrix.nnz < ordinary_nnz
        torch.testing.assert_close(compact,ordinary,rtol=0,atol=1e-7)
        assert torch.equal(net.matrix.coalesce().values(),original)
        net.sparse_backend='scipy_csr_cpu';net._advance_tensor(drive,state)
        assert net._scipy_matrix.nnz==ordinary_nnz
    with pytest.raises(ValueError,match='no_grad'):
        net.sparse_backend='scipy_csr_compact_cpu';net._advance_tensor(drive,state)


def test_numba_csr_backend_preserves_batch_state_and_restores_threads():
    numba=pytest.importorskip('numba')
    if numba.config.NUMBA_NUM_THREADS < 8: pytest.skip('8 threads not available')
    args=constructor();signs=np.array(args['signs']);signs[2]=0
    net=SignedRateCircuit(graph(),signs,args['incoming_totals'],dt=.01,tau=.05,sparse_backend='scipy_csr_cpu')
    torch.manual_seed(8);state=torch.rand(2,9);drive=torch.rand(2,9)*net.input_mask;threads=numba.get_num_threads()
    with torch.no_grad():
        ordinary=state.clone()
        for _ in range(20):_,ordinary=net._advance_tensor(drive,ordinary)
        net.sparse_backend='numba_csr8_cpu';parallel=state.clone()
        for _ in range(20):_,parallel=net._advance_tensor(drive,parallel)
        torch.testing.assert_close(parallel,ordinary,rtol=0,atol=1e-7)
        assert numba.get_num_threads()==threads
    with pytest.raises(ValueError,match='no_grad'):
        net._advance_tensor(drive,state)
