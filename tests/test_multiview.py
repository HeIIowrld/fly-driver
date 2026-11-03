"""Synthetic multi-source optics/time/fusion checks; not an AC driving benchmark."""

import copy

import numpy as np
import pytest
import torch

from flybrain_drive.body_organs import BodyOrganEncoder
from flybrain_drive.compound_eye import PerspectiveView
from flybrain_drive.multiview_circuit import MultiViewCircuitController, MultiViewCircuitRunner
from flybrain_drive.multiview_sensory import MultiViewPacker, ViewMap
from flybrain_drive.sensor_contract import ActorObservation, SampleTime, VisualViewSample


def maps():
    result = []
    rays = np.array([[1, 0, 0], [1, .1, 0], [1, -.1, 0], [1, 0, .1]], float)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    for name, angle in zip(("front", "mirror_left", "mirror_right", "mirror_rear"),
                            (0, 135, -135, 180), strict=True):
        yaw = np.deg2rad(angle)
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        result.append(ViewMap(name, rays @ rotation.T, np.full(4, 1e-5),
                              PerspectiveView(rotation, [0, 0, 0], 90),
                              "synthetic direct view, not a physical reflected mirror"))
    return result


def sample(view, frame=0, time=0, value=.2):
    return VisualViewSample(view.source_id, frame, np.full((4, 3), value), [1] * 4,
                            SampleTime(time, time + 100), view.sha256, "declared_sensor_model")


def observation(seq, time, samples=(), episode="a"):
    return ActorObservation(episode, seq, time, time + 100, (), (), views=tuple(samples))


def packer():
    return MultiViewPacker(maps(), BodyOrganEncoder([], rotation_scale=1), max_view_age=.2)


def model(runtime, **kwargs):
    graph = {"ids": np.array(list(map(str, range(10)))), "types": np.array(["n"] * 10),
             "roles": np.array(["input"] * 5 + ["internal"] + ["output"] * 4),
             "pre": np.array([0, 1, 2, 3, 4, 5, 5, 5, 5]),
             "post": np.array([5] * 5 + [6, 7, 8, 9]),
             "weights": np.array([1.] * 5 + [2., 1., 1., 1.])}
    return MultiViewCircuitController(graph, ["0", "1", "2", "3"], {
        "body": {"body_ids": ["4"], "feature_indices": [0]}},
        np.stack([v.directions_body for v in runtime.views]), np.ones((4, 4)), [0, 1, 2, 3],
        [1] * 10, np.bincount(graph["post"], weights=graph["weights"], minlength=10),
        {"steering": ["6", "7"], "forward": ["8"], "halt": ["9"]}, [-1, 1],
        runtime.calibration, neighbors=2, **kwargs)


def test_direct_projection_preserves_source_identity_and_optical_calibration():
    for i, view in enumerate(maps()):
        image = np.full((21, 31, 3), (i + 1) / 5)
        encoded, coverage = view.project(image, frame_id=0, time=SampleTime(0, 100),
                                         provenance="declared_sensor_model")
        assert encoded.source_id == view.source_id and encoded.calibration_sha256 == view.sha256
        assert encoded.valid.all() and (coverage > .99).all()
        np.testing.assert_allclose(encoded.response, image[0, 0][None] * np.ones((4, 1)))
    assert len({v.sha256 for v in maps()}) == 4


def test_independent_exposures_reuse_ages_and_gap_do_not_fabricate_motion():
    runtime = packer()
    front, rear = runtime.views[0], runtime.views[3]
    a, b = sample(front), sample(rear)
    first = runtime.pack(observation(0, 0, [a, b]))
    assert not first["contrast_valid"].any()
    changed = sample(front, 1, .1, .4)
    second = runtime.pack(observation(1, .1, [changed, b]))
    assert second["contrast_valid"][0].all() and not second["contrast_valid"][3].any()
    np.testing.assert_allclose(second["view_contrast"][0], np.tanh(.2))
    held = runtime.pack(observation(2, .15, [changed, b]))
    np.testing.assert_array_equal(held["view_contrast"], second["view_contrast"])
    np.testing.assert_allclose(held["view_age"][[0, 3]], [.05, .15])
    missing = runtime.pack(observation(3, .2))
    assert not missing["view_valid"].any()
    returned = runtime.pack(observation(4, .25, [changed]))
    assert returned["view_valid"][0].all() and not returned["contrast_valid"].any()
    stale = runtime.pack(observation(5, .5, [changed]))
    assert not stale["view_valid"].any() and not stale["view_contrast"].any()
    with pytest.raises(ValueError, match="backwards"):
        runtime.pack(observation(6, .6, [a]))


def test_bad_sources_or_changed_reused_frames_reject_before_committing_history():
    runtime = packer()
    a = sample(runtime.views[0])
    runtime.pack(observation(0, 0, [a]))
    with pytest.raises(ValueError, match="Reused"):
        runtime.pack(observation(1, .1, [sample(runtime.views[0], value=.8)]))
    good = runtime.pack(observation(1, .1, [sample(runtime.views[0], 1, .1, .4)]))
    assert good["contrast_valid"][0].all()
    changed = VisualViewSample("front", 2, np.zeros((4, 3)), [1] * 4, SampleTime(.2, 100.2),
                               "f" * 64, "declared_sensor_model")
    with pytest.raises(ValueError, match="calibration"):
        runtime.pack(observation(2, .2, [changed]))
    with pytest.raises(ValueError, match="Duplicate"):
        observation(2, .2, [a, a])
    with pytest.raises(ValueError, match="future"):
        observation(2, .2, [sample(runtime.views[0], 2, .3)])


def test_each_local_adapter_reads_one_view_and_fusion_occurs_in_internal_state():
    runtime = packer()
    net = model(runtime)
    with torch.no_grad():
        net.adapter.visual_adapter.weight.fill_(.1)
        for parameter in net.adapter.organ_adapters.parameters():
            parameter.zero_()
    initial = runtime.pack(observation(0, 0, [sample(v) for v in runtime.views]))
    values = {k: torch.tensor(v)[None] for k, v in initial.items()}
    drive = net.prepare_drive(values)
    altered = {k: v.clone() for k, v in values.items()}
    altered["view_rgb"][:, 2] += .2
    delta = net.prepare_drive(altered) - drive
    assert delta[0, 2] != 0 and not delta[:, [0, 1, 3, 4, 5, 6, 7, 8, 9]].any()

    def state_with(sources):
        obs = {k: v.clone() for k, v in values.items()}
        obs["view_valid"].zero_()
        obs["view_valid"][:, sources] = 1
        state = net.initial_state()
        injection = net.prepare_drive(obs)
        for _ in range(60):
            _, state = net.advance(injection, state)
        return state

    a, b, together = state_with([0]), state_with([1]), state_with([0, 1])
    assert together[0, 5] > 0
    assert not torch.isclose(together[0, 5], a[0, 5] + b[0, 5], atol=1e-6)
    # The interaction is already present at internal node 5, before the motor readout.
    rgb = values["view_rgb"].clone().requires_grad_()
    values["view_rgb"] = rgb
    state = net.initial_state()
    for _ in range(8):
        action, state = net(values, state)
    action[:, 1:].sum().backward()
    assert all(rgb.grad[:, i].abs().sum() > 0 for i in range(4))
    missing = {k: v.detach().clone() for k, v in values.items()}
    missing["view_valid"].zero_()
    assert not net.prepare_drive(missing).any()
    with pytest.raises(ValueError, match="actor"):
        net.prepare_drive({**missing, "map_progress": torch.ones(1)})


def test_runner_reset_and_checkpoint_restore_preserve_independent_source_state(tmp_path):
    runtime = packer()
    net = model(runtime)
    runner = MultiViewCircuitRunner(net, runtime)
    first, _ = runner.step(observation(0, 0, [sample(v) for v in runtime.views]))
    state = runner.state.clone()
    runner.step(observation(1, .1, [sample(v, 1, .1, .3) for v in runtime.views]))
    assert not torch.equal(state, runner.state)
    reset, _ = runner.step(observation(0, 0, [sample(v) for v in runtime.views], episode="b"))
    np.testing.assert_array_equal(first, reset)
    path = tmp_path / "multiview.pt"
    torch.save(net.state_dict(), path)
    restored = model(packer())
    restored.load_state_dict(torch.load(path, weights_only=True))
    packed = runtime.pack(observation(1, .1, [sample(v, 1, .1, .3) for v in runtime.views], "b"))
    tensors = {k: torch.tensor(v)[None] for k, v in packed.items()}
    a, b = net.initial_state(), restored.initial_state()
    for _ in range(12):
        x, a = net(tensors, a)
        y, b = restored(tensors, b)
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    changed = copy.deepcopy(net)
    changed.sensory_calibration["view_order"].reverse()
    with pytest.raises(ValueError, match="calibrations"):
        MultiViewCircuitRunner(changed, packer())


def test_frozen_photometry_uses_training_rays_without_missing_view_drive(tmp_path):
    from flybrain_drive.multiview_learning import calibrate_photometry

    runtime = packer()
    net = model(runtime, photometric_normalization=True)
    observations = [runtime.pack(observation(i, i * .1,
                    [sample(v, i, i * .1, value) for v in runtime.views]))
                    for i, value in enumerate((.2, .4))]
    episodes = [{"observation": {k: torch.tensor(np.stack([o[k] for o in observations]))
                                  for k in observations[0]}}]
    record = calibrate_photometry(net, episodes, rgb_floor=.03, contrast_floor=.1)
    assert not record["teacher_labels_used"]
    torch.testing.assert_close(net.photometric_mean[..., :3], torch.full((4, 4, 3), .3))
    with torch.no_grad():
        net.adapter.visual_adapter.weight.fill_(1)
    centered = {k: torch.tensor(v)[None] for k, v in observations[0].items()}
    centered["view_rgb"].fill_(.3)
    centered["contrast_valid"].zero_()
    torch.testing.assert_close(net.prepare_drive(centered), torch.zeros(1, 10), atol=1e-5, rtol=0)
    centered["view_rgb"][:, 2] += .1
    drive = net.prepare_drive(centered)
    assert drive[0, 2] > 0
    torch.testing.assert_close(drive[:, [0, 1, 3]], torch.zeros(1, 3), atol=1e-5, rtol=0)
    centered["view_valid"].zero_()
    assert net.prepare_drive(centered).count_nonzero() == 0
    path = tmp_path / "normalized.pt"
    torch.save(net.state_dict(), path)
    restored = model(packer(), photometric_normalization=True)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.prepare_drive(centered), net.prepare_drive(centered))
    with torch.no_grad():
        restored.photometric_scale.zero_()
    with pytest.raises(ValueError, match="photometric calibration"):
        restored.prepare_drive(centered)


def test_cached_sparse_backend_matches_state_gradients_and_invalidates_after_cut():
    runtime = packer()
    old = model(runtime)
    fast = model(runtime, sparse_backend="cached_csr_cpu")
    fast.load_state_dict(old.state_dict())
    packed = runtime.pack(observation(0, 0, [sample(v) for v in runtime.views]))
    tensors = {k: torch.tensor(v)[None] for k, v in packed.items()}
    outcomes = []
    for net in (old, fast):
        initial = net.initial_state().requires_grad_()
        state = initial
        for _ in range(12):
            action, state = net(tensors, state)
        action.sum().backward()
        outcomes.append((state.detach(), initial.grad, net.adapter.visual_adapter.weight.grad))
    for a, b in zip(outcomes[0], outcomes[1], strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    assert not any("cached" in key for key in fast.state_dict())
    before = fast.circuit._cache_key
    fast.circuit.matrix = fast.circuit.matrix * 0
    fast(tensors, fast.initial_state())
    assert before != fast.circuit._cache_key
    assert fast.circuit._cached_matrix.values().count_nonzero() == 0
    fast.load_state_dict(old.state_dict())
    fast(tensors, fast.initial_state())
    assert fast.circuit._cached_matrix.values().count_nonzero() > 0
    fast.circuit.matrix._values().zero_()
    fast(tensors, fast.initial_state())
    assert fast.circuit._cached_transpose.values().count_nonzero() == 0
    assert old.circuit.matrix.values().count_nonzero() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_chunk_validation_preserves_outputs_gradients_and_live_checks(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    from flybrain_drive.multiview_learning import replay_chunk
    runtime = packer()
    first = runtime.pack(observation(0, 0, [sample(v) for v in runtime.views]))
    obs = {k: torch.as_tensor(np.stack([v, v, v])[:, None], device=device,
                             dtype=torch.float32) for k, v in first.items()}
    regular = model(runtime).to(device)
    fast = copy.deepcopy(regular)
    fast.chunk_validated_training = True
    active = torch.tensor([[1.], [1.], [0.]], device=device)
    results = []
    for network in (regular, fast):
        initial = network.initial_state(1) + .1
        pred, state = replay_chunk(network, obs, active, initial, 4)
        (pred.square().sum() + state.square().sum()).backward()
        results.append((pred, state, {k: p.grad for k, p in network.named_parameters()}))
    for a, b in zip(results[0][:2], results[1][:2], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for k, grad in results[0][2].items():
        if grad is not None:
            torch.testing.assert_close(grad, results[1][2][k], rtol=0, atol=0)
    bad = fast.initial_state(1)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        fast.advance(torch.zeros_like(bad), bad)
    with pytest.raises(ValueError):
        fast.validate_training_chunk(torch.zeros((3, 1, 10), device=device), bad)
    with torch.no_grad():
        fast.circuit.alpha.fill_(1.1)
    with pytest.raises(ValueError):
        fast.validate_training_chunk(torch.zeros((3, 1, 10), device=device),
                                     fast.initial_state(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_fixed_gpu_csr_forward_adjoint_and_cache_invalidation():
    runtime = packer()
    networks = [model(runtime, sparse_backend=b).cuda()
                for b in ("coo", "cached_csr_gpu")]
    networks[1].load_state_dict(networks[0].state_dict())
    results = []
    for network in networks:
        drive = torch.zeros(2, 10, device="cuda")
        drive[:, :5] = .3
        drive.requires_grad_(True)
        state = network.initial_state(2)
        for _ in range(8):
            prediction, state = network.advance(drive, state)
        prediction.sum().backward()
        results.append((prediction.detach(), state.detach(), drive.grad))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    fast = networks[1]
    before = fast.circuit._cached_matrix
    with torch.no_grad():
        fast.circuit.matrix.mul_(.5)
        fast.advance(torch.zeros_like(drive), fast.initial_state(2))
    assert fast.circuit._cached_matrix is not before
    assert not any("cached" in k for k in fast.state_dict())
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_policy_anchor_copies_warmed_gpu_sparse_model_without_changing_learner():
    from flybrain_drive.multiview_learning import episode_chunk, replay_chunk
    from flybrain_drive.policy_anchor import prepare_policy_anchors, replay_anchor_mse

    runtime = packer()
    packed = runtime.pack(observation(0, 0, [sample(v) for v in runtime.views]))
    network = model(runtime, sparse_backend='cached_csr_gpu').cuda()
    network.chunk_validated_training = True
    data = dict(source={'sha256': 'a'*64},
        observation={k: torch.tensor(np.stack([v]*3), device='cuda', dtype=torch.float32)
                     for k, v in packed.items()}, action=torch.zeros(3, 3, device='cuda'),
        action_valid=torch.ones(3, 3, device='cuda', dtype=torch.bool))
    obs, _, _, active = episode_chunk([data], 0, 2)
    replay_chunk(network, obs, active, network.initial_state(1), 4)
    assert network.circuit._cached_matrix is not None
    targets = prepare_policy_anchors(network, [data], ['a'*64], 2, 4)
    assert replay_anchor_mse(network, [data], targets, 2, 4) == 0.
    assert any(p.requires_grad for p in network.parameters())
    from flybrain_drive.multiview_learning import replay_metrics
    from flybrain_drive.policy_anchor import policy_anchor_loss
    with torch.no_grad():
        network.adapter.visual_adapter.weight.mul_(1.1)
    expected = replay_anchor_mse(network, [data], targets, 2, 4)
    metrics = replay_metrics(network, [data], 2, 4,
        chunk_metric=lambda prediction, batch, offset:
            policy_anchor_loss(prediction, batch, offset, targets))
    assert metrics['extra_chunk_metric'] == pytest.approx(expected)
    torch.cuda.synchronize()
