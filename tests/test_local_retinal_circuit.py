import copy
import json

import numpy as np
import pytest
import torch

from flybrain_drive.local_retinal_circuit import (
    FrozenRateCircuit,
    LocalRetinalController,
    LocalRetinalLinear,
)
from flybrain_drive.local_retinal_temporal import (
    episode_chunk,
    load_teacher_sequences,
    replay_chunk,
)
from flybrain_drive.policy import SparseCircuit
from flybrain_drive.sensory_policy import SensoryFeatureEncoder
from scripts.evaluate_local_retinal_controller import (
    apply_vision_condition,
    read_training_calibration,
)


def graph():
    return {"ids": np.array(["1", "2", "3", "4", "5", "6"]),
            "types": np.array(["MeTu", "MeTu", "EPG", "DN", "organ", "DN"]),
            "roles": np.array(["input", "input", "internal", "output", "input", "output"]),
            "pre": np.array([0, 1, 2, 4, 2]), "post": np.array([2, 2, 3, 3, 5]),
            "weights": np.array([2., 1., 4., 1., 3.], dtype=np.float32)}


def eyes():
    a = np.linspace(-.3, .3, 4)
    return np.stack([np.column_stack([np.cos(a), np.sin(a), a * 0])] * 2).astype(np.float32)


def model():
    return LocalRetinalController(graph(), ["1", "2"],
        {"position": {"body_ids": ["5"], "feature_indices": [0]}}, eyes(), np.ones((2, 4)), 1,
        output_signs=[-1, 1], neighbors=2)


def observation(batch=1):
    result = {key: torch.zeros(batch, 1) for key in SensoryFeatureEncoder.fields}
    result.update(ray_values=torch.rand(batch, 2, 4, 3), ray_valid=torch.ones(batch, 2, 4),
                  ray_directions=torch.tensor(eyes()).expand(batch, -1, -1, -1),
                  organs=torch.ones(batch, 1), organ_valid=torch.ones(batch, 1))
    return result


def test_fixed_sparse_dynamics_match_reference_and_preserve_state():
    frozen, reference = FrozenRateCircuit(graph(), .01), SparseCircuit(graph(), .01)
    state, other = torch.zeros(2, 6), torch.zeros(2, 6)
    drive = torch.zeros_like(state)
    drive[:, 0] = .3
    for _ in range(25):
        out, state = frozen(drive, state)
        expected, other = reference.step_with_drive(drive, other)
        torch.testing.assert_close(out, expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(state, other, atol=1e-7, rtol=1e-6)
    assert state[:, 3].abs().min() > 0
    invalid = drive.clone()
    invalid[:, 2] = 1
    with pytest.raises(ValueError, match="input ports"):
        frozen(invalid, state)


def test_full_source_normalization_preserves_missing_input_mass():
    net = FrozenRateCircuit(graph(), .01, incoming_totals=[0, 0, 6, 10, 0, 3])
    rows = net.matrix.to_dense().sum(1)
    torch.testing.assert_close(rows, torch.tensor([0., 0, .5, .5, 0, 1]))
    for invalid in ([0, 0, 2, 10, 0, 3], [0, 0, np.nan, 10, 0, 3], [1]):
        with pytest.raises(ValueError, match="incoming totals"):
            FrozenRateCircuit(graph(), .01, incoming_totals=invalid)


def test_local_features_cannot_use_pixels_outside_the_declared_neighborhood():
    layer = LocalRetinalLinear(eyes(), np.ones((2, 4)), 1, 1, 42)
    flat = torch.zeros(1, 24, requires_grad=True)
    layer(flat).sum().backward()
    touched = torch.nonzero(flat.grad[0] != 0, as_tuple=True)[0]
    assert set(touched.tolist()) == set(layer.indices[0].tolist())
    same = LocalRetinalLinear(eyes(), np.ones((2, 4)), 1, 1, 42)
    torch.testing.assert_close(layer.indices, same.indices)


def test_signal_and_gradient_reach_outputs_only_through_the_circuit():
    net, obs = model(), observation(2)
    drive = net.prepare_drive(obs)
    assert not drive[:, ~net.circuit.input_mask].any()
    state = net.initial_state(2)
    for _ in range(25):
        steer, state = net.advance(drive, state)
    steer.square().mean().backward()
    assert net.adapter.visual_adapter.weight.grad.abs().sum() > 0
    assert not any(p.requires_grad for a in net.adapter.organ_adapters for p in a.parameters())
    cut = copy.deepcopy(net)
    cut.circuit.matrix = cut.circuit.matrix * 0
    state = cut.initial_state(2)
    for _ in range(25):
        steer, state = cut(obs, state)
    assert not steer.any()
    assert state[:, cut.adapter.visual_ports].abs().sum() > 0


def test_missing_eyes_cannot_be_replaced_by_normalization_or_oracle_fields():
    net, obs = model(), observation()
    net.feature_mean.fill_(.5)
    obs["ray_valid"].zero_()
    assert not net.prepare_drive(obs).any()
    obs["route_progress"] = torch.ones(1)
    with pytest.raises(ValueError, match="actor fields"):
        net.prepare_drive(obs)


def test_checkpoint_restores_identical_persistent_decisions(tmp_path):
    net, obs = model(), observation()
    path = tmp_path / "model.pt"
    torch.save(net.state_dict(), path)
    restored = model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    a, b = net.initial_state(), restored.initial_state()
    for _ in range(12):
        x, a = net(obs, a)
        y, b = restored(obs, b)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("mismatch", [None, "pose_hash", "configuration"])
def test_live_evaluation_rejects_different_optics_or_training_source(tmp_path, mismatch):
    from flybrain_drive.graph import sha256

    poses = tmp_path / "poses.npz"
    poses.write_bytes(b"test-pose-source")
    config = {"eye": {"sigma_degrees": 1}}
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"configuration": config}}))
    (tmp_path / "sensory-calibration.json").write_text(json.dumps({"eye_map_hashes": "test"}))
    payload = {"source_sha256": {"poses": sha256(poses)}}
    if mismatch == "pose_hash":
        poses.write_bytes(b"different-source")
    if mismatch == "configuration":
        config["eye"]["sigma_degrees"] = 2
    if mismatch:
        with pytest.raises(ValueError, match="differs from training"):
            read_training_calibration(config, payload, poses)
    else:
        assert read_training_calibration(config, payload, poses) == {"eye_map_hashes": "test"}


def test_temporal_chunk_detach_preserves_identical_numerical_state_and_predictions():
    net = model()
    frames = [observation(2) for _ in range(9)]
    sequence = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
    active = torch.ones(9, 2)
    expected, final = replay_chunk(net, sequence, active, net.initial_state(2))
    first, state = replay_chunk(net, {k: v[:4] for k, v in sequence.items()}, active[:4],
                                net.initial_state(2))
    second, state = replay_chunk(net, {k: v[4:] for k, v in sequence.items()}, active[4:],
                                 state.detach())
    torch.testing.assert_close(torch.cat([first, second]), expected, atol=0, rtol=0)
    torch.testing.assert_close(state, final, atol=0, rtol=0)
    assert state.abs().sum() > 0


def test_frozen_vision_retains_available_first_image_while_body_can_change():
    first = observation()
    expected = first["ray_values"].clone()
    _, frozen = apply_vision_condition(first, "vision_frozen")
    first["ray_values"].zero_()
    second = observation()
    second["organs"].fill_(.7)
    transformed, _ = apply_vision_condition(second, "vision_frozen", frozen)
    torch.testing.assert_close(transformed["ray_values"], expected, atol=0, rtol=0)
    assert transformed["ray_valid"].all() and transformed["organs"][0, 0] == .7


def test_missing_vision_ablation_masks_input_without_changing_body():
    packed = observation()
    expected = packed["organs"].clone()
    transformed, frozen = apply_vision_condition(packed, "vision_missing")
    assert frozen is None and not transformed["ray_valid"].any()
    torch.testing.assert_close(transformed["organs"], expected, atol=0, rtol=0)


def test_ended_episode_padding_never_advances_or_resets_its_state():
    net = model()
    episodes = []
    for length in (5, 2):
        obs = observation(length)
        episodes.append({"observation": obs, "label": torch.zeros(length)})
    first, _, active = episode_chunk(episodes, 0, 2)
    _, before = replay_chunk(net, first, active, net.initial_state(2))
    rest, _, active = episode_chunk(episodes, 2, 3)
    assert not active[:, 1].any()
    _, after = replay_chunk(net, rest, active, before)
    torch.testing.assert_close(after[1], before[1], atol=0, rtol=0)
    assert not torch.equal(after[0], before[0])


@pytest.mark.parametrize("tamper", [None, "hash", "actor_fields"])
def test_temporal_reader_requires_verified_actor_only_expert_data(tmp_path, tamper):
    from flybrain_drive.graph import sha256

    obs = observation(4)
    file = tmp_path / "episode.npz"
    np.savez(file, **{"obs_" + k: v.numpy() for k, v in obs.items()},
             action=np.zeros((4, 2)), teacher_selected=np.ones(4, bool),
             observation_time=np.arange(4) * .01, exposure_ids=np.zeros((4, 2)))
    calibration = {"test": "same"}
    manifest = {"config": {"policy_configuration": {"sensory_encoder": calibration}}}
    metadata = {"teacher_beta": 1, "student_state_sha256": None, "actor_fields": list(obs),
                "episodes": [{"file": file.name, "sha256": sha256(file), "steps": 4}],
                "decisions": 4}
    if tamper == "hash":
        file.write_bytes(b"changed")
    if tamper == "actor_fields":
        metadata["actor_fields"].append("route_progress")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "dataset.json").write_text(json.dumps(metadata))
    if tamper:
        with pytest.raises(ValueError, match="hash|Expert-only"):
            load_teacher_sequences(tmp_path, calibration)
    else:
        episodes, _, _ = load_teacher_sequences(tmp_path, calibration)
        assert len(episodes) == 1 and len(episodes[0]["label"]) == 4
