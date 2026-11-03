import numpy as np
import pytest
import torch

from flybrain_drive.body_organs import BodyOrganEncoder, PositionOrganSpec
from flybrain_drive.compound_eye import synthetic_eye
from flybrain_drive.graph import demo_graph
from flybrain_drive.policy import RecurrentPolicy
from flybrain_drive.sensor_contract import ActorObservation, BodySample, EyeSample, SampleTime
from flybrain_drive.sensory_policy import ObservationPacker, SensoryPolicyRunner


def setup(kind="gru"):
    eyes = [synthetic_eye(side, rows=2, columns=3) for side in ("left", "right")]
    organs = BodyOrganEncoder([PositionOrganSpec("steering_angle", 1, -.5, .5, .15, 1, 3)],
                              rotation_scale=5)
    packer = ObservationPacker(eyes, organs, specific_force_scale=20, wheel_speed_scale=50)
    policy = RecurrentPolicy(kind=kind, graph=demo_graph() if kind == "graph" else None,
                             sensory_encoder={"organ_size": packer.organ_size,
                                              "eye_map_hashes": {e.side: e.sha256 for e in eyes}})
    return policy, packer, eyes


def observation(eyes, seq=0, episode="a", value=.3):
    stamp = SampleTime(seq*.01, 100+seq*.01)
    samples = tuple(EyeSample(e.side, seq, np.full((6, 3), value), np.ones(6), stamp, e.sha256)
                    for e in eyes)
    body = (BodySample("steering_angle", [.01*seq], [1], stamp,
                       "rad", "roadwheel", "declared_sensor_model"),)
    return ActorObservation(episode, seq, stamp.simulation_seconds,
                            stamp.received_monotonic_seconds, samples, body)


@pytest.mark.parametrize("kind", ["gru", "graph"])
def test_eye_organs_reach_recurrent_actor_and_reset_is_reproducible(kind):
    torch.manual_seed(1)
    policy, packer, eyes = setup(kind)
    runner = SensoryPolicyRunner(policy, packer)
    a, _ = runner.step(observation(eyes))
    runner.step(observation(eyes, 1))
    b, _ = runner.step(observation(eyes, 0, episode="b"))
    np.testing.assert_allclose(a, b, atol=1e-7)
    runner2 = SensoryPolicyRunner(policy, setup(kind)[1])
    reference = SensoryPolicyRunner(policy, setup(kind)[1])
    # Graph signals take recurrent steps to propagate from input to output ports.
    for i in range(8):
        changed, _ = runner2.step(observation(eyes, i, value=.8))
        unchanged, _ = reference.step(observation(eyes, i))
    assert np.linalg.norm(unchanged-changed) > 1e-6


def test_sequence_training_replays_exact_boundary_and_has_eye_and_body_gradients():
    policy, packer, eyes = setup()
    packed = [packer.pack(observation(eyes, i)) for i in range(3)]
    tensors = {k: torch.tensor(np.stack([p[k] for p in packed]))[:, None] for k in packed[0]}
    resets = torch.tensor([[1], [0], [0]], dtype=torch.bool)
    means, values, state = policy.sequence_batch(tensors, resets, return_state=True)
    hidden = policy.initial_state()
    for i in range(3):
        mean, _, hidden = policy({k: v[i] for k, v in tensors.items()}, hidden, resets[i])
        torch.testing.assert_close(mean, means[i])
    torch.testing.assert_close(state, hidden)
    (means.square().sum()+values.square().sum()).backward()
    assert policy.encoder.ray[0].weight.grad.abs().sum() > 0
    assert policy.encoder.fusion[0].weight.grad[:, 132:132+packer.organ_size].abs().sum() > 0


def test_map_mismatch_and_extra_privileged_tensors_are_rejected():
    policy, packer, eyes = setup()
    tensors = {k: torch.tensor(v)[None] for k, v in packer.pack(observation(eyes)).items()}
    with pytest.raises(ValueError, match="only"):
        policy({**tensors, "route_progress": torch.tensor([.5])}, policy.initial_state())
    wrong = list(packer.eyes.values())
    wrong[0] = synthetic_eye("left", rows=2, columns=3, sigma_degrees=2)
    mismatched = ObservationPacker(
        wrong, BodyOrganEncoder(packer.organs.positions, rotation_scale=5),
        specific_force_scale=20, wheel_speed_scale=50,
    )
    with pytest.raises(ValueError, match="match"):
        SensoryPolicyRunner(policy, mismatched)


def test_missing_eye_and_stale_samples_are_masked_without_inventing_new_exposures():
    _, packer, eyes = setup()
    first = observation(eyes)
    packer.pack(first)
    missing = ActorObservation("a", 1, .01, 100.01, (first.eyes[1],), ())
    packed = packer.pack(missing)
    assert not packed["ray_valid"][0].any() and not packed["ray_values"][0].any()
    assert packed["ray_valid"][1].all()
    returned = ActorObservation("a", 2, .02, 100.02, first.eyes, ())
    assert packer.pack(returned)["ray_valid"].all()
    stale = ActorObservation("a", 3, 1, 101, first.eyes, ())
    packed = packer.pack(stale)
    assert not packed["ray_valid"].any() and not packed["organ_valid"].any()


def test_optional_spatial_branch_preserves_grid_and_checkpoint_configuration():
    _, packer, eyes = setup()
    policy = RecurrentPolicy(sensory_encoder={
        "organ_size": packer.organ_size, "eye_map_hashes": {e.side: e.sha256 for e in eyes},
        "grid_shape": [2, 3],
    })
    packed = packer.pack(observation(eyes))
    tensors = {k: torch.tensor(v)[None] for k, v in packed.items()}
    mean, _, _ = policy(tensors, policy.initial_state())
    mean.square().sum().backward()
    assert policy.encoder.grid[0].weight.grad.abs().sum() > 0
    invalid = {**tensors, "ray_values": tensors["ray_values"][:, :, :5],
               "ray_directions": tensors["ray_directions"][:, :, :5],
               "ray_valid": tensors["ray_valid"][:, :, :5]}
    with pytest.raises(ValueError, match="Ray count"):
        policy(invalid, policy.initial_state())
