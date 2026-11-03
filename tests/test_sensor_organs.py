import numpy as np
import pytest

from flybrain_drive.body_organs import BodyOrganEncoder, PositionOrganSpec
from flybrain_drive.sensor_contract import (
    ActorObservation,
    BodySample,
    EyeSample,
    SampleTime,
    TrainingOnly,
)


def sample(value, time, valid=True, name="steering_angle"):
    unit, frame = ("rad", "roadwheel") if name == "steering_angle" else ("rad/s", "body")
    value = np.atleast_1d(value)
    return BodySample(name, value, np.full(value.shape, valid), SampleTime(time, time + 10),
                      unit, frame, "declared_sensor_model")


def observation(sequence, time, samples=(), episode="a"):
    return ActorObservation(episode, sequence, time, time + 10, (), tuple(samples))


def encoder():
    return BodyOrganEncoder([PositionOrganSpec("steering_angle", 1, -1, 1, .2, 1, 3)],
                            rotation_scale=2, max_sample_age=.2)


def test_future_and_privileged_inputs_are_rejected_before_actor_use():
    with pytest.raises(ValueError, match="future"):
        observation(0, 1, [sample(0, 2)])
    late = BodySample("steering_angle", [0], [1], SampleTime(0, 20),
                      "rad", "roadwheel", "instrumented")
    with pytest.raises(ValueError, match="not-yet"):
        observation(0, 1, [late])
    with pytest.raises(ValueError, match="privileged"):
        BodySample("route_progress", [0], [1], SampleTime(0, 0), "m", "world", "instrumented")
    with pytest.raises(ValueError, match="provenance"):
        BodySample("steering_angle", [0], [1], SampleTime(0, 0),
                   "rad", "roadwheel", "simulator_truth")
    with pytest.raises(ValueError, match="BodySample"):
        observation(0, 0, [TrainingOnly("a", 0, route_progress=123)])


def test_clock_domains_are_not_subtracted_and_sensor_storage_is_copied():
    source = np.array([.2])
    body = BodySample("steering_angle", source, [True], SampleTime(3, 100000),
                      "rad", "roadwheel", "instrumented")
    obs = ActorObservation("a", 0, 3.1, 100001, (), (body,))
    source[:] = 9
    np.testing.assert_allclose(body.values, [.2])
    assert obs.ages()["steering_angle"] == pytest.approx(.1)
    with pytest.raises(ValueError):
        body.values[0] = 2


def test_position_is_tonic_movement_direction_is_phasic_and_exposure_reuse_holds():
    enc = encoder()
    first = enc.encode(observation(0, 0, [sample(0, 0)]))
    assert first.values[1] == 1  # Preferred central position.
    assert not first.valid[3:5].any()  # No invented derivative at first sample.
    increased = enc.encode(observation(1, .1, [sample(.1, .1)]))
    assert increased.values[3] > 0 and increased.values[4] == 0
    held = enc.encode(observation(2, .15, [sample(.1, .1)]))
    np.testing.assert_array_equal(held.values, increased.values)
    still = enc.encode(observation(3, .2, [sample(.1, .2)]))
    np.testing.assert_array_equal(still.values[3:5], [0, 0])
    decreased = enc.encode(observation(4, .3, [sample(0, .3)]))
    assert decreased.values[3] == 0 and decreased.values[4] > 0


def test_missing_stale_invalid_and_episode_reset_do_not_create_movement():
    enc = encoder()
    enc.encode(observation(0, 0, [sample(0, 0)]))
    stale = enc.encode(observation(1, 1, [sample(0, 0)]))
    assert not stale.valid.any() and not stale.values.any()
    returned = enc.encode(observation(2, 1.1, [sample(.5, 1.1)]))
    assert not returned.valid[3:5].any()
    missing = enc.encode(observation(3, 1.2))
    assert not missing.valid.any()
    reset = enc.encode(observation(0, 0, [sample(-.5, 0)], episode="b"))
    assert not reset.valid[3:5].any()
    invalid = enc.encode(observation(1, .1, [sample(.9, .1, False)], episode="b"))
    assert not invalid.valid.any() and not invalid.values.any()


def test_rotation_channels_and_declared_dimensions_are_respected():
    enc = encoder()
    result = enc.encode(observation(0, 0, [sample([1, -2, 0], 0, name="angular_velocity")]))
    np.testing.assert_allclose(result.values[-6:], [np.tanh(.5), 0, 0, np.tanh(1), 0, 0])
    assert result.valid[-6:].all() and not result.valid[:5].any()
    with pytest.raises(ValueError, match="dimensions"):
        enc.encode(observation(1, .1, [sample([0, 0], .1)]))


def test_nonmonotonic_sequence_and_changed_reused_sample_are_rejected():
    enc = encoder()
    enc.encode(observation(0, 0, [sample(0, 0)]))
    with pytest.raises(ValueError, match="sequence"):
        enc.encode(observation(0, .1))
    with pytest.raises(ValueError, match="Reused"):
        enc.encode(observation(1, .1, [sample(.5, 0)]))


@pytest.mark.parametrize("valid", [[float("nan")], [2], ["False"]])
def test_invalid_validity_cannot_become_a_valid_measurement_by_boolean_cast(valid):
    with pytest.raises(ValueError, match="Validity"):
        BodySample("steering_angle", [0], valid, SampleTime(0, 0),
                   "rad", "roadwheel", "instrumented")


def test_eye_provenance_shapes_and_arrival_boundary():
    eye = EyeSample("left", 5, [[.5], [.7]], [1, 0], SampleTime(1, 11), "a" * 64)
    obs = ActorObservation("a", 0, 1.04, 11.1, (eye,), ())
    assert obs.ages()["eye_left"] == pytest.approx(.04)
    with pytest.raises(ValueError, match="SHA-256"):
        EyeSample("left", 5, [[.5]], [1], SampleTime(1, 11), "unreviewed")
    with pytest.raises(ValueError, match="Duplicate"):
        ActorObservation("a", 0, 1.04, 11.1, (eye, eye), ())
