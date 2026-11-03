"""Synthetic sensor-contract tests, not native AC or biological validation."""

import numpy as np
import pytest
import torch

from flybrain_drive.body_organs import BodyOrganEncoder, PositionOrganSpec, ScalarOrganSpec
from flybrain_drive.compound_eye import synthetic_eye
from flybrain_drive.modality_ports import ModalityPortAdapter
from flybrain_drive.policy import RecurrentPolicy, SparseCircuit
from flybrain_drive.sensor_contract import ActorObservation, BodySample, SampleTime
from flybrain_drive.sensory_policy import ObservationPacker, SensoryPolicyRunner


def specs(temperature_width=15):
    return [ScalarOrganSpec("tyre_core_temperature", 273.15, 433.15, temperature_width, 10),
            ScalarOrganSpec("wheel_normal_load", 0, 10000, 1000, 10000)]


def observation(seq, time, *, temperature=(353.15,) * 4, load=(2000,) * 4,
                valid=(True,) * 4, stamp=None, episode="synthetic"):
    stamp = time if stamp is None else stamp
    body = []
    for name, values, unit, frame in (
        ("tyre_core_temperature", temperature, "K", "tyre_core"),
        ("wheel_normal_load", load, "N", "contact_normal_compression_positive"),
    ):
        if values is not None:
            body.append(BodySample(name, values, valid, SampleTime(stamp, 10 + stamp),
                                   unit, frame, "declared_sensor_model"))
    return ActorObservation(episode, seq, time, 10 + time, (), tuple(body))


def packer(organs):
    eyes = [synthetic_eye(side, rows=2, columns=3) for side in ("left", "right")]
    return ObservationPacker(eyes, organs, specific_force_scale=20, wheel_speed_scale=50)


def test_four_wheel_levels_and_signed_changes_preserve_missingness_and_history():
    enc = BodyOrganEncoder([], scalars=specs(), max_sample_age=.2)
    initial = enc.encode(observation(0, 0, load=(0, 2000, 2000, 2000)))
    assert len(initial.values) == enc.feature_count == 72
    values, valid = initial.values.reshape(8, 9), initial.valid.reshape(8, 9)
    assert values[4, 0] == 1 and valid[4, 0]  # Zero contact load is a valid reading.
    assert not valid[:, -2:].any()  # No invented initial derivative.
    moved = enc.encode(observation(1, .1, temperature=(354.15, 352.15, 353.15, 353.15),
                                    load=(1000, 1000, 2000, 2000)))
    rates = moved.values.reshape(8, 9)[:, -2:]
    assert rates[0, 0] > 0 and rates[0, 1] == 0
    assert rates[1, 1] > 0 and rates[1, 0] == 0
    assert rates[4, 0] > 0 and rates[5, 1] > 0
    held = enc.encode(observation(2, .15, stamp=.1,
                                  temperature=(354.15, 352.15, 353.15, 353.15),
                                  load=(1000, 1000, 2000, 2000)))
    np.testing.assert_array_equal(held.values, moved.values)
    missing = enc.encode(observation(3, .2, temperature=None, load=None))
    assert not missing.valid.any() and not missing.values.any()
    returned = enc.encode(observation(4, .3, valid=(True, False, True, True)))
    assert not returned.valid.reshape(8, 9)[:, -2:].any()
    assert not returned.values.reshape(8, 9)[[1, 5]].any()
    stale = enc.encode(observation(5, .7, stamp=.3, valid=(True, False, True, True)))
    assert not stale.valid.any()
    reset = enc.encode(observation(0, 0, episode="new-synthetic"))
    assert not reset.valid.reshape(8, 9)[:, -2:].any()


def test_changed_sample_reuse_rejects_before_mutating_any_sensor_history():
    enc = BodyOrganEncoder([], scalars=specs())
    initial = enc.encode(observation(0, 0))
    with pytest.raises(ValueError, match="Reused sample"):
        enc.encode(observation(1, .1, stamp=0, load=(3000,) * 4))
    recovered = enc.encode(observation(1, .1, stamp=0))
    np.testing.assert_array_equal(initial.values, recovered.values)


def test_packer_and_policy_seal_tuning_as_well_as_width_and_keep_legacy_layout():
    position = [PositionOrganSpec("steering_angle", 1, -1, 1, .2, 1)]
    legacy = BodyOrganEncoder(position, rotation_scale=5)
    empty = BodyOrganEncoder(position, rotation_scale=5, scalars=[])
    np.testing.assert_array_equal(legacy.encode(observation(0, 0)).values,
                                  empty.encode(observation(0, 0)).values)
    assert legacy.calibration_sha256 is None
    organs = BodyOrganEncoder(position, rotation_scale=5, scalars=specs())
    runtime = packer(organs)
    policy = RecurrentPolicy(kind="gru", sensory_encoder={
        "organ_size": runtime.organ_size,
        "eye_map_hashes": {side: eye.sha256 for side, eye in runtime.eyes.items()},
        "organ_calibration_sha256": organs.calibration_sha256,
    })
    runner = SensoryPolicyRunner(policy, runtime)
    action, packed = runner.step(observation(0, 0))
    assert np.isfinite(action).all() and packed["organs"].shape == (87,)
    changed = packer(BodyOrganEncoder(position, rotation_scale=5, scalars=specs(20)))
    assert changed.organ_size == runtime.organ_size
    with pytest.raises(ValueError, match="contract"):
        SensoryPolicyRunner(policy, changed)
    with pytest.raises(ValueError, match="contract"):
        SensoryPolicyRunner(policy, packer(legacy))


def test_thermal_and_load_can_reach_disjoint_recurrent_ports_without_motor_bypass():
    # Deliberately synthetic IDs and connectivity. No measured port assignment.
    graph = {"ids": np.array(["eye", "thermal", "load", "relay", "output"]),
             "types": np.array(["v", "t", "l", "n", "o"]),
             "roles": np.array(["input", "input", "input", "internal", "output"]),
             "pre": np.array([0, 1, 2, 3]), "post": np.array([3, 3, 3, 4]),
             "weights": np.ones(4, np.float32)}
    enc = BodyOrganEncoder([], scalars=specs())
    adapter = ModalityPortAdapter(graph, ["eye"], {
        "thermal": {"body_ids": ["thermal"], "feature_indices": list(range(36))},
        "load": {"body_ids": ["load"], "feature_indices": list(range(36, 72))},
    }, organ_size=enc.feature_count, visual_size=1)
    circuit = SparseCircuit(graph, .01)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.fill_(.01)

    def drive(obs):
        encoded = enc.encode(obs)
        return adapter(torch.zeros(1, 1), torch.zeros(1, 1),
                       torch.tensor(encoded.values.copy(), dtype=torch.float32)[None],
                       torch.tensor(encoded.valid.copy(), dtype=torch.float32)[None])

    baseline = drive(observation(0, 0))
    missing = drive(observation(1, .1, temperature=None))
    assert missing[0, 1] == 0 and missing[0, 2] > 0
    assert not baseline[:, [0, 3, 4]].any()
    state = torch.zeros(1, 5)
    for _ in range(4):
        output, state = circuit.step_with_drive(baseline, state)
    assert output.abs().sum() > 0
    output.sum().backward()
    assert all(layer.weight.grad.abs().sum() > 0 for layer in adapter.organ_adapters)


@pytest.mark.parametrize("kwargs", [
    {"sensor": "route_progress"}, {"minimum": -1}, {"change_scale": 0},
    {"populations": 1}, {"tuning_width": float("nan")},
])
def test_invalid_scalar_calibration_is_rejected(kwargs):
    settings = dict(sensor="wheel_normal_load", minimum=0, maximum=10000,
                    tuning_width=1000, change_scale=10000)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        ScalarOrganSpec(**settings)
