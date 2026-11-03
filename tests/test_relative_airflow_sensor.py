import numpy as np
import pytest

from flybrain_drive.relative_airflow_sensor import missing_airflow, relative_airflow


def observe(**changes):
    data = dict(wind_world_m_s=[0., 2., 0.], vehicle_world_m_s=[10., 0., 0.],
                body_axes_world=np.eye(3), wind_receipt=10., body_receipt=10.,
                decision=10.02, source_verified=True)
    data.update(changes)
    return relative_airflow(**data)


def test_world_air_minus_vehicle_projects_to_body_axes():
    result = observe()
    np.testing.assert_array_equal(result.velocity_body_m_s, [-10., 2., 0.])
    assert result.valid.all()
    rotated = observe(body_axes_world=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(rotated.velocity_body_m_s, [2., 10., 0.])
    assert not hasattr(result, 'heading') and not result.velocity_body_m_s.flags.writeable


def test_unverified_or_stale_sources_remain_missing_without_default_wind():
    for result in (observe(source_verified=False), observe(decision=10.06),
                   observe(wind_receipt=9.98)):
        assert not result.valid.any()
        np.testing.assert_array_equal(result.velocity_body_m_s, [0., 0., 0.])
    with pytest.raises(ValueError, match='Future'):
        observe(wind_receipt=10.03)
    with pytest.raises(ValueError, match='right-handed'):
        observe(body_axes_world=[[1, 0, 0], [0, -1, 0], [0, 0, 1]])


def test_missing_airflow_needs_no_fabricated_source_and_cannot_be_reused_as_fresh():
    sample = missing_airflow(10.02)
    assert sample.source_verified is False and sample.decision_time == 10.02
    assert not sample.valid.any() and not sample.valid.flags.writeable
    assert not sample.velocity_body_m_s.flags.writeable
    np.testing.assert_array_equal(sample.velocity_body_m_s, [0, 0, 0])
    for time in (True, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='decision time'):
            missing_airflow(time)


def test_validity_survives_sensor_to_flight_adapter_boundary():
    import torch

    from flybrain_drive.flight_airflow_feedback import RelativeAirflowAdapter

    adapter = RelativeAirflowAdapter(2)
    with torch.no_grad():
        adapter.projection.weight.fill_(1.)
    missing = observe(source_verified=False)
    present = observe()
    def project(sample):
        values = torch.tensor(sample.velocity_body_m_s.copy())[None]
        valid = torch.tensor(sample.valid.copy())[None]
        return adapter(values, valid)
    assert torch.equal(project(missing), torch.zeros(1, 2))
    assert torch.any(project(present) != 0)


def test_finite_inputs_cannot_overflow_into_a_valid_airflow():
    with pytest.raises(ValueError, match='representable'):
        observe(wind_world_m_s=[1e100, 0, 0])
    with pytest.raises(ValueError, match='representable'):
        observe(wind_world_m_s=[1e308, 0, 0], vehicle_world_m_s=[-1e308, 0, 0])
    missing = observe(wind_world_m_s=[1e308, 0, 0], source_verified=False)
    assert not missing.valid.any()


def test_joint_boundary_rejects_reused_or_unverified_valid_airflow():
    from types import SimpleNamespace

    import torch

    from flybrain_drive.flight_airflow_feedback import RelativeAirflowAdapter
    from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
    from flybrain_drive.relative_airflow_sensor import AirflowObservation

    def capture(*args):
        return args[-2:]

    joint = SimpleNamespace(airflow=RelativeAirflowAdapter(2), drive=capture)
    body = torch.zeros((1, 21))
    sample = observe()

    def send(observation, time):
        return ExperimentalVisualBodyAirflow.drive_observed(
            joint, None, None, body, None, None, None, None, [observation], [time]
        )

    value, valid = send(sample, 10.02)
    torch.testing.assert_close(value, torch.tensor([[-10., 2., 0.]]))
    assert valid.all()
    with pytest.raises(ValueError, match='exact decision'):
        send(sample, 10.03)
    stale = observe(decision=10.06)
    value, valid = send(stale, 10.06)
    assert not valid.any() and not value.any()
    missing = missing_airflow(10.06)
    missing_value, missing_valid = send(missing, 10.06)
    torch.testing.assert_close(missing_value, value)
    assert torch.equal(missing_valid, valid)
    with pytest.raises(ValueError, match='exact decision'):
        send(missing, 10.07)
    forged = AirflowObservation(sample.velocity_body_m_s, sample.valid, False, 10.02)
    with pytest.raises(ValueError, match='source validity'):
        send(forged, 10.02)
