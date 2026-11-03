"""Synthetic observation times: do not invent latency when a target is already present."""

import pytest

from scripts.probe_ac_control_latency import summarize_steps


def sample(stamp, brake):
    return dict(source='native', before_perf=stamp-.001, after_perf=stamp,
                values=dict(brake=brake))


def test_first_target_observation_is_in_command_interval_and_baseline_must_differ():
    command = dict(index=0, axis='brake', target=.1,
                   before_perf=1., after_perf=1.001, end_perf=1.15)
    samples = [sample(.99, 1.), sample(1.01, 1.), sample(1.02, .1), sample(1.14, .1)]
    result = summarize_steps([command], samples)[0]['native']
    assert result['first_target_observed_ms'] == pytest.approx(20.)
    assert result['steady_median'] == .1 and result['target_differs_from_baseline']
    unchanged = summarize_steps([command], [sample(.99, .1), sample(1.1, .1)])[0]['native']
    assert unchanged['first_target_observed_ms'] is None
    assert not unchanged['target_differs_from_baseline']


def test_missing_or_unreached_target_cannot_be_assigned_a_delay():
    command = dict(index=0, axis='brake', target=.01,
                   before_perf=1., after_perf=1.001, end_perf=1.15)
    result = summarize_steps([command], [sample(.99, 1.), sample(1.1, 0.)])[0]
    assert result['native']['first_target_observed_ms'] is None
    assert result['sdk']['samples'] == 0 and result['sdk']['steady_median'] is None
