import pytest

from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler, bracket_render, diagnostic_profile


def test_brake_profile_reserves_owned_camera_and_child_deadline():
    profile = diagnostic_profile('brake30')
    assert profile['command'] == 'expert-brake-pulse30'
    assert profile['steer'] == .9
    assert profile['camera_deadline'] >= 30 + 5 + 8
    assert profile['frames'] * .08 >= 30 + 5
    assert profile['child_deadline'] > 30 + 7
    assert profile['maximum_samples'] * .002 >= profile['camera_deadline']
    assert diagnostic_profile('motion3')['command'] == 'expert-motion-probe'
from tests.test_ac_teacher_physics import Mapping, packet


def row(packet, start, end):
    return {'authority': {'packet_id': packet},
            'copy': {'first_copy_start': start, 'second_copy_end': end}}


def test_brackets_render_only_with_fully_prior_and_fully_later_copies():
    samples = [row(1, 1., 1.1), row(2, 1.2, 1.4), row(3, 1.5, 1.6)]
    actual = bracket_render(samples, 1.25, 1.45)
    assert actual['status'] == 'bracketed_host_render_call'
    assert actual['earlier_packet'] == 1 and actual['later_packet'] == 3
    assert actual['copy_gap_seconds'] == pytest.approx(.4)


def test_missing_or_unordered_copy_is_not_silently_bracketed():
    samples = [row(1, 1., 1.1), row(2, 1.2, 1.3)]
    assert bracket_render(samples, .5, .8)['status'] == 'missing_host_bracket'
    with pytest.raises(ValueError, match='increasing'):
        bracket_render(list(reversed(samples)), 1.11, 1.15)


def test_sampler_is_ready_before_consumer_and_releases_mapping():
    class ContextMapping(Mapping):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    mapping = ContextMapping([packet(active=0)])
    sampler = NativeCopySampler(lambda: mapping)
    sampler.start()
    assert sampler.rows and sampler.rows[0]['authority']['packet_id'] == 7
    sampler.stop()
    assert mapping.closed and not sampler.thread.is_alive() and sampler.error is None
