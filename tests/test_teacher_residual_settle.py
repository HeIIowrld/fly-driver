import copy

import pytest

from scripts.settle_ac_teacher_residual import validate_residual_pair


def fixture():
    state = dict(generation='same', epoch=0, paused=True, ai_controlled=False,
                 expert_probe_id='b'*32,
                 camera_active=False, expert_probe_active=False, is_online=False,
                 car_id='bmw_z4_gt3', track_id='monza', speed_kmh=.007, gas=0., brake=1.)
    first = dict(csp=state, csp_age=.1, body=dict(speedKmh=.149, gas=0., brake=1.),
        graphics_status=3, graphics_packet=100, released=True, background=True,
        native_ai=True, packet_id=12, prefix_sha256='same')
    second = copy.deepcopy(first)
    second['graphics_packet'] = 101
    terminal = dict(id='b'*32, status='api_returned_true', command='expert-motion-speed60',
                    stop_reason='owned motion probe deadline', after=copy.deepcopy(state))
    return first, second, terminal


def test_frozen_residual_allows_only_corrective_pause_resume_preflight():
    validate_residual_pair(*fixture())


@pytest.mark.parametrize('section,key,value', [
    ('csp', 'paused', False), ('csp', 'ai_controlled', True), ('csp', 'camera_active', True),
    ('csp', 'expert_probe_active', None), ('csp', 'is_online', True),
    ('csp', 'generation', 'other'), ('csp', 'gas', .1), ('csp', 'brake', .5),
    ('csp', 'expert_probe_id', 'c'*32),
    ('body', 'speedKmh', .2), ('body', 'speedKmh', .01), ('body', 'gas', .1),
    ('body', 'brake', .9), (None, 'graphics_packet', 100), (None, 'graphics_status', 2),
    (None, 'packet_id', 13), (None, 'prefix_sha256', 'changed'), (None, 'released', False),
    (None, 'background', False), (None, 'csp_age', 3.)])
def test_live_unknown_stale_or_moving_state_rejected(section, key, value):
    first, second, terminal = fixture()
    (second if section is None else second[section])[key] = value
    with pytest.raises(ValueError):
        validate_residual_pair(first, second, terminal)


@pytest.mark.parametrize('key,value', [('status', 'api_returned_false'),
    ('command', 'resume'), ('stop_reason', 'owned camera ended or became stale')])
def test_unproven_or_failed_teacher_never_qualifies(key, value):
    first, second, terminal = fixture()
    terminal[key] = value
    with pytest.raises(ValueError):
        validate_residual_pair(first, second, terminal)
