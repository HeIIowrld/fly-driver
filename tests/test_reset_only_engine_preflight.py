import copy

import pytest

from scripts.reset_ac_engine_framecap import validate_reset_only_pair
from scripts.reset_ac_engine_framecap import deploy_pinned_publisher
from flybrain_drive.graph import sha256


def pair():
    first = dict(csp=dict(generation='old', paused=True, ai_controlled=False,
        car_id='bmw_z4_gt3', track_id='monza', camera_active=False,
        expert_probe_active=False, is_online=False), csp_age_seconds=.1,
        native_ai=dict(builtin_ai_controlled=True, packet_id=10, prefix_sha256='same'),
        graphics=dict(status=3, packet_id=100), body=dict(speedKmh=0., gas=0., brake=1.),
        released_controller=True, background_engine=True)
    second = copy.deepcopy(first)
    second['graphics']['packet_id'] += 1
    return first, second


def test_paused_frozen_ambiguity_permits_only_reset_preflight():
    validate_reset_only_pair(*pair())


@pytest.mark.parametrize('failure', ['none', 'state', 'source', 'installed'])
def test_publisher_replacement_revalidates_state_and_hashes(tmp_path, failure):
    source, target, backup = (tmp_path/name for name in ('new.py', 'installed.py', 'backup.py'))
    source.write_bytes(b'new reviewed source')
    target.write_bytes(b'old reviewed installation')
    spec = dict(source=str(source), target=str(target), source_sha256=sha256(source),
                expected_installed_sha256=sha256(target))

    def validate():
        if failure == 'state':
            raise ValueError('Engine resumed')
        if failure == 'source':
            source.write_bytes(b'concurrent source change')
        if failure == 'installed':
            target.write_bytes(b'foreign installation change')

    if failure == 'none':
        result = deploy_pinned_publisher(spec, backup, validate)
        assert sha256(target) == result['installed_sha256'] == spec['source_sha256']
    else:
        with pytest.raises(ValueError):
            deploy_pinned_publisher(spec, backup, validate)
        assert target.read_bytes() == (b'foreign installation change' if failure == 'installed'
                                      else b'old reviewed installation')
    assert backup.read_bytes() == b'old reviewed installation'
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.parametrize('section,key,value', [
    ('graphics', 'status', 2), ('graphics', 'packet_id', 100),
    ('csp', 'paused', False), ('csp', 'ai_controlled', True),
    ('csp', 'is_online', True), ('csp', 'generation', 'other'),
    ('csp', 'camera_active', None), ('csp', 'expert_probe_active', None),
    ('csp', 'is_online', None), (None, 'released_controller', 1),
    ('native_ai', 'packet_id', 11), ('native_ai', 'prefix_sha256', 'changed'),
    ('native_ai', 'builtin_ai_controlled', False), ('body', 'gas', .1),
    ('body', 'speedKmh', 1.), ('body', 'brake', .9),
    (None, 'released_controller', False), (None, 'background_engine', False),
    (None, 'csp_age_seconds', 2.1),
])
def test_reset_rejects_live_stale_changing_foreign_or_nonstationary_state(section, key, value):
    first, second = pair()
    target = second if section is None else second[section]
    target[key] = value
    with pytest.raises(ValueError):
        validate_reset_only_pair(first, second)
