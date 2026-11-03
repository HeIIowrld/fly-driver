import copy
import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.ac_csp_retina import VIEW_ORDER, CSPCameraRetina


def retina():
    config = json.loads(Path('configs/ac-csp-retina.json').read_text(encoding='utf8'))
    config['native_retina'].update(receptors_per_view=24, columns=6)
    return CSPCameraRetina(config)


def samples(index=0, stamp=10.):
    return {name: dict(rgb=np.full((180, 320, 3), (i+1)*40, np.uint8),
        capture_id=index, generation='g', render_started_lower=stamp, received=stamp+.02)
        for i, name in enumerate(VIEW_ORDER)}


def body():
    return dict(suspensionTravel=[.02]*4, tyreCoreTemperature=[30]*4,
                wheelLoad=[3000]*4, wheelAngularSpeed=[0]*4)


def test_acquisition_precompute_is_pixel_only_and_preserves_decision_time_transduction():
    cold, prepared = retina(), retina()
    rng = np.random.default_rng(862837)
    for index in range(2):
        source = samples(index, 10+index*.08)
        for item in source.values():
            item['rgb'] = rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
            prepared.precompute_pixels(item['rgb'])
        if index == 0:
            assert prepared.last_decision is None and not prepared.history
            assert not prepared.body.scalar_history
        arguments = dict(decision_time=10.04+index*.08,
                         native_scalars=body(), native_receipt=10.03+index*.08)
        expected, actual = cold.pack(source, **arguments), prepared.pack(source, **arguments)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    assert prepared.pixel_cache_hits == 8 and prepared.pixel_cache_misses == 0
    stale = samples(2, 10.16)
    for item in stale.values():
        prepared.precompute_pixels(item['rgb'])
    result = prepared.pack(stale, decision_time=10.4)
    assert not result['view_valid'].any() and not result['contrast_valid'].any()


def test_preprocessing_cache_is_bounded_and_never_replaces_source_identity_checks():
    r = retina()
    for value in range(65):
        r.precompute_pixels(np.full((180, 320, 3), value, np.uint8))
    assert len(r.pixel_cache) <= 64
    source = samples()
    r.pack(source, decision_time=10.04)
    changed = copy.deepcopy(source)
    changed['front']['rgb'][:] = 1
    r.precompute_pixels(changed['front']['rgb'])
    with pytest.raises(ValueError, match='identity'):
        r.pack(changed, decision_time=10.05)


def test_independent_sources_gaps_and_conservative_age_without_pose_inputs():
    r = retina()
    first = r.pack(samples(), decision_time=10.04,
                   native_scalars=body(), native_receipt=10.03)
    np.testing.assert_allclose(first['view_rgb'][:, 0, 0], np.arange(1, 5)*40/255)
    assert first['view_valid'].sum() == 96 and not first['contrast_valid'].any()
    assert first['organ_valid'].sum() == 98  # stationary wheel levels are valid, rates absent
    np.testing.assert_allclose(np.linalg.norm(r.directions, axis=-1), 1)
    fresh = samples(1, 10.12)
    fresh['right'] = None
    fresh['rear']['render_started_lower'] = 10.04
    second = r.pack(fresh, decision_time=10.24,
                    native_scalars=body(), native_receipt=10.23)
    np.testing.assert_array_equal(second['view_valid'].sum(1), [24, 24, 0, 0])
    np.testing.assert_array_equal(second['contrast_valid'].sum(1), [24, 24, 0, 0])
    again = r.pack(samples(2, 10.25), decision_time=10.28,
                   native_scalars=body(), native_receipt=10.27)
    np.testing.assert_array_equal(again['contrast_valid'].sum(1), [24, 24, 0, 0])
    assert again['organ_valid'].sum() == 126
    bad = samples(3, 10.32)
    bad['front']['world_position'] = [1, 2, 3]
    with pytest.raises(ValueError, match='Only pixels'):
        r.pack(bad, decision_time=10.36)


def test_identity_generation_and_body_failure_roll_back_all_histories():
    r = retina()
    initial = samples()
    r.pack(initial, decision_time=10.04, native_scalars=body(), native_receipt=10.03)
    changed = copy.deepcopy(initial)
    changed['front']['rgb'][0, 0] = 0
    with pytest.raises(ValueError, match='identity'):
        r.pack(changed, decision_time=10.05)
    assert r.last_decision == 10.04
    next_samples = samples(1, 10.08)
    invalid_body = dict(body(), world_position=[1, 2, 3])
    with pytest.raises(ValueError, match='Only declared'):
        r.pack(next_samples, decision_time=10.12,
               native_scalars=invalid_body, native_receipt=10.11)
    assert r.last_decision == 10.04 and r.identities['front'][0] == 0
    new_generation = samples(1, 10.08)
    new_generation['rear']['generation'] = 'another'
    with pytest.raises(ValueError, match='producer changed'):
        r.pack(new_generation, decision_time=10.12)
    packed = r.pack(next_samples, decision_time=10.12,
                    native_scalars=body(), native_receipt=10.11)
    assert packed['contrast_valid'].all()
def test_async_cache_keeps_all_original_sensory_arrays_and_histories_exact():
    from flybrain_drive.ac_pixel_precompute import PixelPrecomputeQueue

    cold, prepared = retina(), retina()
    queue = PixelPrecomputeQueue(prepared.precompute_pixels)
    rng = np.random.default_rng(863420)
    try:
        for index in range(12):
            source = samples(index, 10+index*.08)
            for item in source.values():
                item['rgb'] = rng.integers(0, 256, size=(180, 320, 3), dtype=np.uint8)
                item['rgb'].setflags(write=False)
                queue.submit(item['rgb'])
            arguments = dict(decision_time=10.04+index*.08,
                             native_scalars=body(), native_receipt=10.03+index*.08)
            expected = cold.pack(source, **arguments)
            actual = prepared.pack(source, **arguments)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        assert cold.last_decision == prepared.last_decision
        assert cold.identities == prepared.identities
    finally:
        queue.close()




def test_explicit_episode_restart_discards_temporal_history():
    r = retina()
    r.pack(samples(1, 10.), decision_time=10.03)
    assert r.pack(samples(2, 10.08), decision_time=10.11)['contrast_valid'].all()
    r.body.scalar_history['example'] = (1.,)
    r.reset_episode()
    assert not r.body.scalar_history
    fresh = samples(1, 20.)
    for sample in fresh.values():
        sample['generation'] = 'new-generation'
    actual = r.pack(fresh, decision_time=20.03)
    expected = retina().pack(fresh, decision_time=20.03)
    assert not actual['contrast_valid'].any()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
