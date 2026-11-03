import copy
import json
from pathlib import Path

import numpy as np
import pytest

from flybrain_drive.ac_csp_retina_producer_time import CSPProducerTimeRetina


def source(capture, lower, source_time, pixel):
    return {name: {'rgb': np.full((180, 320, 3), pixel, np.uint8),
                   'capture_id': capture, 'generation': 'g', 'render_started_lower': lower,
                   'received': lower + .01, 'producer_render_started': source_time}
            for name in ('front', 'left', 'right', 'rear')}


def make_retina():
    config = json.loads(Path('configs/ac-csp-retina.json').read_text(encoding='utf8'))
    config['native_retina'].update(receptors_per_view=24, columns=6)
    return CSPProducerTimeRetina(config)


def test_source_interval_ignores_offset_bound_change_and_keeps_host_age():
    retina = make_retina()
    first = retina.pack(source(1, 10., 1., 50), decision_time=10.02)
    assert not first['contrast_valid'].any()
    second = retina.pack(source(2, 10.12, 1.1, 100), decision_time=10.14)
    expected = np.tanh((100/255-50/255)/.1/retina.body.scale)
    np.testing.assert_allclose(second['view_contrast'], expected, atol=1e-6)
    np.testing.assert_allclose(second['view_age'], .02, atol=1e-6)
    assert second['contrast_valid'].all()
    held = retina.pack(source(2, 10.12, 1.1, 100), decision_time=10.15)
    np.testing.assert_array_equal(held['view_contrast'], second['view_contrast'])


def test_producer_clock_identity_gap_and_missing_frame_are_explicit():
    retina = make_retina()
    retina.pack(source(1, 10., 1., 50), decision_time=10.02)
    bad = source(2, 10.1, 1., 100)
    with pytest.raises(ValueError, match='source time'):
        retina.pack(bad, decision_time=10.12)
    assert retina.last_decision == 10.02
    missing = copy.deepcopy(source(2, 10.1, 1.1, 100))
    del missing['front']['producer_render_started']
    with pytest.raises(ValueError, match='Producer time'):
        retina.pack(missing, decision_time=10.12)
    packed = retina.pack(source(2, 10.2, 1.2, 100), decision_time=10.22)
    assert not packed['contrast_valid'].any()
    assert retina.calibration['physical_exposure_sync_verified'] is False


def test_recorder_receipt_to_producer_retina_and_legacy_selection():
    import json
    import threading
    from types import SimpleNamespace

    from flybrain_drive.ac_csp_vision import CSPVisionRecorder

    source_frames = source(1, 10., 1., 50)
    recorder = object.__new__(CSPVisionRecorder)
    recorder.lock = threading.Lock()
    recorder.clock = SimpleNamespace(generation='g')
    recorder.pixels = {1: {name: frame['rgb'] for name, frame in source_frames.items()}}
    recorder.preserve_producer_time = True
    receipt = {'capture_id': 1, 'generation': 'g', 'received': 10.01,
               'ideal_line_shown': False, 'aid_probe_only': False,
               'views': [{'view_id': name, 'render_started_lower': 10.,
                          'producer_render_started': 1.} for name in source_frames]}
    receipt = json.loads(json.dumps(receipt))
    samples = recorder.samples_for(receipt)
    assert all(s['producer_render_started'] == 1. for s in samples.values())
    assert make_retina().pack(samples, decision_time=10.02)['view_valid'].all()
    broken = copy.deepcopy(receipt)
    del broken['views'][0]['producer_render_started']
    with pytest.raises(KeyError):
        recorder.samples_for(broken)
    recorder.preserve_producer_time = False
    assert all('producer_render_started' not in s for s in recorder.samples_for(receipt).values())


def test_explicit_episode_restart_clears_base_and_source_clock_history():
    retina = make_retina()
    retina.pack(source(1, 10., 1., 50), decision_time=10.02)
    active = retina.pack(source(2, 10.1, 1.1, 100), decision_time=10.12)
    assert active['contrast_valid'].all()
    retina.body.scalar_history['example'] = (1.,)
    retina.reset_episode()
    assert not retina.body.scalar_history
    restarted = source(1, 20., 1., 100)
    for sample in restarted.values():
        sample['generation'] = 'new-generation'
    actual = retina.pack(restarted, decision_time=20.02)
    expected = make_retina().pack(restarted, decision_time=20.02)
    assert not actual['contrast_valid'].any()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert retina.generation == 'new-generation'
