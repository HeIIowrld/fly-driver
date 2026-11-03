import base64
import hashlib
import time
from types import SimpleNamespace

import pytest

from flybrain_drive.immutable_body_shadow import ImmutableBodyShadow
from tests.test_per_view_body_history import IDENTITY, sample


def test_single_camera_owner_connects_original_body_and_stops_on_ticker_failure(monkeypatch):
    rows = []
    calls = []

    class Body:
        def __init__(self, **kwargs):
            assert kwargs['native_rows'] is rows

        def _select_context(self, event, *, camera_receipt, bounds):
            calls.append((camera_receipt, bounds))
            event['decision'] = time.perf_counter()
            return 'original-body-context'

    class Camera:
        clock = SimpleNamespace(generation='session')
        host_clock = time.perf_counter
        config = {'camera': 'contract'}
        received = time.perf_counter()-.01
        sample_calls = 0

        def candidate_before(self, now):
            return {'capture_id': 4, 'generation': 'session', 'received': self.received,
                    'views': [{'view_id': 'front', 'rgb_sha256': 'original',
                               'render_started_lower': self.received-.01}]}

        def samples_for(self, candidate):
            self.sample_calls += 1
            return {'original': candidate['capture_id']}

    class Neural:
        ticker = SimpleNamespace(is_alive=lambda: True)
        ticker_error = ticker_stop_reason = None
        native_rows, episode_id = rows, 'episode'

        def configure_immutable(self, config, path):
            assert config == {'camera': 'contract'} and path == 'config'

        def process_immutable(self, candidate, samples, callback):
            assert samples == {'original': 4}
            event = {'camera_receipt': candidate['received'],
                     'render_start_lower': [candidate['received']-.01]*4}
            assert callback(event) == 'original-body-context'
            return event, {'status': 'visual_published'}

    monkeypatch.setattr('flybrain_drive.immutable_body_shadow.OnlineBodyShadow', Body)
    neural = Neural()
    camera = Camera()
    shadow = ImmutableBodyShadow(camera=camera, neural_shadow=neural, native_rows=rows,
        native_identity={}, episode_id='episode', config_path='config')
    result = shadow.consume()
    assert result['neural']['status'] == 'visual_published' and len(calls) == 1
    assert shadow.consume()['neural']['status'] == 'duplicate_not_reprocessed'
    assert camera.sample_calls == 1
    camera.received += .001
    with pytest.raises(ValueError, match='changed original metadata'):
        shadow.consume()
    assert shadow.error and len(shadow.events) == 3
    with pytest.raises(RuntimeError, match='cannot be resumed'):
        shadow.consume()


def test_ticker_stop_is_recorded_before_candidate_access(monkeypatch):
    rows = []
    class Body:
        def __init__(self, **kwargs):
            pass
    class Camera:
        clock = SimpleNamespace(generation='session')
        host_clock = time.perf_counter
        config = {}
    class Neural:
        ticker = SimpleNamespace(is_alive=lambda: True)
        ticker_error, ticker_stop_reason = None, 'current_body_receipt_stale'
        native_rows, episode_id = rows, 'episode'
        def configure_immutable(self, *_):
            pass
    monkeypatch.setattr('flybrain_drive.immutable_body_shadow.OnlineBodyShadow', Body)
    neural = Neural()
    shadow = ImmutableBodyShadow(camera=Camera(), neural_shadow=neural, native_rows=rows,
        native_identity={}, episode_id='episode', config_path='config')
    neural.ticker_stop_reason = 'current_body_receipt_stale'
    with pytest.raises(RuntimeError, match='stopped'):
        shadow.consume()
    assert shadow.error and len(shadow.events) == 1
    with pytest.raises(RuntimeError, match='cannot be resumed'):
        shadow.consume()


def test_wrong_native_source_is_rejected():
    camera = SimpleNamespace(host_clock=time.perf_counter)
    neural = SimpleNamespace(ticker=SimpleNamespace(is_alive=lambda: True), native_rows=[],
                             episode_id='episode')
    with pytest.raises(ValueError, match='ticker'):
        ImmutableBodyShadow(camera=camera, neural_shadow=neural, native_rows=[],
                            native_identity={}, episode_id='episode', config_path='config')


def test_original_native_packets_reach_per_view_context():
    now = time.perf_counter()
    rows = []
    for packet, stamp in [(1, now-.04), (2, now-.01)]:
        original = sample(packet, stamp)
        raw = original.prefix_bytes
        rows.append({'authority': {'prefix_base64': base64.b64encode(raw).decode('ascii'),
                                   'prefix_sha256': hashlib.sha256(raw).hexdigest(),
                                   'received_monotonic_seconds': stamp, 'packet_id': packet},
                     'copy': {'second_copy_end': stamp}})

    class Camera:
        clock = SimpleNamespace(generation='session')
        host_clock = time.perf_counter
        config = {}

        def candidate_before(self, _):
            return {'capture_id': 1, 'generation': 'session', 'received': now-.005,
                    'views': [{'view_id': 'front', 'rgb_sha256': 'original',
                               'render_started_lower': now-.02}]}

        def samples_for(self, _):
            return {}

    class Neural:
        ticker = SimpleNamespace(is_alive=lambda: True)
        ticker_error = ticker_stop_reason = None
        native_rows, episode_id = rows, 'episode'

        def configure_immutable(self, *_):
            pass

        def process_immutable(self, candidate, _, callback):
            event = {'camera_receipt': candidate['received'],
                     'render_start_lower': [now-.02]*4}
            context = callback(event)
            assert context.decision_body.packet_id == 2
            assert all(view is None for view in context.visual_contexts)
            return event, {'status': 'context_observed'}

    shadow = ImmutableBodyShadow(camera=Camera(), neural_shadow=Neural(), native_rows=rows,
        native_identity=IDENTITY, episode_id='episode', config_path='config')
    assert shadow.consume()['neural']['status'] == 'context_observed'
