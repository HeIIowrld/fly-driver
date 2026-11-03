import json
import threading
import time
from types import SimpleNamespace

import pytest

from flybrain_drive.graph import sha256
from flybrain_drive.multirate_live_shadow import MultirateLiveShadow, SplitCudaMultirateLiveShadow
from tests.test_immutable_camera_group import group


def test_mailbox_never_silently_overwrites_unconsumed_frame():
    shadow = MultirateLiveShadow.__new__(MultirateLiveShadow)
    shadow.mail_lock = threading.Lock()
    shadow.mail_condition = threading.Condition(shadow.mail_lock)
    shadow.mail, shadow.mail_version, shadow.consumed_version = None, 0, 0
    shadow.ticker_error = shadow.ticker_stop_reason = None
    context = SimpleNamespace(render_start_lowers=[time.perf_counter()]*4)
    shadow._publish_visual('scene-one', context, 1)
    with pytest.raises(RuntimeError, match='unconsumed'):
        shadow._publish_visual('scene-two', context, 2)
    assert shadow.mail[1:3] == ('scene-one', context)
    shadow._acknowledge_visual(1)
    shadow._publish_visual('scene-two', context, 2)
    assert shadow.mail[0] == 2 and shadow.mail[1:3] == ('scene-two', context)
    with pytest.raises(RuntimeError, match='changed'):
        shadow._acknowledge_visual(1)


def test_pending_scene_waits_for_ack_without_overwrite():
    shadow = MultirateLiveShadow.__new__(MultirateLiveShadow)
    shadow.mail_lock = threading.Lock()
    shadow.mail_condition = threading.Condition(shadow.mail_lock)
    shadow.mail, shadow.mail_version, shadow.consumed_version = None, 0, 0
    shadow.ticker_error = shadow.ticker_stop_reason = None
    context = SimpleNamespace(render_start_lowers=[time.perf_counter()]*4)
    shadow._publish_visual('first', context, 1)
    result = []
    worker = threading.Thread(target=lambda: result.append(shadow._publish_visual('second', context, 2)))
    worker.start()
    time.sleep(.005)
    assert shadow.mail[1] == 'first'
    shadow._acknowledge_visual(1)
    worker.join(timeout=1)
    assert not worker.is_alive() and result[0][1] > 0
    assert shadow.mail[1] == 'second' and shadow.mail_version == 2


def test_stale_scene_rejected_before_publication():
    shadow = MultirateLiveShadow.__new__(MultirateLiveShadow)
    shadow.mail_lock = threading.Lock()
    shadow.mail_condition = threading.Condition(shadow.mail_lock)
    shadow.mail, shadow.mail_version, shadow.consumed_version = None, 0, 0
    shadow.ticker_error = shadow.ticker_stop_reason = None
    context = SimpleNamespace(render_start_lowers=[time.perf_counter()-.16]*4)
    with pytest.raises(ValueError, match='stale before publication'):
        shadow._publish_visual('stale', context, 1)
    assert shadow.mail is None


def test_waiting_producer_stops_if_body_ticker_fails():
    shadow = MultirateLiveShadow.__new__(MultirateLiveShadow)
    shadow.mail_lock = threading.Lock()
    shadow.mail_condition = threading.Condition(shadow.mail_lock)
    shadow.mail, shadow.mail_version, shadow.consumed_version = None, 0, 0
    shadow.ticker_error = shadow.ticker_stop_reason = None
    context = SimpleNamespace(render_start_lowers=[time.perf_counter()]*4)
    shadow._publish_visual('first', context, 1)
    shadow.ticker_stop_reason = 'current_body_receipt_stale'
    with pytest.raises(RuntimeError, match='ticker stopped'):
        shadow._publish_visual('second', context, 2)
    assert shadow.mail[1] == 'first'


def test_split_stream_sync_is_scoped_to_each_worker(monkeypatch):
    class FakeStream:
        def __init__(self):
            self.syncs = 0

        def synchronize(self):
            self.syncs += 1

    shadow = SplitCudaMultirateLiveShadow.__new__(SplitCudaMultirateLiveShadow)
    shadow.visual_cuda_stream, shadow.body_cuda_stream = FakeStream(), FakeStream()
    monkeypatch.setattr('torch.cuda.stream', lambda stream: stream)
    assert shadow._visual_scope() is shadow.visual_cuda_stream
    assert shadow._body_scope() is shadow.body_cuda_stream
    shadow._sync_visual()
    assert shadow.visual_cuda_stream.syncs == 1 and shadow.body_cuda_stream.syncs == 0
    shadow._sync_body()
    assert shadow.visual_cuda_stream.syncs == 1 and shadow.body_cuda_stream.syncs == 1


def test_timing_can_only_be_armed_before_ticker_start():
    shadow = SplitCudaMultirateLiveShadow.__new__(SplitCudaMultirateLiveShadow)
    shadow.enable_diagnostic_cuda_timing()
    assert shadow.diagnostic_cuda_timing is True
    shadow.ticker = object()
    with pytest.raises(ValueError, match='before ticker'):
        shadow.enable_diagnostic_cuda_timing()


def test_immutable_bridge_pins_geometry_and_preserves_duplicate(tmp_path, monkeypatch):
    candidate, samples = group()
    geometry = {'view_order': ['front', 'left', 'right', 'rear'],
                'views': {name: {} for name in ('front', 'left', 'right', 'rear')}}
    camera_path = tmp_path/'camera.json'
    camera_path.write_text(json.dumps(geometry), encoding='utf8')
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps({'camera_config': str(camera_path),
                           'camera_config_sha256': sha256(camera_path)}), encoding='utf8')
    shadow = MultirateLiveShadow.__new__(MultirateLiveShadow)
    shadow.failed, shadow.previous_time = False, None
    shadow.provenance = {'config_sha256': sha256(config_path)}
    shadow.configure_immutable(geometry, config_path)
    monkeypatch.setattr('flybrain_drive.multirate_live_shadow.time.perf_counter', lambda: 1.04)
    def fake_process(frame, event, cb, directory, **kw):
        pixels = kw['_immutable_group']['pixels']['front']
        return {'bytes_immutable': not pixels.flags.writeable}

    monkeypatch.setattr(shadow, 'process_late', fake_process)
    event, output = shadow.process_immutable(candidate, samples, lambda event: None)
    assert event['index'] == 3 and event['missing_intermediate_captures'] == 0
    assert output['bytes_immutable']
    _, repeated = shadow.process_immutable(candidate, samples, lambda event: None)
    assert repeated['status'] == 'duplicate_not_reprocessed' and not shadow.failed
    candidate['views'][0]['render_started_lower'] = .99
    samples['front']['render_started_lower'] = .99
    with pytest.raises(ValueError, match='Repeated capture changed'):
        shadow.process_immutable(candidate, samples, lambda event: None)
    assert shadow.failed
