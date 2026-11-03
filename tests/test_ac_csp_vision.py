import pytest
import threading

from flybrain_drive.ac_csp_vision import candidate_at


def test_prepared_camera_gate_can_be_cancelled_without_a_capture_request():
    from flybrain_drive.ac_csp_vision import await_owned_capture_start
    gate,stop=threading.Event(),threading.Event()
    observed=[]
    worker=threading.Thread(target=lambda:observed.append(await_owned_capture_start(gate,stop)))
    worker.start()
    stop.set()
    worker.join(1)
    assert not worker.is_alive() and observed==[False] and not gate.is_set()
    stop.clear();gate.set()
    assert await_owned_capture_start(gate,stop)


@pytest.mark.parametrize('period,frames',[(.04,1600),(.09,1601)])
def test_extended_recording_budget_does_not_relax_sensor_cadence_or_unbounded_frames(period,frames):
    from flybrain_drive.ac_csp_vision import CSPVisionRecorder
    with pytest.raises(ValueError,match='Source-body recording'):
        CSPVisionRecorder(0,{},source_body_recording=True,mmf_metadata=True,
            period=period,max_frames=frames,expected_source_sha256='a'*64)


@pytest.mark.parametrize('pin', ['', 'a' * 63, 'G' * 64, 1])
def test_explicit_source_pin_must_be_sha256_before_accessing_game(pin):
    from flybrain_drive.ac_csp_vision import CSPVisionRecorder
    with pytest.raises(ValueError, match='SHA-256'):
        CSPVisionRecorder(0, {}, expected_source_sha256=pin)


@pytest.mark.parametrize('frames,async_mode,mmf', [(1201, True, True), (8, False, True),
                                                 (8, True, 1)])
def test_live_mmf_requires_bounded_explicit_persistence_before_accessing_game(
        frames, async_mode, mmf):
    from flybrain_drive.ac_csp_vision import CSPVisionRecorder

    with pytest.raises(ValueError, match='MMF stream'):
        CSPVisionRecorder(0, {}, max_frames=frames, image_format='bmp', period=.04,
                          shared_transport=True, batched_shared_copy=True,
                          async_frame_persistence=async_mode, mmf_metadata=mmf)


@pytest.mark.parametrize('batched,shared,reference,option', [
    (False, True, False, True), (True, True, True, True),
    (False, False, False, True), (True, True, False, 1),
])
def test_async_persistence_rejects_unreviewed_transport_before_game(
        batched, shared, reference, option):
    from flybrain_drive.ac_csp_vision import CSPVisionRecorder

    with pytest.raises(ValueError, match='Async persistence'):
        CSPVisionRecorder(0, {}, max_frames=8, image_format='bmp', period=.04,
                          shared_transport=shared, batched_shared_copy=batched,
                          verify_png_reference=reference, async_frame_persistence=option)


@pytest.mark.parametrize('frame', [
    {}, {'ideal_line_shown': False}, {'ideal_line_shown': None, 'aid_probe_only': False},
    {'ideal_line_shown': True, 'aid_probe_only': False},
    {'ideal_line_shown': 0, 'aid_probe_only': False},
    {'ideal_line_shown': False, 'aid_probe_only': 0},
    {'ideal_line_shown': False, 'aid_probe_only': True},
])
def test_actor_camera_rejects_unknown_enabled_or_diagnostic_aid_metadata(frame):
    from flybrain_drive.ac_csp_vision import validate_frame_aid_state

    with pytest.raises(ValueError, match='explicit ideal-line OFF'):
        validate_frame_aid_state(frame)


def test_actor_camera_accepts_only_explicit_off_non_diagnostic_metadata():
    from flybrain_drive.ac_csp_vision import validate_frame_aid_state

    validate_frame_aid_state(dict(ideal_line_shown=False, aid_probe_only=False))


def row(index, received, earliest):
    return dict(capture_id=index, received=received,
                views=[dict(render_started_lower=earliest+i*.001) for i in range(4)])


def test_latest_frame_reuse_is_causal_and_uses_oldest_source_age_bound():
    rows = [row(0, 10.02, 10), row(1, 10.09, 10.07)]
    assert candidate_at(rows, 10.04, .15)['capture_id'] == 0
    assert candidate_at(rows, 10.08, .15)['capture_id'] == 0
    assert candidate_at(rows, 10.12, .15)['capture_id'] == 1
    with pytest.raises(ValueError, match='preceding'):
        candidate_at(rows, 10.01, .15)
    with pytest.raises(ValueError, match='conservative'):
        candidate_at(rows, 10.23, .15)


def test_extended_camera_budget_remains_finite_in_duration_and_storage():
    from flybrain_drive.ac_csp_vision import validate_camera_budget
    validate_camera_budget(1650, .03, 1024**3)
    validate_camera_budget(3250, .03, 2*1024**3)
    for frames, period, size in [(4001, .03, 1024**3), (4000, .04, 1024**3),
                                 (1650, .03, 3*1024**3), (1650, .01, 1024**3)]:
        with pytest.raises(ValueError):
            validate_camera_budget(frames, period, size)


@pytest.mark.parametrize('fmt,paired,frames', [('jpeg', False, 8), ('bmp', True, 9),
                                             ('png', True, 8), ('bmp', 'yes', 8)])
def test_transport_options_fail_before_accessing_game(fmt, paired, frames):
    from flybrain_drive.ac_csp_vision import CSPVisionRecorder

    with pytest.raises(ValueError, match='lossless format'):
        CSPVisionRecorder(0, {}, max_frames=frames, image_format=fmt, verify_png_reference=paired)


@pytest.mark.parametrize('fmt,period,shared', [('png', .04, True), ('bmp', .03, True),
                                             ('bmp', .04, 1)])
def test_shared_transport_rejects_ambiguous_codec_and_period_before_game(fmt, period, shared):
    from flybrain_drive.ac_csp_vision import CSPVisionRecorder

    with pytest.raises(ValueError, match='Shared transport'):
        CSPVisionRecorder(0, {}, image_format=fmt, period=period, shared_transport=shared)


def test_camera_bundle_preserves_bmp_bytes(tmp_path):
    from flybrain_drive.ac_camera_archive import CameraArchive, bundle_camera_files

    source = tmp_path/'source'
    source.mkdir()
    (source/'0000-front.bmp').write_bytes(b'lossless-bitmap-provenance')
    bundle_camera_files([source], tmp_path/'archive')
    with CameraArchive(tmp_path/'archive') as archive:
        archive.verify_all()
        assert archive.read('0000-front.bmp') == b'lossless-bitmap-provenance'


def test_metadata_publication_retry_preserves_original_times(tmp_path):
    from flybrain_drive.ac_csp_vision import read_json

    path = tmp_path/'frame.json'
    path.write_text('', encoding='utf8')
    stamp, waits = [0.], []

    def sleep(dt):
        stamp[0] += dt
        waits.append(dt)
        path.write_text('{"index":7,"render_started_clock":1.25}', encoding='utf8')

    assert read_json(path, clock=lambda: stamp[0], sleep=sleep) == dict(
        index=7, render_started_clock=1.25)
    assert waits == [.001]


def test_persistent_incomplete_metadata_fails_at_bounded_deadline(tmp_path):
    import json

    from flybrain_drive.ac_csp_vision import read_json

    path = tmp_path/'frame.json'
    path.write_text('{', encoding='utf8')
    stamp = [0.]

    def sleep(dt):
        stamp[0] += dt

    with pytest.raises(json.JSONDecodeError):
        read_json(path, clock=lambda: stamp[0], sleep=sleep)
    assert stamp[0] == pytest.approx(.02)
    assert path.read_text(encoding='utf8') == '{'
