import pytest

from flybrain_drive import ac_csp_vision as vision


def fake_recorder(monkeypatch, publish_at=None):
    clock = [1.2]
    recorder = object.__new__(vision.CSPVisionRecorder)
    recorder.host_clock = lambda: clock[0]
    monkeypatch.setattr(vision.time, 'sleep',
                        lambda duration: clock.__setitem__(0, clock[0]+duration))

    def select(receipt, *, max_age):
        rows = [dict(capture_id=1, received=1.19, views=[{'render_started_lower': 1.088}])]
        if publish_at is not None and clock[0] >= publish_at:
            rows.append(dict(capture_id=2, received=publish_at,
                             views=[{'render_started_lower': 1.16}]))
        return vision.candidate_at(rows, receipt, max_age)

    recorder.candidate_before = select
    return recorder, clock


def test_new_frame_can_arrive_during_bounded_wait_without_relaxing_age(monkeypatch):
    recorder, clock = fake_recorder(monkeypatch, publish_at=1.209)
    candidate, selected, waited = recorder.wait_for_candidate(max_age=.11, timeout=.02)
    assert candidate['capture_id'] == 2
    assert candidate['received'] <= selected == clock[0]
    assert 0 < waited <= .02
    assert 0 <= selected-candidate['views'][0]['render_started_lower'] <= .11


def test_timeout_does_not_reuse_the_old_frame(monkeypatch):
    recorder, clock = fake_recorder(monkeypatch)
    with pytest.raises(vision.NoFreshCSPFrame):
        recorder.wait_for_candidate(max_age=.11, timeout=.02)
    assert clock[0] == pytest.approx(1.22)


def test_invalid_budget_and_worker_error_are_not_retried(monkeypatch):
    recorder, clock = fake_recorder(monkeypatch)
    with pytest.raises(ValueError, match='budget'):
        recorder.wait_for_candidate(max_age=0, timeout=.02)
    assert clock[0] == 1.2


    def failed(*args, **kwargs):
        raise RuntimeError('worker failed')

    recorder.candidate_before = failed
    with pytest.raises(RuntimeError, match='worker failed'):
        recorder.wait_for_candidate()
    assert clock[0] == 1.2


def test_late_scheduler_wakeup_cannot_accept_a_frame_after_wait_deadline(monkeypatch):
    recorder, clock = fake_recorder(monkeypatch, publish_at=1.229)
    monkeypatch.setattr(vision.time, 'sleep', lambda _: clock.__setitem__(0, clock[0]+.03))
    with pytest.raises(vision.NoFreshCSPFrame, match='deadline'):
        recorder.wait_for_candidate(timeout=.02)


def test_40ms_wait_accepts_new_fresh_data_after_20ms_but_keeps_the_same_age_limit(monkeypatch):
    recorder, _ = fake_recorder(monkeypatch, publish_at=1.229)
    with pytest.raises(vision.NoFreshCSPFrame):
        recorder.wait_for_candidate(max_age=.11, timeout=.02)
    recorder, _ = fake_recorder(monkeypatch, publish_at=1.229)
    candidate, selected, waited = recorder.wait_for_candidate(max_age=.11, timeout=.04)
    assert candidate['capture_id'] == 2 and .02 < waited <= .04
    assert selected-candidate['views'][0]['render_started_lower'] <= .11


def test_40ms_wait_never_accepts_stale_data_or_ignores_producer_failure(monkeypatch):
    recorder, clock = fake_recorder(monkeypatch)
    with pytest.raises(vision.NoFreshCSPFrame):
        recorder.wait_for_candidate(max_age=.11, timeout=.04)
    assert clock[0] == pytest.approx(1.24)

    def failed(*args, **kwargs):
        raise RuntimeError('producer failed')

    recorder.candidate_before = failed
    with pytest.raises(RuntimeError, match='producer failed'):
        recorder.wait_for_candidate(max_age=.11, timeout=.04)
