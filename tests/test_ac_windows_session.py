"""Session gate truth table; never opens native desktops or captures pixels."""

import pytest

from flybrain_drive.ac_windows_session import classify_capture_session


def test_disconnected_same_session_is_explicit_and_cannot_be_cured_by_focus():
    record = dict(session_id=1, connect_state=4)
    result = classify_capture_session(record, record, False)
    assert not result['session_ready_for_capture_attempt']
    assert result['reasons'] == ['collector Windows session is disconnected',
        'engine Windows session is disconnected', 'input desktop is inaccessible']
    assert not result['foreground_activation_attempted']


def test_active_same_session_only_allows_capture_attempt_not_success_claim():
    record = dict(session_id=1, connect_state=0)
    result = classify_capture_session(record, record, True)
    assert result['session_ready_for_capture_attempt'] and not result['reasons']
    assert not result['capture_success_verified']
    other = classify_capture_session(record, dict(session_id=2, connect_state=0), True)
    assert not other['session_ready_for_capture_attempt']
    assert other['reasons'] == ['collector and engine are in different Windows sessions']


@pytest.mark.parametrize('state', [None, True, -1, 10])
def test_unknown_state_never_assumed_ready(state):
    with pytest.raises(ValueError):
        classify_capture_session(dict(session_id=1, connect_state=state),
                                 dict(session_id=1, connect_state=0), True)
