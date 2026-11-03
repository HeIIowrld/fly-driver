import pytest

from flybrain_drive.ac_lifecycle_commit import CONTEXT_KEYS, fresh_unpublished_request


def fixture():
    before = dict(schema='flybrain.csp.session.status.v1', generation='g', epoch=1,
        window_handle='w', car_id='bmw_z4_gt3', track_id='monza', is_online=False,
        paused=True, live=False, in_menu=False, ai_controlled=False, expert_probe_active=False,
        camera_active=False, ideal_line_shown=False, physics_api_allowed=True,
        speed_kmh=0., gas=0., brake=1., script_clock=10.)
    request = dict(schema='flybrain.csp.session.request.v1', generation='g', epoch=1,
                   id='original_unique_id', command='resume', expires_clock=13.)
    return before, request


def test_slow_archival_before_publication_does_not_consume_new_request_deadline():
    before, request = fixture()
    fresh = dict(before, script_clock=40.)
    actual = fresh_unpublished_request(before, fresh, request, publication_started=False)
    assert actual == dict(request, expires_clock=43.)
    assert request['expires_clock'] == 13. and before['script_clock'] == 10.


def test_issued_id_can_never_be_renewed_even_if_all_context_matches():
    before, request = fixture()
    with pytest.raises(ValueError, match='Never renew'):
        fresh_unpublished_request(before, before, request, publication_started=True)


@pytest.mark.parametrize('key', CONTEXT_KEYS)
def test_any_context_change_aborts_before_a_request_can_be_published(key):
    before, request = fixture()
    with pytest.raises(ValueError, match='context changed'):
        fresh_unpublished_request(before, dict(before, **{key: 'changed'}), request,
                                  publication_started=False)


@pytest.mark.parametrize('changed', [dict(speed_kmh=.1), dict(gas=.02), dict(brake=.98),
    dict(script_clock=9.), dict(script_clock=float('nan')), dict(gas=float('inf')),
    dict(pending_receipt_id='active'), dict(error='publisher error')])
def test_no_fresh_commit_from_moving_stale_or_pending_state(changed):
    before, request = fixture()
    with pytest.raises(ValueError):
        fresh_unpublished_request(before, dict(before, **changed), request,
                                  publication_started=False)
