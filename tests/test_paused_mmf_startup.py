import pytest
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate


def status(**overrides):
    value = dict(generation='gen', epoch=1, paused=True, live=False, is_online=False,
        in_menu=False, camera_active=False, ai_controlled=False, expert_probe_active=False,
        ideal_line_shown=False, speed_kmh=0., gas=0., brake=1.)
    value.update(overrides)
    if value['camera_active'] and 'camera_request_id' not in overrides:
        value['camera_request_id'] = 'mmf-stream-test'
    return value


def test_terminal_receipt_and_order_required():
    with pytest.raises(ValueError, match='terminal'):
        PausedMMFStartupGate(status(), {'id': 'old'}, {'request_id': 'old', 'state': 'stopped'}, None)
    gate = PausedMMFStartupGate(status(), {'id':'old'}, {'request_id':'old','state':'stopped'}, {'request_id':'old','status':'stopped'})
    camera = dict(generation='gen', request_id='mmf-stream-test', state='capturing')
    with pytest.raises(RuntimeError):
        gate.resume_observed(status(), camera, first_body_packet=1, later_body_packet=2)
    gate.camera_ready(status(camera_active=True), camera, 'mmf-stream-test')
    live = status(paused=False, live=True, camera_active=True)
    with pytest.raises(ValueError, match='advancing'):
        gate.resume_observed(live, camera, first_body_packet=3, later_body_packet=3)
    with pytest.raises(ValueError):
        gate.resume_observed(status(paused=False, live=True, generation='other'), camera, first_body_packet=3, later_body_packet=4)
    gate.resume_observed(live, camera, first_body_packet=3, later_body_packet=4)
    assert gate.phase == 'live'


def test_reject_unpaused_or_mismatched_camera():
    with pytest.raises(ValueError):
        PausedMMFStartupGate(status(paused=False,live=True),None,None,None)
    gate=PausedMMFStartupGate(status(),None,None,None)
    with pytest.raises(ValueError):
        gate.camera_ready(status(camera_active=True),dict(generation='wrong',request_id='mmf-stream-test',state='capturing'),'mmf-stream-test')
    assert gate.phase == 'admitted'


def test_session_camera_owner_must_match_camera_status():
    gate=PausedMMFStartupGate(status(),None,None,None)
    camera=dict(generation='gen',request_id='mmf-stream-test',state='capturing')
    with pytest.raises(ValueError):
        gate.camera_ready(status(camera_active=True,camera_request_id='mmf-stream-other'),camera,'mmf-stream-test')
    gate.camera_ready(status(camera_active=True),camera,'mmf-stream-test')
    with pytest.raises(ValueError):
        gate.resume_observed(status(paused=False,live=True,camera_active=True,camera_request_id='mmf-stream-other'),camera,first_body_packet=1,later_body_packet=2)


@pytest.mark.parametrize('changed', [
    {'speed_kmh': .1}, {'speed_kmh': float('nan')}, {'gas': .02},
    {'brake': .98}, {'expert_probe_active': True}, {'ideal_line_shown': True},
    {'error': 'pending error'}, {'pending_receipt_id': 'other'},
])
@pytest.mark.parametrize('phase', ['camera_ready', 'resume'])
def test_stationary_contract_is_rechecked_during_handoff(changed, phase):
    gate = PausedMMFStartupGate(status(), None, None, None)
    camera = dict(generation='gen', request_id='mmf-stream-test', state='capturing')
    if phase == 'camera_ready':
        with pytest.raises(ValueError):
            gate.camera_ready(status(camera_active=True, **changed), camera, 'mmf-stream-test')
        assert gate.phase == 'admitted'
    else:
        gate.camera_ready(status(camera_active=True), camera, 'mmf-stream-test')
        with pytest.raises(ValueError):
            gate.resume_observed(status(paused=False, live=True, camera_active=True, **changed),
                                 camera, first_body_packet=1, later_body_packet=2)
        assert gate.phase == 'camera_ready'
