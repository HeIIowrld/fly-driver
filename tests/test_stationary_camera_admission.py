import pytest
from flybrain_drive.stationary_camera_admission import validate_stationary_camera_admission, validate_stationary_file_camera_admission


def status():
    return dict(paused=True,live=False,is_online=False,in_menu=False,camera_active=False,ai_controlled=False,expert_probe_active=False,ideal_line_shown=False,speed_kmh=0.,gas=0.,brake=1.)


def test_requires_paused_and_refuses_unknown_pending_camera():
    live=status();live.update(paused=False,live=True)
    with pytest.raises(ValueError,match='Paused idle'):validate_stationary_camera_admission(live,None,None,None)
    with pytest.raises(ValueError,match='terminal state unknown'):
        validate_stationary_camera_admission(status(),{'id':'mmf-stream-123'},{'request_id':'old','state':'failed'},None)


def test_accepts_only_safe_known_completed_request_and_full_brake():
    req={'id':'mmf-stream-123'}
    validate_stationary_camera_admission(status(),req,{'request_id':'mmf-stream-123','state':'failed'},None)
    validate_stationary_camera_admission(status(),req,None,{'request_id':'mmf-stream-123','status':'stopped'})
    with pytest.raises(ValueError,match='identity unknown'):
        validate_stationary_camera_admission(status(),{'id':'../other'},None,None)
    bad=status();bad['brake']=.5
    with pytest.raises(ValueError,match='full-brake'):validate_stationary_camera_admission(bad,None,None,None)


def test_file_camera_requires_live_stationary_same_generation_and_terminal():
    live=status();live.update(paused=False,live=True,generation='g')
    req={'id':'old'};camera={'request_id':'old','state':'failed','generation':'g'}
    validate_stationary_file_camera_admission(live,req,camera,None)
    with pytest.raises(ValueError):validate_stationary_file_camera_admission(status(),req,camera,None)
    with pytest.raises(ValueError,match='generation'):
        validate_stationary_file_camera_admission(live,req,dict(camera,generation='other'),None)
    with pytest.raises(ValueError,match='terminal state unknown'):
        validate_stationary_file_camera_admission(live,{'id':'pending'},camera,None)


def test_conflicting_terminal_and_heartbeat_are_not_silently_accepted():
    live=status();live.update(paused=False,live=True,generation='g')
    request={'id':'old'}
    camera={'request_id':'old','state':'failed','generation':'g'}
    terminal={'request_id':'old','status':'stopped'}
    with pytest.raises(ValueError,match='terminal and heartbeat disagree'):
        validate_stationary_file_camera_admission(live,request,camera,terminal)


def test_known_failure_recovery_is_explicit_fresh_and_narrow():
    from flybrain_drive.stationary_camera_admission import validate_known_stopped_metadata_failure_probe as check
    live=status();live.update(paused=False,live=True,generation='g')
    request={'id':'old'};terminal={'request_id':'old','status':'stopped'}
    camera={'request_id':'old','state':'failed','generation':'g','reason':'Cannot commit camera metadata'}
    clock={'generation':'g','uncertainty_seconds':.005,'exchanges':[{'sent':9.8,'received':9.9,'producer_clock':2.} for _ in range(4)]}
    check(live,request,camera,terminal,clock,now=10.)
    with pytest.raises(ValueError):check(live,request,camera,terminal,clock,now=12.)
    with pytest.raises(ValueError):check(live,request,dict(camera,reason='other'),terminal,clock,now=10.)
    with pytest.raises(ValueError):check(live,request,camera,dict(terminal,status='failed'),clock,now=10.)
    with pytest.raises(ValueError):check(dict(live,camera_active=True),request,camera,terminal,clock,now=10.)
    with pytest.raises(ValueError):check(live,request,camera,terminal,None,now=10.)
    with pytest.raises(ValueError):check(live,request,camera,terminal,dict(clock,generation='other'),now=10.)
