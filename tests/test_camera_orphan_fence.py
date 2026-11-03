import copy
import pytest
from flybrain_drive.camera_orphan_fence import validate_orphan_fence


def sample():
    return dict(request={'id':'mmf-stream-abc','expected_generation':'old'},orphan={'request_id':'mmf-stream-abc','expected_generation':'old','status':'prior_generation_unresolved','actor_admissible':False},quarantine={'status':'request_slot_quarantined_outcome_still_unknown','original_terminal_known':False,'generation':'old-release'},restart={'status':'completed_engine_restart_after_camera_quarantine','restart_outcome_verified':True,'old_engine_pid':1,'new_engine_pid':2,'original_process':{'pid':1,'active':False,'termination_signaled':True}},restart_request={'old_engine_pid':1,'generation':'old-release'},release={'generation':'old-release','active_before':False,'active_after':False},current_session={'generation':'new','camera_active':False,'ai_controlled':False,'is_online':False,'in_menu':False},current_camera={'generation':'new','state':'idle'},engine_pid=2,request_slot_exists=False,old_terminal_exists=False)


def test_valid_fence_keeps_old_terminal_unknown():
    r=validate_orphan_fence(**sample());assert r['original_terminal']=='unknown' and r['actor_qualified'] is False and r['camera_requested'] is False


@pytest.mark.parametrize('section,key,value',[
 ('restart','restart_outcome_verified',False),('restart','new_engine_pid',3),('orphan','actor_admissible',True),('release','active_before',True),('release','active_after',True),('current_camera','state','capturing'),('current_session','camera_active',True),('current_session','is_online',True),('request','expected_generation','new')])
def test_mismatched_or_active_state_rejected(section,key,value):
    d=sample();d[section][key]=value
    with pytest.raises(ValueError):validate_orphan_fence(**d)


def test_exit_requested_is_not_process_termination():
    d=sample();d['restart']['original_process'].update(active=True,termination_signaled=False,exit_requested=True)
    with pytest.raises(ValueError):validate_orphan_fence(**d)


@pytest.mark.parametrize('key',['request_slot_exists','old_terminal_exists'])
def test_publication_or_terminal_change_requires_review(key):
    d=sample();d[key]=True
    with pytest.raises(ValueError):validate_orphan_fence(**d)
