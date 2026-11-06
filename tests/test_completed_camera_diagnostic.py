import copy
import pytest
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate


def case():
    status=dict(generation='g',epoch=7,paused=True,live=False,is_online=False,in_menu=False,camera_active=False,
        ai_controlled=False,expert_probe_active=False,ideal_line_shown=False,speed_kmh=0.,gas=0.,brake=1.)
    request=dict(id='mmf-stream-known',frames=240)
    camera=dict(generation='g',request_id=request['id'],state='failed',reason='Cannot commit camera metadata')
    terminal=dict(request_id=request['id'],status='completed',frames=240,actor_stream_created=False,actuator_messages_sent=0)
    clock=dict(generation='g',uncertainty_seconds=.005,exchanges=[dict(sent=90+i*.2,received=90.01+i*.2,producer_clock=i+1.) for i in range(4)])
    evidence=dict(producer_terminal=copy.deepcopy(terminal),expected_bundle_sha256='a'*64,archive_bundle_sha256='a'*64,archive_verified=True,
        prior_sources_closed=True,source_sha256='b'*64,expected_source_sha256='b'*64)
    return status,request,camera,terminal,clock,evidence


def test_default_gate_still_rejects_completed_failed_conflict():
    s,r,c,t,_,_=case()
    with pytest.raises(ValueError,match='disagree'):PausedMMFStartupGate(s,r,c,t)


def test_narrow_diagnostic_preserves_original_failed_and_completed_records():
    args=case();before=copy.deepcopy(args)
    gate=PausedMMFStartupGate.completed_metadata_diagnostic(*args,now=91.)
    assert args==before and gate.phase=='admitted'
    assert gate.diagnostic_admission['previous_run_reclassified'] is False
    assert gate.diagnostic_admission['camera_health_verified'] is False


@pytest.mark.parametrize('defect',['incomplete','different_error','generation','unclosed','bundle','source','expired','online','actuator'])
def test_diagnostic_refuses_other_conflicts_and_unverified_sources(defect):
    s,r,c,t,clock,evidence=case()
    if defect=='incomplete':t['frames']=239
    if defect=='different_error':c['reason']='Unknown renderer failure'
    if defect=='generation':c['generation']='other'
    if defect=='unclosed':evidence['prior_sources_closed']=False
    if defect=='bundle':evidence['archive_bundle_sha256']='c'*64
    if defect=='source':evidence['source_sha256']='c'*64
    if defect=='expired':clock['exchanges'][-1]['received']=88.
    if defect=='online':s['is_online']=True
    if defect=='actuator':t['actuator_messages_sent']=1
    with pytest.raises(ValueError):PausedMMFStartupGate.completed_metadata_diagnostic(s,r,c,t,clock,evidence,now=91.)
