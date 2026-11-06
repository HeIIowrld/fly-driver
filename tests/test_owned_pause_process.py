from pathlib import Path
import pytest
from flybrain_drive.owned_pause_process import DeferredPauseEvidence, await_owned_camera_idle

def test_nas_artifact_flush_cannot_consume_api_heartbeat_budget(tmp_path,monkeypatch):
    import flybrain_drive.owned_pause_process as mod
    out=tmp_path/'run';out.mkdir();local=tmp_path/'local-request.json';writes=[];clock=[0.]
    def writer(path,value):
        writes.append(Path(path));clock[0]+=10 if Path(path).is_relative_to(out) else .001
    monkeypatch.setattr(mod,'write_json',writer)
    evidence=DeferredPauseEvidence(out,{'source_sha256':{},'source_root':'x','git_revision':'sha'})
    evidence.begin(out,{});evidence.write(out/'before.json',{});evidence.write(local,{'id':'original'})
    evidence.write(out/'request.json',{});evidence.write(out/'after.json',{'paused':True})
    assert clock[0]<2 and writes==[local]
    evidence.flush();assert clock[0]>2 and out/'request.json' in writes

def test_frozen_evidence_is_scoped_to_owned_directory(tmp_path):
    e=DeferredPauseEvidence(tmp_path/'one',{})
    with pytest.raises(ValueError):e.begin(tmp_path/'other',{})


def test_pause_waits_for_camera_idle_of_exact_owned_epoch():
    states = iter([dict(camera_active=True,camera_request_id='owned',script_clock=5),
                   dict(camera_active=False,camera_request_id=None,script_clock=5.25)])
    clock = [0.]
    validations = []

    def read(**kwargs):
        assert 0 < kwargs['maximum_wait'] <= .5
        return dict(generation='g',epoch=7,**next(states)), 1., 0

    def validate(state,age,*,mutating):
        validations.append((age,mutating))

    observations = await_owned_camera_idle(
        read,validate,generation='g',epoch=7,request_id='owned',
        clock=lambda:clock[0],wall=lambda:1.1,sleep=lambda t:clock.__setitem__(0,clock[0]+t))
    assert [row['active'] for row in observations] == [True,False]
    assert len(validations) == 2 and all(m for _,m in validations)


@pytest.mark.parametrize("generation,request_id,message", [
    ('new','owned','Session changed'),
    ('g','other','Unowned camera'),
    ('g','owned','idle heartbeat not observed'),
])
def test_pause_camera_admission_fails_closed(generation,request_id,message):
    clock = [0.]
    with pytest.raises((ValueError,TimeoutError),match=message):
        await_owned_camera_idle(
            lambda **_: (dict(generation=generation,epoch=7,camera_active=True,
                               camera_request_id=request_id,script_clock=5),1.,0),
            lambda *_,**__: None,generation='g',epoch=7,request_id='owned',
            clock=lambda:clock[0],wall=lambda:1.1,sleep=lambda t:clock.__setitem__(0,clock[0]+t))
