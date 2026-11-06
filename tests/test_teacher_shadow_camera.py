import threading
import time
from types import SimpleNamespace
import pytest
from flybrain_drive.teacher_shadow_camera import TeacherShadowCameraCoordinator,require_teacher_motion_state
from flybrain_drive.owned_source_camera import OwnedSourceCamera
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json


def test_shadow_coordinator_uses_original_teacher_owner_protocol(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path));directory=tmp_path/'flybrain/camera-coordination/owned';directory.mkdir(parents=True)
    lua=tmp_path/'source.lua';lua.write_text('fixture')
    camera=SimpleNamespace(request_id='mmf-stream-'+'a'*32,clock=SimpleNamespace(generation='fixture'),count=0,triggered=False,error=None)
    gate=threading.Event();coordinator=TeacherShadowCameraCoordinator(camera,gate,directory,window_handle=1,source_sha256=sha256(lua),recording='fixture')
    try:
        owner=OwnedSourceCamera(directory/'ready.json',generation='fixture',window_handle=1,installed_source=lua)
        # The actual teacher creates this trigger after its paused native/thermal
        # preflight; the coordinator itself never resumes or writes controls.
        (directory/'trigger').touch();assert gate.wait(1)
        coordinator.check();owner.check(False,None)
    finally:coordinator.close()


def state():return dict(generation='g',epoch=1,paused=False,live=True,ai_controlled=False,is_online=False,
    expert_probe_active=False,camera_request_id='r',camera_active=True,speed_kmh=20,error=None,pending_receipt_id=None)


def test_original_stationary_brake_gate_is_not_used_as_motion_admission():
    require_teacher_motion_state(state(),generation='g',epoch=1,request_id='r')


def test_teacher_trigger_cannot_start_camera_before_explicit_native_readiness(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path));directory=tmp_path/'flybrain/camera-coordination/barrier';directory.mkdir(parents=True)
    camera=SimpleNamespace(request_id='mmf-stream-'+'b'*32,clock=SimpleNamespace(generation='fixture'),count=0,triggered=False,error=None)
    gate=threading.Event();coordinator=TeacherShadowCameraCoordinator(camera,gate,directory,window_handle=1,source_sha256='fixture',recording='fixture',native_ready_required=True)
    try:
        (directory/'trigger').touch();assert not gate.wait(.15)
        write_json(directory/'native-ready.json',dict(request_id=camera.request_id,generation='fixture',prior_actor_steps=0))
        assert gate.wait(1);coordinator.check()
    finally:coordinator.close()


@pytest.mark.parametrize('key,value',[('epoch',2),('generation','other'),('camera_request_id','other'),
    ('speed_kmh',40.1),('speed_kmh',float('nan')),('ai_controlled',True),('paused',True),('is_online',True)])
def test_motion_admission_keeps_identity_authority_and_speed_envelope(key,value):
    row=state();row[key]=value
    with pytest.raises(ValueError):require_teacher_motion_state(row,generation='g',epoch=1,request_id='r')
