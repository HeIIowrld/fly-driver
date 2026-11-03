import json
import time
import os
from pathlib import Path

import pytest

from flybrain_drive.graph import sha256
from flybrain_drive.owned_source_camera import OwnedSourceCamera


def make_plan(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    owned=tmp_path/'flybrain/camera-coordination/test-owned'
    owned.mkdir(parents=True)
    source=tmp_path/'camera.lua';source.write_text('pinned camera')
    nonce='mmf-stream-'+'a'*32
    plan=dict(schema='flybrain.owned.source_body_camera.v1',generation='g',
        window_handle=2,request_id=nonce,source_sha256=sha256(source),
        trigger_file=str(owned/'trigger'),stop_file=str(owned/'stop'),
        health_file=str(owned/'health.json'))
    (owned/'ready.json').write_text(json.dumps(plan),encoding='utf8')
    (owned/'health.json').write_text(json.dumps(dict(state='prepared',request_id=nonce,
        observed_host_time=time.perf_counter())),encoding='utf8')
    return owned,source,plan


def test_owned_identity_trigger_and_stop_do_not_accept_foreign_camera(tmp_path,monkeypatch):
    root,source,plan=make_plan(tmp_path,monkeypatch)
    owner=OwnedSourceCamera(root/'ready.json',generation='g',window_handle=2,installed_source=source)
    owner.trigger_near_initial_finish(.9,0)
    assert not (root/'trigger').exists()
    owner.trigger_near_initial_finish(.98,0)
    assert (root/'trigger').exists()
    owner.trigger_near_initial_finish(.99,0)
    with pytest.raises(RuntimeError,match='camera'):
        owner.check(True,'foreign')
    owner.request_stop();owner.request_stop()
    assert (root/'stop').exists()


def test_camera_generation_and_coordination_escape_are_rejected(tmp_path,monkeypatch):
    root,source,plan=make_plan(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='identity'):
        OwnedSourceCamera(root/'ready.json',generation='other',window_handle=2,installed_source=source)
    plan['stop_file']=str(tmp_path/'unowned-stop')
    (root/'ready.json').write_text(json.dumps(plan),encoding='utf8')
    with pytest.raises(ValueError,match='escaped'):
        OwnedSourceCamera(root/'ready.json',generation='g',window_handle=2,installed_source=source)
    assert not (tmp_path/'unowned-stop').exists()


def test_health_uses_original_qpc_publication_instead_of_filesystem_wall_clock(tmp_path,monkeypatch):
    root,source,plan=make_plan(tmp_path,monkeypatch)
    os.utime(root/'health.json',(time.time()+60,time.time()+60))
    owner=OwnedSourceCamera(root/'ready.json',generation='g',window_handle=2,installed_source=source)
    owner.check(False,None)
    row=dict(state='prepared',request_id=plan['request_id'],observed_host_time=time.perf_counter()+1)
    (root/'health.json').write_text(json.dumps(row),encoding='utf8')
    with pytest.raises(RuntimeError,match='qpc_publication_age'):
        owner.check(False,None)
