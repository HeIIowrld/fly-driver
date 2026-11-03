import json,time
from pathlib import Path
import pytest
import flybrain_drive.control_guard_metadata_process as module
from flybrain_drive.control_guard_metadata_process import ControlGuardMetadataProcess
from flybrain_drive.graph import sha256


def make(tmp_path):
 hb=tmp_path/'status.json';profile=tmp_path/'controls.ini';health=tmp_path/'health.json';stop=tmp_path/'stop'
 hb.write_text(json.dumps(dict(generation='g',epoch=1,paused=False,in_menu=False,camera_request_id='camera')))
 profile.write_text('original controls')
 health.write_text(json.dumps(dict(observed_host_time=time.perf_counter(),state='capturing',request_id='camera')))
 return ControlGuardMetadataProcess(heartbeat_path=hb,profile_path=profile,stop_path=stop,camera_health_path=health),profile,stop


def test_separate_process_preserves_read_brackets_and_detects_original_stop(tmp_path):
 reader,profile,stop=make(tmp_path)
 try:
  first=reader.snapshot();started=first['read_started']
  assert first['profile_sha256']==sha256(profile)
  assert started<=first['read_finished']<=time.perf_counter()
  assert first['read_started']==started
  stop.touch();deadline=time.perf_counter()+2
  while not reader.latest['stop_requested'] and time.perf_counter()<deadline:time.sleep(.01)
  with pytest.raises(RuntimeError,match='stop requested'):reader.snapshot()
 finally:result=reader.close()
 assert result['status']=='completed' and not result['process_alive'] and not result['receiver_alive'] and not result['forced_termination']

def test_any_watched_profile_change_invalidates_original_hash_without_refreshing_it(tmp_path):
 reader,profile,_=make(tmp_path)
 try:
  first=reader.snapshot();profile.write_text('changed controls')
  deadline=time.perf_counter()+2
  while reader.terminal is None and time.perf_counter()<deadline:time.sleep(.01)
  with pytest.raises(RuntimeError,match='Controls profile changed'):reader.snapshot()
  assert reader.latest['profile_sha256']==first['profile_sha256']
 finally:result=reader.close()
 assert result['status']=='failed' and not result['forced_termination'] and not result['process_alive']


def test_cache_retrieval_cannot_renew_old_original_observation_clock(tmp_path,monkeypatch):
 reader,_,_=make(tmp_path)
 try:
  original=time.perf_counter
  with monkeypatch.context() as scoped:
   scoped.setattr(module.time,'perf_counter',lambda:original()+1.)
   with pytest.raises(TimeoutError,match='Original guard metadata'):reader.snapshot()
 finally:result=reader.close()
 assert not result['process_alive'] and not result['receiver_alive']
