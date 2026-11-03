import threading,time
import pytest
from flybrain_drive.owned_stop_monitor import OwnedStopMonitor


def test_blocked_nas_observation_does_not_block_control_check_and_expires_original_age(tmp_path):
 release=threading.Event();blocked=threading.Event();calls=[]
 def lookup():
  calls.append(1)
  if len(calls)>1:blocked.set();release.wait(2)
  return False
 monitor=OwnedStopMonitor(tmp_path/'stop',maximum_age=.08,period=.005,exists=lookup)
 try:
  assert blocked.wait(1)
  start=time.perf_counter();monitor.check()
  assert time.perf_counter()-start<.01
  time.sleep(.09)
  with pytest.raises(TimeoutError,match='Original stop'):monitor.check()
 finally:
  release.set();result=monitor.close()
 assert not result['worker_alive'] and result['maximum_read_ms']>=80


def test_real_owned_stop_file_and_read_failures_are_not_ignored(tmp_path):
 stop=tmp_path/'stop';monitor=OwnedStopMonitor(stop,period=.005)
 try:
  stop.touch();assert monitor.requested.wait(1)
  with pytest.raises(RuntimeError,match='stop requested'):monitor.check()
 finally:assert not monitor.close()['worker_alive']
 def failed():raise OSError('unavailable share')
 with pytest.raises(RuntimeError,match='unavailable share'):
  OwnedStopMonitor(tmp_path/'other',exists=failed)
