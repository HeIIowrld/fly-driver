import threading,time
from types import SimpleNamespace
import pytest
from test_ac_tcp import load_app


def test_tcp_suspension_requires_real_inflight_file_completion_and_no_new_writes(tmp_path):
 app=load_app();writer=app.LatestFilePublisher(str(tmp_path))
 entered=threading.Event();release=threading.Event();writes=[];rows=[]
 def slow_write(name,row):
  writes.append(row);entered.set();assert release.wait(2)
 writer.file._write=slow_write
 hybrid=object.__new__(app.HybridPublisher)
 hybrid.tcp=SimpleNamespace(connection=None,publish=lambda row:rows.append(dict(row)),close=lambda:None)
 hybrid.file=writer;hybrid.last_file=-1.
 try:
  hybrid.publish({'speed':0.});assert entered.wait(1)
  hybrid.tcp.connection=object()
  hybrid.publish({'speed':1.})
  assert rows[-1]['publisher_file_compatibility_suspended'] is True
  assert rows[-1]['publisher_file_io_quiescent'] is False
  assert writer.pending is None
  with pytest.raises(RuntimeError,match='suspended'):writer.publish({'speed':2.})
  release.set();deadline=time.perf_counter()+1
  while writer.writing and time.perf_counter()<deadline:time.sleep(.001)
  hybrid.publish({'speed':3.})
  assert rows[-1]['publisher_file_io_quiescent'] is True
  assert len(writes)==1 and writer.pending is None
  hybrid.tcp.connection=None;hybrid.last_file=-1
  hybrid.publish({'speed':4.})
 finally:release.set();hybrid.close()
 assert not writer.thread.is_alive()
