import json,threading,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from flybrain_drive.camera_receive_observer import CameraReceiveObserver

def test_file_and_window_queries_stay_out_of_consumer_and_original_terminal_survives(tmp_path,monkeypatch):
 window=SimpleNamespace(hwnd=1);path=tmp_path/'result.json';original=Path.exists;calls=[]
 def exists(self):
  calls.append(threading.current_thread().name)
  assert threading.current_thread().name=='camera-window-terminal-observer'
  return original(self)
 monkeypatch.setattr(Path,'exists',exists)
 native=SimpleNamespace(probe=lambda *args,**kwargs:window)
 reader=CameraReceiveObserver(native,window,path)
 try:
  first=reader.snapshot();before=first['started']
  for _ in range(100):assert reader.snapshot()['started']>=before
  terminal=dict(status='stopped',frames=3);path.write_text(json.dumps(terminal))
  deadline=time.perf_counter()+2
  while reader.snapshot()['terminal'] is None and time.perf_counter()<deadline:time.sleep(.01)
  assert reader.snapshot()['terminal']==terminal and first['started']==before
 finally:result=reader.close()
 assert calls and not result['worker_alive'] and result['error'] is None

def test_old_observation_cannot_be_renewed_by_cache_access(tmp_path):
 window=SimpleNamespace(hwnd=1);reader=CameraReceiveObserver(SimpleNamespace(probe=lambda *a,**k:window),window,tmp_path/'result.json')
 try:
  reader.clock=lambda:time.perf_counter()+1
  with pytest.raises(TimeoutError,match='Original camera'):reader.snapshot()
 finally:reader.close()
