"""Read-only window/terminal observations outside finite GPU copy reception."""
import threading,time
from pathlib import Path
from .ac_published_state import read_published_json

class CameraReceiveObserver:
 def __init__(self,native,window,terminal_path,*,clock=time.perf_counter):
  self.native,self.window,self.path,self.clock=native,window,Path(terminal_path),clock
  self.latest=None;self.error=None;self.records=[];self.stop=threading.Event();self.ready=threading.Event()
  self.thread=threading.Thread(target=self._run,name='camera-window-terminal-observer',daemon=True);self.thread.start()
  if not self.ready.wait(5):self.close();raise TimeoutError('Original camera side observations not ready')
  self.snapshot()
 def _run(self):
  try:
   while not self.stop.is_set() and len(self.records)<12000:
    started=self.clock()
    if self.native.probe(self.window.hwnd,require_foreground=False)!=self.window:raise ValueError('Original camera engine changed')
    identity_finished=self.clock();terminal=None
    if self.path.exists():terminal,_=read_published_json(self.path)
    finished=self.clock();row=dict(started=started,finished=finished,terminal=terminal,
     phases_ms=dict(window_identity=(identity_finished-started)*1000,terminal_read=(finished-identity_finished)*1000))
    self.records.append(row);self.latest=row;self.ready.set()
    if terminal is not None:break
    if self.stop.wait(.025):break
   if len(self.records)>=12000:raise RuntimeError('Finite camera observation budget exhausted')
  except BaseException as error:self.error=error;self.ready.set()
 def snapshot(self):
  if self.error is not None:raise RuntimeError('Original camera side observation failed') from self.error
  if self.latest is None or not 0<=self.clock()-self.latest['started']<=.5:raise TimeoutError('Original camera side observation age exceeded500ms')
  return self.latest
 def close(self):
  self.stop.set();self.thread.join(2)
  return dict(worker_alive=self.thread.is_alive(),error=str(self.error) if self.error else None,
   observations=len(self.records),maximum_read_ms=max((r['finished']-r['started'] for r in self.records),default=0)*1000,
   original_observations=self.records)
