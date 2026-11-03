"""Bounded original stop-file observation outside the40ms control loop.

The NAS pathname can block in Windows filesystem I/O. Only the observer reads
it; control consumes its original completion clock and fails closed if that
metadata is unavailable for500ms. No controls or lifecycle APIs exist here.
"""
import threading,time
from pathlib import Path

class OwnedStopMonitor:
 def __init__(self,path,*,maximum_age=.5,period=.05,exists=None):
  if not .02<=maximum_age<=.5 or not .001<=period<=.1:raise ValueError('Bounded stop metadata polling required')
  self.path=Path(path);self.exists=exists or self.path.exists
  self.maximum_age,self.period=maximum_age,period
  self.stopped=threading.Event();self.requested=threading.Event();self.ready=threading.Event()
  self.observed=None;self.error=None;self.maximum_read_seconds=0.;self.reads=0
  self.thread=threading.Thread(target=self._observe,name='owned-nas-stop-observer',daemon=True)
  self.thread.start()
  if not self.ready.wait(1):
   self.stopped.set();raise TimeoutError('Stop metadata observer did not initialize before control')
  try:self.check()
  except BaseException:
   self.stopped.set();self.thread.join(.5);raise
 def _observe(self):
  try:
   while not self.stopped.is_set():
    started=time.perf_counter();requested=self.exists();completed=time.perf_counter()
    self.maximum_read_seconds=max(self.maximum_read_seconds,completed-started)
    self.reads+=1;self.observed=started
    if requested:self.requested.set()
    self.ready.set()
    if self.stopped.wait(self.period):break
  except BaseException as error:
   self.error=str(error);self.ready.set()
 def check(self):
  if self.requested.is_set():raise RuntimeError('Owned run stop requested')
  if self.error:raise RuntimeError('Stop metadata observer failed: '+self.error)
  if self.observed is None or not 0<=time.perf_counter()-self.observed<=self.maximum_age:
   raise TimeoutError('Original stop metadata observation exceeded500ms')
 def close(self):
  self.stopped.set();self.thread.join(.5)
  return dict(reads=self.reads,maximum_read_ms=self.maximum_read_seconds*1000,
   original_observation_time=self.observed,stop_requested=self.requested.is_set(),
   error=self.error,worker_alive=self.thread.is_alive())
