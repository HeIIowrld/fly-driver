import pytest
from flybrain_drive.unified_circuit_worker import UnifiedCircuitWorker

class Pipe:
 def __init__(self,ready=True,response=None):self.ready=ready;self.response=response;self.sent=[]
 def poll(self,timeout):return self.ready
 def recv(self):return self.response
 def send(self,value):self.sent.append(value)
class Process:
 def is_alive(self):return True

def make(pipe):
 worker=UnifiedCircuitWorker.__new__(UnifiedCircuitWorker);worker.pipe=pipe;worker.timeout=.01;worker.poisoned=False;worker.sequence=0;worker.process=Process();return worker

def test_timeout_fences_retry_and_reset():
 worker=make(Pipe(False))
 with pytest.raises(TimeoutError,match='unknown'):worker.step(when=1.,scene=None,context=None)
 with pytest.raises(RuntimeError,match='no retry'):worker.step(when=1.,scene=None,context=None)
 with pytest.raises(RuntimeError,match='no reset'):worker.reset()
 assert len(worker.pipe.sent)==1

def test_wrong_sequence_fences_worker():
 worker=make(Pipe(True,{'status':'ok','sequence':2}))
 with pytest.raises(RuntimeError,match='identity'):worker.step(when=1.,scene=None,context=None)
 assert worker.poisoned
