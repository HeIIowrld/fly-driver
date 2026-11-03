import numpy as np
import pytest
from flybrain_drive.body_circuit_process import _command_loop

class Pipe:
    def __init__(self,messages):self.messages=iter(messages);self.sent=[]
    def recv(self):return next(self.messages)
    def send(self,value):self.sent.append(value)
class Thread:
    def __init__(self,alive=True):self.alive=alive
    def is_alive(self):return self.alive
class Ticker:
    def __init__(self,error=None,alive=True):self.error=error;self.thread=Thread(alive);self.inputs=[]
    def publish(self,scene,context,capture):
        self.inputs.append((scene.numpy().copy(),context,capture))
        return 14.25

def test_publications_preserve_context_order_and_explicit_stop():
    image=np.arange(576,dtype=np.float32).reshape(1,576);ctx=object();ticker=Ticker()
    pipe=Pipe([{'kind':'publish','sequence':1,'scene':image,'context':ctx,'capture':19},{'kind':'status'},{'kind':'stop'}])
    assert _command_loop(pipe,ticker)==1
    assert pipe.sent==[{'status':'published','sequence':1,'publication':14.25},{'status':'status','sequence':1,'error':None,'alive':True}]
    assert np.array_equal(ticker.inputs[0][0],image) and ticker.inputs[0][1] is ctx and ticker.inputs[0][2]==19

def test_failed_ticker_and_out_of_order_publication_are_rejected():
    scene=np.zeros((1,576),dtype=np.float32)
    with pytest.raises(RuntimeError,match='ticker unavailable'):
        _command_loop(Pipe([{'kind':'publish','sequence':1,'scene':scene}]),Ticker(error={'reason':'stale'}))
    with pytest.raises(ValueError,match='Ordered'):
        _command_loop(Pipe([{'kind':'publish','sequence':2,'scene':scene}]),Ticker())

def test_invalid_scene_is_not_delivered():
    ticker=Ticker()
    with pytest.raises(ValueError,match='Finite'):
        _command_loop(Pipe([{'kind':'publish','sequence':1,'scene':np.full((1,576),np.nan,dtype=np.float32)}]),ticker)
    assert ticker.inputs==[]

def test_failed_owner_receipt_requires_process_exit_and_matching_hashes(tmp_path):
    import json
    from flybrain_drive.body_circuit_process import BodyCircuitProcess
    from flybrain_drive.graph import sha256
    events=tmp_path/'ticker-events.json'; native=tmp_path/'native-copy.json'
    events.write_text('[]',encoding='utf8');native.write_text('[]',encoding='utf8')
    receipt={'status':'failed','sequence':2,'events_sha256':sha256(events),'native_sha256':sha256(native)}
    (tmp_path/'result.json').write_text(json.dumps(receipt),encoding='utf8')
    class Process:
        def join(self,timeout):pass
        def is_alive(self):return False
    class Connection:
        def close(self):pass
    owner=BodyCircuitProcess.__new__(BodyCircuitProcess);owner.poisoned=True;owner.process=Process();owner.pipe=Connection();owner.evidence_dir=tmp_path
    assert owner.collect_failure()['sequence']==2
    events.write_text('[1]',encoding='utf8')
    with pytest.raises(ValueError,match='evidence'):
        owner.collect_failure()
