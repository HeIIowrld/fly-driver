from unittest.mock import patch
import pytest
import flybrain_drive.flyvis_process as module


class Pipe:
    def __init__(self, messages): self.messages=iter(messages); self.sent=[]; self.closed=False
    def recv(self): return next(self.messages)
    def send(self, value): self.sent.append(value)
    def close(self): self.closed=True

class Engine:
    def close(self): pass
    def __init__(self, *args): self.ids=[]
    def step(self, capture, pixels):
        self.ids.append(capture)
        return {'scene':capture}

def run_service(messages):
    pipe=Pipe(messages); exits=[]
    with patch.object(module, 'FlyvisEngine', Engine), patch.object(module.os, '_exit', lambda code: exits.append(code)):
        module._serve(pipe, None, None, None, None, 'cpu')
    return pipe, exits

def test_visual_service_keeps_one_ordered_owner_and_clean_stop():
    pipe,exits=run_service([{'kind':'step','sequence':1,'capture':7,'pixels':None}, {'kind':'step','sequence':2,'capture':8,'pixels':None}, {'kind':'stop'}])
    assert [x['status'] for x in pipe.sent]==['ready','ok','ok','stopped']
    assert [x['sequence'] for x in pipe.sent[1:]]==[1,2,2]
    assert pipe.sent[-1]['device_released'] is True
    assert exits==[0] and pipe.closed

def test_visual_service_rejects_skipped_sequence_and_exits_failure():
    pipe,exits=run_service([{'kind':'step','sequence':2,'capture':7,'pixels':None}])
    assert pipe.sent[-1]['status']=='failed'
    assert 'ordered' in pipe.sent[-1]['error']
    assert exits==[1] and pipe.closed
