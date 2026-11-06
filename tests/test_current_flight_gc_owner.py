import gc,json
from flybrain_drive import current_flight_gc_owner as module

def test_gc_deferred_during_owner_and_restored_after(tmp_path,monkeypatch):
    before=gc.isenabled();callbacks=list(gc.callbacks);seen=[]
    def serve(*args):seen.append(gc.isenabled())
    monkeypatch.setattr(module,'_serve_current_owner',serve)
    module._serve_gc_owner(None,'checkpoint','hash',tmp_path,True)
    assert seen==[False] and gc.isenabled()==before and gc.callbacks==callbacks
    value=json.loads((tmp_path/'owner-gc-scope.json').read_text())
    assert value['previous_state_restored'] is True
