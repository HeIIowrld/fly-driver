import subprocess
from scripts import run_owned_native_motion_shadow as launcher


def test_metadata_permission_error_does_not_abandon_or_restart_owned_child(tmp_path,monkeypatch):
    class Child:
        pid=123
        def __init__(self):self.waits=0
        def wait(self,timeout):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('fixture',timeout)
            return 7
        def poll(self):return None
    def denied(*args,**kwargs):raise PermissionError('synthetic sharing violation')
    monkeypatch.setattr(launcher,'write',denied)
    child=Child();errors=[]
    assert launcher.wait_owned_child(child,tmp_path/'owned',errors)==7
    assert child.waits==2 and len(errors)==1
