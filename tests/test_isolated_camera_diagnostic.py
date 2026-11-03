import pytest
from scripts.isolated_camera_diagnostic import IsolatedCamera
from flybrain_drive.ac_csp_vision import NoFreshCSPFrame

class FakePipe:
    def __init__(self, ready, response=None):
        self.ready, self.response, self.sent = ready, response, []
    def poll(self, timeout):
        return self.ready
    def recv(self):
        return self.response
    def send(self, value):
        self.sent.append(value)

def test_unknown_owner_response_fences_all_later_commands():
    camera = IsolatedCamera.__new__(IsolatedCamera)
    camera.pipe, camera.poisoned = FakePipe(False), False
    with pytest.raises(TimeoutError, match='unknown'):
        camera.call('candidate', timeout=0)
    with pytest.raises(RuntimeError, match='no further commands'):
        camera.call('close')
    assert camera.pipe.sent == [('candidate', None)]

def test_no_fresh_candidate_does_not_poison_owner():
    camera = IsolatedCamera.__new__(IsolatedCamera)
    camera.pipe, camera.poisoned = FakePipe(True, ('no_frame', 'old')), False
    with pytest.raises(NoFreshCSPFrame):
        camera.call('candidate')
    assert not camera.poisoned
