import gc
from types import SimpleNamespace
import pytest
from flybrain_drive.visual_phase_probe import VisualPhaseProbe


class Worker:
    def simulate(self):
        gc.collect(0)
        return 7
    def sample(self): return 3
    def forward(self): return 11


class Stream:
    def __init__(self):
        self.projector=Worker();self.network=Worker();self.pool=Worker()
    def process(self, *, fail=False):
        value=self.projector.sample()+self.network.simulate()+self.pool.forward()
        if fail: raise ValueError('rejected original frame')
        return value


@pytest.mark.parametrize('fail',[False,True])
def test_observation_preserves_return_or_error_and_restores_methods_and_callbacks(fail):
    stream=Stream();before=list(gc.callbacks)
    probe=VisualPhaseProbe(stream,synchronize=lambda:None);record={}
    if fail:
        with pytest.raises(ValueError,match='original frame'):probe.run('process',record,fail=True)
    else:assert probe.run('process',record)==21
    assert gc.callbacks==before and not probe.active
    assert all(not obj.__dict__ for obj in [stream.projector,stream.network,stream.pool])
    assert len(record['phases'])==3 and record['gc_collections']
    assert record['wall_seconds']>=0 and record['final_synchronize_seconds']>=0
    assert ('error' in record)==fail


def test_instance_override_restored_after_probe():
    stream=Stream();override=lambda:19;stream.network.simulate=override
    probe=VisualPhaseProbe(stream,synchronize=lambda:None)
    assert probe.run('process',{})==33
    assert stream.network.simulate is override
