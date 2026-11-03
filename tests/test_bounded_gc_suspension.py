import gc
import pytest
from flybrain_drive.bounded_gc_suspension import BoundedGCSuspension

def test_suspension_restores_existing_state():
    original=gc.isenabled()
    try:
        gc.enable();control=BoundedGCSuspension();receipt=control.begin()
        assert receipt['previously_enabled'] and not gc.isenabled()
        with pytest.raises(RuntimeError):control.begin()
        assert control.restore()['restored'] and gc.isenabled()
        gc.disable();control=BoundedGCSuspension();control.begin();assert control.restore()['restored'] and not gc.isenabled()
    finally:
        gc.enable() if original else gc.disable()
