"""Bounded automatic-GC suspension for short inference diagnostics."""
import gc

class BoundedGCSuspension:
    def __init__(self):
        self.active=False
        self.previously_enabled=None
        self.collected=None
    def begin(self):
        if self.active: raise RuntimeError('GC suspension already active')
        self.previously_enabled=gc.isenabled()
        # Complete outstanding collection before the bounded observation starts.
        self.collected=gc.collect()
        gc.disable()
        self.active=True
        return dict(previously_enabled=self.previously_enabled,collected=self.collected,auto_gc_enabled=gc.isenabled())
    def restore(self):
        if not self.active: raise RuntimeError('GC suspension is not active')
        if self.previously_enabled:gc.enable()
        else:gc.disable()
        self.active=False
        return dict(auto_gc_enabled=gc.isenabled(),restored=gc.isenabled()==self.previously_enabled)
