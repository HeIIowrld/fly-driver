"""Finite control-loop scheduling; reference counting and explicit gc.collect remain active."""

import gc
import time
from contextlib import contextmanager


@contextmanager
def defer_automatic_cyclic_gc():
    """Caller must bound the scope and close actuators before leaving it, including errors."""
    before = gc.isenabled()
    started = time.perf_counter()
    receipt = dict(automatic_cyclic_gc_enabled_before=before,
                   scope_started_perf_counter_seconds=started,
                   scope='finite control loop, including neutral acquisition and device release',
                   reference_counting_disabled=False,
                   explicit_collection_blocked=False)
    gc.disable()
    try:
        yield receipt
    finally:
        if before:
            gc.enable()
        else:
            gc.disable()
        receipt.update(scope_finished_perf_counter_seconds=time.perf_counter(),
                       automatic_cyclic_gc_enabled_after=gc.isenabled(),
                       previous_state_restored=gc.isenabled() == before)
