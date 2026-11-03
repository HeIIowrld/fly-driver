import gc

import pytest

from flybrain_drive.ac_runtime_gc import defer_automatic_cyclic_gc


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('raises', [True, False])
def test_finite_scope_restores_prior_gc_state_on_normal_and_failed_control(enabled, raises):
    original = gc.isenabled()
    try:
        gc.enable() if enabled else gc.disable()
        try:
            with defer_automatic_cyclic_gc() as receipt:
                assert not gc.isenabled()
                if raises:
                    raise RuntimeError('synthetic control abort')
        except RuntimeError:
            assert raises
        assert gc.isenabled() == enabled
        assert receipt['previous_state_restored']
        assert receipt['scope_finished_perf_counter_seconds'] >= (
            receipt['scope_started_perf_counter_seconds'])
    finally:
        gc.enable() if original else gc.disable()
