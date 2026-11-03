import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flybrain_drive.win_process_receipt import ProcessReceipt

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows native process receipt')


def test_receipt_observes_same_child_handle_through_actual_exit():
    child = subprocess.Popen([sys.executable, '-c',
        'import time,sys; time.sleep(.25); sys.exit(7)'],
        creationflags=subprocess.CREATE_NO_WINDOW)
    receipt = ProcessReceipt(child.pid)
    try:
        assert receipt.snapshot()['active']
        assert child.wait(timeout=5) == 7
        assert receipt.snapshot() == dict(pid=child.pid, active=False, exit_code=7,
            exit_code_reported=7, termination_signaled=True, exit_requested=False)
    finally:
        receipt.close()


def test_closed_receipt_cannot_be_used_as_evidence_of_process_termination():
    receipt = ProcessReceipt(os.getpid())
    assert receipt.snapshot()['active']
    receipt.close()
    with pytest.raises(ValueError, match='closed'):
        receipt.snapshot()


def test_exit_code_alone_cannot_mark_a_process_with_pending_teardown_terminated():
    def exit_code(handle, value):
        value._obj.value = 1
        return True

    receipt = object.__new__(ProcessReceipt)
    receipt.pid, receipt.handle = 123, 456  # Synthetic handle protocol; no native access.
    receipt.api = SimpleNamespace(GetExitCodeProcess=exit_code,
                                  WaitForSingleObject=lambda handle, timeout: 258)
    assert receipt.snapshot() == dict(pid=123, active=True, exit_code=None,
        exit_code_reported=1, termination_signaled=False, exit_requested=True)
