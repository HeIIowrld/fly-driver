from types import SimpleNamespace

import pytest

from flybrain_drive.cpu_body_process import CpuBodyProcess


def worker(connection):
    instance = object.__new__(CpuBodyProcess)
    instance.connection = connection
    instance.process = SimpleNamespace(is_alive=lambda: True)
    instance.timeout, instance.sequence, instance.failed = 1, 0, False
    instance.closed, instance.forced_termination = False, False
    return instance


def test_timeout_marks_worker_failed_and_does_not_retry():
    sent = []
    conn = SimpleNamespace(send=sent.append, poll=lambda timeout: False)
    subject = worker(conn)
    with pytest.raises(TimeoutError):
        subject.update(when=1, body=None, scene=None, context=None)
    assert subject.failed and len(sent) == 1 and sent[0]['sequence'] == 1
    with pytest.raises(RuntimeError, match='no automatic reset'):
        subject.update(when=1, body=None, scene=None, context=None)
    assert len(sent) == 1


def test_worker_eof_and_sequence_mismatch_fail_closed():
    def eof():
        raise EOFError()
    conn = SimpleNamespace(send=lambda value: None, poll=lambda timeout: True, recv=eof)
    subject = worker(conn)
    with pytest.raises(RuntimeError, match='connection closed'):
        subject.update(when=1, body=None, scene=None, context=None)
    assert subject.failed
    conn.recv = lambda: {'status': 'updated', 'sequence': 3}
    subject = worker(conn)
    with pytest.raises(RuntimeError, match='sequence mismatch'):
        subject.update(when=1, body=None, scene=None, context=None)
    assert subject.failed and subject.sequence == 0


def test_failed_owned_worker_is_terminated_after_bounded_join():
    events = []
    class Process:
        exitcode = None
        alive = True
        def is_alive(self): return self.alive
        def join(self, timeout): events.append(('join', timeout))
        def terminate(self):
            events.append(('terminate',))
            self.alive, self.exitcode = False, 1
    conn = SimpleNamespace(close=lambda: events.append(('close',)))
    subject = worker(conn)
    subject.process, subject.failed = Process(), True
    assert subject.close() == 1
    assert subject.forced_termination and subject.closed
    assert events == [('close',), ('join', 1), ('terminate',), ('join', 5)]
    assert subject.close() == 1
