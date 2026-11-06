import multiprocessing as mp
import sys
from unittest.mock import Mock
import pytest
from test_current_flight_owner_protocol import OwnerProtocolTests
from flybrain_drive.current_flight_actuator_process import CurrentFlightActuatorProcess,start_with_lean_main,_probe_cpu_bootstrap

class ActuatorProtocolTests(OwnerProtocolTests):
    def owner(self):
        owner=CurrentFlightActuatorProcess.__new__(CurrentFlightActuatorProcess)
        owner.pipe=Mock();owner.poisoned=False;owner.pending_unknown=False;owner.stages=[];owner.stopped=False;owner.stop_requested=False
        return owner


def test_control_process_bootstraps_without_torch_even_when_parent_imported_it():
    import torch
    original=sys.modules['__main__'];context=mp.get_context('spawn');parent,child=context.Pipe()
    process=context.Process(target=_probe_cpu_bootstrap,args=(child,))
    try:
        start_with_lean_main(process);child.close()
        assert sys.modules['__main__'] is original
        assert parent.poll(10);value=parent.recv();assert not value['torch_loaded']
        process.join(2);assert process.exitcode==0
    finally:
        parent.close()
        if process.is_alive():process.terminate();process.join(2)


def test_bootstrap_restores_gpu_main_after_spawn_failure():
    original=sys.modules['__main__'];process=Mock();process.start.side_effect=OSError('spawn failure')
    with pytest.raises(OSError):start_with_lean_main(process)
    assert sys.modules['__main__'] is original
