from unittest.mock import Mock
from test_current_flight_owner_protocol import OwnerProtocolTests
from flybrain_drive.current_flight_control_owner import CurrentFlightControlOwnerProcess

class ControlOwnerProtocolTests(OwnerProtocolTests):
    def owner(self):
        owner=CurrentFlightControlOwnerProcess.__new__(CurrentFlightControlOwnerProcess)
        owner.pipe=Mock();owner.poisoned=False;owner.pending_unknown=False;owner.stages=[];owner.stopped=False
        return owner


def test_idle_history_maintenance_does_not_advance_inference_or_modify_source_receipts():
    from flybrain_drive.current_flight_control_owner import receive_with_history
    pipe=Mock();selector=Mock();pipe.poll.side_effect=[False,False,True]
    expected={'kind':'context','request_id':3};pipe.recv.return_value=expected
    assert receive_with_history(pipe,selector,'live_readonly') is expected
    assert selector._drain_native.call_count==3
    pipe.poll.assert_called_with(.005)


def test_replay_receives_without_live_history_maintenance():
    from flybrain_drive.current_flight_control_owner import receive_with_history
    pipe=Mock();selector=Mock();receive_with_history(pipe,selector,'recorded_replay')
    selector._drain_native.assert_not_called();pipe.poll.assert_not_called()
