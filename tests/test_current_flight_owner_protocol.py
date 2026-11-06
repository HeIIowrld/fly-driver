import unittest
from unittest.mock import Mock,patch
from flybrain_drive.current_flight_owner import CurrentFlightOwnerProcess

class OwnerProtocolTests(unittest.TestCase):
    def owner(self):
        owner=CurrentFlightOwnerProcess.__new__(CurrentFlightOwnerProcess)
        owner.pipe=Mock();owner.poisoned=False;owner.pending_unknown=False;owner.stages=[];owner.stopped=False
        return owner

    def test_stage_messages_share_absolute_deadline(self):
        owner=self.owner();owner.pipe.poll.side_effect=[True,False]
        owner.pipe.recv.return_value=dict(type='stage',request_id=3,stage='inference')
        with patch('flybrain_drive.current_flight_owner.time.perf_counter',side_effect=[10.,10.01,10.02]):
            with self.assertRaises(TimeoutError):owner._receive(3,'publish',10.05)
        self.assertEqual(len(owner.stages),1)
        self.assertAlmostEqual(owner.pipe.poll.call_args_list[0].args[0],.05)
        self.assertAlmostEqual(owner.pipe.poll.call_args_list[1].args[0],.03)
        self.assertTrue(owner.pending_unknown)
        with self.assertRaises(RuntimeError):owner.request('status')
        owner.pipe.send.assert_not_called()

    def test_wrong_sequence_poisoned(self):
        owner=self.owner();owner.pipe.poll.return_value=True
        owner.pipe.recv.return_value=dict(type='response',request_id=2,status='published')
        with self.assertRaises(RuntimeError):owner._receive(1,'publish',float('inf'))
        self.assertTrue(owner.poisoned);self.assertTrue(owner.pending_unknown)

    def test_eof_is_unknown_without_retry(self):
        owner=self.owner();owner.pipe.poll.return_value=True;owner.pipe.recv.side_effect=EOFError()
        with self.assertRaisesRegex(RuntimeError,'no retry'):owner._receive(1,'status',float('inf'))
        self.assertTrue(owner.pending_unknown);self.assertTrue(owner.poisoned)

    def test_known_rejection_poisoned_but_not_unknown(self):
        owner=self.owner();owner.pipe.poll.return_value=True
        owner.pipe.recv.return_value=dict(type='response',request_id=1,status='rejected',reason='invalid body')
        with self.assertRaisesRegex(RuntimeError,'invalid body'):owner._receive(1,'publish',float('inf'))
        self.assertTrue(owner.poisoned);self.assertFalse(owner.pending_unknown)

if __name__=='__main__':unittest.main()
