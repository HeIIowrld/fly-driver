"""Synthetic algebra/gradient checks, not measured physiology evidence."""
import unittest

import torch

from flybrain_drive.flight_anticipation_readout import FlightAnticipationReadout


class FlightAnticipationTests(unittest.TestCase):
    def test_target_only_loss_and_masked_absence(self):
        head = FlightAnticipationReadout(torch.zeros(6), torch.ones(6), 2)
        muscles = torch.ones(1, 6)
        before = head(muscles).detach().clone()
        target = torch.tensor([[[1., .5, 0.], [float('nan')] * 3]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        prediction = head(muscles)
        loss = head.masked_loss(prediction, target, mask)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(head.projection.weight.grad).all())
        self.assertTrue(torch.equal(before, head(muscles)))
        self.assertEqual(float(head.projection.weight.grad[3:].abs().sum()), 0.)

    def test_auxiliary_gradients_reach_existing_muscle_state(self):
        head = FlightAnticipationReadout(torch.zeros(6), torch.ones(6), 1)
        with torch.no_grad():
            head.projection.weight.fill_(.1)
        muscles = torch.ones(1, 6, requires_grad=True)
        loss = head.masked_loss(head(muscles), torch.zeros(1, 1, 3), torch.ones(1, 1, dtype=torch.bool))
        loss.backward()
        self.assertGreater(float(muscles.grad.abs().sum()), 0.)

    def test_no_supervision_has_zero_gradient(self):
        head = FlightAnticipationReadout(torch.zeros(6), torch.zeros(6), 1)
        muscles = torch.ones(1, 6, requires_grad=True)
        loss = head.masked_loss(head(muscles), torch.full((1, 1, 3), float('nan')),
                                torch.zeros(1, 1, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(muscles.grad.abs().sum()), 0.)

    def test_calibration_is_copied_and_bad_inputs_rejected(self):
        ref, scale = torch.zeros(6), torch.ones(6)
        head = FlightAnticipationReadout(ref, scale)
        ref.fill_(10)
        scale.fill_(0)
        self.assertTrue(torch.equal(head.reference, torch.zeros(6)))
        self.assertTrue(torch.equal(head.scale, torch.ones(6)))
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 576))
        with self.assertRaises(ValueError):
            head.masked_loss(head(torch.zeros(1, 6)), torch.zeros(1, 4, 3), torch.ones(1, 4))


if __name__ == '__main__':
    unittest.main()
