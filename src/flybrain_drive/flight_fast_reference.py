"""Motor-reference changes only; no direct vehicle action or learned-policy target fitting."""

import torch


def relative_motor_reference(readout, muscles):
    if muscles.ndim != 2 or muscles.shape[1] != 6:
        raise ValueError('Original b1L/R, b2L/R and powerL/R motor activities required')
    lateral = torch.stack([muscles[:, 1]-muscles[:, 0],
                           muscles[:, 3]-muscles[:, 2]], dim=1)
    delta = lateral-lateral[:1]
    references = readout.lateral_reference[None]+delta
    preactivation_shift = readout.turn_log_gain.exp()*(delta@readout.steering_mix.softmax(0))
    return references, preactivation_shift
