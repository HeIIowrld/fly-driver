"""Explicit projection onto measured pedal capabilities; never a learned controller.

t1674 established zero below a cutoff and identity at/above .02, rather than
an affine deadzone. The conservative feasible set is {0} union [.02, 1].
The unknown interval below .02 is deliberately excluded. A nearest projection
minimizes instantaneous request error; a midpoint tie selects zero. No duty
cycling, additive correction, steering change, or hidden state is introduced.
"""

import numpy as np

CODEC_CONTRACT = "flybrain.actuator.minimum_positive_projection.v1"


def project_pedals(canonical, *, minimum_positive=.02):
    """Return an auditable encoding and the cost available to future training.

    This is a request-side capability model. Only separate native observations
    establish application. Small signed intents remain available unchanged.
    """
    values = np.asarray(canonical, dtype=float)
    if (values.shape != (3,) or not np.isfinite(values).all()
            or not -1 <= values[0] <= 1 or np.any(values[1:] < 0)
            or np.any(values[1:] > 1) or not np.isfinite(minimum_positive)
            or not 0 < minimum_positive <= 1):
        raise ValueError("Finite canonical controls and explicit positive pedal floor required")
    encoded = values.copy()
    gap = (values[1:] > 0) & (values[1:] < minimum_positive)
    encoded[1:] = np.where(gap,
        np.where(values[1:] > minimum_positive / 2, minimum_positive, 0.), values[1:])
    error = encoded - values
    return dict(contract=CODEC_CONTRACT, minimum_positive=float(minimum_positive),
        canonical_request=values.tolist(), encoded_request=encoded.tolist(),
        encoding_error=error.tolist(), squared_projection_distance=float(error @ error),
        pedal_in_gap=gap.tolist(), game_application_verified=False,
        policy="nearest feasible value; midpoint tie zero; above floor identity")
