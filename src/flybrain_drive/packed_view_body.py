"""Optional batched per-view body tensor construction, not installed in the actor."""

import numpy as np
import torch


def pack_view_body(contexts, *, device, dtype):
    if len(contexts) != 4:
        raise ValueError('Four original per-view body contexts required')
    # Preserve the original per-view conversion for float64 experimental streams.
    values = np.zeros((1, 4, 21), dtype=np.float64 if dtype == torch.float64 else np.float32)
    valid = np.zeros((1, 4, 21), dtype=np.bool_)
    for index, observation in enumerate(contexts):
        if observation is not None:
            if observation.values.shape != (21,) or observation.valid.shape != (21,):
                raise ValueError('Original 21-channel view body and mask required')
            values[0, index] = observation.values
            valid[0, index] = observation.valid
    return (torch.as_tensor(values, dtype=dtype, device=device),
            torch.as_tensor(valid, dtype=torch.bool, device=device))
