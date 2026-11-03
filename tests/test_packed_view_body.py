from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.packed_view_body import pack_view_body


def test_missing_view_and_masks_remain_explicit():
    values = np.arange(21, dtype=np.float32)
    mask = np.arange(21) % 2 == 0
    one = SimpleNamespace(values=values, valid=mask)
    packed, valid = pack_view_body((one, None, one, None), device='cpu', dtype=torch.float32)
    assert torch.equal(packed[0, 0], torch.from_numpy(values))
    assert torch.equal(valid[0, 2], torch.from_numpy(mask))
    assert not packed[0, 1].any() and not valid[0, 1].any()
    with pytest.raises(ValueError, match='Four'):
        pack_view_body((one,), device='cpu', dtype=torch.float32)


def test_float64_context_is_not_silently_rounded_to_float32():
    precise = np.full(21, 1.0000000001, dtype=np.float64)
    observation = SimpleNamespace(values=precise, valid=np.ones(21, dtype=bool))
    value, _ = pack_view_body((observation, None, None, None), device='cpu', dtype=torch.float64)
    assert value[0, 0, 0].item() == precise[0]
