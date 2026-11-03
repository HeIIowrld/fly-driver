import pytest

from scripts.select_remote_egl import choose_device


def test_egl_numbering_can_differ_from_cuda_and_slurm_numbering():
    devices = [
        {"egl_index": 0, "cuda_uuid": None},
        {"egl_index": 1, "cuda_uuid": "other-gpu"},
        {"egl_index": 2, "cuda_uuid": "allocated-gpu", "cuda_device": 0},
    ]
    assert choose_device(devices, "allocated-gpu") == 2


@pytest.mark.parametrize("devices", [[], [{"egl_index": 0, "cuda_uuid": None}], [
    {"egl_index": 0, "cuda_uuid": "allocated-gpu"},
    {"egl_index": 1, "cuda_uuid": "allocated-gpu"},
]])
def test_missing_or_ambiguous_uuid_is_rejected_without_gpu_fallback(devices):
    with pytest.raises(RuntimeError, match="Expected one EGL"):
        choose_device(devices, "allocated-gpu")
