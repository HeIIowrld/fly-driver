import pytest
import torch

from flybrain_drive.cpu_body_live_shadow import CpuBodyLiveShadow


def test_cpu_shadow_rejects_invalid_source_and_gpu_body_timing():
    shadow = object.__new__(CpuBodyLiveShadow)
    with pytest.raises(ValueError, match='GPU scene'):
        shadow._prepare_body_scene(torch.zeros(1, 576))
    with pytest.raises(ValueError, match='CPU body'):
        shadow.enable_diagnostic_cuda_timing()
    with shadow._body_device_scope():
        assert torch.zeros(1).device.type == 'cpu'
    assert shadow._body_thread_time() >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU scene transfer required')
def test_cpu_shadow_copies_completed_gpu_scene_without_alias():
    shadow = object.__new__(CpuBodyLiveShadow)
    original = torch.arange(576, device='cuda', dtype=torch.float32).reshape(1, 576)
    result = shadow._prepare_body_scene(original)
    assert result.device.type == 'cpu'
    torch.testing.assert_close(result, original.cpu(), atol=0, rtol=0)
    original.zero_()
    assert result[0, -1] == 575
    with pytest.raises(ValueError, match='GPU scene'):
        shadow._prepare_body_scene(torch.zeros(1, 575, device='cuda'))
