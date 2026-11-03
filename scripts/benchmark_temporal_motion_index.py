"""Offline comparison of identical motion-channel extraction operations."""

import json
import time

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.manual_seed(863009)
    scene = torch.randn(1, 576, device='cuda')
    mean = torch.randn_like(scene)
    scale = torch.rand_like(scene) + .1
    mask = torch.zeros((4, 9, 16), dtype=torch.bool, device='cuda')
    mask[:, 1:] = True
    mask = mask.flatten()

    def old():
        return ((scene-mean)/scale).clamp(-5, 5)[:, mask]

    def sliced():
        return ((scene-mean)/scale).clamp(-5, 5).reshape(-1, 4, 9, 16)[:, :, 1:, :].reshape(-1, 512)

    if not torch.equal(old(), sliced()):
        raise AssertionError('Motion features differ')
    for _ in range(30):
        old(); sliced()
    times = {'mask': [], 'slice': []}
    for i in range(200):
        ops = [('mask', old), ('slice', sliced)]
        if i % 2:
            ops.reverse()
        for name, operation in ops:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            operation()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter_ns()-start)/1e6)
    run = begin_run('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-temporal-motion-index-benchmark',
                    {'program_sha256': sha256(__file__), 'seed': 863009, 'pairs': 200,
                     'device': torch.cuda.get_device_name(), 'order': 'alternating',
                     'scope': 'isolated extraction, no concurrent visual workload'})
    write_json(run/'samples.json', times)
    output = {'status': 'completed', 'identical': True,
              'milliseconds_p50_p95_max': {k: np.percentile(v, [50, 95, 100]).tolist() for k, v in times.items()},
              'samples_sha256': sha256(run/'samples.json'), 'vehicle_control': False,
              'concurrent_gpu_or_live_freshness_verified': False}
    write_json(run/'result.json', output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
