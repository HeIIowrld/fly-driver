"""Compare equivalent body-state validation predicates on GPU; no simulator."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def original(state, drive):
    return (not torch.isfinite(state).all() or not torch.isfinite(drive).all()
            or torch.any((state < 0) | (state > 1)))


def combined(state, drive):
    return not (torch.isfinite(state).all() & torch.isfinite(drive).all()
                & (~((state < 0) | (state > 1)).any()))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.manual_seed(863009)
    state = torch.rand((1, 9728), device="cuda")
    drive = torch.rand_like(state)
    variants = [(state, drive)]
    for index, value, which in ((5, float('nan'), 'state'), (7, float('inf'), 'state'),
                                 (8, -0.001, 'state'), (9, 1.001, 'state'),
                                 (10, float('nan'), 'drive'), (11, float('inf'), 'drive')):
        s, d = state.clone(), drive.clone()
        (s if which == 'state' else d)[0, index] = value
        variants.append((s, d))
    decisions = [(bool(original(s, d)), bool(combined(s, d))) for s, d in variants]
    if decisions != [(False, False)] + [(True, True)] * 6:
        raise AssertionError("Validation differs or admits invalid input")
    for _ in range(30):
        original(state, drive); combined(state, drive)
    durations = {"original": [], "combined": []}
    for i in range(200):
        functions = [("original", original), ("combined", combined)]
        if i % 2: functions.reverse()
        for name, fn in functions:
            torch.cuda.synchronize(); start = time.perf_counter_ns()
            assert not fn(state, drive)
            torch.cuda.synchronize()
            durations[name].append((time.perf_counter_ns() - start) / 1e6)
    out = Path("C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-body-validation-sync-benchmark")
    run = begin_run(out, {"program_sha256": sha256(__file__), "seed": 863009,
        "state_size": 9728, "pairs": 200, "order": "alternating", "device": torch.cuda.get_device_name(),
        "scope": "isolated predicate only, no concurrent visual workload"})
    write_json(run / "samples.json", durations)
    result = {"status": "completed", "invalid_case_decisions": decisions,
        "milliseconds_p50_p95_max": {k: np.percentile(v, [50, 95, 100]).tolist() for k, v in durations.items()},
        "samples_sha256": sha256(run / "samples.json"), "vehicle_control": False,
        "concurrent_gpu_or_live_freshness_verified": False}
    write_json(run / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
