"""Compare batched and original per-view body transfers on original recorded contexts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.packed_view_body import pack_view_body
from scripts.replay_multirate_flight import original_view


def original(contexts):
    history = torch.zeros((1, 4, 21), dtype=torch.float32, device="cuda")
    valid = torch.zeros((1, 4, 21), dtype=torch.bool, device="cuda")
    for index, observation in enumerate(contexts):
        if observation is not None:
            if observation.values.shape != (21,) or observation.valid.shape != (21,):
                raise ValueError("Original 21-channel view body and mask required")
            history[0, index] = torch.as_tensor(
                observation.values.copy(), dtype=torch.float32, device="cuda"
            )
            valid[0, index] = torch.as_tensor(
                observation.valid.copy(), dtype=torch.bool, device="cuda"
            )
    return history, valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--body-audit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, audit = Path(a.recording), Path(a.body_audit)

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf8"))

    result, proof, manifest = (
        read(root / "result.json"),
        read(audit / "result.json"),
        read(audit / "manifest.json"),
    )
    if (
        result["status"] != "completed"
        or proof["status"] != "completed_online_body_shadow_audit"
        or manifest["config"]["recording_sha256"] != sha256(root / "result.json")
        or sha256(root / "online-body-shadow.json") != result["shadow_sha256"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact audited original shadow and GPU required")
    rows = read(root / "online-body-shadow.json")
    contract = sha256("configs/ac-virtual-navigation-body-v2.json")
    contexts = [
        tuple(
            original_view(
                view, episode=result["request_id"], camera=e["camera_receipt"], contract=contract
            )
            for view in e["view_body"]
        )
        for e in rows
        if e.get("status") == "context_built"
    ]
    if not contexts:
        raise ValueError("Original body contexts required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "candidate_sha256": sha256("src/flybrain_drive/packed_view_body.py"),
            "recording_sha256": sha256(root / "result.json"),
            "shadow_sha256": result["shadow_sha256"],
            "audit_sha256": sha256(audit / "result.json"),
            "seed": 863009,
            "scope": "synchronized isolated input-packing only; not live concurrency",
        },
    )
    records = []
    with torch.no_grad():
        for iteration in range(4):
            for index, context in enumerate(contexts):
                outputs, durations = {}, {}
                order = (
                    ("original", "packed")
                    if (iteration + index) % 2 == 0
                    else ("packed", "original")
                )
                for name in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs[name] = (
                        original(context)
                        if name == "original"
                        else pack_view_body(context, device="cuda", dtype=torch.float32)
                    )
                    torch.cuda.synchronize()
                    durations[name] = 1000 * (time.perf_counter() - start)
                if not all(
                    torch.equal(x, y)
                    for x, y in zip(outputs["original"], outputs["packed"], strict=False)
                ):
                    raise ValueError("Original values or missing-sensor masks differ")
                if iteration:
                    records.append({"context": index, **durations})
    write_json(run / "samples.json", records)
    out = {
        "status": "completed_packed_view_body_benchmark",
        "contexts": len(contexts),
        "timed_samples_per_path": len(records),
        "exact_tensor_equality": True,
        "original_ms_p50_p95_max": np.percentile(
            [x["original"] for x in records], [50, 95, 100]
        ).tolist(),
        "packed_ms_p50_p95_max": np.percentile(
            [x["packed"] for x in records], [50, 95, 100]
        ).tolist(),
        "samples_sha256": sha256(run / "samples.json"),
        "live_contention_verified": False,
        "actor_installed": False,
        "training_admissible": False,
        "vehicle_control_sent": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
