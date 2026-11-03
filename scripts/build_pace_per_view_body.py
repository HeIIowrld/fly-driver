"""Select each pace camera view's original pre-render native receipt independently."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.causal_render_body_match import select_prior_body_receipts_per_view
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("recording", "raw-audit", "common-context", "output"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    root, audit, common = map(Path, (a.recording, a.raw_audit, a.common_context))
    ar, am = read(audit / "result.json"), read(audit / "manifest.json")["config"]
    cr, cm = read(common / "result.json"), read(common / "manifest.json")["config"]
    events_path, samples_path = root / "events.jsonl", root / "control/native-motion-samples.json"
    if (
        ar.get("status") != "completed"
        or not ar.get("camera_aid_state_verified")
        or any(ar["actor_replay_max_errors"].values())
        or am["events_sha256"] != sha256(events_path)
        or cr.get("status") != "completed_temporal_body_context"
        or cr.get("training_admissible") is not False
        or cm["events_sha256"] != sha256(events_path)
        or sha256(common / "body-pairs.npz") != cr["body_pairs_sha256"]
    ):
        raise ValueError("Exact original raw audit and conservative common context required")
    events = [json.loads(x) for x in events_path.read_text(encoding="utf8").splitlines()]
    rows, samples, last, seen = read(samples_path), [], -1.0, {}
    for row in rows:
        raw = base64.b64decode(row["prefix_base64"], validate=True)
        stamp = row["received_monotonic_seconds"]
        if hashlib.sha256(raw).hexdigest() != row["prefix_sha256"] or stamp <= last:
            raise ValueError("Unchanged native bytes and advancing receipt required")
        sample = decode_navigation_body_prefix(raw, stamp)
        if sample.legacy.packet_id != row["packet_id"]:
            raise ValueError("Native packet identity differs")
        if samples and sample.legacy.packet_id < samples[-1].legacy.packet_id:
            raise ValueError("Native packet reset requires a new episode")
        if sample.legacy.packet_id in seen:
            if seen[sample.legacy.packet_id] != row["prefix_sha256"]:
                raise ValueError("Duplicate packet changed bytes")
        else:
            samples.append(sample)
            seen[sample.legacy.packet_id] = row["prefix_sha256"]
        last = stamp
    receipts = [s.legacy.received_monotonic_seconds for s in samples]
    if len(events) != ar['decisions'] or len(events) != cr['observations'] or len(samples) < 2:
        raise ValueError("Complete original audited and native histories required")
    identity = read(root / "identity.json")
    encoders = [
        NavigationBodyEncoder(
            "configs/ac-virtual-navigation-body-v2.json", native_identity=identity
        )
        for _ in VIEW_ORDER
    ]
    value = np.zeros((len(events), 4, 21), dtype=np.float64)
    valid = np.zeros((len(events), 4, 21), dtype=bool)
    packets = np.full((len(events), 4), -1, dtype=np.int64)
    received = np.full((len(events), 4), -1.0, dtype=np.float64)
    bounds, reasons = [], []
    for i, event in enumerate(events):
        camera = event["camera"]
        views = {v["view_id"]: v for v in camera["views"]}
        if event["index"] != i or set(views) != set(VIEW_ORDER):
            raise ValueError("Original ordered event and four views required")
        lower = [views[v]["render_started_lower"] for v in VIEW_ORDER]
        matches = select_prior_body_receipts_per_view(receipts, lower, camera["received"])
        for j, match in enumerate(matches):
            if match.native_index is None:
                continue
            sample = samples[match.native_index]
            observation = encoders[j].encode(
                sample, episode_id="pace-original", decision_monotonic_seconds=camera["received"]
            )
            value[i, j], valid[i, j] = observation.values, observation.valid
            packets[i, j], received[i, j] = (
                observation.packet_id,
                observation.original_received_monotonic_seconds,
            )
        bounds.append(lower)
        reasons.append([match.reason for match in matches])
    with np.load(common / "body-pairs.npz", allow_pickle=False) as c:
        common_packets = c["packets"][:, 1].copy()
        decisions = c["decision_monotonic_seconds"].copy()
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "events_sha256": sha256(events_path),
            "native_samples_sha256": sha256(samples_path),
            "raw_audit_sha256": sha256(audit / "result.json"),
            "common_context_sha256": cr["body_pairs_sha256"],
            "contract_sha256": sha256("configs/ac-virtual-navigation-body-v2.json"),
            "selector_sha256": sha256("src/flybrain_drive/causal_render_body_match.py"),
            "seed": None,
            "scope": "host receipt before each view; physical exposure not established",
        },
    )
    np.savez_compressed(
        run / "per-view-body.npz",
        values=value,
        valid=valid,
        packets=packets,
        receipts=received,
        render_start_lower=np.asarray(bounds),
        decisions=decisions,
    )
    write_json(run / "match-reasons.json", reasons)
    differences = (
        (packets != common_packets[:, None]) & (packets >= 0) & (common_packets[:, None] >= 0)
    )
    result = {
        "status": "completed_per_view_body_receipt_audit",
        "observations": len(events),
        "views": len(events) * 4,
        "matched_per_view": (packets >= 0).sum(axis=0).tolist(),
        "valid_yaw_per_view": valid[:, :, 4].sum(axis=0).tolist(),
        "packet_differs_from_common_per_view": differences.sum(axis=0).tolist(),
        "decision_groups_with_distinct_view_packets": sum(
            len(set(row[row >= 0])) > 1 for row in packets
        ),
        "arrays_sha256": sha256(run / "per-view-body.npz"),
        "reasons_sha256": sha256(run / "match-reasons.json"),
        "physical_sample_or_exposure_time_verified": False,
        "training_admissible": False,
        "actor_deployed": False,
        "vehicle_control_sent": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
