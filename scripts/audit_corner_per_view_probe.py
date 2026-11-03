"""Independently verify the retained corner probe and original per-view receipts."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("recording", "common", "per-view", "replay", "output"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    root, common, view, replay = map(Path, (a.recording, a.common, a.per_view, a.replay))
    cr, vr, rr = (read(x / "result.json") for x in (common, view, replay))
    config = read(replay / "manifest.json")["config"]
    if (
        cr["status"] != "completed_temporal_body_context"
        or vr["status"] != "completed_per_view_body_receipt_audit"
        or rr["status"] != "completed_corner_per_view_probe_replay"
        or config["events_sha256"] != sha256(root / "events.jsonl")
        or config["common_sha256"] != sha256(common / "body-pairs.npz")
        or config["per_view_sha256"] != sha256(view / "per-view-body.npz")
        or sha256(replay / "events.json") != rr["events_sha256"]
    ):
        raise ValueError("Exact original inputs and completed exploratory replay required")
    with np.load(common / "body-pairs.npz", allow_pickle=False) as c:
        common_values, common_valid, common_packets = (
            c[k].copy() for k in ("values", "valid", "packets")
        )
    with np.load(view / "per-view-body.npz", allow_pickle=False) as v:
        values, valid, packets = (v[k].copy() for k in ("values", "valid", "packets"))
    records = read(replay / "events.json")
    if len(records) != 739 or values.shape != (739, 4, 21):
        raise ValueError("Complete original decision and four-view histories required")
    equal = packets == common_packets[:, 1, None]
    original = np.broadcast_to(common_values[:, 1, None, :], values.shape)
    masks = np.broadcast_to(common_valid[:, 1, None, :], valid.shape)
    if not np.array_equal(values[equal], original[equal]) or not np.array_equal(
        valid[equal], masks[equal]
    ):
        raise ValueError("Same original packet did not preserve exact values and masks")
    changed = (~equal) & (packets >= 0) & (common_packets[:, 1, None] >= 0)
    yaw_difference = np.abs(values[:, :, 4] - original[:, :, 4])
    intents = np.asarray([x["intents"] for x in records])
    if intents.shape != (739, 2, 2) or not np.isfinite(intents).all():
        raise ValueError("Complete finite paired original intents required")
    delta = np.abs(intents[:, 1] - intents[:, 0])
    if not np.allclose(delta.max(axis=0), rr["max_abs_intent_difference"], atol=1e-12, rtol=0):
        raise ValueError("Saved summary differs from actual output")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "source_events_sha256": sha256(root / "events.jsonl"),
            "replay_result_sha256": sha256(replay / "result.json"),
            "seed": None,
        },
    )
    summary = {
        "status": "completed_corner_per_view_probe_audit",
        "observations": len(records),
        "same_packet_views": int(equal.sum()),
        "changed_packet_views": int(changed.sum()),
        "changed_view_yaw_abs_mean": float(yaw_difference[changed].mean()),
        "changed_view_yaw_abs_max": float(yaw_difference[changed].max()),
        "output_max_abs": delta.max(axis=0).tolist(),
        "output_groups_above_2e_6": int(np.any(delta > 2e-6, axis=1).sum()),
        "source_selected_exploratorily": True,
        "probe_is_untrained": True,
        "physical_sample_or_exposure_time_verified": False,
        "fit_or_control": False,
    }
    write_json(run / "result.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
