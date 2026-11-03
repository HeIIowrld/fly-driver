"""Attribute all retained shared-camera delays, including rows the actor could not use."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def stats(values):
    x = np.asarray(values, float)
    return dict(
        count=len(x),
        median=float(np.median(x)),
        p95=float(np.quantile(x, 0.95)),
        maximum=float(x.max()),
        above80ms=int((x > 0.08).sum()),
        above150ms=int((x > 0.15).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.recording)
    bundle = source / "vision/camera.zip"
    declared = json.loads((source / "vision/bundle.json").read_text(encoding="utf8"))
    assert sha256(bundle) == declared["sha256"]
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            script_sha256=sha256(__file__),
            bundle_sha256=sha256(bundle),
            recording=str(source),
        ),
    )
    with zipfile.ZipFile(bundle) as archive:
        clock = json.loads(archive.read("clock-final.json"))
        stream = [
            json.loads(line) for line in archive.read("stream.jsonl").decode("utf8").splitlines()
        ]
        rows = []
        for index, row in enumerate(stream):
            assert row["capture_id"] == index and row["generation"] == clock["generation"]
            frame = json.loads(archive.read(f"frame-{index:04d}.json"))
            copies = row["shared_copy_receipts"]
            pipeline = row["shared_pipeline"]
            first = min(v["render_started_clock"] for v in frame["views"])
            age = copies[0]["started_at"] - first - clock["offset_lower_seconds"]
            between = sum(
                n["started_at"] - p["finished_at"]
                for p, n in zip(copies[:-1], copies[1:], strict=True)
            )
            rows.append(
                dict(
                    index=index,
                    render_start_clock=first,
                    render_to_copy_start_upper=age,
                    copy_seconds=sum(c["finished_at"] - c["started_at"] for c in copies),
                    maximum_individual_copy_seconds=max(
                        c["finished_at"] - c["started_at"] for c in copies
                    ),
                    between_copy_seconds=between,
                    copy_finished_to_ack_stamp_seconds=pipeline["acknowledged_copy_clock"]
                    - copies[-1]["finished_at"],
                    ack_stamp_to_persistence_seconds=pipeline["persistence_started_clock"]
                    - pipeline["acknowledged_copy_clock"],
                    persistence_seconds=pipeline["persistence_finished_clock"]
                    - pipeline["persistence_started_clock"],
                    publication_age_upper=row["received"] - first - clock["offset_lower_seconds"],
                    publication_clock=row["received"],
                    first_copy_clock=copies[0]["started_at"],
                )
            )
    for previous, current in zip(rows[:-1], rows[1:], strict=True):
        current["previous_persistence_to_first_copy_seconds"] = (
            current["first_copy_clock"] - previous["publication_clock"]
        )
        current["previous_persistence_overlapped_render_seconds"] = previous[
            "publication_clock"
        ] - (current["render_start_clock"] + clock["offset_lower_seconds"])
    keys = (
        "render_to_copy_start_upper",
        "copy_seconds",
        "maximum_individual_copy_seconds",
        "between_copy_seconds",
        "copy_finished_to_ack_stamp_seconds",
        "ack_stamp_to_persistence_seconds",
        "persistence_seconds",
        "publication_age_upper",
    )
    result = dict(
        status="completed_all_original_shared_stream_timing",
        frames=len(rows),
        statistics={k: stats([r[k] for r in rows]) for k in keys},
        previous_persistence_overlap=stats(
            [r["previous_persistence_overlapped_render_seconds"] for r in rows[1:]]
        ),
        highest_start_delay=sorted(
            rows, key=lambda r: r["render_to_copy_start_upper"], reverse=True
        )[:8],
        highest_copy_delay=sorted(rows, key=lambda r: r["copy_seconds"], reverse=True)[:8],
        highest_persistence_delay=sorted(
            rows, key=lambda r: r["persistence_seconds"], reverse=True
        )[:8],
        original_data_or_clocks_modified=False,
        no_resource_cause_claim=True,
        optimizer_updates=0,
        actuator_messages_sent=0,
        ui_actions_sent=0,
        goal_achieved=False,
    )
    write_json(run / "result.json", result)
    print(json.dumps(result["statistics"]))
    print(
        json.dumps(
            dict(
                worst_start=result["highest_start_delay"][0],
                worst_persistence=result["highest_persistence_delay"][0],
                worst_copy=result["highest_copy_delay"][0],
            )
        )
    )


if __name__ == "__main__":
    main()
