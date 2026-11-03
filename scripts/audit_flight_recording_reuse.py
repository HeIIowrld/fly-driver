"""Check complete original histories before reusing old labels in the flight circuit."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def inspect(root, episodes, dataset):
    rows = [json.loads(line) for line in (root / "events.jsonl").open(encoding="utf8")]
    observations, samples, conflicts = [], {}, 0
    for row in rows:
        native = row.get("native", {})
        if "prefix_base64" in native:
            sample = decode_virtual_body_prefix(
                base64.b64decode(native["prefix_base64"], validate=True),
                native["received_monotonic_seconds"],
            )
            if (sample.prefix_sha256 != native["prefix_sha256"]
                    or sample.packet_id != native["packet_id"]):
                raise ValueError("Original native prefix identity changed")
            old = samples.get(sample.packet_id)
            conflicts += int(old is not None and old.prefix_sha256 != sample.prefix_sha256)
            if old is None or sample.received_monotonic_seconds < old.received_monotonic_seconds:
                samples[sample.packet_id] = sample
        kind = row.get("kind")
        if kind in ("student_request", "handoff_wait_observation"):
            actor = row["actor_input"]
            observations.append((actor["decision_monotonic_seconds"],
                                 actor["native_packet_id"], actor["image"]))
        elif kind == "expert_observation":
            observations.append((row["decision_monotonic_seconds"],
                                 row["native_packet_id"], row["camera"]))
        elif kind is None and all(k in row for k in ("camera", "native",
                                                     "decision_monotonic_seconds")):
            observations.append((row["decision_monotonic_seconds"],
                                 row["native"]["packet_id"], row["camera"]))
    times = np.array([row[0] for row in observations])
    gaps = np.diff(times)
    ages = np.array([clock - samples[packet].received_monotonic_seconds
                     for clock, packet, _ in observations if packet in samples])
    missing = sum(packet not in samples for _, packet, _ in observations)
    camera_bad, generations = 0, set()
    for clock, _, image in observations:
        generations.add(image["generation"])
        views = image["views"]
        camera_bad += int(len(views) != 4 or {v["view_id"] for v in views}
                          != {"front", "left", "right", "rear"})
        camera_bad += sum(not 0 <= clock - v["render_started_lower"] <= .15 for v in views)
    reasons = []
    if len(times) < 2 or not np.isfinite(times).all() or np.any(gaps <= 0):
        reasons.append("original clocks not finite and strictly increasing")
    if np.any(gaps > .08):
        reasons.append("original gaps exceed80ms; no slicing or reset repair")
    if missing or conflicts:
        reasons.append("missing or conflicting original native prefixes")
    if not np.isfinite(ages).all() or np.any((ages < 0) | (ages > .05)):
        reasons.append("required body input outside50ms first-receipt age")
    identity_path = root / "identity.json"
    expected = read(Path("configs/ac-virtual-body-v1.json"))["native_identity"]
    identity_valid = (identity_path.exists()
                      and all(read(identity_path).get(k) == v for k, v in expected.items()))
    if not identity_valid:
        reasons.append("original static car/version identity missing or different")
    if camera_bad or len(generations) != 1:
        reasons.append("invalid four-view exposure age or camera generation")
    bundle_path = root / "vision/bundle.json"
    bundle = read(bundle_path) if bundle_path.exists() else None
    if bundle is None or not (root / "vision" / bundle["file"]).exists():
        reasons.append("RGB bundle missing")
    labels = []
    for episode in episodes:
        if sha256(resolve_episode_path(dataset, episode)) != episode["sha256"]:
            raise ValueError("Original sealed episode changed")
        start, stop = episode["original_observation_start"], episode["original_observation_stop"]
        with load_array(resolve_episode_path(dataset, episode), allow_pickle=False) as archive:
            exact = bool(np.array_equal(archive["observation_time"], times[start:stop]))
            actions, valid = archive["action"], archive["action_valid"]
            labels.append(dict(file=episode["file"], original_slice=[start, stop],
                               clocks_exact=exact,
                               left_labels=int(((actions[:, 0] < -.01) & valid[:, 0]).sum()),
                               brake_labels=int(((actions[:, 2] > 0) & valid[:, 2]).sum())))
            if not exact:
                reasons.append("sealed slice differs from full original clocks")
    return dict(
        recording=str(root), events_sha256=sha256(root / "events.jsonl"),
        observations=len(times), native_packets=len(samples),
        missing_native_observations=missing, conflicting_native_packets=conflicts,
        maximum_gap_seconds=float(gaps.max()) if len(gaps) else None,
        gaps_over80ms=int((gaps > .08).sum()),
        body_ages_over50ms=int((ages > .05).sum()), body_from_future=int((ages < 0).sum()),
        maximum_body_age_seconds=float(ages.max()) if len(ages) else None,
        camera_invalid_observations_or_views=camera_bad, identity_valid=identity_valid,
        bundle_metadata_sha256=sha256(bundle_path) if bundle else None,
        archive_bytes_hash_checked=False, episodes=labels,
        eligible_for_current_flight_replay=not reasons, rejection_reasons=sorted(set(reasons)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/ac-shared-lateral-recovery-merged-20261029-t946")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    manifest = read(dataset / "dataset.json")
    run = begin_run(args.output, dict(arguments=vars(args),
                                    source_manifest_sha256=sha256(dataset / "dataset.json"),
                                    maximum_gap_seconds=.08, maximum_body_age_seconds=.05,
                                    ui_actions_sent=0, actuator_messages_sent=0))
    grouped = {}
    for episode in manifest["episodes"]:
        name = episode.get("original_recording") or episode.get("original_trial")
        if name is None:
            raise ValueError("Explicit original source required")
        grouped.setdefault(name, []).append(episode)
    records = []
    for name, episodes in grouped.items():
        record = inspect(Path(name), episodes, dataset)
        records.append(record)
        write_json(run / "progress.json", dict(completed=len(records), total=len(grouped)))
        print(json.dumps({key: record[key] for key in (
            "recording", "gaps_over80ms", "identity_valid", "eligible_for_current_flight_replay"
        )}), flush=True)
    write_json(run / "result.json", dict(
        status="completed", termination_reason="all full original histories checked unchanged",
        recordings=records, eligible_recordings=sum(r["eligible_for_current_flight_replay"]
                                                    for r in records),
        archive_bytes_hash_checked=False,
        scope="necessary preflight only; bundle existence is not byte/hash verification",
        ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False,
    ))


if __name__ == "__main__":
    main()
