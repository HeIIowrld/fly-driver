"""Seal a compatible sensory union without duplicating episode payloads by default."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from flybrain_drive.episode_storage import reference_record, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage", choices=("reference", "copy"), default="reference",
                        help="Reuse sealed sibling episodes (default); copy for standalone export")
    args = parser.parse_args()
    manifests = [json.loads((Path(p) / "dataset.json").read_text(encoding="utf8"))
                 for p in args.input]
    keys = ("schema", "action_profile", "sensory_calibration", "actor_fields",
            "decision_dt_seconds", "brain_ticks_per_decision")
    first = manifests[0]
    contracts = [manifest.get('teacher_label_contract') for manifest in manifests]
    if any(contract is not None for contract in contracts) and any(
            contract != contracts[0] for contract in contracts):
        raise ValueError('Teacher task contracts differ; reconcile labels before merging')
    counts = [0, 0, 0]
    for path, manifest in zip(args.input, manifests, strict=True):
        if any(manifest[k] != first[k] for k in keys):
            raise ValueError("Only equal sensory, timing and action contracts may be merged")
        episodes, _ = load_episodes(path, first["sensory_calibration"])
        for episode in episodes:
            for action in range(3):
                counts[action] += int(episode['action_valid'][:, action].sum())
    references = {}
    if args.storage == "reference":
        for source_index, (path, manifest) in enumerate(zip(args.input, manifests, strict=True)):
            for index in range(len(manifest['episodes'])):
                references[source_index, index] = reference_record(
                    path, manifest, index, args.output)
    run = begin_run(args.output, {"sources": [{"directory": p,
        "manifest_sha256": sha256(Path(p) / "dataset.json")} for p in args.input],
        "scope": "read-only input validation and exact episode union",
        "episode_storage": args.storage})
    records = []
    for source_index, (path, manifest) in enumerate(zip(args.input, manifests, strict=True)):
        for i, episode in enumerate(manifest["episodes"]):
            if args.storage == "reference":
                record = references[source_index, i]
            else:
                record = copy.deepcopy(episode)
                record.pop('source_reference', None)
                target = run / f"source-{source_index:02d}-episode-{i:03d}.npz"
                shutil.copyfile(resolve_episode_path(path, episode, manifest), target)
                if sha256(target) != episode["sha256"]:
                    raise ValueError("Episode bytes changed during copy")
                record['file'] = target.name
            record.update(episode_id=f"source-{source_index}-"+episode["episode_id"],
                          source_manifest_sha256=sha256(Path(path) / "dataset.json"))
            records.append(record)
    merged = {k: copy.deepcopy(first[k]) for k in keys}
    merged.update(episodes=records, decisions=sum(e["steps"] for e in records),
        episode_storage=args.storage,
        teacher_kind=f"{args.storage} compatible episodes; providers in sealed source manifests",
        native_AC_training_data=all(m.get('native_AC_training_data', False) for m in manifests)
            if all('native_AC_training_data' in m for m in manifests) else None,
        supervised_action_counts=counts,
        pedal_labels_available=bool(counts[1] and counts[2]),
        expert_labels_available=all(m.get('expert_labels_available', True) for m in manifests),
        experimental_offline_replay_only=any(m.get('experimental_offline_replay_only', False)
                                              for m in manifests),
        sources=[{"directory": p, "sha256": sha256(Path(p) / "dataset.json")} for p in args.input])
    if contracts[0] is not None:
        merged['teacher_label_contract'] = copy.deepcopy(contracts[0])
        merged['labels_are_executed_requests'] = all(
            m.get('labels_are_executed_requests', False) for m in manifests)
        merged['labels_are_executed_applied_controls'] = all(
            m.get('labels_are_executed_applied_controls', False) for m in manifests)
    write_json(run / "dataset.json", merged)
    load_episodes(run, first["sensory_calibration"])
    write_json(run / "result.json", {"status": "completed", "episodes": len(records),
                                    "decisions": merged["decisions"], "goal_achieved": False})


if __name__ == "__main__":
    main()
