"""Append explicitly versioned speed/wheel feedback; preserve every original history and label."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--body-contract", default="configs/ac-virtual-navigation-body-v2.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent = Path(args.data)
    manifest = read(parent / "dataset.json")
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            arguments=vars(args),
            program_sha256=sha256(__file__),
            parent_dataset_sha256=sha256(parent / "dataset.json"),
            body_contract_sha256=sha256(args.body_contract),
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for entry in manifest["episodes"]:
        assert entry["role"] in ("train", "development"), "No evaluation access"
        assert sha256(parent / entry["file"]) == entry["sha256"]
        with load_array(parent / entry["file"], allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        root = Path(entry["original"]["recording"])
        assert sha256(root / "events.jsonl") == entry["source_events_sha256"]
        encoder = NavigationBodyEncoder(
            args.body_contract, native_identity=read(root / "identity.json")
        )
        events = [
            json.loads(line)
            for line in (root / "events.jsonl").read_text(encoding="utf8").splitlines()
        ]
        assert len(events) == len(arrays["observation_time"])
        values, valid = [], []
        for index, event in enumerate(events):
            assert event["decision_monotonic_seconds"] == arrays["observation_time"][index]
            native = event["native"]
            sample = decode_navigation_body_prefix(
                base64.b64decode(native["prefix_base64"], validate=True),
                native["received_monotonic_seconds"],
            )
            assert sample.legacy.packet_id == native["packet_id"]
            assert sample.legacy.prefix_sha256 == native["prefix_sha256"]
            body = encoder.encode(
                sample,
                episode_id=root.name,
                decision_monotonic_seconds=event["decision_monotonic_seconds"],
            )
            np.testing.assert_array_equal(
                body.values[:16].astype(np.float32), arrays["body"][index]
            )
            np.testing.assert_array_equal(body.valid[:16], arrays["body_valid"][index])
            assert body.valid[16:].all()
            values.append(body.values)
            valid.append(body.valid)
        arrays["body"] = np.array(values, np.float32)
        arrays["body_valid"] = np.array(valid, bool)
        path = run / entry["file"]
        np.savez_compressed(path, **arrays)
        with (
            load_array(path, allow_pickle=False) as saved,
            load_array(parent / entry["file"], allow_pickle=False) as old,
        ):
            for key in old.files:
                if key not in ("body", "body_valid"):
                    np.testing.assert_array_equal(saved[key], old[key])
        records.append(
            dict(
                entry,
                sha256=sha256(path),
                parent_episode_sha256=entry["sha256"],
                body_feature_count=21,
                original_labels_masks_clocks_and_scene_unchanged=True,
            )
        )
        write_json(
            run / "progress.json", dict(completed=len(records), total=len(manifest["episodes"]))
        )
    dataset = dict(
        schema="flybrain.flight.embodied_navigation.episodes.v1",
        episodes=records,
        parent_dataset=str(parent),
        parent_dataset_sha256=sha256(parent / "dataset.json"),
        parent_goal_contract=manifest["goal_contract"],
        goal_contract=COMMAND_GOAL_CONTRACT,
        original_numeric_goals_unchanged=True,
        body_contract=args.body_contract,
        body_contract_sha256=sha256(args.body_contract),
        body_feature_count=21,
        evaluation_arrays_opened=False,
        inherited_source_aid_qualification_upgraded=False,
        no_temporal_slicing=True,
        no_original_target_change=True,
        actor_qualified=False,
    )
    write_json(run / "dataset.json", dataset)
    write_json(
        run / "result.json",
        dict(
            status="completed_explicit_body21_dataset_conversion",
            episodes=len(records),
            observations=sum(e["observations"] for e in records),
            dataset_manifest_sha256=sha256(run / "dataset.json"),
            optimizer_updates=0,
            actuator_messages_sent=0,
            ui_actions_sent=0,
            goal_achieved=False,
        ),
    )


if __name__ == "__main__":
    main()
