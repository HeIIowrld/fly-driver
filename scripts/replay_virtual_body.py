"""Replay every retained actor observation through a new body contract without controls."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import VirtualBodyEncoder, decode_virtual_body_prefix
from flybrain_drive.virtual_navigation_body import (
    NavigationBodyEncoder,
    decode_navigation_body_prefix,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--contract", default="configs/ac-virtual-body-v1.json")
    parser.add_argument('--sealed-dataset')
    parser.add_argument('--navigation-contract',
                        help='Explicit new21 contract; requires unchanged whole legacy replay')
    parser.add_argument('--legacy-reference')
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    identity = json.loads((root / "identity.json").read_text(encoding="utf8"))
    if bool(args.navigation_contract) != bool(args.legacy_reference):
        raise ValueError('Navigation extension requires an explicit whole legacy reference')
    encoder = (NavigationBodyEncoder(args.navigation_contract, native_identity=identity)
               if args.navigation_contract else
               VirtualBodyEncoder(args.contract, native_identity=identity))
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=None,
            contract_sha256=sha256(args.contract),
            navigation_contract_sha256=(sha256(args.navigation_contract)
                                        if args.navigation_contract else None),
            legacy_reference_sha256=(sha256(Path(args.legacy_reference)/'virtual-body.npz')
                                     if args.legacy_reference else None),
            events_sha256=sha256(root / "events.jsonl"),
            identity=identity,
            actuator_messages_sent=0,
            optimizer_updates=0,
        ),
    )
    samples, observations = {}, []
    for row in (json.loads(line) for line in (root / "events.jsonl").open(encoding="utf8")):
        kind = row.get("kind")
        engine_ai_row = kind is None and all(
            key in row for key in ("native", "camera", "decision_monotonic_seconds"))
        if kind in ("raw_pair", "expert_raw_native") or engine_ai_row:
            native = row["native"]
            decode = decode_navigation_body_prefix if args.navigation_contract else (
                decode_virtual_body_prefix)
            sample = decode(
                base64.b64decode(native["prefix_base64"], validate=True),
                native["received_monotonic_seconds"],
            )
            legacy = sample.legacy if args.navigation_contract else sample
            if (
                legacy.packet_id != native["packet_id"]
                or legacy.prefix_sha256 != native["prefix_sha256"]
            ):
                raise ValueError("Original native identity/hash mismatch")
            previous = samples.get(legacy.packet_id)
            old = previous.legacy if previous is not None and args.navigation_contract else previous
            if old is not None and old.prefix_sha256 != legacy.prefix_sha256:
                raise ValueError("One native packet has conflicting bytes")
            if (
                previous is None
                or legacy.received_monotonic_seconds < old.received_monotonic_seconds
            ):
                samples[legacy.packet_id] = sample
        if kind in ("student_request", "handoff_wait_observation"):
            actor = row["actor_input"]
            observations.append(
                (kind, actor["native_packet_id"], actor["decision_monotonic_seconds"])
            )
        elif kind == "expert_observation":
            observations.append(
                (kind, row["native_packet_id"], row["decision_monotonic_seconds"])
            )
        elif engine_ai_row:
            observations.append(("engine_ai_observation", row["native"]["packet_id"],
                                 row["decision_monotonic_seconds"]))
    values, valid, receipts, packets, clocks, phases = [], [], [], [], [], []
    for phase, packet, clock in observations:
        body = encoder.encode(
            samples[packet], episode_id=root.name, decision_monotonic_seconds=clock
        )
        values.append(body.values)
        valid.append(body.valid)
        receipts.append(body.original_received_monotonic_seconds)
        packets.append(packet)
        clocks.append(clock)
        phases.append(phase)
    values, valid, clocks = np.array(values), np.array(valid), np.array(clocks)
    legacy_exact = False
    if args.legacy_reference:
        legacy_root = Path(args.legacy_reference)
        reference_result = json.loads((legacy_root/'result.json').read_text(encoding='utf8'))
        if (reference_result['status'] != 'completed'
                or sha256(legacy_root/'virtual-body.npz') != reference_result[
                    'virtual_body_sha256']):
            raise ValueError('Completed sealed legacy body replay required')
        with load_array(legacy_root/'virtual-body.npz', allow_pickle=False) as old:
            for actual, key in [(values[:, :16], 'values'), (valid[:, :16], 'valid'),
                    (clocks, 'decision_monotonic_seconds'),
                    (np.array(receipts), 'received_monotonic_seconds'),
                    (np.array(packets), 'native_packet_ids')]:
                np.testing.assert_array_equal(actual, old[key])
        legacy_exact = True
    sealed_match = False
    if args.sealed_dataset:
        dataset = Path(args.sealed_dataset)
        manifest = json.loads((dataset/'dataset.json').read_text(encoding='utf8'))
        if len(manifest['episodes']) != 1:
            raise ValueError('Exactly one original sealed episode required')
        episode = manifest['episodes'][0]
        if sha256(resolve_episode_path(dataset, episode)) != episode['sha256']:
            raise ValueError('Sealed episode hash mismatch')
        with load_array(resolve_episode_path(dataset, episode), allow_pickle=False) as old:
            np.testing.assert_array_equal(clocks, old['observation_time'])
        sealed_match = True
    np.savez_compressed(
        run / "virtual-body.npz",
        values=values,
        valid=valid,
        received_monotonic_seconds=np.array(receipts),
        decision_monotonic_seconds=clocks,
        native_packet_ids=np.array(packets),
        phases=np.array(phases),
        feature_names=np.array(body.feature_names),
    )
    write_json(
        run / "result.json",
        dict(
            status="completed",
            termination_reason="all original actor observations encoded without filtering",
            observations=len(observations),
            phases={phase: phases.count(phase) for phase in set(phases)},
            contract_sha256=encoder.contract_sha256,
            legacy16_whole_replay_exact=legacy_exact,
            navigation_feature_count=21 if args.navigation_contract else 16,
            virtual_body_sha256=sha256(run / "virtual-body.npz"),
            feature_names=body.feature_names,
            valid_counts_per_feature=valid.sum(axis=0).tolist(),
            all_invalid_observations=int((~valid.any(axis=1)).sum()),
            maximum_original_decision_gap_seconds=float(np.diff(clocks).max()),
            observations_removed=0,
            clock_retiming=False,
            encoder_resets_at_handoff=0,
            neural_model_executed=False,
            sealed_episode_clocks_exactly_match=sealed_match,
            actor_heading_or_world_position_supplied=False,
            actual_actor_integration=False,
            physiological_registration=False,
            actuator_messages_sent=0,
            optimizer_updates=0,
            goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(
                observations=len(observations), valid_counts_per_feature=valid.sum(axis=0).tolist()
            )
        )
    )


if __name__ == "__main__":
    main()
