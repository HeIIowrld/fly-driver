"""Diagnostic horizontal path coverage only; no actor input or fitting."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def load(root):
    manifest = json.loads((root/'dataset.json').read_text(encoding='utf8'))
    result, proof = [], []
    for entry in manifest['episodes']:
        assert entry['role'] in ('train', 'development')
        path = Path(entry['original']['recording'])/'control/api-receipt.json'
        rows = json.loads(path.read_text(encoding='utf8'))['telemetry']
        positions = np.array([r['position'] for r in rows if r['phase'] == 'driving'
            and r['ai_controlled'] and r['speed_kmh'] > 1 and r['wheels_outside'] == 0])[:, [0, 2]]
        assert np.isfinite(positions).all()
        result.append((entry['file'], entry['role'], positions))
        proof.append(dict(file=entry['file'], role=entry['role'], receipt=str(path),
                          receipt_sha256=sha256(path)))
    return result, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original, new = Path(args.original), Path(args.new)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        original_dataset_sha256=sha256(original/'dataset.json'),
        new_dataset_sha256=sha256(new/'dataset.json'),
        method='Per moving driving/AI/on-track sample, nearest XZ sample in original role',
        scope='Path sample geometry, not pixel duplicate or target equivalence proof'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    old, old_proof = load(original)
    added, added_proof = load(new)
    summary = []
    for file, role, positions in added:
        row = dict(file=file, role=role, moving_teacher_telemetry_samples=len(positions))
        for oldrole in ['train', 'development']:
            reference = np.concatenate([p for _, r, p in old if r == oldrole])
            distance = []
            for start in range(0, len(positions), 128):
                delta = positions[start:start+128, None]-reference[None]
                distance.extend(np.sqrt((delta**2).sum(2).min(1)).tolist())
            distance = np.array(distance)
            row['nearest_original_'+oldrole] = dict(
                within1m_fraction=float((distance <= 1).mean()),
                within5m_fraction=float((distance <= 5).mean()),
                median_distance_m=float(np.median(distance)),
                max_distance_m=float(distance.max()))
        summary.append(row)
    write_json(run/'result.json', dict(status='completed_diagnostic_path_sample_overlap',
        rows=summary, original_source_proof=old_proof, new_source_proof=added_proof,
        optimizer_updates=0, actuator_messages_sent=0, ui_actions_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
