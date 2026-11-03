"""Audit historical TRAIN-only preview coverage; never fit or open development payloads."""
import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

IDS = ['episode-04-train', 'episode-05-train', 'episode-06-train']


def bouts(mask):
    return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    source = json.loads(dataset.read_text(encoding='utf-8'))
    entries = [e for e in source['episodes'] if e['role'] == 'train']
    if [e['id'] for e in entries] != IDS:
        raise ValueError('Expected exactly the three original ordered TRAIN histories')
    run = begin_run(args.output, dict(dataset=str(dataset), dataset_sha256=sha256(dataset),
        program_sha256=sha256(__file__), threshold_per_m=0.001,
        development_payloads_opened=False, optimizer_updates=0))
    rows = []
    for entry in entries:
        if sha256(entry['targets_file']) != entry['targets_sha256']:
            raise ValueError('Original target hash mismatch: ' + entry['id'])
        with load_array(entry['targets_file'], allow_pickle=False) as data:
            y = data['preview'][:, 2, 1]
            valid = data['preview_valid'][:, 2].astype(bool)
            clock = data['observation_time']
            if not (np.isfinite(y[valid]).all() and np.isfinite(clock).all()
                    and np.all(np.diff(clock) > 0)):
                raise ValueError('Invalid target or observation clock')
            masks = {'left': valid & (y < -.001),
                     'near_straight': valid & (np.abs(y) <= .001),
                     'right': valid & (y > .001)}
            rows.append(dict(id=entry['id'], target_sha256=entry['targets_sha256'],
                observations=len(y), valid=int(valid.sum()),
                groups={name: dict(frames=int(mask.sum()), contiguous_bouts=bouts(mask))
                        for name, mask in masks.items()},
                clock_step_min_s=float(np.diff(clock).min()),
                clock_step_max_s=float(np.diff(clock).max())))
    write_json(run/'result.json', dict(status='completed', episodes=rows,
        note='Bouts are threshold crossings in correlated recordings, not independent corners; '
             'targets are teacher paths, not optimal racing lines.',
        development_payloads_opened=False, fit_performed=False))
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
