"""Audit unchanged preview probe, including full/late errors and target coverage."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, data = Path(args.probe), Path(args.data)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        original_probe_sha256=sha256(root/'probe.npz'),
        original_result_sha256=sha256(root/'result.json'), no_fitting=True,
        full_and_after3seconds_reported=True, original_roles_unchanged=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    with load_array(root/'probe.npz', allow_pickle=False) as archive:
        probe = {k: archive[k].copy() for k in archive.files}
    train_y, train_mask = [], []
    for entry in result['episodes']:
        assert entry['role'] in ('train', 'development')
        assert sha256(root/entry['file']) == entry['auxiliary_sha256']
        if entry['role'] == 'train':
            with load_array(root/entry['file'], allow_pickle=False) as a:
                train_y.append(a['preview'].copy())
                train_mask.append(a['preview_valid'].copy())
    ty, tm = np.concatenate(train_y), np.concatenate(train_mask)
    means = np.array([ty[tm[:, h], h].mean(0) for h in range(3)])
    rows = []
    for entry in result['episodes']:
        assert sha256(data/entry['file']) == entry['original_episode_sha256']
        with load_array(root/entry['file'], allow_pickle=False) as a:
            ys, valid, t = (a[k].copy() for k in
                            ('preview', 'preview_valid', 'observation_time'))
        with load_array(data/entry['file'], allow_pickle=False) as a:
            x = a['scene_features'].copy()
            np.testing.assert_array_equal(t, a['observation_time'])
        z = ((x-probe['mean'])/probe['scale']).clip(-5, 5)
        pred = np.einsum('nf,hfc->nhc', z, probe['weights'])+probe['biases']
        np.savez_compressed(run/entry['file'], prediction=pred, target=ys,
                            target_valid=valid, observation_time=t)
        stats = []
        for h in range(3):
            item = dict(distance_m=[5, 10, 20][h])
            for label, keep in [('all', valid[:, h]),
                                ('after3s', valid[:, h] & (t >= t[0]+3))]:
                y, p = ys[keep, h], pred[keep, h]
                item[label] = dict(n=int(keep.sum()),
                    bearing_range_degrees=np.degrees([y[:, 0].min(), y[:, 0].max()]).tolist(),
                    bearing_std_degrees=float(np.degrees(y[:, 0].std())),
                    above_5deg=int((abs(y[:, 0]) > np.radians(5)).sum()),
                    rmse=np.sqrt(((p-y)**2).mean(0)).tolist(),
                    mean_rmse=np.sqrt(((means[h]-y)**2).mean(0)).tolist())
            stats.append(item)
        rows.append(dict(file=entry['file'], role=entry['role'], stats=stats))
    write_json(run/'result.json', dict(status='completed_no_fit_coverage_and_late_audit',
        rows=rows, optimizer_updates=0, actuator_messages_sent=0,
        ui_actions_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
