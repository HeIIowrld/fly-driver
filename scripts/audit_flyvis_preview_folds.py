"""Independent CPU moment/model audit of TRAIN-only fold initialization."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_body_registered_preview import RegisteredPreview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    result = read(root/'result.json')
    plan = read(root/'manifest.json')['config']['plan']
    assert result['status'] == 'completed_cpu_train_history_fold_setup'
    assert result['readout_optimizer_updates'] == 1200
    assert result['author_optimizer_updates'] == 0 and result['gpu_used'] is False
    assert all('development' not in path.lower() for path in plan['input_sha256']
               if Path(path).suffix == '.npz')
    references = {e['id']: e for e in result['source_references']}
    torch.set_num_threads(1)
    torch.set_default_device('cpu')
    records = []
    for fold in result['folds']:
        holdout, fitted = fold['heldout_history'], fold['fit_history_ids']
        assert holdout not in fitted and len(fitted) == 2
        assert fold['readout_fit_ids'] == fold['all_moments_fit_ids'] == fitted
        total, square, count = np.zeros(14), np.zeros(14), 0
        bodies, masks, labels, validity = [], [], [], []
        for name in fitted:
            entry = references[name]
            assert sha256(entry['source_responses']) == entry['source_responses_sha256']
            with load_array(entry['source_responses'], allow_pickle=False) as a:
                original_clock = a['decision_monotonic_seconds'].copy()
                for key in ('additional_activity', 'motion_activity'):
                    values = a[key]
                    offset = 0 if key == 'additional_activity' else 6
                    for start in range(0, len(values), 64):
                        x = values[start:start+64].astype(float)
                        total[offset:offset+x.shape[2]] += x.sum(axis=(0, 1, 3))
                        square[offset:offset+x.shape[2]] += (x*x).sum(axis=(0, 1, 3))
                    if offset == 0:
                        count += len(values)*4*721
            with load_array(entry['source_body'], allow_pickle=False) as a:
                np.testing.assert_array_equal(original_clock, a['decision_monotonic_seconds'])
                bodies.append(a['values'].copy())
                masks.append(a['valid'].copy())
            assert sha256(entry['targets_file']) == entry['targets_sha256']
            with load_array(entry['targets_file'], allow_pickle=False) as a:
                np.testing.assert_array_equal(original_clock, a['observation_time'])
                labels.append(a['preview'].reshape(-1, 6).copy())
                validity.append(np.repeat(a['preview_valid'], 2, 1))
        cm = total/count
        cs = np.maximum(np.sqrt(np.maximum(square/count-cm**2, 0)), plan['channel_scale_floor'])
        bv, bmasks = np.concatenate(bodies), np.concatenate(masks)
        bm, bs = np.zeros(21), np.ones(21)
        for h in range(21):
            if bmasks[:, h].any():
                selected = bv[bmasks[:, h], h]
                bm[h], bs[h] = selected.mean(), max(selected.std(), 1e-5)
        y, valid = np.concatenate(labels), np.concatenate(validity)
        ym = np.array([y[valid[:, h], h].mean() for h in range(6)])
        ys = np.maximum([y[valid[:, h], h].std() for h in range(6)], plan['target_scale_floor'])
        with load_array(root/holdout/'normalization.npz', allow_pickle=False) as norms:
            errors = {}
            for key, expected in dict(channel_mean=cm, channel_scale=cs, body_mean=bm,
                body_scale=bs, target_mean=ym, target_scale=ys).items():
                error = float(np.abs(norms[key]-expected).max())
                assert error < 1e-12, (holdout, key, error)
                errors[key] = error
            assert norms['channel_count'] == count
            np.testing.assert_array_equal(norms['body_valid_counts'], bmasks.sum(0))
            np.testing.assert_array_equal(norms['target_valid_counts'], valid.sum(0))
            np.testing.assert_array_equal(norms['feature_mean'][896:], 0)
            np.testing.assert_array_equal(norms['feature_scale'][896:], 1)
        torch.manual_seed(plan['seed'])
        initial = RegisteredPreview().state_dict()
        changed = {}
        for condition, reference in fold['models'].items():
            assert sha256(reference['path']) == reference['sha256']
            saved = torch.load(reference['path'], map_location='cpu', weights_only=False)
            assert saved['fitted_history_ids'] == fitted and saved['heldout_history'] == holdout
            assert saved['actuator_policy'] is False
            assert len(fold['learning'][condition]['losses']) == 200
            changed[condition] = sum(int(torch.count_nonzero(saved['model'][k]-v))
                                     for k, v in initial.items())
            assert changed[condition] > 0
            assert all(torch.isfinite(v).all() for v in saved['model'].values())
        records.append(dict(heldout_history=holdout, fitting_ids=fitted,
            independent_normalization_max_errors=errors, changed_model_values=changed,
            original_fit_observations=len(y), all_original_fit_clocks_exact=True))
    audit = dict(status='passed_independent_fold_moment_model_audit', records=records,
        known_development_arrays_opened=0, gpu_used=False, actuator_messages_sent=0,
        result_sha256=sha256(root/'result.json'), program_sha256=sha256(__file__),
        scope='Independent raw channel/body/target moments and changed readout weights. '
              'Feature projection numerics and holdout score recomputation are not repeated here.',
        independent_final_validation=False, author_optimizer_updates=0)
    write_json(root/'independent-audit.json', audit)
    (root/'executed-independent-audit.py').write_bytes(Path(__file__).read_bytes())
    print(audit['status'])


if __name__ == '__main__':
    main()
