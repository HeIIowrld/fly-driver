"""Summarize saved TRAIN-only fold scores without re-fitting or claiming re-execution."""
import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

IDS = ['episode-04-train', 'episode-05-train', 'episode-06-train']
CONDITIONS = ['vision_body__intact', 'body_only__intact', 'zero',
    'vision_body__neural_feature_mean', 'vision_body__front_mean',
    'vision_body__left_mean', 'vision_body__right_mean', 'vision_body__rear_mean']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds', required=True)
    p.add_argument('--coverage', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.folds)
    coverage = json.loads(Path(a.coverage).read_text(encoding='utf-8'))
    if [e['id'] for e in coverage['episodes']] != IDS:
        raise ValueError('Unexpected TRAIN-only coverage')
    run = begin_run(a.output, dict(folds=str(root), coverage_sha256=sha256(a.coverage),
        program_sha256=sha256(__file__), optimizer_updates=0, inference_reexecuted=False))
    rows = []
    for identifier, counts in zip(IDS, coverage['episodes']):
        path = root/identifier/'result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if (data['heldout_history'] != identifier or data['heldout_external_role'] != 'train'
                or set(data['fit_history_ids']) != set(IDS)-{identifier}):
            raise ValueError('Invalid fold roles or training history')
        metrics = data['holdout_metrics']
        conditions = {}
        for name in CONDITIONS:
            score = metrics[name]
            if score['heads'][5]['count'] != counts['valid']:
                raise ValueError('20m valid count does not match coverage audit')
            groups = {}
            for group in ('left', 'near_straight', 'right'):
                result = score['groups'][group]
                if result['rows'] != counts['groups'][group]['frames']:
                    raise ValueError('Direction count does not match coverage audit')
                groups[group] = result
            conditions[name] = dict(head20=score['heads'][5], groups=groups)
        rows.append(dict(heldout=identifier, fit_history_ids=data['fit_history_ids'],
            result_sha256=sha256(path), conditions=conditions))
    write_json(run/'result.json', dict(status='completed', folds=rows,
        scope='Summary of previously saved scores, not independent inference or new learning',
        cautions=['Held-out histories are geographically adjacent and remain TRAIN role',
                  'Frame scores are correlated', 'Ablations may be out of distribution',
                  'Body-only has independent fitted parameters']))
    for row in rows:
        print(row['heldout'], {k: round(v['head20']['rmse'], 6)
                               for k, v in row['conditions'].items()})


if __name__ == '__main__':
    main()
