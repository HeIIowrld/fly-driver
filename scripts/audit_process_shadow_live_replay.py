"""Recompute saved live-versus-replay intent errors from original source rows."""

import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root, replay = base/'20261031-process-body-owned-pause-live', base/'20261031-process-shadow-live-original-replay'
    original, result = read(root/'result.json'), read(replay/'result.json')
    if (sha256(root/'body-ticker.json') != original['ticker_sha256']
            or sha256(replay/'comparisons.json') != result['comparisons_sha256']
            or result['status'] != 'completed'):
        raise ValueError('Exact original ticker and completed replay required')
    ticks, rows = read(root/'body-ticker.json'), read(replay/'comparisons.json')
    if len(ticks) != len(rows) or len(rows) != 365:
        raise ValueError('Complete original replay required')
    errors = []
    for i, (tick, row) in enumerate(zip(ticks, rows, strict=True)):
        if (row['index'] != i or row['packet'] != tick['packet'] or row['decision'] != tick['decision']
                or row['capture'] != tick['visual_frame_index']):
            raise ValueError('Original event order or identity changed')
        delta = np.abs(np.asarray(row['intent']) - np.asarray(tick['intent']))
        if not np.allclose(delta, row['abs_error'], atol=0, rtol=0):
            raise ValueError('Saved error differs from source intents')
        errors.append(delta)
    errors = np.asarray(errors)
    if (not np.allclose(errors.max(axis=0), result['max_abs_intent_error'], atol=0, rtol=0)
            or not np.allclose(errors.mean(axis=0), result['mean_abs_intent_error'], atol=0, rtol=0)
            or int(np.any(errors > result['tolerance'], axis=1).sum()) != result['over_tolerance']):
        raise ValueError('Original error summary differs')
    run = begin_run(base/'20261031-process-shadow-live-original-replay-audit',
        {'program_sha256':sha256(__file__), 'original_sha256':sha256(root/'result.json'),
         'replay_sha256':sha256(replay/'result.json'), 'scope':'source event identity and numeric output errors'})
    write_json(run/'result.json', {'status':'completed','decisions':len(rows),
        'max_abs_intent_error':errors.max(axis=0).tolist(),'mean_abs_intent_error':errors.mean(axis=0).tolist(),
        'over_tolerance':result['over_tolerance'],'independent_model_reexecution':False,
        'physical_source_or_exposure_verified':False,'vehicle_control_sent':False})
    print(json.dumps(read(run/'result.json')))


if __name__ == '__main__':
    main()
