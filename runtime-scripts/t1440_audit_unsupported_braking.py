"""Audit raw unsupported brake predictions against original valid development labels."""
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
evaluation = base/'20261029-t1439-embodied-development-evaluation'
root = base/'20261029-t1434-body21-flight-cache'
run = begin_run(base/'20261029-t1440-unsupported-braking-audit', dict(scope=__doc__,
    program_sha256=sha256(__file__), predictions_sha256=sha256(evaluation/'development-predictions.npz'),
    dataset_sha256=sha256(root/'dataset.json'), threshold=-.01))
(run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
records = []
with load_array(evaluation/'development-predictions.npz', allow_pickle=False) as predictions:
    for entry in json.loads((root/'dataset.json').read_text(encoding='utf8'))['episodes']:
        if entry['role'] != 'development':
            continue
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as a:
            p = predictions[entry['file'].removesuffix('.npz')+'_intact'][:, 1]
            valid = a['target_valid'][:, 1]
            late = valid & (a['observation_time']-a['observation_time'][0] >= 3)
            brake = p < -.01
            records.append(dict(file=entry['file'], original_observations=len(p),
                valid_forward_targets=int(valid.sum()), late_valid_targets=int(late.sum()),
                actual_brake_labels=int(((a['target'][:, 1] < 0) & valid).sum()),
                all_negative_forward_predictions=int(brake.sum()),
                valid_negative_forward_predictions=int((brake & valid).sum()),
                late_valid_negative_forward_predictions=int((brake & late).sum()),
                minimum_raw_forward_prediction=float(p.min()),
                late_forward_rmse=float(np.sqrt(np.mean((p[late]-a['target'][late, 1])**2)))))
write_json(run/'result.json', dict(status='completed_original_unsupported_braking_audit',
    records=records, predictions_not_clipped_or_replaced=True,
    checkpoint_promoted=False, optimizer_updates=0, actuator_messages_sent=0, goal_achieved=False))
print(json.dumps(records))
