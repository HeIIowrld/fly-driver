import json

import numpy as np
import pytest

from flybrain_drive.graph import sha256
from scripts.analyze_flyvis_probe import load_dataset


def make_raw_run(directory, nonfinite=False):
    activity = np.ones((2, 3, 1), dtype=np.float32)
    if nonfinite:
        activity[0, 1, 0] = np.nan
    np.savez(directory/'responses.npz', activity=activity,
             central_cell_types=np.array(['T4a']), dt=np.array(.005))
    (directory/'result.json').write_text(json.dumps(dict(status='completed',
        responses_sha256=sha256(directory/'responses.npz'))), encoding='utf8')
    (directory/'stimulus.json').write_text(json.dumps(dict(mode='flash', parameters=[
        dict(intensity=0, radius=-1), dict(intensity=1, radius=-1)])), encoding='utf8')
    return dict(radius=[-1], dynamic_range=[0, 1], t_pre=1)


def test_modified_raw_archive_is_rejected_before_analysis(tmp_path):
    config = make_raw_run(tmp_path)
    with (tmp_path/'responses.npz').open('ab') as stream:
        stream.write(b'changed after receipt')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_dataset(tmp_path, config)


def test_actual_nonfinite_response_cannot_be_skipped_by_nan_aware_metrics(tmp_path):
    config = make_raw_run(tmp_path, nonfinite=True)
    with pytest.raises(ValueError, match='inside actual stimulus'):
        load_dataset(tmp_path, config)


def test_declared_protocol_cannot_silently_use_a_smaller_grid(tmp_path):
    config = make_raw_run(tmp_path)
    config['radius'] = [-1, 6]
    with pytest.raises(ValueError, match='declared radius grid'):
        load_dataset(tmp_path, config)
