import json

import pytest

from flybrain_drive.graph import sha256
from scripts.audit_failed_camera_capture import analyze


def save(path, data):
    path.write_text(json.dumps(data), encoding='utf8')


def test_audit_retains_partial_failure_and_rejects_changed_ticker(tmp_path):
    recording, camera = tmp_path / 'run', tmp_path / 'camera'
    recording.mkdir()
    camera.mkdir()
    (recording / 'control').mkdir()
    save(recording / 'body-ticker.json', [{'decision': 1.0}])
    save(recording / 'online-body-shadow.json', [{'index': 0}])
    save(recording / 'result.json', {'request_id': 'owned',
         'shadow_sha256': sha256(recording / 'online-body-shadow.json'),
         'ticker_sha256': sha256(recording / 'body-ticker.json')})
    save(recording / 'control/api-receipt.json', {'status': 'api_returned_false',
         'stop_reason': 'stale', 'api_calls': []})
    save(camera / 'result.json', {'request_id': 'owned', 'status': 'stopped', 'frames': 2})
    for index, clock in enumerate((1., 1.3)):
        save(camera / f'frame-{index:04d}.json', {'index': index, 'request_id': 'owned',
             'script_clock': clock, 'views': [{'view_id': 'front',
             'render_call_finished_clock': clock, 'save_finished_clock': clock + .01}]})
    output = analyze(recording, camera)
    assert output['control_status'] == 'api_returned_false'
    assert output['largest_camera_script_clock_gaps'][0]['seconds'] == pytest.approx(.3)
    assert output['training_admissible'] is False
    save(recording / 'body-ticker.json', [{'decision': 2.0}])
    with pytest.raises(ValueError, match='hash differs'):
        analyze(recording, camera)
