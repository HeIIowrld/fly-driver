import json

import pytest

from flybrain_drive.ac_pixel_retina import digest
from flybrain_drive.ac_view_contract import validate_declared_live_view


def test_declared_camera_change_blocks_old_frontend_before_student_use(tmp_path):
    path = tmp_path / 'current-view.json'
    dash = dict(camera='dash', regions=dict(front=[10, 20, 30, 40], rear=[1, 2, 3, 4]))
    with pytest.raises(ValueError, match='Declare'):
        validate_declared_live_view(dash, path)
    contract = dict(schema='flybrain.ac.live_view.v1',
                    retina_configuration_sha256=digest(dash),
                    automated_camera_identity_verified=False)
    path.write_text(json.dumps(contract))
    assert validate_declared_live_view(dash, path) == contract
    old = dict(dash, camera='cockpit')
    with pytest.raises(ValueError, match='differs'):
        validate_declared_live_view(old, path)
    changed_crop = dict(dash, regions=dict(front=[10, 20, 31, 40], rear=[1, 2, 3, 4]))
    with pytest.raises(ValueError, match='differs'):
        validate_declared_live_view(changed_crop, path)


def test_live_contract_uses_canonical_json_digest_not_serialized_file_bytes(tmp_path):
    import hashlib

    configuration = dict(camera='engine', gain=1)
    serialized = json.dumps(configuration, indent=2).replace('\n', '\r\n').encode('utf8')
    file_hash = hashlib.sha256(serialized).hexdigest()
    assert file_hash != digest(configuration)
    path = tmp_path/'contract.json'
    contract = dict(schema='flybrain.ac.live_view.v1', retina_configuration_sha256=file_hash)
    path.write_text(json.dumps(contract), encoding='utf8')
    with pytest.raises(ValueError, match='differs'):
        validate_declared_live_view(configuration, path)
    contract['retina_configuration_sha256'] = digest(configuration)
    path.write_text(json.dumps(contract), encoding='utf8')
    assert validate_declared_live_view(json.loads(serialized), path) == contract
