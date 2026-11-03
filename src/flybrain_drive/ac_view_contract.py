"""Reject a declared camera change before running an old native student frontend.

This is a local configuration interlock, not automatic recognition of the live
camera. Engine camera identity and optics still require independent verification.
"""

import json
from pathlib import Path

from .ac_pixel_retina import digest


def validate_declared_live_view(retina_config, contract_path='configs/ac-live-view-contract.json'):
    path = Path(contract_path)
    if not path.exists():
        raise ValueError('Declare and verify the current live camera contract before student use')
    contract = json.loads(path.read_text(encoding='utf8'))
    if (contract.get('schema') != 'flybrain.ac.live_view.v1'
            or contract.get('retina_configuration_sha256') != digest(retina_config)):
        raise ValueError('Live camera declaration differs from the student retina; '
                         'recollect/retrain')
    return contract
