"""Pin the read-only diagnostic transport proof independently of actor inference."""

import base64
import hashlib
import json
from pathlib import Path

from .graph import sha256


def verified_diagnostic_transport(plan):
    path = plan.get('diagnostic_transport_verification')
    if path is None:
        return None
    root = Path(path)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    file = 'src/flybrain_drive/ac_file.py'
    if (result['status'] != 'completed_readonly_diagnostic_probe' or result['samples'] != 50
            or result['actuator_messages_sent'] != 0 or result['maximum_original_age'] > .05
            or manifest['source_sha256'][file] != sha256(file)):
        raise ValueError('Same successful bounded diagnostic transport proof required')
    proof = dict(path=str(root), result_sha256=sha256(root/'result.json'),
                 implementation_sha256=sha256(file))
    if plan.get('publisher_restart_verification'):
        restart = Path(plan['publisher_restart_verification'])
        receipt = json.loads((restart/'result.json').read_text(encoding='utf8'))
        provenance = json.loads((restart/'manifest.json').read_text(encoding='utf8'))
        after = json.loads((restart/'after.json').read_text(encoding='utf8'))
        publisher = 'integrations/ac/flybrain_telemetry/flybrain_telemetry.py'
        assert receipt['status'] == 'completed_engine_publisher_restart'
        assert receipt['restart_outcome_verified'] and receipt['publisher_installed']
        assert not receipt['original_process']['active']
        assert receipt['original_process']['exit_code'] == 0
        assert receipt['new_engine_pid'] != receipt['old_engine_pid']
        assert provenance['config']['new_publisher_sha256'] == sha256(publisher)
        assert manifest['source_sha256'][publisher] == sha256(publisher)
        raw = base64.b64decode(after['packet']['wire_base64'], validate=True)
        assert hashlib.sha256(raw).hexdigest() == after['packet']['wire_sha256']
        packet = json.loads(raw)
        assert packet['publisher_replace_retry_limit_seconds'] == .003
        assert packet['producer_session_id'] == receipt['new_producer_session']
        assert result['publisher_error_count_change'] == 0
        proof['publisher_restart'] = dict(path=str(restart),
            result_sha256=sha256(restart/'result.json'),
            publisher_sha256=sha256(publisher), producer_session=receipt['new_producer_session'])
    return proof
