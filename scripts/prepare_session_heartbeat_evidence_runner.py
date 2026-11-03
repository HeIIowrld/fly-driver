"""Create a NEW evidence-only runner from exact v2; no live action/deployment."""

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path('scripts/qualify_session_heartbeat_service_v2.py')
BASE_SHA = '3ed890e79f4eedf851899528a3732d727eb0bcd96ff4a4465f618da1b7472878'
OUTPUT = Path('scripts/qualify_session_heartbeat_evidence.py')


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned evidence-only replacement count differs: '+old[:60])
    return source.replace(old, new)


def build():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Frozen v2 runner changed; review a new source binding')
    source = raw.decode().replace('\r\n', '\n')
    source = once(source, '    from flybrain_drive.session_heartbeat_runtime import (\n',
        '''    from flybrain_drive.session_heartbeat_evidence_v1 import (
        EvidenceDeferred,
        HeartbeatEvidence,
        require_local_c_output,
    )
    from flybrain_drive.session_heartbeat_runtime import (
''')
    source = once(source,
        "    session_root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'",
        "    require_local_c_output(args.output)\n    evidence = HeartbeatEvidence()\n"
        "    session_root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'")
    source = once(source,
        '        context = HeartbeatSessionContext(status_path, calibration)\n',
        '        captured_calibration = evidence.capture_calibration(calibration)\n'
        '        context = HeartbeatSessionContext(status_path, calibration,\n'
        '            map_factory=evidence.map_factory, control_factory=evidence.control_factory)\n'
        '        if captured_calibration != context.calibration.calibration_sha256:\n'
        "            raise ValueError('Captured original calibration fingerprint differs')\n")
    source = once(source, "        write_json(session_root/'heartbeat-transport.json', plan)\n",
        "        write_json(session_root/'heartbeat-transport.json', plan)\n"
        "        evidence.capture_arm(session_root/'heartbeat-transport.json')\n")
    source = once(source, "        result['handoff'] = context.complete_sync_return()\n",
        '        evidence.capture_sync(context)\n'
        "        result['handoff'] = context.complete_sync_return()\n"
        "        evidence.bind_handoff(result['handoff'])\n")
    source = once(source, "        write_json(output/'result.json', result)\n\n\ndef main():",
        '''        try:
            result['heartbeat_evidence'] = evidence.persist(
                output/'heartbeat-evidence', context=context)
        except EvidenceDeferred as error:
            result['heartbeat_evidence'] = evidence.defer_persistence(
                output/'heartbeat-evidence', context=context,
                receipt_path=output/'heartbeat-evidence-terminal-receipt.json')
            result['heartbeat_evidence']['reason'] = str(error)
        except BaseException as error:
            result['heartbeat_evidence'] = dict(status='persistence_failed',
                error=type(error).__name__, reason=str(error),
                partial_owned_refs=evidence.pending_report(context))
        if (result['status'].startswith('completed')
                and result['heartbeat_evidence']['status'] != 'complete'):
            result.update(status='failed', error='IncompleteHeartbeatEvidence',
                reason='Original evidence incomplete; retain this run, never rerun automatically')
        write_json(output/'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


def main():''')
    source = once(source, "choices=['stationary-smoke', 'camera3', 'pace30']",
        "choices=['camera3', 'pace30']")
    return source.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    content = build()
    OUTPUT.write_bytes(content)
    receipt = dict(base=str(BASE), base_sha256=BASE_SHA, output=str(OUTPUT),
        output_sha256=hashlib.sha256(content).hexdigest(),
        frozen_base_unchanged=hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA,
        protocol_runtime_changed=False, live_actions=0)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
