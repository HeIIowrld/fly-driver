"""Narrow diagnostic admission after an owned completed camera's metadata failure.

This authorizes one new finite stationary diagnostic, never repairs old evidence
or converts the previous failed observation into a passing run.
"""
import math
from .stationary_camera_admission import validate_stationary_camera_admission


def validate_completed_metadata_diagnostic(status,request,camera,terminal,clock,evidence,*,now):
    validate_stationary_camera_admission(status,request,None,terminal)
    rid=request['id']
    if (not isinstance(camera,dict) or camera.get('state')!='failed'
            or camera.get('generation')!=status.get('generation') or camera.get('request_id')!=rid
            or not isinstance(camera.get('reason'),str) or 'Cannot commit camera metadata' not in camera['reason']
            or terminal.get('status')!='completed' or type(request.get('frames')) is not int
            or not 1<=request['frames']<=1200 or terminal.get('frames')!=request['frames']
            or terminal.get('actor_stream_created') is not False or terminal.get('actuator_messages_sent')!=0):
        raise ValueError('Exact completed owned producer and failed metadata heartbeat required')
    if (evidence.get('producer_terminal')!=terminal or evidence.get('archive_bundle_sha256')!=evidence.get('expected_bundle_sha256')
            or not isinstance(evidence.get('expected_bundle_sha256'),str) or len(evidence['expected_bundle_sha256'])!=64
            or evidence.get('archive_verified') is not True or evidence.get('prior_sources_closed') is not True
            or evidence.get('source_sha256')!=evidence.get('expected_source_sha256')
            or not isinstance(evidence.get('source_sha256'),str) or len(evidence['source_sha256'])!=64):
        raise ValueError('Hash-bound original archive/source and terminated prior owners required')
    exchanges=clock.get('exchanges');width=clock.get('uncertainty_seconds')
    if (clock.get('generation')!=status['generation'] or not isinstance(width,(int,float))
            or not math.isfinite(width) or not 0<=width<=.01 or not isinstance(exchanges,list) or len(exchanges)<4
            or not math.isfinite(now)):
        raise ValueError('Fresh same-generation bounded clock exchanges required')
    for p in exchanges:
        values=[p.get(k) for k in ['sent','received','producer_clock']]
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in values) or not 0<=values[0]<=values[1]<=now:
            raise ValueError('Original causal clock exchange required')
    if now-exchanges[-1]['received']>1:raise ValueError('Clock witness expired')
    return dict(scope='One finite stationary diagnostic only; original completed/failed discrepancy retained',
        request_id=rid,generation=status['generation'],original_terminal_status=terminal['status'],original_camera_state=camera['state'],
        previous_run_reclassified=False,camera_health_verified=False,actor_qualified=False)
