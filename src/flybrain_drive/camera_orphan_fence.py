"""Explicit read-only orphan fence validation; never fabricates old terminal."""
import re


def validate_orphan_fence(*, request, orphan, quarantine, restart, restart_request,
                          release, current_session, current_camera, engine_pid,
                          request_slot_exists, old_terminal_exists):
    """Validate captured evidence for a new-generation diagnostic, not driving.

    Caller must validate fresh source reads, hashes, OS handle provenance and
    native sensors independently. The old request outcome remains unknown.
    """
    rid=request.get('id') if isinstance(request,dict) else None
    if not isinstance(rid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',rid):
        raise ValueError('Exact original request identity required')
    old_gen=request.get('expected_generation');new_gen=current_session.get('generation')
    old=restart.get('original_process',{})
    if (quarantine.get('status')!='request_slot_quarantined_outcome_still_unknown'
            or quarantine.get('original_terminal_known') is not False
            or orphan.get('request_id')!=rid or orphan.get('expected_generation')!=old_gen
            or orphan.get('status')!='prior_generation_unresolved' or orphan.get('actor_admissible') is not False
            or not isinstance(old_gen,str) or not old_gen or not isinstance(new_gen,str) or not new_gen or old_gen==new_gen
            or request_slot_exists is not False or old_terminal_exists is not False):
        raise ValueError('Original quarantine identity and unknown terminal required')
    if (restart.get('status')!='completed_engine_restart_after_camera_quarantine'
            or restart.get('restart_outcome_verified') is not True or old.get('pid')!=restart.get('old_engine_pid')
            or old.get('termination_signaled') is not True or old.get('active') is not False
            or restart.get('new_engine_pid')!=engine_pid or engine_pid==old.get('pid')
            or restart_request.get('old_engine_pid')!=old.get('pid')
            or restart_request.get('generation')!=quarantine.get('generation')
            or release.get('generation')!=restart_request.get('generation')
            or release.get('active_before') is not False or release.get('active_after') is not False):
        raise ValueError('Old process termination and owned release evidence required')
    if (current_session.get('camera_active') is not False or current_session.get('ai_controlled') is not False
            or current_session.get('is_online') is not False or current_session.get('in_menu') is not False
            or current_camera.get('generation')!=new_gen or current_camera.get('state')!='idle'):
        raise ValueError('Fresh new-generation idle offline session required')
    return {'status':'diagnostic_fence_evidence_valid','original_request_id':rid,
            'original_terminal':'unknown','old_engine_terminated':True,
            'new_generation':new_gen,'engine_pid':engine_pid,
            'actor_qualified':False,'camera_requested':False}
