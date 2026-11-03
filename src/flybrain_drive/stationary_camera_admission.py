"""Pure fail-closed checks before an owned paused stationary camera capture."""
import math
import re


def validate_stationary_camera_admission(status, request, camera_status, terminal):
    if (not isinstance(status,dict) or status.get('paused') is not True
            or status.get('live') is not False or status.get('is_online') is not False
            or status.get('in_menu') is not False or status.get('camera_active') is not False
            or status.get('ai_controlled') is not False or status.get('expert_probe_active') is not False
            or status.get('ideal_line_shown') is not False or status.get('error')
            or status.get('pending_receipt_id')):
        raise ValueError('Paused idle offline camera and AI-off state required')
    speed,gas,brake=status.get('speed_kmh'),status.get('gas'),status.get('brake')
    if (not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                for v in (speed,gas,brake)) or abs(speed)>=.1 or gas>=.02 or brake<=.98):
        raise ValueError('Stationary full-brake condition required')
    if request is None:return
    if not isinstance(request,dict) or not isinstance(request.get('id'),str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',request['id']):
        raise ValueError('Previous camera request identity unknown')
    request_id=request['id']
    terminal_ok=(isinstance(terminal,dict) and terminal.get('request_id')==request_id
                 and terminal.get('status') in ('completed','stopped','failed'))
    known_last=(isinstance(camera_status,dict) and camera_status.get('request_id')==request_id
                and camera_status.get('state') in ('completed','failed','stopped'))
    if terminal_ok and known_last and terminal['status'] != camera_status['state']:
        raise ValueError('Previous camera terminal and heartbeat disagree')
    if not terminal_ok and not known_last:
        raise ValueError('Previous camera request terminal state unknown')


def validate_stationary_file_camera_admission(status, request, camera_status, terminal):
    """File camera only: stationary unpaused original session and no pending request."""
    if not isinstance(status,dict) or status.get('paused') is not False or status.get('live') is not True:
        raise ValueError('Unpaused live stationary file-camera session required')
    # Reuse every idle/AI/brake/request check without changing the original status.
    checked=dict(status,paused=True,live=False)
    validate_stationary_camera_admission(checked,request,camera_status,terminal)
    if camera_status is not None and camera_status.get('generation') != status.get('generation'):
        raise ValueError('Same generation camera heartbeat required')


def validate_known_stopped_metadata_failure_probe(status, request, camera_status, terminal, clock, *, now):
    """Permit one diagnostic capture after preserving this specific stopped/error conflict.

    Clock exchanges demonstrate current reply writes, not camera health. This is not
    normal admission, a repair claim, or permission to ignore other failures.
    """
    validate_stationary_file_camera_admission(status, request, None, terminal)
    rid = request['id']
    if (not isinstance(camera_status, dict) or camera_status.get('generation') != status.get('generation')
            or camera_status.get('request_id') != rid or camera_status.get('state') != 'failed'
            or not isinstance(camera_status.get('reason'), str)
            or 'Cannot commit camera metadata' not in camera_status['reason']
            or terminal.get('request_id') != rid or terminal.get('status') != 'stopped'):
        raise ValueError('Exact known stopped metadata failure required')
    exchanges = clock.get('exchanges') if isinstance(clock, dict) else None
    width = clock.get('uncertainty_seconds') if isinstance(clock, dict) else None
    if (not isinstance(clock, dict) or not isinstance(now, (int, float)) or isinstance(now, bool) or not math.isfinite(now)
            or clock.get('generation') != status.get('generation')
            or not isinstance(width, (int, float)) or isinstance(width, bool)
            or not math.isfinite(width) or not 0 <= width <= .01
            or not isinstance(exchanges, list) or len(exchanges) < 4):
        raise ValueError('Fresh same-generation clock exchange required')
    for item in exchanges:
        if (not isinstance(item, dict) or not all(isinstance(item.get(k), (int, float))
                and not isinstance(item[k], bool) and math.isfinite(item[k]) for k in ('sent', 'received', 'producer_clock'))
                or not 0 <= item['sent'] <= item['received'] <= now):
            raise ValueError('Causal clock exchange required')
    if now-exchanges[-1]['received'] > 1:
        raise ValueError('Fresh same-generation clock exchange required')
