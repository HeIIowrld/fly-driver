"""Pure admission gate for a future paused-start stationary MMF shadow.

This gate issues no lifecycle or camera requests and does not establish physical
sensor timing. An unresolved preceding request must never be overwritten.
"""
import re
from .stationary_camera_admission import validate_stationary_camera_admission


def _require_stationary_session(status):
    """Recheck the original stationary contract without mutating a heartbeat.

    Camera ownership and paused/live state are checked separately by the gate.
    This does not establish heartbeat freshness or issue any lifecycle action.
    """
    if not isinstance(status, dict):
        raise ValueError('Original stationary session heartbeat required')
    idle_view = dict(status, paused=True, live=False, camera_active=False)
    validate_stationary_camera_admission(idle_view, None, None, None)


class PausedMMFStartupGate:
    """Enforce ownership and ordering across paused admission and live resume."""

    def __init__(self, status, previous_request, previous_camera, previous_terminal):
        validate_stationary_camera_admission(status, previous_request, previous_camera, previous_terminal)
        generation = status.get('generation')
        epoch = status.get('epoch')
        if not isinstance(generation, str) or not generation or type(epoch) is not int or epoch < 0:
            raise ValueError('Exact session generation and epoch required')
        if previous_request is not None and (not isinstance(previous_terminal, dict)
                or previous_terminal.get('request_id') != previous_request.get('id')
                or previous_terminal.get('status') not in ('completed', 'stopped', 'failed')):
            raise ValueError('Matching preceding camera terminal receipt required')
        self.generation, self.epoch = generation, epoch
        self.request_id, self.phase = None, 'admitted'

    def camera_ready(self, status, camera, request_id):
        if self.phase != 'admitted':
            raise RuntimeError('Camera readiness only once after admission')
        _require_stationary_session(status)
        if (not isinstance(request_id, str) or not re.fullmatch(r'mmf-stream-[A-Za-z0-9_-]{1,69}', request_id)
                or status.get('generation') != self.generation or status.get('epoch') != self.epoch
                or status.get('paused') is not True or status.get('live') is not False
                or status.get('camera_active') is not True or status.get('ai_controlled') is not False
                or status.get('camera_request_id') != request_id
                or status.get('is_online') is not False or status.get('in_menu') is not False
                or camera.get('generation') != self.generation
                or camera.get('request_id') != request_id or camera.get('state') != 'capturing'):
            raise ValueError('Owned MMF camera must be capturing in the same paused session')
        self.request_id, self.phase = request_id, 'camera_ready'

    def resume_observed(self, status, camera, *, first_body_packet, later_body_packet):
        if self.phase != 'camera_ready':
            raise RuntimeError('Owned paused camera must be ready before resume')
        _require_stationary_session(status)
        if (status.get('generation') != self.generation or status.get('epoch') != self.epoch
                or status.get('paused') is not False or status.get('live') is not True
                or status.get('camera_active') is not True or status.get('ai_controlled') is not False or status.get('is_online') is not False
                or status.get('camera_request_id') != self.request_id
                or status.get('in_menu') is not False
                or camera.get('generation') != self.generation
                or camera.get('request_id') != self.request_id or camera.get('state') != 'capturing'
                or type(first_body_packet) is not int or type(later_body_packet) is not int
                or later_body_packet <= first_body_packet):
            raise ValueError('Original live session, owned camera and advancing native packets required')
        self.phase = 'live'
