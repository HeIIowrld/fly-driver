"""Same-process vJoy-to-engine-AI transfer; explicit finite probes, no UI input."""

import json
import os
import time
import uuid
from pathlib import Path

from .evidence import write_json
from .session_heartbeat_provenance import (
    HeartbeatProvenanceError,
    capture_lease,
    validate_opened_status,
)


def validate_handoff_preset(context, decisions, seconds, native_receipts, profile,
                            backend, background):
    """Only explicit finite data probes; extended launch keeps the motion guards."""
    routes = {'launch': 0, 'curve200': 200, 'curve320': 320}
    if context not in routes:
        raise ValueError('Unknown handoff data context')
    curve = context != 'launch'
    if (type(decisions) is not int or backend != 'csp' or background is not True
            or profile != ('receipt15_route480' if curve else 'receipt15_route240')
            or decisions not in ((2000,) if curve else (800, 1200))
            or (curve or decisions == 1200) and (
                seconds != 30 or native_receipts is not True)):
        raise ValueError('Explicit bounded launch800/1200 or curve200/320 data preset required')
    return routes[context]


def validate_owned_handoff(session, status, age, *, seconds=3):
    if type(seconds) is not int or seconds not in (3, 30):
        raise ValueError('Explicit3second or30second handoff preset required')
    if (not session.owned or session.sdk.GetVJDStatus(session.device_id) != 0
            or not 0 <= age <= 2 or status.get('error') or status.get('pending_receipt_id')
            or status.get('paused') is not False or status.get('live') is not True
            or status.get('ai_controlled') is not False or status.get('expert_probe_active')
            or status.get('physics_api_allowed') is not True or status.get('is_online')
            or status.get('car_id') != 'bmw_z4_gt3' or status.get('track_id') != 'monza'
            or status.get('capabilities', {}).get('expert_moving_handoff'+str(seconds)) is not True
            or not 0 <= status.get('speed_kmh', -1) <= 18.5):
        raise ValueError('Same-process owned vJoy and explicit live16..18.5kmh handoff required')



def validate_curve_handoff_evidence(arguments, request, prefix_result):
    """Verify the opt-in collection context from independently replayed route evidence."""
    context = arguments.get('handoff_context', 'launch')
    if context == 'launch':
        return
    thresholds = {'curve200': 200, 'curve320': 320}
    if context not in thresholds:
        raise ValueError('Unknown moving-handoff collection context')
    route = prefix_result.get('route_progress_evidence') or {}
    distance = route.get('distance_m')
    if (arguments.get('trial_profile') != 'receipt15_route480'
            or arguments.get('decisions') != 2000 or request.get('hold_seconds') != 30
            or not arguments.get('handoff_native_receipts')
            or route.get('continuity_error') is not None
            or isinstance(distance, bool) or not isinstance(distance, (int, float))
            or not thresholds[context] <= distance < float('inf')):
        raise ValueError('Pinned curve context and independently replayed route progress required')


def effective_motion_after_release(sample, released_at):
    return (sample.received_monotonic_seconds >= released_at+.3
            and sample.fields['speedKmh'] > 1 and sample.fields['brake'] < .98)


def handoff_observation_records(events, roles):
    """One immutable prefix/suffix index mapping; coordinator role is not an expert label mask."""
    records, pairs, requested, transferred, released = [], {}, False, False, False
    for event in events:
        kind = event['kind']
        if kind in ('raw_pair', 'expert_raw_native'):
            key = event['native']['packet_id']
            if key in pairs:
                raise ValueError('Duplicate raw native packet in handoff')
            pairs[key] = event['native']
        elif kind == 'expert_handoff_requested':
            if requested or transferred:
                raise ValueError('Repeated handoff intent')
            requested = True
        elif kind == 'expert_handoff':
            if transferred:
                raise ValueError('Repeated handoff')
            transferred = True
        elif kind == 'handoff_vjoy_released':
            if not transferred or released or event['vjoy_status'] != 1:
                raise ValueError('Unordered or unverified vJoy release')
            released = True
        elif kind in ('student_request', 'expert_observation', 'handoff_wait_observation'):
            expert = kind == 'expert_observation'
            bridge = kind == 'handoff_wait_observation'
            if (expert and not released or not expert and transferred
                    or bridge and not requested or kind == 'student_request' and requested):
                raise ValueError('Student/expert actions interleaved across handoff')
            if bridge and event.get('prediction_applied') is not False:
                raise ValueError('Waiting observations must not submit model predictions')
            meta = event if expert else event['actor_input']
            index = meta['observation_index']
            if index != len(records) or index >= len(roles) or bool(roles[index]) != expert:
                raise ValueError('Exact contiguous observation role/index mapping required')
            records.append(dict(index=index, expert=expert, bridge=bridge,
                prediction=None if expert else event['prediction'],
                native_packet_id=meta['native_packet_id'],
                camera=meta['camera' if expert else 'image'],
                decision_monotonic_seconds=meta['decision_monotonic_seconds']))
    if (not transferred or not released or len(records) != len(roles)
            or not any(roles) or all(roles)):
        raise ValueError('One complete student prefix and expert suffix required')
    return records, pairs


class OwnedExpertHandoff:
    def __init__(self, camera, output, *, seconds=3):
        if type(seconds) is not int or seconds not in (3, 30):
            raise ValueError('Explicit3second or30second handoff preset required')
        self.seconds = seconds
        self.camera, self.output = camera, Path(output)
        self.root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
        self.request, self.terminal, self.deadline = None, None, None

    def status(self):
        path = self.root/'status.json'
        deadline = time.monotonic()+.15
        while True:
            try:
                captured = capture_lease(path)
                with path.open('rb') as stream:
                    content = stream.read()
                    value = json.loads(content.decode('utf8'))
                    modified = os.fstat(stream.fileno()).st_mtime
                validate_opened_status(path, content, value, captured_lease=captured)
                break
            except (FileNotFoundError, PermissionError, json.JSONDecodeError,
                    HeartbeatProvenanceError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.002)
        if (not 0 <= time.time()-modified <= 2 or value.get('error')
                or self.request is not None and (
                    value['generation'] != self.request['generation']
                    or value['epoch'] != self.request['epoch'])):
            raise ValueError('Fresh same-generation/epoch expert handoff status required')
        return value

    def activate(self, session, reading, *, observe_wait=None):
        if self.request is not None:
            raise ValueError('Never repeat an accepted or uncertain handoff')
        status = self.status()
        validate_owned_handoff(session, status,
                              time.time()-(self.root/'status.json').stat().st_mtime,
                              seconds=self.seconds)
        if (status.get('camera_request_id') != self.camera.request_id
                or self.camera.clock.generation != status['generation']
                or not 16 <= reading.diagnostic['speed_kmh'] <= 18.5
                or reading.diagnostic['tyres_out'] != 0):
            raise ValueError('Same owned camera and bounded zero-tyre-out student required')
        # The heartbeat can lag by250ms; the fresh native sample above and the
        # server's actual live snapshot both still require16..18.5kmh.
        self.output.mkdir(exist_ok=False)
        self.request = dict(schema='flybrain.csp.session.request.v1', id=uuid.uuid4().hex,
            command='expert-moving-handoff'+str(self.seconds), generation=status['generation'],
            epoch=status['epoch'],
            expires_clock=status['script_clock']+3, hold_seconds=self.seconds, speed_limit_kmh=15,
            ai_steer_multiplier=.9, camera_request_id=self.camera.request_id,
            student_owner_verified=True, student_owner_pid=os.getpid())
        self.deadline = time.monotonic()+self.seconds+7
        write_json(self.output/'request.json', self.request)
        write_json(self.output/'before.json', status)
        write_json(self.root/'request.json', self.request)
        activation_deadline = time.monotonic()+.8
        while time.monotonic() < activation_deadline:
            if self.poll() is not None:
                raise ValueError('Handoff ended before ownership was observed')
            current = self.status()
            if current.get('expert_probe_id') == self.request['id'] \
                    and current.get('ai_controlled') is True and current.get('paused') is False:
                receipt = dict(request_id=self.request['id'], student_owner_pid=os.getpid(),
                    ai_active_observed=True, host_clock=time.perf_counter())
                write_json(self.output/'activation.json', receipt)
                return receipt
            if observe_wait is not None:
                observe_wait()
            else:
                time.sleep(.005)
        raise TimeoutError('Handoff activation unknown; never issue another activation')

    def poll(self):
        if self.request is None:
            return None
        from scripts.control_ac_session import terminal_receipt

        receipt = terminal_receipt(self.root, self.request)
        if receipt is not None:
            self.terminal = receipt
            write_json(self.output/'api-receipt.json', receipt)
        return receipt

    def finish(self, *, cancel=False):
        if self.request is None:
            return None
        if cancel and self.terminal is None:
            (self.root/(self.request['id']+'.stop')).touch(exist_ok=True)
        while self.poll() is None:
            if time.monotonic() >= self.deadline:
                raise TimeoutError('Same finite handoff outcome unknown; inspect owned receipt')
            time.sleep(.01)
        if (self.terminal.get('status') != 'api_returned_true'
                or self.terminal.get('moving_handoff_passed') is not True):
            raise ValueError('Moving handoff did not verify its native brake/pause teardown')
        return self.terminal
