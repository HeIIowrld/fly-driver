"""Corrective pause/resume only for a proven stopped teacher's frozen native residual."""

import argparse
import json
import math
import os
import struct
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, repeated_snapshot
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def validate_residual_pair(first, second, terminal):
    if (terminal.get('status') != 'api_returned_true'
            or not isinstance(terminal.get('id'), str) or len(terminal['id']) != 32
            or any(c not in '0123456789abcdef' for c in terminal['id'])
            or terminal.get('command') not in ('expert-motion-speed60', 'expert-collect30-speed60')
            or terminal.get('stop_reason') != 'owned motion probe deadline'):
        raise ValueError('Normal immutable60kmh teacher terminal required')
    for row in (first, second):
        s, b = row['csp'], row['body']
        if (not 0 <= row['csp_age'] <= 2 or s.get('paused') is not True
                or any(s.get(k) is not False for k in ('ai_controlled', 'camera_active',
                                                      'expert_probe_active', 'is_online'))
                or s.get('error') or s.get('pending_receipt_id')
                or s.get('car_id') != 'bmw_z4_gt3' or s.get('track_id') != 'monza'
                or s['generation'] != terminal['after']['generation']
                or s['epoch'] != terminal['after']['epoch']
                or s.get('expert_probe_id') != terminal['id']
                or row['graphics_status'] != 3 or row['released'] is not True
                or row['background'] is not True or row['native_ai'] is not True
                or not all(math.isfinite(b[k]) for k in ('speedKmh', 'gas', 'brake'))
                or not .1 <= abs(b['speedKmh']) < .2 or b['gas'] >= .02 or b['brake'] <= .98
                or abs(s['speed_kmh']) >= .1 or s['gas'] >= .02 or s['brake'] <= .98):
            raise ValueError('Independent paused full-brake residual envelope required')
    if (second['graphics_packet'] <= first['graphics_packet']
            or second['packet_id'] != first['packet_id']
            or second['prefix_sha256'] != first['prefix_sha256']):
        raise ValueError('Advancing paused graphics and same frozen native prefix required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-control', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    teacher = Path(a.teacher_control)
    terminal = read(teacher/'api-receipt.json')
    run = begin_run(a.output, dict(arguments=vars(a), source_sha256=sha256(__file__),
        teacher_receipt_sha256=sha256(teacher/'api-receipt.json'),
        recovery_only=True, driving_ownership_granted=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
        lifecycle_requests_sent=0, driving_ownership_granted=False, student_deployed=False)
    root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    api = WindowsACWindow()
    windows = api.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one engine required')
    window = windows[0]
    installed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha256(installed) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
        raise ValueError('Deployed lifecycle source differs')
    axis = read('configs/ac-axis-probe.json')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    owned_terminal = root/(terminal['id']+'.terminal.json')
    if read(owned_terminal) != terminal:
        raise ValueError('Original immutable engine teacher terminal differs')
    pause_sent, resume_sent = False, False
    baseline = None

    def state():
        s, modified = read_published_json(root/'status.json')
        if (not 0 <= time.time()-modified <= 2 or s['generation'] != terminal['after'][
                'generation'] or s['epoch'] != terminal['after']['epoch']):
            raise ValueError('Fresh original generation required')
        return s, modified

    def snapshot():
        s, modified = state()
        with NativePhysicsReader(timeout=.15) as reader:
            sample = reader.receive()
        reader = NativeTeacherStatusReader()
        try:
            flag = reader.snapshot()
        finally:
            reader.close()
        with ExistingNamedMapping('Local\\acpmf_graphics', 12) as mapping:
            first = mapping.read()
            second = mapping.read()
        if first != second or sample.packet_id != flag['packet_id']:
            raise ValueError('Independent native snapshots changed')
        packet, status, _ = struct.unpack('<iii', second)
        return dict(csp=s, csp_age=time.time()-modified, body=dict(sample.fields),
            packet_id=sample.packet_id, prefix_sha256=flag['prefix_sha256'],
            native_ai=flag['builtin_ai_controlled'], graphics_status=status,
            graphics_packet=packet, released=sdk.GetVJDStatus(axis['device_id']) == 1,
            background=api.user.GetForegroundWindow() != window.hwnd)

    def issue(command):
        s, _ = state()
        observed = api.probe(window.hwnd, require_foreground=False)
        foreground = api.user.GetForegroundWindow() == window.hwnd
        released = sdk.GetVJDStatus(axis['device_id']) == 1
        result['last_lifecycle_guard'] = dict(command=command, csp=s,
            expected_window=asdict(window), observed_window=asdict(observed) if observed else None,
            foreground=foreground, released=released, passed=False)
        if (s['ai_controlled'] is not False or s['camera_active'] is not False
                or s['expert_probe_active'] is not False or s['is_online'] is not False
                or s['gas'] >= .02 or s['brake'] <= .98 or abs(s['speed_kmh']) >= .2
                or observed != window or foreground or not released):
            raise ValueError('Recovery envelope changed before lifecycle request')
        result['last_lifecycle_guard']['passed'] = True
        request = dict(schema='flybrain.csp.session.request.v1', id=uuid.uuid4().hex,
            command=command, generation=s['generation'], epoch=s['epoch'],
            expires_clock=s['script_clock']+3)
        write_json(run/(command+'-request.json'), request)
        write_json(root/'request.json', request)
        result['lifecycle_requests_sent'] += 1
        return request

    try:
        first = snapshot()
        time.sleep(.3)
        second = snapshot()
        validate_residual_pair(first, second, terminal)
        write_json(run/'preflight.json', dict(first=first, second=second))
        baseline = second
        if not a.execute:
            result['status'] = 'read_only_residual_preflight'
            return
        resume = issue('resume')
        resume_sent = True
        samples = []
        deadline = time.perf_counter()+.35
        with NativePhysicsReader(timeout=.15) as reader:
            while time.perf_counter() < deadline:
                sample = repeated_snapshot(reader.mapping, clock=time.perf_counter)
                samples.append(physics_record(sample))
                if (sample.packet_id != baseline['packet_id'] and abs(sample.fields['speedKmh'])
                        < .01 and sample.fields['gas'] < .02 and sample.fields['brake'] > .98):
                    break
                time.sleep(.01)
        # Pause once even if the requested native settling was not observed.
        pause = issue('pause')
        pause_sent = True
        write_json(run/'native-samples.json', samples)
        for request in (resume, pause):
            deadline = time.perf_counter()+3
            path = root/(request['id']+'.terminal.json')
            while not path.exists() and time.perf_counter() < deadline:
                time.sleep(.02)
            receipt = read(path)
            write_json(run/(request['command']+'-receipt.json'), receipt)
            if receipt['status'] != 'api_returned_true':
                raise ValueError('Recovery lifecycle request refused; never retry')
        final = snapshot()
        write_json(run/'after.json', final)
        if (final['csp']['paused'] is not True or abs(final['body']['speedKmh']) >= .1
                or final['body']['gas'] >= .02 or final['body']['brake'] <= .98
                or final['packet_id'] == baseline['packet_id']):
            raise ValueError('Independent stationary native packet not established')
        result.update(status='completed_teacher_residual_settle', final_speed_kmh=final[
            'body']['speedKmh'], foreground_after=api.user.GetForegroundWindow() == window.hwnd)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        if resume_sent and not pause_sent:
            # One compensating pause, no resend of an uncertain request.
            pause_sent = True
            try:
                issue('pause')
            except Exception as error:
                result['compensating_pause_error'] = str(error)
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
