"""Recovery-only engine reset from independently paused state; never grants driving ownership."""

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.win_process_receipt import ProcessReceipt
from scripts.record_ac_wgc import physics_record


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def deploy_pinned_publisher(spec, backup, validate):
    """Back up reviewed bytes, then revalidate paused state before one replacement."""
    source, target = Path(spec['source']), Path(spec['target'])
    if (sha256(source) != spec['source_sha256']
            or sha256(target) != spec['expected_installed_sha256']):
        raise ValueError('Pinned publisher source or installation differs')
    backup = Path(backup)
    if backup.exists():
        raise ValueError('Publisher backup already exists')
    backup.write_bytes(target.read_bytes())
    if sha256(backup) != spec['expected_installed_sha256']:
        raise ValueError('Publisher backup differs')
    staged = target.with_name(target.name+'.flybrain-'+uuid.uuid4().hex+'.partial')
    try:
        staged.write_bytes(source.read_bytes())
        validate()
        if (sha256(staged) != spec['source_sha256']
                or sha256(source) != spec['source_sha256']
                or sha256(target) != spec['expected_installed_sha256']):
            raise ValueError('Publisher changed during deployment')
        os.replace(staged, target)
        if sha256(target) != spec['source_sha256']:
            raise ValueError('Publisher deployment readback differs')
    finally:
        if staged.exists():
            staged.unlink()
    return dict(installed_sha256=sha256(target), backup_sha256=sha256(backup))


def validate_reset_only_pair(before, after):
    """Fresh paused graphics plus frozen ambiguous physics permits reset only, never driving."""
    for row in (before, after):
        csp, native, body, graphics = (row[k] for k in ('csp', 'native_ai', 'body', 'graphics'))
        if (not 0 <= row['csp_age_seconds'] <= 2 or csp.get('paused') is not True
                or csp.get('ai_controlled') is not False or csp.get('camera_active') is not False
                or csp.get('expert_probe_active') is not False or csp.get('is_online') is not False
                or csp.get('error') or csp.get('pending_receipt_id')
                or csp.get('car_id') != 'bmw_z4_gt3' or csp.get('track_id') != 'monza'
                or native['builtin_ai_controlled'] is not True
                or graphics['status'] != 3 or type(graphics['packet_id']) is not int
                or graphics['packet_id'] < 0 or row['released_controller'] is not True
                or row['background_engine'] is not True
                or not all(math.isfinite(body[k]) for k in ('speedKmh', 'gas', 'brake'))
                or abs(body['speedKmh']) >= .1 or body['gas'] >= .02 or body['brake'] <= .98):
            raise ValueError('Recovery-only reset requires independently paused offline engine')
    if (before['csp']['generation'] != after['csp']['generation']
            or after['graphics']['packet_id'] <= before['graphics']['packet_id']
            or after['native_ai']['packet_id'] != before['native_ai']['packet_id']
            or after['native_ai']['prefix_sha256'] != before['native_ai']['prefix_sha256']):
        raise ValueError('Fresh paused graphics and same frozen ambiguous physics required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), recovery_only=True, driving_ownership_granted=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', recovery_only=True, driving_ownership_granted=False,
        video_changed=False, lifecycle_requests_sent=0, ui_actions_sent=0,
        actuator_messages_sent=0, student_deployed=False, goal_achieved=False)
    old_process = None
    try:
        for file, digest in plan['source_sha256'].items():
            if sha256(file) != digest:
                raise ValueError('Preregistered recovery source changed')
        api = WindowsACWindow()
        windows = api.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original engine required')
        window = windows[0]
        deployed = Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        if sha256(deployed) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
            raise ValueError('Same reviewed lifecycle source required')
        axis = read('configs/ac-axis-probe.json')
        sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
        root = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'

        def snapshot():
            status, modified = read_published_json(root/'status.json')
            with NativePhysicsReader(timeout=.15) as reader:
                sample = reader.receive()
            native = NativeTeacherStatusReader()
            try:
                flag = native.snapshot()
            finally:
                native.close()
            with ExistingNamedMapping('Local\\acpmf_graphics', 12) as mapping:
                for _ in range(8):
                    first, second = mapping.read(), mapping.read()
                    if first == second:
                        break
                else:
                    raise ValueError('Native graphics changed during all repeated copies')
            packet, phase, session = struct.unpack('<iii', second)
            if sample.packet_id != flag['packet_id']:
                raise ValueError('Independent paused physics samples differ')
            if (int(status['window_handle'].removesuffix('ULL')) != window.hwnd
                    or api.probe(window.hwnd, require_foreground=False) != window):
                raise ValueError('Original paused window identity changed')
            return dict(csp=status, csp_age_seconds=time.time()-modified,
                native_ai=flag, body=dict(sample.fields), native_body=physics_record(sample),
                graphics=dict(packet_id=packet, status=phase, session=session,
                    prefix_base64=base64.b64encode(second).decode(),
                    prefix_sha256=hashlib.sha256(second).hexdigest()),
                background_engine=api.user.GetForegroundWindow() != window.hwnd,
                released_controller=sdk.GetVJDStatus(axis['device_id']) == 1)

        before = snapshot()
        time.sleep(.3)
        after = snapshot()
        validate_reset_only_pair(before, after)
        write_json(run/'reset-only-preflight.json', dict(before=before, after=after,
            native_ai_off_not_claimed=True, original_window=asdict(window)))
        original = Path(plan['actual_video_path'])
        prepared = Path(plan['prepared_video'])
        if (sha256(original) != plan['expected_actual_video_sha256']
                or sha256(prepared) != plan['prepared_video_sha256']):
            raise ValueError('Prepared reversible video change differs')
        backup = run/'video-before.ini'
        backup.write_bytes(original.read_bytes())
        if not args.execute:
            result.update(status='read_only_reset_preflight')
            return
        old_process = ProcessReceipt(window.pid)
        if not old_process.snapshot()['active']:
            raise ValueError('Original process already ended')
        if plan.get('publisher_deployment'):
            spec = plan['publisher_deployment']
            expected_target = (Path(window.executable).parent
                / 'apps/python/flybrain_telemetry/flybrain_telemetry.py')
            if Path(spec['target']) != expected_target:
                raise ValueError('Publisher target must belong to the observed engine')
            deployment = deploy_pinned_publisher(spec, run/'publisher-installed-before.py',
                lambda: validate_reset_only_pair(before, snapshot()))
            result.update(publisher_installed=True, publisher_deployment=deployment)
        changed = sha256(original) != plan['prepared_video_sha256']
        if changed:
            part = original.with_name(original.name+'.flybrain-'+uuid.uuid4().hex+'.partial')
            part.write_bytes(prepared.read_bytes())
            validate_reset_only_pair(before, snapshot())
            os.replace(part, original)
        if sha256(original) != plan['prepared_video_sha256']:
            raise ValueError('Video replacement readback differs')
        result.update(video_changed=changed, video_sha256=sha256(original))
        fresh = snapshot()
        validate_reset_only_pair(before, fresh)
        request = dict(schema='flybrain.csp.session.request.v1', id=uuid.uuid4().hex,
            command='restart-engine', generation=fresh['csp']['generation'],
            epoch=fresh['csp']['epoch'], expires_clock=fresh['csp']['script_clock']+3,
            old_engine_pid=window.pid, released_controller_verified=True)
        write_json(run/'request.json', request)
        write_json(root/'request.json', request)
        result.update(lifecycle_requests_sent=1, old_engine_pid=window.pid,
                      request_id=request['id'], restart_outcome_verified=False)
        deadline, progress_at = time.monotonic()+90, 0.
        while time.monotonic() < deadline:
            receipt = old_process.snapshot()
            terminal = root/(request['id']+'.terminal.json')
            if terminal.exists():
                intent = read(terminal)
                if (intent['id'] != request['id']
                        or intent['status'] != 'engine_restart_intent_committed'):
                    raise ValueError('Different or rejected restart intent')
                if not (run/'restart-intent.json').exists():
                    write_json(run/'restart-intent.json', intent)
            candidates = api.discover()
            if time.monotonic() >= progress_at:
                write_json(run/'progress.json', dict(original_process=receipt,
                    windows=[asdict(w) for w in candidates], request_id=request['id']))
                progress_at = time.monotonic()+1
            newer = [w for w in candidates if w.pid != window.pid
                     and Path(w.executable) == Path(window.executable)]
            if not receipt['active'] and len(newer) == 1:
                try:
                    state, modified = read_published_json(root/'status.json')
                except (OSError, ValueError):
                    time.sleep(.1)
                    continue
                if (0 <= time.time()-modified <= 2 and state['generation'] != after['csp'][
                        'generation'] and int(state['window_handle'].removesuffix('ULL')) == (
                            newer[0].hwnd)):
                    if plan.get('publisher_deployment'):
                        try:
                            client = FileTelemetryClient(Path(os.environ['LOCALAPPDATA'])
                                / 'flybrain/telemetry', timeout=.02, clock=time.perf_counter)
                            packet = client.snapshot()
                        except (OSError, ValueError, TimeoutError):
                            time.sleep(.1)
                            continue
                        if (not client.session_id.startswith('ac-file-'+str(newer[0].pid)+'-')
                                or packet.raw.get('publisher_previous_io_timing', {}).get(
                                    'schema') != 'flybrain.telemetry.host_stage_timing.v1'):
                            time.sleep(.1)
                            continue
                        if (plan.get('expected_publisher_delivery') is not None and
                                packet.raw.get('publisher_delivery') != plan[
                                    'expected_publisher_delivery']):
                            time.sleep(.1)
                            continue
                        write_json(run/'new-publisher-packet.json', dict(
                            session=client.session_id, packet=record_packet(packet)))
                        result['publisher_loaded_verified'] = True
                    write_json(run/'after.json', dict(session=state,
                        window=asdict(newer[0]), original_process=receipt))
                    result.update(status='completed_reset_only_engine_restart',
                        restart_outcome_verified=True, new_engine_pid=newer[0].pid,
                        generation=state['generation'], foreground_after=(
                            api.user.GetForegroundWindow() == newer[0].hwnd))
                    break
            time.sleep(.1)
        else:
            raise TimeoutError('Same restart outcome unknown; do not resend intent')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if old_process is not None:
            result['original_process'] = old_process.snapshot()
            old_process.close()
        write_json(run/'result.json', result)
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
