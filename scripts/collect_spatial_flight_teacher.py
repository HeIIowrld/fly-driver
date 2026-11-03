"""Collect whole finite teacher histories with prospectively assigned geographic roles."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.ac_camera_mount import configured_mount
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.spatial_teacher_plan import next_lease, validate_lease_route


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def recovery_candidates(previous, audited):
    """Reuse every completed fixed-role TRAIN, never the failed tail or a chosen subset."""
    entries = [e for e in previous['episodes'] if e['lease']['role'] != 'transit']
    if (previous['status'] != 'failed'
            or previous['completed_counts'] != dict(train=3, development=0)
            or audited['status'] != 'completed_partial_spatial_history_raw_audit'
            or len(entries) != 3 or len(audited['episodes']) != 3
            or len({e['path'] for e in entries}) != 3
            or len({e['path'] for e in audited['episodes']}) != 3
            or any(e['lease']['role'] != 'train' for e in entries)
            or {e['path']: e['lease']['role'] for e in entries} != {
                e['path']: e['role'] for e in audited['episodes']}):
        raise ValueError('All three completed preassigned TRAIN histories and partial audit required')
    return entries


def idle(generation):
    state, modified = read_published_json(
        Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
    if (not 0 <= time.time()-modified <= 2 or state['generation'] != generation
            or not state['paused'] or state['ai_controlled'] or state['camera_active']
            or state['expert_probe_active'] or state['is_online'] or state['in_menu']
            or state.get('error') or state.get('pending_receipt_id')
            or state['ideal_line_shown'] is not False):
        raise ValueError('Fresh original idle generation and explicit aid-OFF required')
    axis = read('configs/ac-axis-probe.json')
    sdk = load_vjoy_sdk(axis['dll_path'], axis['dll_sha256'])
    native = WindowsACWindow()
    windows = native.discover()
    if (len(windows) != 1 or native.user.GetForegroundWindow() == windows[0].hwnd
            or sdk.GetVJDStatus(axis['device_id']) != 1):
        raise ValueError('Background original engine and released vJoy required')
    with NativePhysicsReader(timeout=.15) as reader:
        sample = reader.receive()
    if (abs(sample.fields['speedKmh']) >= .1 or sample.fields['gas'] >= .02
            or sample.fields['brake'] <= .98):
        raise ValueError('Native stopped full-brake state required')
    return dict(session=state, native=dict(sample.fields), engine_pid=windows[0].pid,
                student_owner_qualification=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, plan_hash = read(args.plan), sha256(args.plan)
    recovery = plan['schema'] == 'flybrain.spatial_teacher_collection.v4'
    exterior = recovery or plan['schema'] == 'flybrain.spatial_teacher_collection.v3'
    mmf = plan['schema'] in ('flybrain.spatial_teacher_collection.v2',
                             'flybrain.spatial_teacher_collection.v3',
                             'flybrain.spatial_teacher_collection.v4')
    if (plan['schema'] not in ('flybrain.spatial_teacher_collection.v1',
                              'flybrain.spatial_teacher_collection.v2',
                              'flybrain.spatial_teacher_collection.v3',
                              'flybrain.spatial_teacher_collection.v4')
            or not 1 <= plan['maximum_leases'] <= 18 or plan['teacher_speed_kmh'] != 60
            or plan['required_histories'] != dict(train=3, development=1)
            or mmf != plan.get('mmf_metadata', False)):
        raise ValueError('Explicit finite spatial collection plan required')
    if not exterior and 'camera_configuration' in plan:
        raise ValueError('Explicit exterior camera configuration requires collection schema v3')
    prior = Path(plan['initial_predecessor'])
    if sha256(prior/'control/api-receipt.json') != plan['initial_receipt_sha256']:
        raise ValueError('Initial known predecessor differs')
    predecessor = read(prior/'control/api-receipt.json')
    generation = predecessor['after']['generation']
    for source, digest in plan['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Prospectively configured source differs: '+source)
    audit = read(Path(plan['prerequisite_raw_audit'])/'result.json')
    if (audit['status'] != 'completed' or any(audit['actor_replay_max_errors'].values())
            or not audit['deferred_persistence_verified'] or audit['intervals_above_80ms']
            or mmf and audit.get('raw_mmf_slots_and_original_metadata_verified') is not True):
        raise ValueError('Qualified immutable persistence calibration required')
    camera_config = None
    if exterior:
        config_path = Path(plan['camera_configuration'])
        camera_config = read(config_path)
        audit_root = Path(plan['prerequisite_raw_audit'])
        if (sha256(config_path) != plan['camera_configuration_sha256']
                or configured_mount(camera_config) != 'exterior_front_v3'
                or read(prior/'manifest.json')['config']['sensory_configuration'] != camera_config
                or sha256(audit_root/'result.json') != plan['prerequisite_raw_audit_sha256']
                or read(audit_root/'manifest.json')['config']['recording_manifest_sha256'] != (
                    sha256(prior/'manifest.json'))
                or audit.get('preactivation_baseline_advance_verified') is not True):
            raise ValueError('Exact new-mount calibration and baseline audit required')
        visual_root = Path(plan['prerequisite_visual_replay'])
        visual = read(visual_root/'result.json')
        visual_config = read(visual_root/'manifest.json')['config']
        if (sha256(visual_root/'result.json') != plan['prerequisite_visual_replay_sha256']
                or visual['status'] != 'completed' or visual['weights_changed']
                or visual_config['camera_config_sha256'] != sha256(config_path)
                or visual_config['events_sha256'] != sha256(prior/'events.jsonl')):
            raise ValueError('Completed frozen new-camera visual calibration required')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=plan_hash, scope=__doc__,
                                    program_sha256=sha256(__file__)))
    shutil.copy2(__file__, run/'executed-program.py')
    shutil.copy2(args.plan, run/'collection-plan.json')
    result = dict(status='failed', episodes=[], fit_allowed=False, student_deployed=False,
                  goal_achieved=False, ui_actions_sent=0, optimizer_updates=0)
    counts = dict(train=0, development=0)
    try:
        if recovery:
            source, proof = Path(plan['recovery_collection']), Path(plan['recovery_audit'])
            if (sha256(source/'result.json') != plan['recovery_collection_sha256']
                    or sha256(proof/'result.json') != plan['recovery_audit_sha256']
                    or read(proof/'manifest.json')['config']['collection_result_sha256'] != (
                        sha256(source/'result.json'))):
                raise ValueError('Pinned failed batch and its complete partial audit required')
            previous, audited = read(source/'result.json'), read(proof/'result.json')
            entries = recovery_candidates(previous, audited)
            lookup = {e['path']: e for e in audited['episodes']}
            for entry in entries:
                root, checked = Path(entry['path']), lookup[entry['path']]
                raw = Path(checked['raw_audit'])
                report = read(raw/'result.json')
                if (sha256(root/'result.json') != entry['result_sha256']
                        or checked['original_result_sha256'] != entry['result_sha256']
                        or sha256(raw/'result.json') != checked['raw_audit_result_sha256']
                        or report['intervals_above_80ms'] or any(
                            report['actor_replay_max_errors'].values())
                        or report['camera_aid_state_verified'] is not True
                        or report['preactivation_baseline_advance_verified'] is not True
                        or read(root/'result.json')['camera_wait_misses']
                        or read(root/'manifest.json')['config']['sensory_configuration'] != (
                            camera_config)):
                    raise ValueError('Whole original recovered TRAIN failed raw qualification')
                shutil.copy2(source/(root.name+'-assignment.json'),
                             run/(root.name+'-assignment.json'))
            result['episodes'] = entries
            counts['train'] = 3
            result['recovery_provenance'] = dict(source=str(source), original_status='failed',
                every_complete_train_retained=True, failed_tail_fit=False, roles_changed=False)
            prior = Path(plan['recovery_predecessor'])
            if sha256(prior/'control/api-receipt.json') != plan['recovery_predecessor_sha256']:
                raise ValueError('Actual stopped recovery predecessor changed')
            predecessor = read(prior/'control/api-receipt.json')
            if predecessor['after']['generation'] != generation:
                raise ValueError('Recovery generation differs from qualified calibration')
            idle(generation)
        for index in range(plan['maximum_leases']):
            if counts == plan['required_histories']:
                break
            if sha256(args.plan) != plan_hash:
                raise ValueError('Collection plan changed during execution')
            for source, digest in plan['source_sha256'].items():
                if sha256(source) != digest:
                    raise ValueError('Execution sources changed during collection')
            if shutil.disk_usage(os.environ['LOCALAPPDATA']).free < 40_000_000_000:
                raise ValueError('Insufficient reviewed local raw-storage budget')
            before = idle(generation)
            spline = predecessor['after']['diagnostic_route']['spline_position']
            lease = next_lease(spline, plan['regions'])
            if lease['role'] != 'transit' and counts[lease['role']] >= (
                    plan['required_histories'][lease['role']]):
                # Already-filled regions remain transit; never create extra selectable fits.
                lease.update(role='transit', region=None)
            name = f'episode-{index:02d}-{lease["role"]}'
            output = run/name
            write_json(run/(name+'-assignment.json'), dict(lease=lease, before=before,
                       predecessor_receipt_sha256=sha256(prior/'control/api-receipt.json')))
            command = [sys.executable, '-m', 'scripts.record_ac_engine_ai_probe',
                '--image-format', 'bmp', '--shared-transport', '--batched-shared-copy',
                '--async-pixel-precompute', '--async-frame-persistence',
                '--ai-steer-multiplier', '.9', '--speed-hint-kmh', '60',
                '--duration-seconds', str(lease['duration_seconds']), '--local-event-spool',
                '--output', str(output)]
            if mmf:
                command.append('--mmf-metadata')
            if exterior:
                command.extend(['--config', plan['camera_configuration']])
            with (run/(name+'.log')).open('x', encoding='utf8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(run/'progress.json', dict(episode=name, lease=lease, pid=child.pid,
                           completed_counts=counts, active=True))
                print(json.dumps(dict(episode=name, pid=child.pid, lease=lease)), flush=True)
                code = child.wait()
            recording = read(output/'result.json')
            result['last_child'] = dict(path=str(output), exit_code=code,
                                        status=recording['status'])
            if code or recording['status'] != 'completed' or recording['vision']['error']:
                raise ValueError('Owned finite recording failed; preserve original failure')
            after = idle(generation)
            receipt = read(output/'control/api-receipt.json')
            admission = validate_lease_route(lease, receipt)
            entry = dict(index=index, lease=lease, admission=admission, path=str(output),
                         result_sha256=sha256(output/'result.json'), after=after,
                         decisions=recording['decisions'],
                         camera_wait_misses=recording['camera_wait_misses'])
            if exterior:
                if read(output/'manifest.json')['config']['sensory_configuration'] != camera_config:
                    raise ValueError('Child used a different camera configuration')
                raw_output = run/(name+'-raw-audit')
                raw_command = [sys.executable, '-m', 'scripts.audit_ac_engine_ai_probe',
                    '--recording', str(output), '--require-aid-off', '--output', str(raw_output)]
                with (run/(name+'-raw-audit.log')).open('x', encoding='utf8') as log:
                    raw = subprocess.Popen(raw_command, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                    write_json(run/'progress.json', dict(episode=name, phase='raw_audit',
                               pid=raw.pid, completed_counts=counts, active=True))
                    raw_code = raw.wait()
                proof = read(raw_output/'result.json')
                if (raw_code or proof['status'] != 'completed'
                        or proof['intervals_above_80ms']
                        or any(proof['actor_replay_max_errors'].values())
                        or not proof['preactivation_baseline_advance_verified']
                        or recording['camera_wait_misses']):
                    raise ValueError('Whole new-mount history failed continuous raw qualification')
                entry.update(raw_audit=str(raw_output),
                             raw_audit_result_sha256=sha256(raw_output/'result.json'))
            result['episodes'].append(entry)
            if lease['role'] != 'transit':
                counts[lease['role']] += 1
            write_json(run/'partial-result.json', dict(**result, completed_counts=counts))
            prior, predecessor = output, receipt
        if counts != plan['required_histories']:
            raise ValueError('Finite lease budget ended before declared geographic coverage')
        result.update(status='completed_spatial_teacher_collection', completed_counts=counts,
                      raw_audits_and_targets_required=True)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), completed_counts=counts)
        write_json(run/'result.json', result)
        raise
    finally:
        write_json(run/'progress.json', dict(active=False, completed_counts=counts,
                                            status=result['status']))
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
