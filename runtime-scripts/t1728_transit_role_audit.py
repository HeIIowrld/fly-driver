"""Read-only CPU audit of completed t1720 transit metadata, never live interfaces."""

import base64
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path.cwd()
BASE = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
SOURCE = BASE / '20261030-t1720-old60-transit-to-train'
BASELINE = BASE / '20261030-t1725-pre-train-paused-baseline/result.json'
OUTPUT = BASE / '20261030-t1728-transit-role-audit'
GENERATION = '37294426ULL:0.0014905'
PID = 45604
FAILURES = []
CHECKS = []
REFERENCES = {}


def check(condition, description):
    CHECKS.append(description)
    if not condition:
        FAILURES.append(description)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    REFERENCES[str(path)] = digest(path)
    return json.loads(path.read_text(encoding='utf8'))


def near(a, b):
    return abs(a - b) <= 1e-10


def idle(state, name):
    session, native = state['session'], state['native']
    check(state['engine_pid'] == PID, name + ': engine PID')
    check(session['generation'] == GENERATION, name + ': generation')
    check(session['paused'] and not session['ai_controlled']
          and not session['is_online'] and not session['in_menu']
          and not session['camera_active'] and not session['expert_probe_active']
          and not session.get('brake_settle_active', False), name + ': inactive paused lifecycle')
    check(session['gas'] == 0 and session['brake'] == 1
          and native['gas'] == 0 and native['brake'] == 1
          and native['speedKmh'] < .1, name + ': native full-brake baseline')
    check(state['student_owner_qualification'] is False, name + ': no student admission')


def native_prefix(row, name):
    prefix = base64.b64decode(row['prefix_base64'], validate=True)
    check(len(prefix) == 308, name + ': original physics prefix length308')
    check(hashlib.sha256(prefix).hexdigest() == row['prefix_sha256'], name + ': prefix SHA')
    check(struct.unpack_from('<i', prefix)[0] == row['packet_id'], name + ': packet')
    for field, offset in [('gas', 4), ('brake', 8), ('steerAngle', 24), ('speedKmh', 28)]:
        check(struct.unpack_from('<f', prefix, offset)[0] == row['fields'][field],
              name + ': same-packet ' + field)
    check(row['source_time_seconds'] is None and row['simulation_clock_verified'] is False,
          name + ': host receipt is not an exact simulation clock')


def main():
    OUTPUT.mkdir(exist_ok=False)
    plan = read(SOURCE / 'transit-plan.json')
    manifest = read(SOURCE / 'manifest.json')
    result = read(SOURCE / 'result.json')
    program = SOURCE / 'executed-program.py'
    plan_sha = digest(SOURCE / 'transit-plan.json')
    REFERENCES[str(program)] = digest(program)
    check(manifest['config']['plan_sha256'] == plan_sha, 'sealed plan SHA')
    check(manifest['config']['plan'] == plan, 'manifest contains exact sealed plan')
    check(manifest['config']['program_sha256'] == digest(program)
          == plan['input_sha256']['scripts/transit_to_pace_region.py'], 'executed wrapper SHA')
    check(plan['role'] == result['role'] == 'transit'
          and plan['fit_allowed'] is result['fit_allowed'] is False
          and plan['role_reassignment_allowed'] is result['role_reassignment_allowed'] is False,
          'whole-history transit role and no fit/reassignment')
    check(result['status'] == 'completed_at_eligible_paused_start', 'parent terminal completion')
    check(result['actor_world_input'] is result['student_deployed'] is False
          and result['ui_actions_sent'] == result['optimizer_updates']
          == result['original_payloads_copied'] == 0
          and result['wrapper_timeouts_or_pid_kills'] is False, 'parent no fit/deployment/UI/copies/kills')
    check(plan['engine_pid'] == PID and plan['generation'] == GENERATION
          and plan['maximum_stages'] == 12 and plan['child_duration_seconds'] == [3, 30]
          and plan['teacher_speed_kmh'] == 60 and plan['ai_steer_multiplier'] == .9
          and plan['eligible_start_interval'] == [.275, .44]
          and plan['preferred_start_interval'] == [.285, .425], 'reviewed finite old60 contract')

    snapshot_rows = []
    current_differences = []
    for index, (name, sha) in enumerate(plan['input_sha256'].items()):
        current = Path(name).resolve()
        relative = (current.relative_to(ROOT.resolve()) if current.is_relative_to(ROOT.resolve())
                    else Path('external-metadata') / f'{index:03d}-{current.name}')
        snapshot = SOURCE / 'executed-inputs' / relative
        check(snapshot.suffix in ('.py', '.json', '.lua') and digest(snapshot) == sha,
              'pinned preactivation source snapshot: ' + name)
        REFERENCES[str(snapshot)] = digest(snapshot)
        if digest(current) != sha:
            current_differences.append(name)
        snapshot_rows.append(dict(source=name, sha256=sha, snapshot=str(snapshot)))
    check(not current_differences, 'all declared current inputs remain source-identical')

    pipeline_names = [name for name in plan['input_sha256']
                      if not Path(name).is_absolute() and Path(name).suffix == '.py']
    pilot = read(BASE / '20261030-t1715-pace30-diagnostic-pilot-v2/recording/manifest.json')
    pipeline_changes_from_pilot = [name for name in pipeline_names
                                  if name in pilot['source_sha256']
                                  and pilot['source_sha256'][name] != plan['input_sha256'][name]]
    check(not pipeline_changes_from_pilot, 'declared shared Python pipeline unchanged from t1715')
    episode_rows = []
    previous = result['initial_position']['spline_position']
    wraps = 0
    all_splines = []
    for i, entry in enumerate(result['episodes']):
        name = f'transit-{i:02d}'
        child = SOURCE / name
        assignment_path = SOURCE / (name + '-assignment.json')
        assignment = read(assignment_path)
        cm = read(child / 'manifest.json')
        recording = read(child / 'result.json')
        control = read(child / 'control/result.json')
        receipt = read(child / 'control/api-receipt.json')
        before = read(child / 'control/native-before.json')
        after = read(child / 'control/native-after.json')
        check(entry['role'] == assignment['role'] == assignment['decision']['role'] == 'transit'
              and entry['fit_allowed'] is assignment['fit_allowed']
              is assignment['decision']['fit_allowed'] is False, name + ': preassigned transit-only')
        check(assignment['plan_sha256'] == plan_sha and entry['child_exit_code'] == 0,
              name + ': plan identity and terminal0')
        # Exact executed source seals the assignment before Popen; file chronology is additional evidence.
        created = dt.datetime.fromisoformat(cm['created_utc']).timestamp()
        check(assignment_path.stat().st_mtime <= created, name + ': assignment precedes child manifest')
        args = cm['config']['arguments']
        check(args['duration_seconds'] == entry['duration_seconds'] == 30
              and args['speed_hint_kmh'] == 60 and args['ai_steer_multiplier'] == .9
              and args['image_format'] == 'bmp'
              and all(args[key] is True for key in ['shared_transport', 'batched_shared_copy',
                                                   'async_pixel_precompute', 'async_frame_persistence',
                                                   'mmf_metadata'])
              and all(args[key] is False for key in ['recover_to15', 'brake_pulse', 'pace_task',
                                                    'lateral_recovery', 'local_event_spool'])
              and args['flight_shadow_config'] is args['flight_shadow_verification'] is None,
              name + ': original recorder transport and old60 profile')
        command = assignment['command']
        expected_command = [command[0], '-m', 'scripts.record_ac_engine_ai_probe', '--config',
                            plan['camera_configuration'], '--image-format', 'bmp', '--shared-transport',
                            '--batched-shared-copy', '--async-pixel-precompute', '--async-frame-persistence',
                            '--mmf-metadata', '--duration-seconds', '30', '--speed-hint-kmh', '60',
                            '--ai-steer-multiplier', '.9', '--output', str(child)]
        check(command == expected_command, name + ': exact presealed old60 argv')
        source_checks = [key for key in pipeline_names if key in cm['source_sha256']]
        check(all(cm['source_sha256'][key] == plan['input_sha256'][key] for key in source_checks),
              name + ': all declared Python manifest hashes agree')
        check(cm['config']['sensory_configuration'] == read(ROOT / plan['camera_configuration']),
              name + ': exact sensory configuration')
        for filename, field in [('result.json', 'result_sha256'),
                                ('control/result.json', 'control_result_sha256'),
                                ('control/api-receipt.json', 'receipt_sha256')]:
            check(digest(child / filename) == entry[field], name + ': parent ledger ' + filename)
        check(recording['status'] == control['status'] == 'completed'
              and recording['training_admissible'] is recording['expert_labels_available'] is False
              and recording['actor_input_pose_free'] is True and not recording['vision']['error']
              and recording['decisions'] == entry['decisions']
              and recording['camera_wait_misses'] == entry['camera_wait_misses'],
              name + ': preserved complete diagnostic counts/no labels')
        check(control['target_state_observed'] is True and control['lifecycle_requests_sent']
              == control['lifecycle_publication_attempts'] == 1
              and control['foreground_before'] is control['foreground_after'] is False
              and control['ui_actions_sent'] == control['actuator_messages_sent'] == 0,
              name + ': single background lifecycle')
        check(receipt['status'] == 'api_returned_true' and receipt['command'] == 'expert-collect30-speed60'
              and receipt['motion_hard_speed_kmh'] == 80
              and receipt['maximum_displacement_guard_m'] == 720
              and receipt['maximum_speed_kmh'] < 80
              and receipt['maximum_position_delta_m'] < 720
              and receipt['requested_ai_speed_hint_kmh'] == 60
              and receipt['ai_steer_multiplier'] == .9 and not receipt['explicit_pace_task'],
              name + ': unchanged old60 physical guards')
        calls = receipt['api_calls']
        check([call['name'] for call in calls] == ['enable_ai', 'set_ai_steer_multiplier',
              'set_ai_speed_limit', 'resume_owned_ai', 'disable_ai', 'pause_after_brake_hold',
              'restore_default_ai_speed_limit', 'restore_default_ai_steer_multiplier']
              and all(call['returned'] for call in calls)
              and all(a['script_clock'] <= b['script_clock'] for a, b in zip(calls, calls[1:])),
              name + ': existing API lifecycle only')
        idle(assignment['before'], name + ' assignment before')
        idle(entry['after'], name + ' parent after')
        native_prefix(before, name + ' native before')
        native_prefix(after, name + ' native after')
        check(after['fields']['gas'] == 0 and after['fields']['brake'] == 1
              and after['fields']['speedKmh'] < .1, name + ': recorded native cleanup')
        start = entry['before_position']['spline_position']
        end = entry['after_position']['spline_position']
        check(near(start, previous) and near(start, assignment['diagnostic_position']['spline_position'])
              and near(start, receipt['before']['diagnostic_route']['spline_position'])
              and near(end, receipt['after']['diagnostic_route']['spline_position']),
              name + ': whole before-after chain')
        if start - end > .5:
            wraps += 1
        check(not .275 <= start <= .44, name + ': no activation at eligible start')
        states = [receipt['before']] + receipt['telemetry'] + [receipt['after']]
        routes = [state['diagnostic_route'] for state in states]
        splines = [route['spline_position'] for route in routes]
        all_splines.extend(splines)
        check(all(math.isfinite(x) and 0 <= x < 1 for x in splines)
              and all(route['actor_input'] is False for route in routes),
              name + ': every before/driving/stopping/after route stays diagnostic')
        phases = {phase: sum(state.get('phase') == phase for state in receipt['telemetry'])
                  for phase in ['driving', 'stopping']}
        check(sum(phases.values()) == len(receipt['telemetry']) and all(phases.values()),
              name + ': driving and cleanup samples both retained')
        check(all(receipt[key]['generation'] == GENERATION and receipt[key]['paused']
                  and receipt[key]['ai_controlled'] is False
                  and receipt[key]['gas'] == 0 and receipt[key]['brake'] == 1
                  for key in ['before', 'after']), name + ': same-generation receipt endpoints')
        episode_rows.append(dict(index=i, role='transit', fit_allowed=False, child_pid=entry['child_pid'],
                                 child_exit_code=entry['child_exit_code'], before_spline=start,
                                 after_spline=end, decisions=recording['decisions'],
                                 camera_wait_misses=recording['camera_wait_misses'],
                                 telemetry_phases=phases, maximum_wheels_outside=max(
                                     state['wheels_outside'] for state in states),
                                 maximum_speed_kmh=receipt['maximum_speed_kmh'],
                                 maximum_position_delta_m=receipt['maximum_position_delta_m'],
                                 renderer_origin_before=receipt['before']['renderer_origin_shift_diagnostic'],
                                 renderer_origin_after=receipt['after']['renderer_origin_shift_diagnostic'],
                                 observed_lap_count_values=sorted(set(route['lap_count'] for route in routes)),
                                 declared_python_manifest_hashes_checked=len(source_checks)))
        previous = end

    live = read(SOURCE / 'live-child.json')
    final = read(BASELINE)
    check(len(episode_rows) == 4 and [row['decisions'] for row in episode_rows] == [733, 737, 735, 736]
          and [row['camera_wait_misses'] for row in episode_rows] == [1, 0, 1, 1], 'all four whole histories retained')
    check(wraps == result['wraps'] == 1, 'one diagnostic spline wrap; no extra lap attempt')
    check(live['active'] is False and live['terminal_exit_code'] == 0
          and live['pid'] == result['episodes'][-1]['child_pid'], 'last owned child metadata terminal')
    check(near(previous, result['last_position']['spline_position'])
          and .285 <= previous <= .425 and result['preferred_start']
          and result['no_new_activation_at_eligible_position'], 'preferred eligible stop without extra activation')
    check(final['status'] == 'completed_read_only_paused_collection_baseline'
          and final['engine_pid'] == PID and final['generation'] == GENERATION
          and near(final['diagnostic_route']['spline_position'], previous)
          and final['native_gas'] == 0 and final['native_brake'] == 1
          and final['native_speed_kmh'] < .1 and final['vjoy_status'] == 1
          and final['foreground'] is final['controller_admission'] is False
          and final['new_physics_advance_verified'] is False,
          't1725 subsequent paused planning baseline agrees, no motion/student qualification')
    protected = {'known_dev_020_025': [.2, .25], 'known_dev_094_100': [.94, 1],
                 'new_dev_067_084': [.67, .84], 'eval_008_0125': [.08, .125]}
    summary = dict(status='passed' if not FAILURES else 'failed', audit='t1728-transit-role-audit',
                   created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source=str(SOURCE),
                   source_plan_sha256=plan_sha, source_program_sha256=digest(program),
                   failures=FAILURES, checks_passed=len(CHECKS) - len(FAILURES), checks_total=len(CHECKS),
                   role='transit', fit_allowed=False, role_reassignment_allowed=False,
                   source_snapshots_checked=len(snapshot_rows), current_source_differences=current_differences,
                   shared_pipeline_changes_from_t1715=pipeline_changes_from_pilot,
                   episodes=episode_rows, observations=sum(row['decisions'] for row in episode_rows),
                   camera_wait_misses=sum(row['camera_wait_misses'] for row in episode_rows),
                   initial_spline=result['initial_position']['spline_position'], final_spline=previous,
                   diagnostic_spline_wraps=wraps, completed_stages=len(episode_rows), maximum_stages=12,
                   sampled_protected_region_visits={key: sum(lo <= x <= hi for x in all_splines)
                                                    for key, (lo, hi) in protected.items()},
                   protected_visit_meaning='Transit CSP route metadata only; never old DEV/EVAL arrays or fit data.',
                   clock_scope='Native endpoints retain original host receipt clocks, source_time null; not exact simulation time.',
                   prefix_bytes=308, valid_laps_claimed=0, optimizer_updates=0, pixel_replay_performed=False,
                   new_train_candidate_opened=False, source_payloads_copied=0, live_interfaces_opened=0,
                   controls_sent=0, gpu_used=False,
                   limitations=['Metadata/source audit, no transit pixel replay or teacher-label qualification.',
                                'Spline wrap is progress metadata; observed lap_count remains0 and lap_valid false.',
                                'Three camera wait misses and final renderer origin shift remain in originals.',
                                'Paused baseline is planning evidence; fresh lifecycle guards still required for a new collection.',
                                'Any new TRAIN history requires a separate presealed role and whole raw/authority qualification.'])
    write = lambda path, value: path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf8')
    write(OUTPUT / 'result.json', summary)
    write(OUTPUT / 'references.json', REFERENCES)
    (OUTPUT / 'executed-audit.py').write_bytes(Path(__file__).read_bytes())
    table = '\n'.join(f"| {row['index']} | {row['child_pid']} | {row['before_spline']:.9f} | {row['after_spline']:.9f} | {row['decisions']} | {row['camera_wait_misses']} |" for row in episode_rows)
    report = f'''# t1728 transit role/source audit

UTC: {summary['created_utc']}. CPU/read-only metadata audit: **{summary['status']}**, {summary['checks_passed']}/{summary['checks_total']} checks.

The four t1720 whole histories stay **transit / fit_allowed=false / no role reassignment**. All before, driving, stopping, cleanup and after states retain that assignment. The sealed parent executed four of at most twelve stages and stopped at spline **{previous:.12f}**, inside preferred [.285,.425], without another activation. There was one diagnostic spline wrap; this is not a valid-lap claim (receipt lap_count0, lap_valid=false).

| Stage | Child PID (terminal0) | Before spline | After spline | Observations | Wait misses |
|---|---:|---:|---:|---:|---:|
{table}

All **2,941 observations and 3 wait misses** remain. Every child used the original recorder/transport and `expert-collect30-speed60`, speed60/steer.9, hard80km/h/720m. No pace task or flight shadow was enabled. All API sequences are one enable/resume, disable/brake-hold/pause, restore pair; all calls returned true. Route metadata is privileged diagnostic input only, including any protected-region crossings. No old DEV/EVAL arrays were opened.

The plan SHA is `{plan_sha}`; executed wrapper SHA `{digest(program)}`. All {len(snapshot_rows)} declared preactivation source/JSON/Lua snapshots and current inputs match; shared declared Python pipeline hashes also match t1715. The four child manifests and parent result/control/receipt hash ledgers agree. Per-file hashes are in `references.json`; no source payload was copied by this audit.

Before/after native prefixes are **308 bytes** and their packet/hash/gas4/brake8/steer24/speed28 decode exactly. Their clocks remain host receipt evidence with null source time, not exact simulation clocks. All recorded parent after states are paused, AI/camera/probe off, native gas0/brake1/speed<.1, same engine45604 and generation `{GENERATION}`. t1725 independently records that final paused location, vJoyFREE/background/fullbrake, while explicitly declining fresh-motion/student admission.

The final receipt preserves renderer-origin diagnostic change `[0,0,0]` → `[0,0,1024]`. Maximum recorded wheels_outside is0 throughout the receipt states; this metadata audit does not replay pixels, certify every native teacher label, or promote any transit data to training. A subsequent TRAIN candidate needs a separate sealed assignment and whole raw/clock/authority audit. t1726/t1727 were not opened.

Audit controls0 / live mappings0 / GPU0 / optimizer0 / payload copies0 / valid laps claimed0. Primary shared docs and collection sources were not changed.
'''
    (OUTPUT / 'REPORT.md').write_text(report, encoding='utf8')
    write(OUTPUT / 'receipt.json', {name: digest(OUTPUT / name) for name in
                                   ['result.json', 'references.json', 'executed-audit.py', 'REPORT.md']})
    print(json.dumps({key: summary[key] for key in ['status', 'checks_passed', 'checks_total', 'failures',
                                                    'observations', 'camera_wait_misses', 'final_spline']}))
    if FAILURES:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
