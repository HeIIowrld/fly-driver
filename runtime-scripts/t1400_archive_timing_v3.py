"""Retain successful and failed evidence; hash-copy terminal runs and stage sources."""
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

local = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
failed_watcher = local/'20261029-t1398-v3-prefix-audits'
(failed_watcher/'startup-failure.json').write_text(json.dumps(dict(
    status='failed_before_watch_loop', exit_code=1,
    reason='Confirmed live collection process handle required; original parent was already terminal',
    collection_restarted=False, recovery_run='20261029-t1399-v3-terminal-prefix-audits'), indent=2), encoding='utf8')
destination = Path('runs/flight-timing-coverage-and-v3-collection-20261029-t1400')
destination.mkdir(exist_ok=False)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


names = [
    '20261029-t1389-timing-coverage-diagnosis',
    '20261029-t1391-v3-moving-preflight',
    '20261029-t1392-v3-moving-preflight-raw-audit',
    '20261029-t1393-v2-route-coverage-audit',
    '20261029-t1394-teacher-corpus-v3',
    '20261029-t1396-v3-train0-raw-audit',
    '20261029-t1397-v3-train0-targets',
    '20261029-t1398-v3-prefix-audits',
    '20261029-t1399-v3-terminal-prefix-audits',
]
rows = []
for name in names:
    source = local/name
    assert source.is_dir()
    for path in sorted(source.rglob('*')):
        if path.is_file():
            target = destination/name/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(path)
            shutil.copy2(path, target)
            assert before == digest(target) == digest(path)
            rows.append(dict(source=str(path), archived=str(target), sha256=before,
                             bytes=path.stat().st_size))
stage = destination/'staged-source-not-game-deployed'
for file in ('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua',
             'tests/test_ac_lua_session_receipts.py',
             'scripts/collect_flight_teacher_corpus.py', 'scripts/audit_flight_teacher_corpus.py',
             'scripts/diagnose_flight_timing_coverage.py', 'scripts/audit_flight_route_coverage.py',
             'configs/flight-aid-free-corpus-v3-20261029.json',
             'configs/flight-v2-timing-coverage-diagnosis-20261029.json'):
    target = stage/file
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, target)
    assert digest(Path(file)) == digest(target)
write = lambda path, value: Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
write(destination/'transfer-receipt.json', dict(files=rows, files_count=len(rows),
    bytes=sum(r['bytes'] for r in rows), local_originals_retained=True,
    failed_collection_and_failed_watcher_retained=True,
    staged_lua_not_deployed=True, tests=dict(command='pytest tests/test_ac_lua_session_receipts.py tests/test_ac_session_control.py -q -p no:cacheprovider', passed=169, exit_code=0)))
plan_file = Path('configs/fly-flight-mechanism-plan-20261029.json')
plan = json.loads(plan_file.read_text(encoding='utf8'))
plan['updated_at_kst'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
data = plan['next_visual_and_pedal_data']
data['currently_active_owned_model_jobs'] = []
data['next_action'] = ('Validate narrowly scoped brake-settle recovery for terminal native residual0.108kmh, '
    'then deploy/test stricter CSP pause<0.01 with unchanged native<0.1. Keep failed v3 and fixed roles; '
    'new missing-route collection only after teardown qualification. Continue fly visual/body fusion, '
    'real braking, load/tyre/setup and student circuit/laptime work.')
data['timing_route_and_v3'] = dict(
    report='docs/flight-timing-coverage-and-v3-collection-20261029.md', archive=destination.as_posix(),
    original_train_development_observed_bins=37, total_route_bins=100,
    original_labels_retimed=False, original_evaluation_opened=False,
    new_collection_status='failed_after_two_completed_train_leases', planned_leases=24,
    completed_train_leases=2, new_completed_observations=1484, valid_proposal_rows=1144,
    failed_development_index=2, failed_development_observations=739, unexecuted_leases=21,
    completed_child_exit_codes=[0, 0], failed_child_and_parent_exit_codes=[1, 1],
    first_two_complete_train_route_spline=[0.26344954967499, 0.35185897350311],
    native_packet_after_failure=222775, native_speed_kmh=0.10807372629642487,
    csp_speed_kmh=0.04587983712554, full_brake=True, paused=True,
    native_stationary_admission_limit_kmh_unchanged=0.1,
    staged_csp_pause_speed_limit_kmh=0.01, staged_lua_deployed=False,
    lua_and_control_tests_passed=169, new_student_training_or_deployment=False,
    teacher_full_lap_verified=False, student_full_lap_verified=False)
write(plan_file, plan)
print(json.dumps(dict(directories=len(names), files=len(rows), bytes=sum(r['bytes'] for r in rows))))
