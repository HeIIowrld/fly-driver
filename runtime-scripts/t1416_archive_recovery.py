"""Archive terminal recovery evidence, retain failures and journal the live data handles."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

local = Path('C:/Users/jeff4/AppData/Local/flybrain')
base = local/'runtime-runs'
destination = Path('runs/ac-strict-stop-recovery-20261029-t1416')
destination.mkdir(exist_ok=False)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


names = [
    '20261029-t1401-brake-settle-deployment', '20261029-t1402-brake-settle-recovery',
    '20261029-t1403-owned-engine-reset', '20261029-t1404-start-reset-engine',
    '20261029-t1405-pause-reset-engine', '20261029-t1407-strict-stop-moving-preflight',
    '20261029-t1408-reset-engine-brake-initialization', '20261029-t1409-pause-initialized-engine',
    '20261029-t1410-initialized-strict-stop-preflight', '20261029-t1411-strict-stop-preflight-raw-audit',
    '20261029-t1413-v4-train0-raw-audit', '20261029-t1414-v4-train0-targets',
]
rows = []
for name in names:
    original = base/name
    assert (original/'result.json').exists()
    for path in sorted(original.rglob('*')):
        if path.is_file():
            target = destination/name/path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(path)
            shutil.copy2(path, target)
            assert before == digest(target) == digest(path)
            rows.append(dict(source=str(path), destination=str(target), bytes=path.stat().st_size,
                             sha256=before))
archive = destination/'configured-execution-source-t1406.zip'
source_hashes = {}
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    snapshot = local/'execution-snapshots/20261029-t1406'
    for path in sorted(snapshot.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json'):
            name = path.relative_to(snapshot).as_posix()
            source_hashes[name] = digest(path)
            zipped.write(path, name)
    for name in ('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua',
                 'tests/test_ac_lua_session_receipts.py', 'configs/flight-aid-free-corpus-v4-20261029.json'):
        source_hashes[name] = digest(Path(name))
        zipped.write(name, name)
with zipfile.ZipFile(archive) as zipped:
    for name, expected in source_hashes.items():
        assert hashlib.sha256(zipped.read(name)).hexdigest() == expected
write(destination/'pre-api-launch-errors.json', dict(
    t1401_initial_invocation='ModuleNotFoundError: scripts, before begin_run; PYTHONPATH corrected',
    t1404_initial_invocation='argparse positional start rejected, before begin_run; --command start corrected',
    lifecycle_requests_in_these_initial_failures=0))
write(destination/'transfer-receipt.json', dict(files=rows, file_count=len(rows),
    bytes=sum(r['bytes'] for r in rows), all_local_originals_retained=True,
    configured_source_hashes=source_hashes, source_zip_sha256=digest(archive),
    configured_source_not_retrospective_process_memory=True,
    current_collection_not_copied_while_mutating=True,
    tests=dict(passed=193, exit_code=0, command='pytest tests/test_ac_lua_session_receipts.py tests/test_ac_session_control.py -q -p no:cacheprovider')))
plan_file = Path('configs/fly-flight-mechanism-plan-20261029.json')
plan = read(plan_file)
plan['updated_at_kst'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
data = plan['next_visual_and_pedal_data']
data['strict_stop_recovery'] = dict(archive=destination.as_posix(),
    report='docs/ac-strict-stop-recovery-20261029.md',
    original_v3_failure_preserved=True, bounded_residual_recovery_passed=False,
    bounded_recovery_stop_seconds=0.022806600000003, bounded_recovery_maximum_displacement_m=0.00073297349772631,
    old_owned_engine_pid=28428, original_process_exit_verified=True,
    new_engine_pid=36644, engine_generation='4525688ULL:0.0017325',
    normal_native_stationary_limit_kmh_unchanged=.1, csp_cleanup_pause_limit_kmh=.01,
    actual_strict_stop_preflight_native_speed_kmh=.009756526909768581,
    actual_first_30s_native_stopped_speed_kmh=.009369644336402416,
    first_30s_observations=739, first_30s_camera_wait_misses=0,
    preflight_and_first30s_raw_audits_passed=True, startup_brake_vjoy_requests=7,
    ui_actions_sent=0, new_student_policy_applied=False, tests_passed=193)
data['currently_active_owned_data_jobs'] = [
    dict(kind='finite_teacher_collection', run=str(base/'20261029-t1412-teacher-corpus-v4'),
         process_pid=46660, tool_session=50310),
    dict(kind='read_only_prefix_audits', run=str(base/'20261029-t1415-v4-prefix-audits'),
         process_pid=23292, tool_session=99226)]
data['currently_active_owned_model_jobs'] = []
data['next_action'] = ('Poll the same owned v4 collection and prefix-audit handles; do not restart on observation timeout. '
    'Preserve all failures and fixed roles. After terminal, archive/hash verify, audit actual route and driving braking, '
    'seal complete roles, replay unchanged author vision/body, and assess fusion/steering. '
    'Student braking, load/tyre/setup, fast-body integration and circuit/laptime remain unqualified.')
write(plan_file, plan)
print(json.dumps(dict(directories=len(names), files=len(rows), bytes=sum(r['bytes'] for r in rows))))
