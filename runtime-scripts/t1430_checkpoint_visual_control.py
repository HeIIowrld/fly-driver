"""Checkpoint terminal metadata and sources; retain large original payloads locally."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

local = Path('C:/Users/jeff4/AppData/Local/flybrain')
base = local/'runtime-runs'
destination = Path('runs/flight-visual-control-checkpoint-20261029-t1430')
destination.mkdir(exist_ok=False)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


names = ['20261029-t1412-teacher-corpus-v4', '20261029-t1415-v4-prefix-audits',
         '20261029-t1417-v4-visual-body-preparation', '20261029-t1418-v4-failed-camera-timing',
         '20261029-t1419-all-original-shared-timing', '20261029-t1421-async-cache-paired-verification',
         '20261029-t1422-async-cache-moving-preflight', '20261029-t1423-async-cache-moving-raw-audit',
         '20261029-t1425-v4-fixed-roles', '20261029-t1426-v5-train0-raw-audit',
         '20261029-t1427-v5-train0-targets', '20261029-t1429-v5-first-shared-timing']
copied, omitted = [], []
for name in names:
    source = base/name
    assert (source/'result.json').exists()
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.suffix not in ('.json', '.jsonl', '.log', '.md', '.py', '.lua') or any(
                'evaluation' in part for part in relative.parts):
            omitted.append(dict(source=str(path), bytes=path.stat().st_size,
                reason='Large payload or held-out evaluation retained locally; no content inspection here'))
            continue
        target = destination/name/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        shutil.copy2(path, target)
        assert before == digest(target) == digest(path)
        copied.append(dict(source=str(path), archived=str(target), sha256=before, bytes=path.stat().st_size))
snapshot = local/'execution-snapshots/20261029-t1420'
source_hashes = {}
archive = destination/'configured-source-t1420.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(snapshot.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix == '.py':
            name = path.relative_to(snapshot).as_posix()
            zipped.write(path, name)
            source_hashes[name] = digest(path)
    for name in ('configs/flight-aid-free-corpus-v5-20261029.json',
                 'configs/flight-v4-replay-preparation-20261029.json',
                 'tests/test_ac_pixel_precompute.py', 'tests/test_ac_csp_retina.py',
                 'scripts/seal_flight_corpus_roles.py', 'scripts/prepare_flight_corpus_replays.py'):
        if name in source_hashes:
            assert source_hashes[name] == digest(Path(name))
            continue
        zipped.write(name, name)
        source_hashes[name] = digest(Path(name))
with zipfile.ZipFile(archive) as zipped:
    for name, expected in source_hashes.items():
        assert hashlib.sha256(zipped.read(name)).hexdigest() == expected
write(destination/'transfer-receipt.json', dict(scope='Compact terminal metadata/source checkpoint only',
    copied=copied, omitted=omitted, copied_files=len(copied), copied_bytes=sum(r['bytes'] for r in copied),
    local_originals_retained=True, large_payloads_not_fully_archived_here=True,
    configured_source_hashes=source_hashes, source_zip_sha256=digest(archive),
    tests=dict(passed=38, exit_code=0, ruff_passed=True)))
plan_file = Path('configs/fly-flight-mechanism-plan-20261029.json')
plan = read(plan_file)
plan['updated_at_kst'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
data = plan['next_visual_and_pedal_data']
data['latest_visual_learning_status'] = dict(
    report='docs/flight-visual-control-learning-status-20261029.md',
    compact_checkpoint=destination.as_posix(), visual_steering_development_passed=False,
    visual_longitudinal_joint_learning_verified=False, v4_positive_driving_brake_targets=0,
    v4_collection_status='failed_camera_stale_after7completed_leases',
    v4_complete_train_development_replayed=6, v4_replayed_observations=4432,
    v4_maximum_central_audit_error=9.5367431640625e-7,
    v4_maximum_motion_audit_error=5.289912223815918e-7,
    v4_sealed_train_rows=2897, v4_sealed_development_rows=574,
    new_student_policy_applied=False, next_model_priority='Learn/test road and preview representation, connect it to flight commands/body and prove causal steering/speed-control contribution; resolve zero/invalid functional self-motion channels and obtain real braking supervision.')
data['async_pixel_cache_profile'] = dict(enabled_in_v5=True, default_old_synchronous_path_preserved=True,
    workers=2, maximum_pending_images=8, exact_actor_cache_miss_path_preserved=True,
    camera_freshness_guards_changed=False, publication_still_after_persistence=True,
    tests_passed=38, same_shot_groups=8, same_shot_views=4,
    moving_preflight_observations=74, moving_preflight_camera_misses=0,
    first30s_observations=739, first30s_camera_misses=0,
    first30s_peak_pending_images=3, first30s_cache_warms_skipped=0,
    first30s_maximum_publication_age_seconds=0.06517470002290793,
    controlled_matched_performance_improvement_claimed=False)
data['currently_active_owned_model_jobs'] = []
data['currently_active_owned_data_jobs'] = [
    dict(run=str(base/'20261029-t1424-teacher-corpus-v5'), process_pid=53320, tool_session=5965,
         kind='finite_teacher_collection'),
    dict(run=str(base/'20261029-t1428-v5-prefix-audits'), process_pid=51748, tool_session=62380,
         kind='read_only_fixed_role_audits')]
data['next_action'] = ('Poll the same active v5 handles and verify actual terminal state; preserve failed v4 and fixed roles. '
    'Then audit route and real braking, seal complete roles and replay frozen circuits. Prioritize actual visual-to-steering '
    'and visual/goal/body-to-longitudinal learning, not more collection as a substitute. Functional self-motion cache remains '
    'zero/invalid. Accel/brake, load/tyre/setup, student circuit/racing line/laptime all remain outstanding.')
write(plan_file, plan)
print(json.dumps(dict(copied_files=len(copied), copied_bytes=sum(r['bytes'] for r in copied),
                      payloads_or_evaluation_files_retained_locally=len(omitted))))
