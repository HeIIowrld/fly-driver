"""Hash-copy original diagnostic outputs and configured execution source; keep originals."""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

local = Path('C:/Users/jeff4/AppData/Local/flybrain')
destination = Path('runs/flight-nonlinear-information-diagnosis-20261029-t1388')
destination.mkdir(exist_ok=True)
kst = timezone(timedelta(hours=9))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


rows = []
for name in ('20261029-t1386-nonlinear-information-diagnosis',
             '20261029-t1387-nonlinear-independent-audit'):
    origin = local/'runtime-runs'/name
    for path in sorted(origin.rglob('*')):
        if not path.is_file():
            continue
        target = destination/name/path.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        if not target.exists():
            shutil.copy2(path, target)
        assert before == digest(target) == digest(path)
        rows.append(dict(source=str(path), archived=str(target), sha256=before,
                         bytes=path.stat().st_size))
snapshot = local/'execution-snapshots/20261029-t1385'
archive = destination/'configured-execution-source-t1385.zip'
sources = {}
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for path in sorted(snapshot.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json'):
            relative = path.relative_to(snapshot).as_posix()
            sources[relative] = digest(path)
            zipped.write(path, relative)
with zipfile.ZipFile(archive) as zipped:
    for name, expected in sources.items():
        assert hashlib.sha256(zipped.read(name)).hexdigest() == expected
(destination/'transfer-receipt.json').write_text(json.dumps(dict(
    created_at_kst=datetime.now(kst).isoformat(),
    files=rows, files_count=len(rows), bytes=sum(r['bytes'] for r in rows),
    snapshot_archive_sha256=digest(archive), snapshot_files=sources,
    snapshot_is_configured_source_not_retrospective_memory_introspection=True,
    local_originals_retained=True), indent=2), encoding='utf8')
plan_path = Path('configs/fly-flight-mechanism-plan-20261029.json')
plan = json.loads(plan_path.read_text(encoding='utf8'))
plan['updated_at_kst'] = datetime.now(kst).isoformat()
plan['next_visual_and_pedal_data']['next_action'] = (
    'Nonlinear comparator also failed development baseline. Separate road spatial cues, teacher '
    'timing and left/right coverage. Validate stationary/photometric stability and visual/body '
    'fusion. Real braking, fast-body integration, tyre/setup and circuit/laptime remain outstanding.')
plan['next_visual_and_pedal_data']['nonlinear_information_diagnosis'] = dict(
    status='completed_exploratory_diagnosis_failed_development_mean_baseline',
    archive=destination.as_posix(), report='docs/flight-nonlinear-information-diagnosis-20261029.md',
    gpu='AMD Radeon RX 9070 XT', train_rows=3497, diagnostic_optimizer_updates=400,
    fly_circuit_optimizer_updates=0, nonlinear_train_rmse=0.027637724029614893,
    nonlinear_development_rmse=[0.09407304972410202, 0.15330739319324493],
    development_train_mean_baseline_rmse=[0.05513088032603264, 0.09666474908590317],
    maximum_independent_prediction_error=6.75704266206445e-7,
    evaluation_role_arrays_opened=False, fly_checkpoint_modified=False,
    actor_deployed=False, ui_actions_sent=0, actuator_messages_sent=0,
    foreign_llama_server_modified=False,
    next='Separate road spatial cues, teacher timing and left/right coverage; verify stationary/photometric reference and visual/body fusion. Braking, tyre/setup, fast-body integration and valid circuit/laptime still outstanding.')
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows),
                      frozen_source_files=len(sources), archive_sha256=digest(archive))))
