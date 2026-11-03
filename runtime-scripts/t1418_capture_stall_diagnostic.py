"""Separate original render scheduling, host acknowledgement and persistence delays."""
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
source = base/'20261029-t1412-teacher-corpus-v4/episode-07-train'
run = begin_run(base/'20261029-t1418-v4-failed-camera-timing', dict(scope=__doc__,
    program_sha256=sha256(__file__), events_sha256=sha256(source/'events.jsonl'),
    camera_bundle_sha256=sha256(source/'vision/camera.zip')))
(run/'executed-program.py').write_bytes(Path(__file__).read_bytes())


def stats(values):
    x = np.asarray(values, float)
    return dict(count=len(x), median=float(np.median(x)), p95=float(np.quantile(x, .95)),
                maximum=float(x.max()), above150ms=int((x > .15).sum()))


rows = []
with zipfile.ZipFile(source/'vision/camera.zip') as archive:
    clock = json.loads(archive.read('clock-final.json'))
    for name in sorted(n for n in archive.namelist() if n.startswith('frame-') and n.endswith('.json')):
        frame = json.loads(archive.read(name))
        index = frame['index']
        ack = json.loads(archive.read(f'ack-{index:04d}.json'))
        assert frame['generation'] == ack['generation'] == clock['generation']
        assert ack['capture_id'] == index
        first = min(v['render_started_clock'] for v in frame['views'])
        finish = max(v['render_call_finished_clock'] for v in frame['views'])
        rows.append(dict(index=index, render_start=first, render_call_finished=finish,
            ack_clock=ack['copied_before_ack'], render_call_span=finish-first,
            ack_age_lower=ack['copied_before_ack']-first-clock['offset_upper_seconds'],
            ack_age_upper=ack['copied_before_ack']-first-clock['offset_lower_seconds']))
assert [r['index'] for r in rows] == list(range(len(rows)))
source_gaps = np.diff([r['render_start'] for r in rows])
ack_to_next_start_lower = np.array([n['render_start']+clock['offset_lower_seconds']-p['ack_clock']
                                   for p, n in zip(rows[:-1], rows[1:], strict=True)])
for i, row in enumerate(rows):
    row['preceding_render_gap'] = float(source_gaps[i-1]) if i else None
    row['previous_ack_to_this_render_lower'] = float(ack_to_next_start_lower[i-1]) if i else None
events = [json.loads(line) for line in (source/'events.jsonl').read_text(encoding='utf8').splitlines()]
copy, between, persist = [], [], []
for event in events:
    receipts = event['camera']['shared_copy_receipts']
    copy.extend(r['finished_at']-r['started_at'] for r in receipts)
    between.extend(n['started_at']-p['finished_at'] for p, n in zip(receipts[:-1], receipts[1:], strict=True))
    pipeline = event['camera']['shared_pipeline']
    persist.append(pipeline['persistence_finished_clock']-pipeline['persistence_started_clock'])
result = dict(status='completed_original_camera_timing_diagnosis', original_frames=len(rows),
    original_actor_observations=len(events), original_camera_gaps=json.loads((source/'camera-gaps.json').read_text(encoding='utf8')),
    render_start_gap=stats(source_gaps), render_call_span=stats([r['render_call_span'] for r in rows]),
    copy_ack_age_upper=stats([r['ack_age_upper'] for r in rows]),
    previous_ack_to_next_render_start_lower=stats(ack_to_next_start_lower),
    accepted_actor_gpu_copy_seconds=stats(copy), accepted_actor_between_copy_seconds=stats(between),
    accepted_actor_persistence_seconds=stats(persist),
    highest_ack_age_frames=sorted(rows, key=lambda r:r['ack_age_upper'], reverse=True)[:8],
    highest_render_gap_frames=sorted(rows[1:], key=lambda r:r['preceding_render_gap'], reverse=True)[:8],
    model_preparation_created_utc=json.loads((base/'20261029-t1417-v4-visual-body-preparation/manifest.json').read_text(encoding='utf8'))['created_utc'],
    original_failed_recording_created_utc=json.loads((source/'manifest.json').read_text(encoding='utf8'))['created_utc'],
    no_guards_or_original_clocks_changed=True, no_retrospective_resource_cause_claim=True,
    optimizer_updates=0, ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False)
write_json(run/'result.json', result)
print(json.dumps({k:result[k] for k in ('render_start_gap','render_call_span','copy_ack_age_upper',
    'previous_ack_to_next_render_start_lower','accepted_actor_gpu_copy_seconds',
    'accepted_actor_between_copy_seconds','accepted_actor_persistence_seconds')}))
print(json.dumps(result['highest_render_gap_frames'][:2]))
