"""Record uncertain side-view road-edge observations on original TRAIN images."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

# Human-inspected vertical intervals at x=160; None means no reliable edge label.
OBSERVATIONS = [
    ('episode-04-train', 0, 'left', (104, 112)),
    ('episode-04-train', 735, 'right', None),
    ('episode-05-train', 588, 'right', (140, 150)),
    ('episode-06-train', 294, 'left', None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--four-view-result', required=True)
    parser.add_argument('--four-view-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if (sha256(args.plan) != args.plan_sha256
            or sha256(args.four_view_result) != args.four_view_sha256):
        raise ValueError('Source evidence changed')
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    sensor = json.loads(Path(plan['sensor_replay']).read_text(encoding='utf8'))
    sources = {e['id']: e for e in sensor['episodes'] if e['role'] == 'train'}
    evidence = json.loads(Path(args.four_view_result).read_text(encoding='utf8'))
    if evidence['status'] != 'completed_four_view_sample_inspection':
        raise ValueError('Four-view inspection required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'four_view_evidence_sha256': args.four_view_sha256, 'actor_deployed': False})
    sheet = Image.new('RGB', (640, 400), 'black')
    draw = ImageDraw.Draw(sheet)
    labels = []
    for i, (episode, index, view_id, y_interval) in enumerate(OBSERVATIONS):
        row = next(r for r in evidence['reports']
                   if r['episode'] == episode and r['event_index'] == index)
        expected = next(v for v in row['views'] if v['view_id'] == view_id)
        root = Path(sources[episode]['recording'])
        with (root/'events.jsonl').open(encoding='utf8') as stream:
            event = next(json.loads(line) for n, line in enumerate(stream) if n == index)
        view = next(v for v in event['camera']['views'] if v['view_id'] == view_id)
        with ZipFile(root/'vision/camera.zip') as archive:
            raw = archive.read(Path(view['file']).name)
        if (hashlib.sha256(raw).hexdigest() != expected['source_bmp_sha256']
                or view['file_sha256'] != expected['source_bmp_sha256']
                or event['camera']['capture_id'] != row['capture_id']):
            raise ValueError('Original side-view evidence mismatch')
        x, y = i % 2 * 320, i // 2 * 200 + 20
        with Image.open(BytesIO(raw)) as image:
            if image.size != (320, 180):
                raise ValueError('Original image shape changed')
            sheet.paste(image.convert('RGB'), (x, y))
        draw.text((x+2, y-18), f'{episode} {index} {view_id}', fill='white')
        if y_interval is not None:
            draw.line((x+160, y+y_interval[0], x+160, y+y_interval[1]),
                      fill=(255, 0, 255), width=3)
        labels.append({'episode': episode, 'event_index': index, 'view': view_id,
            'capture_id': row['capture_id'], 'decision_clock': row['decision_clock'],
            'source_bmp_sha256': expected['source_bmp_sha256'],
            'status': 'uncertain_visible_edge' if y_interval else 'unknown',
            'column_x': 160 if y_interval else None, 'y_inclusive_interval': y_interval,
            'corner_distance_or_direction': 'unknown'})
    image = run/'side-pilot-overlays.png'
    sheet.save(image)
    write_json(run/'result.json', {'status': 'completed_uncertain_side_view_pilot',
        'labels': labels, 'overlay_sha256': sha256(image),
        'manual_not_independently_reviewed': True, 'unseen_edge_is_unknown': True,
        'role': 'train_pilot_only', 'model_fits': 0, 'actor_deployed': False,
        'goal_achieved': False})


if __name__ == '__main__':
    main()
