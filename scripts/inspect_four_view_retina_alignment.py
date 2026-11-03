"""Inspect four original camera views and the matching compound-eye samples."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--alignment', required=True)
    parser.add_argument('--alignment-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256 or sha256(args.alignment) != args.alignment_sha256:
        raise ValueError('Original evidence mismatch')
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    sensor = json.loads(Path(plan['sensor_replay']).read_text(encoding='utf8'))
    alignment = json.loads(Path(args.alignment).read_text(encoding='utf8'))
    sources = {e['id']: e for e in sensor['episodes'] if e['role'] == 'train'}
    chosen = [('episode-04-train', 0), ('episode-04-train', 735),
              ('episode-05-train', 588), ('episode-06-train', 294)]
    view_ids = ['front', 'left', 'right', 'rear']
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'original_alignment_sha256': args.alignment_sha256, 'actor_deployed': False})
    sheet = Image.new('RGB', (1280, 800), 'black')
    draw = ImageDraw.Draw(sheet)
    reports = []
    for row_i, (episode, index) in enumerate(chosen):
        source = sources[episode]
        root = Path(source['recording'])
        with (root/'events.jsonl').open(encoding='utf8') as stream:
            events = [json.loads(line) for line in stream]
        event = events[index]
        saved = next(r for r in alignment['rows']
                     if r['episode'] == episode and r['event_index'] == index)
        response = Path(source['output'])/'responses.npz'
        if sha256(response) != source['responses_sha256']:
            raise ValueError('Original visual response changed')
        with load_array(response, allow_pickle=False) as data:
            capture = int(data['capture_ids'][index])
            clock = float(data['decision_monotonic_seconds'][index])
            retina_shape = list(data['retina_intensity'][index].shape)
            motion_shape = list(data['motion_activity'][index].shape)
            finite = bool(np.isfinite(data['retina_intensity'][index]).all()
                          and np.isfinite(data['motion_activity'][index]).all())
        with load_array(root/'observations.npz', allow_pickle=False) as data:
            view_valid_counts = data['obs_view_valid'][index].sum(axis=1).tolist()
            if float(data['observation_time'][index]) != clock:
                raise ValueError('Observation clock differs')
        registration = Path(source['output'])/'retina-registration.npz'
        with load_array(registration, allow_pickle=False) as data:
            if (data['view_order'].tolist() != view_ids
                    or data['body_directions'].shape != (4, 721, 3)):
                raise ValueError('Original four distinct view geometry required')
        if (event['camera']['capture_id'] != capture or saved['capture_id'] != capture
                or event['decision_monotonic_seconds'] != clock or not finite
                or retina_shape != [4, 721] or motion_shape != [4, 8, 721]):
            raise ValueError('Original capture, shapes or clocks differ')
        view_reports = []
        with ZipFile(root/'vision/camera.zip') as archive:
            for col, view_id in enumerate(view_ids):
                view = next(v for v in event['camera']['views'] if v['view_id'] == view_id)
                raw = archive.read(Path(view['file']).name)
                if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                    raise ValueError('Original camera view hash mismatch: '+view_id)
                with Image.open(BytesIO(raw)) as picture:
                    if picture.size != (320, 180):
                        raise ValueError('Unexpected view geometry')
                    sheet.paste(picture.convert('RGB'), (col*320, row_i*200+20))
                draw.text((col*320+3, row_i*200+3), f'{episode} {index} {view_id}', fill='white')
                view_reports.append({'view_id': view_id, 'source_bmp_sha256': view['file_sha256'],
                                     'valid_compound_eye_samples': view_valid_counts[col]})
        reports.append({'episode': episode, 'event_index': index, 'capture_id': capture,
            'decision_clock': clock, 'views': view_reports,
            'saved_flyvis_retina_shape': retina_shape, 'saved_motion_shape': motion_shape})
        print(json.dumps({'episode': episode, 'index': index,
                          'four_views_verified': True}), flush=True)
    image = run/'four-view-contact-sheet.png'
    sheet.save(image)
    write_json(run/'result.json', {'status': 'completed_four_view_sample_inspection',
        'reports': reports, 'sheet_sha256': sha256(image), 'view_order': view_ids,
        'four_views_not_concatenated_into_one_eye': True, 'student_deployed': False,
        'physiological_function_or_driving_verified': False,
        'exposure_simulation_alignment_verified': False, 'goal_achieved': False})


if __name__ == '__main__':
    main()
